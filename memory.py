"""The room's memory.

- Cognee (graph + vector memory) is the main store. Its LLM is Darkbloom, its embeddings are
  Superlinked SIE running on this machine.
- A small SIE-only index over the same events picks the exact timestamped source lines shown
  under each answer, and answers by itself if Cognee errors (or MEMORY=simple).
"""
import asyncio
import datetime as dt
import json
import os
import re
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
EVENTS = DATA / "events.jsonl"
STATE = DATA / "memory_state.json"

KEY = os.getenv("DARKBLOOM_API_KEY", "")
BASE = os.getenv("DARKBLOOM_BASE_URL", "https://api.darkbloom.dev/v1").rstrip("/")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-oss-20b")
SIE_URL = os.getenv("SIE_URL", "http://localhost:8080").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Snowflake/snowflake-arctic-embed-s")
EMBED_DIMS = os.getenv("EMBED_DIMS", "384")
DATASET = os.getenv("DATASET", "room")
MODE = os.getenv("MEMORY", "cognee")

# --- Point Cognee at Darkbloom (LLM) + Superlinked SIE (embeddings), store data in the repo ---
for k, v in {
    "LLM_PROVIDER": "custom",
    "LLM_MODEL": f"openai/{ANSWER_MODEL}",
    "LLM_ENDPOINT": BASE,
    "LLM_API_KEY": KEY or ".",
    "EMBEDDING_PROVIDER": "openai_compatible",
    "EMBEDDING_MODEL": EMBED_MODEL,
    "EMBEDDING_ENDPOINT": f"{SIE_URL}/v1",
    "EMBEDDING_API_KEY": ".",
    "EMBEDDING_DIMENSIONS": EMBED_DIMS,
    "SYSTEM_ROOT_DIRECTORY": str(ROOT / ".cognee" / "system"),
    "DATA_ROOT_DIRECTORY": str(ROOT / ".cognee" / "data"),
    # one local process, one user: skip multi-tenant auth and the Q&A session cache
    "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    "CACHING": "false",
}.items():
    os.environ.setdefault(k, v)

if MODE == "cognee":
    import cognee
    from cognee import SearchType

ANSWER_PROMPT = (
    "You are the shared memory of a live event. You are given timestamped lines from the "
    "livestream: what speakers said and what was on screen. Answer the question in 1-4 short "
    "sentences using only that material. Always cite the time each fact comes from like "
    "[13:42]. If the material doesn't cover it, say you didn't hear that yet."
)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction


# ---------- events ----------
def read_events() -> list[dict]:
    if not EVENTS.exists():
        return []
    out = []
    for line in EVENTS.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def event_epoch(e: dict) -> float:
    return dt.datetime.fromisoformat(f'{e["date"]}T{e["t"]}').timestamp()


def fmt(e: dict) -> str:
    hhmm = e["t"][:5]
    if e["kind"] == "slide":
        return f"[{hhmm}] On screen: {e['text']}"
    who = e.get("speaker") or "Speaker"
    return f"[{hhmm}] {who} said: {e['text']}"


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"synced": 0}


def synced_count() -> int:
    return _state()["synced"]


# ---------- Superlinked SIE embeddings ----------
async def embed(texts: list[str]) -> np.ndarray:
    vecs = []
    async with httpx.AsyncClient(timeout=60) as c:
        for i in range(0, len(texts), 64):
            r = await c.post(f"{SIE_URL}/v1/embeddings",
                             json={"model": EMBED_MODEL, "input": texts[i:i + 64]})
            r.raise_for_status()
            vecs += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
    m = np.array(vecs, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class Index:
    def __init__(self):
        self.events: list[dict] = []
        self.vecs = np.zeros((0, int(EMBED_DIMS)), dtype=np.float32)
        self.lock = asyncio.Lock()

    async def refresh(self):
        async with self.lock:
            events = read_events()
            new = events[len(self.events):]
            if new:
                self.vecs = np.vstack([self.vecs, await embed([fmt(e) for e in new])])
                self.events = events

    async def top(self, q: str, k: int = 10) -> list[dict]:
        await self.refresh()
        if not self.events:
            return []
        qv = (await embed([QUERY_PREFIX + q]))[0]
        scores = self.vecs @ qv
        return [self.events[i] for i in np.argsort(-scores)[:k]]


INDEX = Index()


# ---------- Darkbloom answer ----------
async def llm_answer(q: str, lines: list[str]) -> str:
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": ANSWER_MODEL, "max_tokens": 600, "messages": [
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": "Livestream lines:\n" + "\n".join(lines) + f"\n\nQuestion: {q}"},
            ]},
        )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


# ---------- public API used by server.py ----------
async def sync() -> int:
    """Push new events into Cognee (add + cognify) and the SIE index. Returns # of new events."""
    await INDEX.refresh()
    events = read_events()
    st = _state()
    new = events[st["synced"]:]
    if not new:
        return 0
    if MODE == "cognee":
        doc = (f"Livestream notes {new[0]['date']}, {new[0]['t'][:5]}-{new[-1]['t'][:5]}\n"
               + "\n".join(fmt(e) for e in new))
        await cognee.add(doc, dataset_name=DATASET)
        await cognee.cognify(datasets=[DATASET])
    st["synced"] = len(events)
    STATE.write_text(json.dumps(st))
    return len(new)


def _flatten(results) -> str:
    parts = []
    for r in results or []:
        v = getattr(r, "search_result", r)
        if isinstance(v, list):
            parts += [x if isinstance(x, str) else json.dumps(x, default=str) for x in v]
        else:
            parts.append(v if isinstance(v, str) else json.dumps(v, default=str))
    return "\n".join(parts).strip()


async def answer(q: str) -> dict:
    hits = await INDEX.top(q, k=10)
    recent = INDEX.events[-8:]  # helps "what's on screen now / last 10 minutes" questions
    used = MODE
    text = ""
    if MODE == "cognee" and synced_count() > 0:
        try:
            res = await cognee.search(query_text=q, query_type=SearchType.HYBRID_COMPLETION,
                                      datasets=[DATASET], system_prompt=ANSWER_PROMPT, top_k=12)
            text = _flatten(res)
        except Exception as e:
            print(f"[memory] cognee search failed, using SIE fallback: {e!r}")
            used = "simple (cognee error)"
    elif MODE == "cognee":
        used = "simple (cognee warming up)"
    if not text:
        ctx = {id(e): e for e in hits + recent}.values()
        lines = [fmt(e) for e in sorted(ctx, key=lambda e: e["t"])]
        text = await llm_answer(q, lines) if lines else "Nothing recorded yet."

    # Sources: lines whose timestamps the answer cites, else the best SIE matches
    cited = set(re.findall(r"\[(\d{1,2}:\d{2})", text))
    pool = INDEX.events if cited else hits[:3]
    sources = [e for e in pool if e["t"][:5] in cited][:6] if cited else pool
    return {"answer": text, "memory": used,
            "sources": [{"t": e["t"], "kind": e["kind"], "text": e["text"]} for e in sources]}
