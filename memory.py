"""The room's memory.

- Cognee (graph + vector memory) is the main store. Its LLM is the chat provider (Superlinked
  Serverless by default), its embeddings are Superlinked SIE running on this machine.
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

import providers

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
EVENTS = DATA / "events.jsonl"
STATE = DATA / "memory_state.json"

SIE_URL = os.getenv("SIE_URL", "http://localhost:8080").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Snowflake/snowflake-arctic-embed-s")
EMBED_DIMS = os.getenv("EMBED_DIMS", "384")
DATASET = os.getenv("DATASET", "room")
MODE = os.getenv("MEMORY", "cognee")

# --- Point Cognee at the chat provider (LLM) + local Superlinked SIE (embeddings) ---
for k, v in {
    "LLM_PROVIDER": "custom",
    "LLM_MODEL": f"openai/{providers.CHAT_MODEL}",
    "LLM_ENDPOINT": providers.BASE,
    "LLM_API_KEY": providers.KEY or ".",
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
    # graph extraction returns a big JSON; the default cap cut it off mid-object
    "LLM_MAX_COMPLETION_TOKENS": "16384",
}.items():
    os.environ.setdefault(k, v)

if MODE == "cognee":
    import cognee
    from cognee import SearchType

BASE_PROMPT = (
    "You are the shared memory of a live event that is still being livestreamed. You are given "
    "timestamped lines from the stream: what speakers said and what was on screen. Answer the "
    "question in 1-4 short sentences using only that material. Always cite the time each fact "
    "comes from like [13:42]. If the material doesn't cover it, say you didn't hear that yet. "
    "The stream has NOT ended: the newest line is just the latest minute processed, so never "
    "say the stream concluded or finished."
)


def answer_prompt() -> str:
    return BASE_PROMPT + f" The time now is {dt.datetime.now():%H:%M}."


ANSWER_PROMPT = BASE_PROMPT  # kept for imports

RECENT = re.compile(r"\blast\s+(\d+|few|couple of)?\s*(min|minute)s?\b|\b(right now|just now|currently|"
                    r"latest|most recent|happening now|on screen now|what'?s happening)\b", re.I)
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


# ---------- answer (direct path) ----------
MAX_CTX_CHARS = 14000  # Superlinked's Qwen3.8-27B has an 8k-token window; ~4k tokens of lines is safe


def fit(lines: list[str], max_chars: int = MAX_CTX_CHARS, keep: str = "end") -> list[str]:
    """Shorten long lines and drop lines until the total fits the model's context window."""
    lines = [l if len(l) <= 300 else l[:297] + "..." for l in lines]
    while lines and sum(len(l) + 1 for l in lines) > max_chars:
        lines.pop(0 if keep == "end" else -1)
    return lines


async def llm_answer(q: str, lines: list[str]) -> str:
    async with httpx.AsyncClient() as c:
        return await providers.chat(c, [
            {"role": "system", "content": answer_prompt()},
            {"role": "user", "content": "Livestream lines:\n" + "\n".join(fit(lines)) + f"\n\nQuestion: {q}"},
        ])


# ---------- public API used by server.py ----------
SYNC_BATCH = 40  # events per add+cognify round
DOC_LINES = 8    # lines per Cognee document (keeps graph-extraction JSON small)


async def sync() -> int:
    """Push new events into Cognee (add + cognify) in small batches. Returns # of events synced."""
    await INDEX.refresh()
    events = read_events()
    st = _state()
    total = 0
    while st["synced"] < len(events):
        batch = events[st["synced"]:st["synced"] + SYNC_BATCH]
        if MODE == "cognee":
            ordered = sorted(batch, key=lambda e: (e["date"], e["t"]))
            docs = [f"Livestream notes {part[0]['date']}, {part[0]['t'][:5]}-{part[-1]['t'][:5]}\n"
                    + "\n".join(fmt(e) for e in part)
                    for part in (ordered[i:i + DOC_LINES] for i in range(0, len(ordered), DOC_LINES))]
            for attempt in (1, 2):
                try:
                    await cognee.add(docs, dataset_name=DATASET)
                    await cognee.cognify(datasets=[DATASET])
                    break
                except Exception as e:
                    print(f"[memory] cognify failed (attempt {attempt}): {str(e)[:200]}")
            # after two failures we move on; the SIE index still has these lines
        st["synced"] += len(batch)
        STATE.write_text(json.dumps(st))
        total += len(batch)
        print(f"[memory] synced {st['synced']}/{len(events)}")
    return total


def _flatten(results) -> str:
    parts = []
    for r in results or []:
        v = getattr(r, "search_result", r)
        if isinstance(v, list):
            parts += [x if isinstance(x, str) else json.dumps(x, default=str) for x in v]
        else:
            parts.append(v if isinstance(v, str) else json.dumps(v, default=str))
    return "\n".join(parts).strip()


VISUAL = re.compile(r"\b(show( me)?|footage|clip|video|watch|picture|photo|image|screenshot|frame)\b|"
                    r"\blooks? like\b|\bwhat did .{1,40} look\b|\bwhat (is|was) .{1,30} wearing\b", re.I)
WANTS_CLIP = re.compile(r"\b(footage|clip|video|watch|play)\b", re.I)

PICK_PROMPT = (
    "The user wants to SEE something from a livestream. Below are timestamped lines from it. "
    "Pick the single moment whose video frame would best show what they ask for. If they name a "
    "person, pick a time while that person was presenting or speaking (people are usually thanked "
    "right AFTER their talk, so pick a few minutes earlier, during their slides). If they ask for a "
    "slide or topic, pick when it was on screen. Reply with only JSON like {\"t\": \"HH:MM:SS\"}."
)


STOP = {"what", "does", "look", "like", "looks", "show", "footage", "clip", "video", "picture", "photo",
        "image", "screenshot", "frame", "from", "with", "that", "this", "when", "where", "they", "them",
        "their", "talk", "about", "give", "please", "watch", "see", "stream", "slide", "slides", "the"}


def _look_prompt(q: str, t: str, named: bool = False) -> str:
    hint = (" The transcript says the person asked about was presenting around this time, so they are "
            "most likely the presenter visible here; describe that presenter.") if named else ""
    return (f"A viewer of a hackathon livestream asked: \"{q}\". This is the stream at {t[:5]}.{hint} "
            "Answer from what is visible, in 1-3 sentences. The presenter is often in the small "
            "camera window in a corner. Describe appearance and what is on screen. If the frame "
            "doesn't show what was asked, say what it does show.")


async def visual_answer(q: str, by_time: list[dict]) -> dict:
    """'Show me X' / 'what does X look like': pick the moment, look at the frame, attach media."""
    import media  # local import: media is only needed for these questions
    hits = await INDEX.top(q, k=16)
    # names ("kaido") are rare words that embeddings rank poorly: add exact-word matches, plus the
    # 12 minutes before each one (people get thanked after their talk, so the talk is before)
    words = {w for w in re.findall(r"[a-z][a-z0-9'-]{3,}", q.lower()) if w not in STOP}
    named = [e for e in by_time if any(re.search(rf"\b{re.escape(w)}\b", e["text"].lower()) for w in words)]
    around = [e for n in named[:3] for e in by_time
              if 0 <= event_epoch(n) - event_epoch(e) <= 720]
    pool, seen_keys = [], set()
    for e in named + around + hits:  # priority order
        k = (e["date"], e["t"], e["kind"])
        if k not in seen_keys:
            seen_keys.add(k)
            pool.append(e)
    chosen, size = [], 0
    for e in pool:
        line = fmt(e)[:300]
        if size + len(line) > MAX_CTX_CHARS:
            break
        chosen.append(e)
        size += len(line) + 1
    lines = [fmt(e)[:300] for e in sorted(chosen, key=lambda e: (e["date"], e["t"]))]
    async with httpx.AsyncClient() as c:
        raw = await providers.chat(c, [{"role": "system", "content": PICK_PROMPT},
                                       {"role": "user", "content": "\n".join(lines) + f"\n\nUser asks: {q}"}],
                                   max_tokens=60)
    m = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", raw or "")
    t = m.group(1) if m else hits[0]["t"]
    t = t if t.count(":") == 2 else t + ":00"
    t = f"{int(t[:t.index(':')]):02d}{t[t.index(':'):]}"
    date = (hits[0] if hits else by_time[-1])["date"]
    try:
        frame = await media.frame(t, date)
        async with httpx.AsyncClient() as c:
            seen = await providers.describe_image(c, frame, _look_prompt(q, t, bool(named)))
    except Exception as e:
        print(f"[memory] visual answer: no frame for {t}: {e!r}")
        seen = "I couldn't get the video for that moment."
    near = [e for e in by_time if abs(event_epoch(e) - dt.datetime.fromisoformat(f"{date}T{t}").timestamp()) <= 90]
    return {"answer": f"{seen} [{t[:5]}]", "memory": "vision",
            "media": {"t": t, "date": date, "play": bool(WANTS_CLIP.search(q))},
            "sources": [{"t": e["t"], "date": e["date"], "kind": e["kind"], "text": e["text"]} for e in near[:4]]}


async def answer(q: str) -> dict:
    await INDEX.refresh()
    by_time = sorted(INDEX.events, key=lambda e: (e["date"], e["t"]))
    used, text = MODE, ""
    if VISUAL.search(q) and by_time:
        return await visual_answer(q, by_time)

    m = RECENT.search(q)
    lagging = len(INDEX.events) - synced_count() > 20
    if m:
        # "last N minutes" / "right now": answer from the time window directly, newest lines
        n = m.group(1)
        mins = int(n) if n and n.isdigit() else 10
        cutoff = dt.datetime.now().timestamp() - mins * 60
        window = [e for e in by_time if event_epoch(e) >= cutoff] or by_time[-8:]
        text = await llm_answer(q, [fmt(e) for e in window[-60:]])
        used = "recent window"
    elif MODE == "cognee" and synced_count() > 0 and not lagging:
        try:
            res = await cognee.search(query_text=q, query_type=SearchType.HYBRID_COMPLETION,
                                      datasets=[DATASET], system_prompt=answer_prompt(), top_k=12)
            text = _flatten(res)
            if "didn't hear" in text.lower() or "did not hear" in text.lower():
                text, used = "", "simple (cognee had nothing)"
        except Exception as e:
            print(f"[memory] cognee search failed, using SIE fallback: {e!r}")
            used = "simple (cognee error)"
    elif MODE == "cognee":
        used = "simple (cognee catching up)"
    if not text:
        hits = await INDEX.top(q, k=14)
        ctx = {id(e): e for e in hits + by_time[-6:]}.values()
        lines = [fmt(e) for e in sorted(ctx, key=lambda e: (e["date"], e["t"]))]
        text = await llm_answer(q, lines) if lines else "Nothing recorded yet."

    # Sources: only the lines whose timestamps the answer actually cites
    cited = set(re.findall(r"(?<![\d:])(\d{1,2}:\d{2})(?!:\d)", text))
    sources = [e for e in by_time if e["t"][:5] in cited][:6]
    return {"answer": text, "memory": used,
            "sources": [{"t": e["t"], "date": e["date"], "kind": e["kind"], "text": e["text"]} for e in sources]}
