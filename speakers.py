"""Who was speaking when, worked out from the stream itself.

People introduce themselves ("hi, I'm Kaido from Darkbloom"), hosts hand over ("next up is Tim"),
speakers get thanked when they finish ("thank you so much Kaido"), and slides carry company names.
This reads the transcript + slide lines in overlapping windows, asks the chat model to mark speaker
segments from those cues, and saves a timeline to data/speakers.json. Names only ever come from
the stream; an unnamed presenter is labelled by company or role.

    uv run speakers.py        # rebuild from scratch and print the timeline
"""
import asyncio
import datetime as dt
import json
import re
from pathlib import Path

import httpx

import providers

ROOT = Path(__file__).parent
EVENTS = ROOT / "data" / "events.jsonl"
OUT = ROOT / "data" / "speakers.json"
WINDOW_CHARS = 11000   # stays well inside Qwen3.8-27B's 8k-token window with the prompt
OVERLAP_LINES = 8

PROMPT = """You label who is speaking in a transcript of a hackathon livestream (speech lines plus
descriptions of slides on screen). Find speaker changes from these cues:
- self-introductions: "I'm X", "my name is X", "I work on/at Y", "we are Y"
- host hand-offs: "next up is X", "please welcome X", "our next speaker"
- thank-yous: "thank you so much X" means X just FINISHED speaking before that line
- slides: company names/logos on slides tell you which company the presenter is from
Rules: use a person's name only if it is said or shown in these lines (or given as "previously
speaking"). If a presenter's name is never said, label them by company, e.g. "Superlinked presenter".
Use "Host" for the MC between talks. Whisper may misspell names; use the most common spelling.
Do not label crowd chatter or breaks. Reply with only JSON:
{"segments": [{"start": "HH:MM:SS", "end": "HH:MM:SS", "speaker": "Kaido", "about": "Darkbloom, talk on private inference on Macs"}]}"""


def _events() -> list[dict]:
    if not EVENTS.exists():
        return []
    ev = [json.loads(l) for l in EVENTS.read_text().splitlines() if l.strip()]
    return sorted(ev, key=lambda e: (e["date"], e["t"]))


def _line(e: dict) -> str:
    text = e["text"].replace("\n", " ")[:260]
    return f"[{e['t']}] {'(slide) ' if e['kind'] == 'slide' else ''}{text}"


def load() -> dict:
    try:
        return json.loads(OUT.read_text())
    except Exception:
        return {"segments": [], "through": None}


_cache = {"mtime": None, "segs": []}


def segments() -> list[dict]:
    """Timeline segments, cached until data/speakers.json changes."""
    m = OUT.stat().st_mtime if OUT.exists() else None
    if m != _cache["mtime"]:
        _cache.update(mtime=m, segs=load()["segments"])
    return _cache["segs"]


def version() -> float | None:
    return OUT.stat().st_mtime if OUT.exists() else None


def speaker_at(date: str, t: str) -> str | None:
    best = None
    for s in segments():
        if s.get("date", date) == date and s["start"] <= t <= s["end"]:
            if best is None or (s["end"] > best["end"]):  # later/longer segment wins ties
                best = s
    return best["speaker"] if best else None


def find(name_words: set[str], include_about: bool = False) -> dict | None:
    """The longest segment whose speaker (or, for talk questions, its topic) mentions these words."""
    def text(s):
        if include_about and s["speaker"].lower() != "host":
            return (s["speaker"] + " " + s.get("about", "")).lower()
        return s["speaker"].lower()
    hits = [s for s in segments() if any(re.search(rf"\b{re.escape(w)}", text(s)) for w in name_words)]
    if not hits:
        return None
    dur = lambda s: (dt.datetime.fromisoformat(f"2000-01-01T{s['end']}") -
                     dt.datetime.fromisoformat(f"2000-01-01T{s['start']}")).total_seconds()
    return max(hits, key=dur)


def summary() -> str:
    """Human/LLM-readable timeline, e.g. for Cognee."""
    rows = [f"{s['start'][:5]}-{s['end'][:5]} {s['speaker']}" + (f" ({s['about']})" if s.get("about") else "")
            for s in segments()]
    return "Speaker timeline of the livestream (who was presenting when):\n" + "\n".join(rows)


async def _label(client: httpx.AsyncClient, lines: list[str], previous: str | None) -> list[dict]:
    user = (f"Previously speaking: {previous}\n\n" if previous else "") + "\n".join(lines)
    raw = await providers.chat(client, [{"role": "system", "content": PROMPT},
                                        {"role": "user", "content": user}], max_tokens=900)
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        segs = json.loads(m.group(0))["segments"] if m else []
    except Exception:
        return []
    ok = re.compile(r"^\d{2}:\d{2}:\d{2}$")
    return [s for s in segs if isinstance(s, dict) and ok.match(str(s.get("start", "")))
            and ok.match(str(s.get("end", ""))) and s.get("speaker") and s["start"] <= s["end"]
            and not re.match(r"(?i)^(unlabeled|unknown|none|n/a|crowd|audience|chatter)", str(s["speaker"]))]


def _windows(ev: list[dict]) -> list[list[dict]]:
    out, cur, size = [], [], 0
    for e in ev:
        n = len(_line(e)) + 1
        if cur and size + n > WINDOW_CHARS:
            out.append(cur)
            cur = cur[-OVERLAP_LINES:]
            size = sum(len(_line(x)) + 1 for x in cur)
        cur.append(e)
        size += n
    if cur:
        out.append(cur)
    return out


async def update(full: bool = False) -> int:
    """Label new lines (or everything with full=True). Returns how many segments changed."""
    state = {"segments": [], "through": None} if full else load()
    ev = _events()
    if not full and state["through"]:
        # re-read a little before where we stopped, so a talk spanning the boundary keeps its name
        start_i = next((i for i, e in enumerate(ev) if f"{e['date']}T{e['t']}" > state["through"]), len(ev))
        ev = ev[max(0, start_i - OVERLAP_LINES):]
        if start_i >= len(_events()):
            return 0
    if not ev:
        return 0
    segs = [s for s in state["segments"]]
    added = 0
    async with httpx.AsyncClient() as client:
        for w in _windows(ev):
            date = w[0]["date"]
            prev = next((f"{s['speaker']} ({s.get('about', '')})" for s in reversed(segs)
                         if s.get("date") == date and s["end"] <= w[0]["t"]), None)
            try:
                new = await _label(client, [_line(e) for e in w], prev)
            except Exception as e:
                print(f"[speakers] window {w[0]['t']}-{w[-1]['t']} failed: {str(e)[:150]}")
                continue
            # the newer reading replaces old segments inside this window
            lo, hi = w[0]["t"], w[-1]["t"]
            segs = [s for s in segs if not (s.get("date") == date and lo <= s["start"] <= hi)]
            for s in new:
                s["date"] = date
            segs += new
            added += len(new)
    segs.sort(key=lambda s: (s["date"], s["start"]))
    last = _events()[-1]
    OUT.write_text(json.dumps({"segments": segs, "through": f"{last['date']}T{last['t']}"}, indent=1))
    return added


if __name__ == "__main__":
    n = asyncio.run(update(full=True))
    print(f"{n} segments\n")
    print(summary())
