"""Watches audio/ and frames/ (written by record.sh), sends them to Darkbloom,
and appends timestamped lines to data/events.jsonl, transcript.md and slides.md.

    uv run ingest.py
"""
import asyncio
import base64
import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

AUDIO, FRAMES, DATA = ROOT / "audio", ROOT / "frames", ROOT / "data"
EVENTS = DATA / "events.jsonl"
STATE = DATA / "ingest_state.json"
for d in (AUDIO, FRAMES, DATA):
    d.mkdir(exist_ok=True)

BASE = os.getenv("DARKBLOOM_BASE_URL", "https://api.darkbloom.dev/v1").rstrip("/")
KEY = os.getenv("DARKBLOOM_API_KEY", "")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.6-35b-a3b-vl-mtp-mxfp8")
HEADERS = {"Authorization": f"Bearer {KEY}"}

# Whisper likes to invent these on silence / music
HALLUCINATIONS = re.compile(
    r"^(thank you( for watching)?|thanks for watching|you|bye|\.+|music|\[.*\])[.!]?$", re.I
)

VISION_PROMPT = (
    "This is a frame from a livestream of a hackathon. Ignore the broadcast overlay: the "
    "LIVE / CAM / LOCALHOST branding, the clock, the mic meter, corner brackets, status text "
    "like 'SCOUT // HOLDING', and the thermal-camera colour effect. If the frame shows a "
    "presentation slide, a shared screen, a demo or a sign with real content, describe it in "
    "at most 80 words and copy the key text exactly (titles, names, numbers, URLs). Otherwise "
    "(just people, a stage, a room, or only the overlay) reply with exactly NONE."
)


def load_schedule():
    try:
        return json.loads((ROOT / "schedule.json").read_text())
    except Exception:
        return []


def speaker_for(hhmm: str, schedule) -> str | None:
    for s in schedule:
        if s["from"] <= hhmm < s["to"]:
            return f'{s["speaker"]} ({s["session"]})'
    return None


def stamp(path: Path) -> str:
    """audio/134200.wav -> '13:42:00' (local wall-clock time the chunk started)."""
    s = path.stem[:6]
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"done": []}


def save_state(state):
    STATE.write_text(json.dumps(state))


def append_event(ev: dict):
    with EVENTS.open("a") as f:
        f.write(json.dumps(ev) + "\n")
    hhmm = ev["t"][:5]
    who = f' **{ev["speaker"]}:**' if ev.get("speaker") else ""
    if ev["kind"] == "speech":
        with (ROOT / "transcript.md").open("a") as f:
            f.write(f"[{hhmm}]{who} {ev['text']}\n\n")
    else:
        with (ROOT / "slides.md").open("a") as f:
            f.write(f"[{hhmm}] {ev['text']}\n\n")


def ready(path: Path, newest: Path | None) -> bool:
    """A file is complete once ffmpeg has started a newer one, or it hasn't changed in 75 s."""
    if newest is not None and path != newest:
        return True
    return time.time() - path.stat().st_mtime > 75


async def transcribe(client: httpx.AsyncClient, wav: Path) -> str:
    with wav.open("rb") as f:
        r = await client.post(
            f"{BASE}/audio/transcriptions",
            headers=HEADERS,
            files={"file": (wav.name, f, "audio/wav")},
            data={"model": TRANSCRIBE_MODEL},
            timeout=120,
        )
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


async def describe(client: httpx.AsyncClient, jpg: Path) -> str:
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    r = await client.post(
        f"{BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": VISION_MODEL,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
        },
        timeout=120,
    )
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    return "" if text.upper().startswith("NONE") else text


async def main():
    if not KEY or "REPLACE_ME" in KEY:
        raise SystemExit("Set DARKBLOOM_API_KEY in .env first")
    state = load_state()
    done = set(state["done"])
    last_slide = ""
    print(f"[ingest] watching {AUDIO.name}/ and {FRAMES.name}/ ...")
    async with httpx.AsyncClient() as client:
        while True:
            schedule = load_schedule()
            today = dt.date.today().isoformat()
            wavs = sorted(AUDIO.glob("*.wav"))
            jpgs = sorted(FRAMES.glob("*.jpg"))
            todo = [p for p in wavs if p.name not in done and ready(p, wavs[-1])]
            todo += [p for p in jpgs if p.name not in done and ready(p, jpgs[-1])]

            for p in todo:
                t = stamp(p)
                try:
                    if p.suffix == ".wav":
                        text = await transcribe(client, p)
                        if len(text) > 3 and not HALLUCINATIONS.match(text):
                            append_event({"date": today, "t": t, "kind": "speech",
                                          "speaker": speaker_for(t[:5], schedule),
                                          "text": text, "src": f"audio/{p.name}"})
                            print(f"[ingest] {t} speech: {text[:90]}")
                    else:
                        text = await describe(client, p)
                        # skip empty frames and the same slide shown again
                        if text and text[:120] != last_slide[:120]:
                            last_slide = text
                            append_event({"date": today, "t": t, "kind": "slide",
                                          "text": text, "src": f"frames/{p.name}"})
                            print(f"[ingest] {t} slide: {text[:90]}")
                    done.add(p.name)
                except httpx.HTTPStatusError as e:
                    print(f"[ingest] {p.name}: HTTP {e.response.status_code} {e.response.text[:200]}")
                except Exception as e:  # network blips: retry next loop
                    print(f"[ingest] {p.name}: {e!r}")
                state["done"] = sorted(done)
                save_state(state)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
