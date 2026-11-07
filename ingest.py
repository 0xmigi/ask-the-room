"""Watches audio/ and frames/ (written by record.sh), sends them to the model provider
(Superlinked Serverless by default, or Darkbloom),
and appends timestamped lines to data/events.jsonl, transcript.md and slides.md.

    uv run ingest.py
"""
import asyncio
import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

import providers

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

AUDIO, FRAMES, DATA = ROOT / "audio", ROOT / "frames", ROOT / "data"
EVENTS = DATA / "events.jsonl"
STATE = DATA / "ingest_state.json"
for d in (AUDIO, FRAMES, DATA):
    d.mkdir(exist_ok=True)


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


REPEAT = re.compile(r"(\b.{6,120}?)(?:[\s,.!?]+\1\b){2,}", re.I | re.S)


def collapse_repeats(text: str) -> str:
    """Whisper sometimes loops on noisy audio ("it's not a lot of money, but it's not a lot of ...")."""
    prev = None
    while prev != text:
        prev, text = text, REPEAT.sub(r"\1", text)
    return text.strip()


async def transcribe(client: httpx.AsyncClient, wav: Path) -> str:
    return collapse_repeats(await providers.transcribe(client, wav))


async def describe(client: httpx.AsyncClient, jpg: Path) -> str:
    text = await providers.describe_image(client, jpg, VISION_PROMPT)
    return "" if text.upper().startswith("NONE") else text


async def main():
    providers.check_key()
    state = load_state()
    done = set(state["done"])
    last_slide = ""
    print(f"[ingest] provider={providers.PROVIDER}; watching {AUDIO.name}/ and {FRAMES.name}/ ...")
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
