"""Phone page + question API. Owns the memory (Cognee) so only one process touches its databases.

    uv run server.py          # http://0.0.0.0:8000
"""
import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import memory  # loads .env and configures Cognee before importing it
import media
import room_mcp
import speakers

ROOT = Path(__file__).parent
COGNIFY_EVERY = int(os.getenv("COGNIFY_EVERY", "180"))



async def _sync_loop():
    while True:
        try:
            changed = await speakers.update()
            if changed:
                print(f"[speakers] timeline updated ({changed} segments)")
        except Exception as e:
            print(f"[speakers] update failed: {e!r}")
        try:
            n = await memory.sync()
            if n:
                print(f"[memory] added {n} new events")
        except Exception as e:
            print(f"[memory] sync failed: {e!r}")
        await asyncio.sleep(COGNIFY_EVERY)


async def _warm_media():
    while True:
        await asyncio.to_thread(media.warm)
        await asyncio.sleep(15)


MCP_APP = room_mcp.asgi_app()


@asynccontextmanager
async def lifespan(app):
    tasks = [asyncio.create_task(_sync_loop()), asyncio.create_task(_warm_media())]
    room_mcp.room_key()  # make sure a key exists before anyone asks for it
    async with room_mcp.mcp.session_manager.run():  # MCP endpoint's session manager
        yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Ask the Room", lifespan=lifespan)


class Ask(BaseModel):
    q: str
    history: list[dict] = []  # earlier turns [{q, a}], so follow-ups like "show me" make sense


@app.get("/api/connect")
def connect():
    """How to plug your own agent into the room's memory (MCP URL, key, ready-to-paste config)."""
    return room_mcp.connect_info()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    events = memory.read_events()
    last = max(events, key=lambda e: (e["date"], e["t"])) if events else None
    age = None
    if last:
        age = time.time() - memory.event_epoch(last)
    return {
        "speech": sum(e["kind"] == "speech" for e in events),
        "slides": sum(e["kind"] == "slide" for e in events),
        "last_t": last["t"] if last else None,
        "last_event_age_s": age,
        "memory": memory.MODE,
        "synced": memory.synced_count(),
        # live = the recorder wrote audio in the last 90 s (quiet stretches still count)
        "recording_age_s": _recording_age(),
    }


def _recording_age():
    wavs = list((ROOT / "audio").glob("*.wav"))
    return time.time() - max(w.stat().st_mtime for w in wavs) if wavs else None


TIME_RE = r"^\d{2}:\d{2}(:\d{2})?$"


async def _media(fn, t: str, date: str | None, mime: str):
    if not re.match(TIME_RE, t) or (date and not re.match(r"^\d{4}-\d{2}-\d{2}$", date)):
        raise HTTPException(400, "t must be HH:MM[:SS], date YYYY-MM-DD")
    try:
        path = await fn(t if t.count(":") == 2 else t + ":00", date)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    return FileResponse(path, media_type=mime, headers={"Cache-Control": "max-age=86400"})


@app.get("/api/frame")
async def frame(t: str, date: str | None = None):
    """The video frame at a moment of the stream (cut on request, then cached)."""
    return await _media(media.frame, t, date, "image/jpeg")


@app.get("/api/clip")
async def clip(t: str, date: str | None = None):
    """A 30 s clip starting at a moment of the stream (cut on request, then cached)."""
    return await _media(media.clip, t, date, "video/mp4")


@app.post("/api/ask")
async def ask(body: Ask):
    q = body.q.strip()
    if not q:
        raise HTTPException(400, "empty question")
    t0 = time.time()
    try:
        result = await memory.answer(q, body.history[-3:])
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    result["ms"] = int((time.time() - t0) * 1000)
    with (ROOT / "data" / "questions.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "q": q, **result}) + "\n")
    return result


# MCP for agents: mounted last so the routes above win; KeyGate checks the room key
app.mount("/", MCP_APP)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
