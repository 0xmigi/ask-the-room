"""Phone page + question API. Owns the memory (Cognee) so only one process touches its databases.

    uv run server.py          # http://0.0.0.0:8000
"""
import asyncio
import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import memory  # loads .env and configures Cognee before importing it

ROOT = Path(__file__).parent
COGNIFY_EVERY = int(os.getenv("COGNIFY_EVERY", "180"))

app = FastAPI(title="Ask the Room")


class Ask(BaseModel):
    q: str


@app.on_event("startup")
async def start_sync_loop():
    async def loop():
        while True:
            try:
                n = await memory.sync()
                if n:
                    print(f"[memory] added {n} new events")
            except Exception as e:
                print(f"[memory] sync failed: {e!r}")
            await asyncio.sleep(COGNIFY_EVERY)

    asyncio.create_task(loop())


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
    }


@app.post("/api/ask")
async def ask(body: Ask):
    q = body.q.strip()
    if not q:
        raise HTTPException(400, "empty question")
    t0 = time.time()
    try:
        result = await memory.answer(q)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    result["ms"] = int((time.time() - t0) * 1000)
    with (ROOT / "data" / "questions.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "q": q, **result}) + "\n")
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
