"""The room's memory as an MCP server, so people's own agents (Claude Code, Cursor, ...) can use it.

Mounted by server.py at /mcp (streamable HTTP). Every request needs the room key:
    Authorization: Bearer <ROOM_KEY>     (or ?key=<ROOM_KEY>)
The key lives in .env as ROOM_KEY (generated on first run) and is shown on the chat page under
"Connect your agent", so being in the room is what gets you in.
"""
import datetime as dt
import json
import os
import secrets
import socket
from pathlib import Path

from mcp.server.mcpserver import MCPServer

import memory
import speakers

ROOT = Path(__file__).parent
PORT = int(os.getenv("PORT", "8000"))


def room_key() -> str:
    key = os.getenv("ROOM_KEY", "").strip()
    if not key:
        key = "room-" + secrets.token_urlsafe(9)
        with (ROOT / ".env").open("a") as f:
            f.write(f"\n# Key agents need for the /mcp endpoint (shown on the chat page)\nROOM_KEY={key}\n")
        os.environ["ROOM_KEY"] = key
    return key


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the outgoing interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def base_url() -> str:
    return os.getenv("PUBLIC_URL", f"http://{lan_ip()}:{PORT}").rstrip("/")


def _links(t: str, date: str) -> dict:
    q = f"t={t}&date={date}"
    return {"frame": f"{base_url()}/api/frame?{q}", "clip": f"{base_url()}/api/clip?{q}"}


def _line(e: dict) -> dict:
    return {"time": e["t"], "date": e["date"], "kind": e["kind"],
            "speaker": speakers.speaker_at(e["date"], e["t"]), "text": e["text"]}


mcp = MCPServer(
    name="ask-the-room",
    title="Ask the Room: LOCALHOST hackathon memory",
    instructions=(
        "Shared memory of the LOCALHOST AI hackathon livestream (Nov 7 2026, Brooklyn): everything "
        "said on stage and every slide shown, with timestamps and speaker names. Use `ask` for a "
        "quick answer, `search` / `transcript` for raw timestamped lines to reason over yourself, "
        "`recent` for what just happened, `speakers` for who presented when, and `moment` for a "
        "link to the video frame and a 30 s clip at a given time."
    ),
)


@mcp.tool()
async def ask(question: str) -> str:
    """Ask the room's memory a question about the livestream. Returns an answer citing [HH:MM] times,
    the source lines, and frame/clip links when the question is about seeing something."""
    r = await memory.answer(question)
    out = {"answer": r["answer"], "sources": r.get("sources", [])}
    if r.get("media"):
        out["moment"] = {"time": r["media"]["t"], **_links(r["media"]["t"], r["media"]["date"])}
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
async def search(query: str, limit: int = 12) -> str:
    """Semantic search over everything said or shown on the stream. Returns timestamped lines
    (speech and slides) with speaker names, best matches first."""
    hits = await memory.INDEX.top(query, k=max(1, min(limit, 40)))
    return json.dumps([_line(e) for e in hits], ensure_ascii=False)


@mcp.tool()
async def recent(minutes: int = 10) -> str:
    """Everything said or shown in the last N minutes of the stream, oldest first."""
    await memory.INDEX.refresh()
    cutoff = dt.datetime.now().timestamp() - max(1, min(minutes, 240)) * 60
    ev = sorted((e for e in memory.INDEX.events if memory.event_epoch(e) >= cutoff),
                key=lambda e: (e["date"], e["t"]))
    return json.dumps([_line(e) for e in ev], ensure_ascii=False)


@mcp.tool()
async def transcript(start: str, end: str, date: str | None = None) -> str:
    """Raw timestamped lines between two local times, e.g. start='11:00', end='11:15'."""
    await memory.INDEX.refresh()
    d = date or dt.date.today().isoformat()
    s, e_ = (start + ":00")[:8], (end + ":59")[:8]
    ev = sorted((e for e in memory.INDEX.events if e["date"] == d and s <= e["t"] <= e_),
                key=lambda e: e["t"])
    return json.dumps([_line(e) for e in ev][:300], ensure_ascii=False)


@mcp.tool()
async def speakers_timeline() -> str:
    """Who was presenting when (from self-introductions, host hand-offs and thank-yous on stream)."""
    return json.dumps(speakers.segments(), ensure_ascii=False)


@mcp.tool()
async def moment(time: str, date: str | None = None) -> str:
    """Links to the video frame and a 30 s clip of the stream at a local time like '11:05' or '11:05:30'."""
    t = time if time.count(":") == 2 else time + ":00"
    d = date or dt.date.today().isoformat()
    return json.dumps({"time": t, "speaker": speakers.speaker_at(d, t), **_links(t, d)})


class KeyGate:
    """Tiny ASGI wrapper: rejects MCP requests without the room key (header or ?key=)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            key = room_key()
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode()
            qs = scope.get("query_string", b"").decode()
            if auth != f"Bearer {key}" and f"key={key}" not in qs.split("&"):
                body = json.dumps({"error": "room key required: Authorization: Bearer <ROOM_KEY>"}).encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def asgi_app():
    # host="0.0.0.0": accept requests addressed to the laptop's LAN IP, not only localhost
    return KeyGate(mcp.streamable_http_app(streamable_http_path="/mcp", stateless_http=True,
                                           json_response=True, host="0.0.0.0"))


def connect_info() -> dict:
    url = f"{base_url()}/mcp"
    key = room_key()
    return {
        "url": url,
        "key": key,
        "claude_code": f'claude mcp add --transport http ask-the-room {url} --header "Authorization: Bearer {key}"',
        "json": {"mcpServers": {"ask-the-room": {"url": url, "headers": {"Authorization": f"Bearer {key}"}}}},
    }
