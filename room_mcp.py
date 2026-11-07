"""The room's memory as an MCP server, so people's own agents (Claude Code, Cursor, ...) can use it.

Mounted by server.py at /mcp (streamable HTTP). Every request needs the room key:
    Authorization: Bearer <ROOM_KEY>     (or ?key=<ROOM_KEY>)
The key lives in .env as ROOM_KEY (generated on first run) and is shown on the chat page under
"Connect your agent", so being in the room is what gets you in.
"""
import datetime as dt
import json
import os
import base64
import secrets
import socket
from urllib.parse import quote, urlencode
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer

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
        "said on stage and every slide shown, with timestamps and speaker names. These tools return "
        "raw material; you do the reasoning. Use `search` to find relevant lines, `transcript` for a "
        "time range, `recent` for what just happened, `speakers_timeline` for who presented when, and "
        "`frame` to look at the stream yourself at a moment (slides, the presenter). `moment` gives "
        "links to share (frame + 30 s clip). Cite times like [11:05] when you answer."
    ),
)


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


@mcp.tool()
async def frame(time: str, date: str | None = None) -> Image:
    """The livestream video frame at a local time like '11:05' (slides on screen, the presenter's
    camera window). Look at it with your own vision to answer questions about what was shown."""
    import media
    t = time if time.count(":") == 2 else time + ":00"
    path = await media.frame(t, date or dt.date.today().isoformat())
    return Image(path=path)


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
    """Everything the 'Connect your agent' panel needs. The key rides in the URL so a single link
    works in every client (KeyGate accepts ?key= as well as the Authorization header)."""
    key = room_key()
    url = f"{base_url()}/mcp?key={key}"
    cfg = {"url": url}
    cursor = "cursor://anysphere.cursor-deeplink/mcp/install?" + urlencode(
        {"name": "ask-the-room", "config": base64.b64encode(json.dumps(cfg).encode()).decode()})
    vscode = "vscode:mcp/install?" + quote(json.dumps({"name": "ask-the-room", "type": "http", "url": url}))
    return {
        "url": url,
        "key": key,
        "cursor": cursor,
        "vscode": vscode,
        "claude_code": f'claude mcp add --transport http ask-the-room "{url}"',
        "json": {"mcpServers": {"ask-the-room": cfg}},
        "context": f"{base_url()}/context.md?key={key}",
    }


def context_markdown() -> str:
    """The whole room as one Markdown file, for AIs that can't connect (paste or upload it)."""
    events = sorted(memory.read_events(), key=lambda e: (e["date"], e["t"]))
    out = ["# LOCALHOST AI hackathon: room memory",
           f"Exported {dt.datetime.now():%Y-%m-%d %H:%M}. Every line is timestamped (local time). "
           "Speech lines say who was speaking; 'On screen' lines describe slides.", "",
           "## Who presented when", ""]
    out += [f"- {sg['start'][:5]}-{sg['end'][:5]} **{sg['speaker']}**" + (f": {sg['about']}" if sg.get("about") else "")
            for sg in speakers.segments()] or ["(not available yet)"]
    out += ["", "## Everything said and shown", ""]
    out += [memory.fmt(e).replace("\n", " ") for e in events]
    return "\n".join(out) + "\n"
