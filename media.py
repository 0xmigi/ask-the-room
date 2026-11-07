"""Cut a frame or a short video clip from the livestream at a given time, on request.

X/Periscope playlists (live or replay) keep every segment of a broadcast, each stamped with its
clock time (EXT-X-PROGRAM-DATE-TIME), so any moment can be found and cut quickly: fetch only the
few segments around it and run ffmpeg locally. Every stream link ever set as STREAM_URL is
remembered in data/streams.json, so moments from an earlier broadcast keep working after the
organizers start a new one. Results are cached in data/media/.
"""
import asyncio
import bisect
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "media"
OUT.mkdir(parents=True, exist_ok=True)
REGISTRY = ROOT / "data" / "streams.json"
CLIP_SECONDS = 30

_hls: dict[str, tuple[str, float]] = {}      # stream page URL -> (signed HLS URL, fetched at)
_pl: dict[str, tuple[list, float, bool]] = {}  # stream page URL -> (segments, fetched at, ended)
_lock = asyncio.Lock()


def stream_urls() -> list[str]:
    """Current STREAM_URL (re-read from .env, so a new link needs no restart) first, then older ones."""
    current = (dotenv_values(ROOT / ".env").get("STREAM_URL") or "").strip()
    try:
        known = json.loads(REGISTRY.read_text())
    except Exception:
        known = []
    if current and current not in known:
        known.append(current)
        REGISTRY.write_text(json.dumps(known, indent=1))
    return ([current] if current else []) + [u for u in reversed(known) if u != current]


def _hls_url(stream: str) -> str:
    # signed URLs expire; refresh every 5 minutes
    url, at = _hls.get(stream, (None, 0.0))
    if not url or time.time() - at > 300:
        out = subprocess.run(["yt-dlp", "-g", "-f", "live-1200/best[height<=480]/worst", stream],
                             capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        if not out:
            raise RuntimeError(f"couldn't get a video URL for {stream}")
        url = out[-1]
        _hls[stream] = (url, time.time())
    return url


def _segments(stream: str) -> list[tuple[dt.datetime, float, str]]:
    """(start time, duration, absolute URL) for every segment of that broadcast."""
    segs, at, ended = _pl.get(stream, ([], 0.0, False))
    # a finished broadcast never changes; a live one grows, so re-read it every 20 s
    if not segs or (not ended and time.time() - at > 20):
        base = _hls_url(stream)
        m3u8 = httpx.get(base, timeout=30).text
        segs, pdt, dur = [], None, None
        for line in m3u8.splitlines():
            if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                pdt = dt.datetime.fromisoformat(line.split(":", 1)[1].replace("Z", "+00:00")).astimezone()
            elif line.startswith("#EXTINF:"):
                dur = float(line[8:].split(",")[0])
            elif line and not line.startswith("#") and dur is not None:
                start = pdt or (segs[-1][0] + dt.timedelta(seconds=segs[-1][1]))
                segs.append((start, dur, urljoin(base, line)))
                pdt, dur = None, None
        _pl[stream] = (segs, time.time(), "#EXT-X-ENDLIST" in m3u8)
    return segs


def _find(when: dt.datetime) -> list[tuple[dt.datetime, float, str]]:
    """Segments of whichever known broadcast was on air at that moment."""
    errors = []
    for stream in stream_urls():
        try:
            segs = _segments(stream)
        except Exception as e:
            errors.append(f"{stream}: {e}")
            continue
        if segs and segs[0][0] - dt.timedelta(seconds=5) <= when <= segs[-1][0] + dt.timedelta(seconds=10):
            return segs
    raise LookupError("no stream covers that time" + (f" ({'; '.join(errors)})" if errors else ""))


def _parse(t: str, date: str | None) -> dt.datetime:
    d = date or dt.date.today().isoformat()
    return dt.datetime.fromisoformat(f"{d}T{t}").astimezone()


async def _cut(kind: str, t: str, date: str | None) -> Path:
    when = _parse(t, date)
    out = OUT / f"{when:%Y%m%d-%H%M%S}.{'mp4' if kind == 'clip' else 'jpg'}"
    if out.exists() and out.stat().st_size > 0:
        return out
    async with _lock:
        segs = await asyncio.to_thread(_find, when)
        i = max(0, bisect.bisect_right([x[0] for x in segs], when) - 1)
        need = CLIP_SECONDS + 4 if kind == "clip" else 6
        picked, total = [], 0.0
        for seg in segs[i:]:
            picked.append(seg)
            total += seg[1]
            if total >= need:
                break
        # fetch just those few segments in parallel and join them (MPEG-TS concatenates cleanly)
        async with httpx.AsyncClient(timeout=30) as c:
            parts = await asyncio.gather(*(c.get(u) for _, _, u in picked))
        ts = out.with_suffix(".ts")
        ts.write_bytes(b"".join(r.raise_for_status().content for r in parts))
        offset = max(0.0, (when - segs[i][0]).total_seconds())
        if kind == "clip":
            args = ["-ss", f"{offset:.2f}", "-t", str(CLIP_SECONDS), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "28", "-vf", "scale=-2:480", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"]
        else:
            args = ["-ss", f"{offset + 1:.2f}", "-frames:v", "1", "-vf", "scale=-2:540", "-q:v", "4"]
        tmp = out.with_suffix(".part" + out.suffix)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(ts), *args, str(tmp),
            stderr=asyncio.subprocess.PIPE)
        _, err = await asyncio.wait_for(proc.communicate(), timeout=90)
        ts.unlink(missing_ok=True)
        if proc.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"ffmpeg failed: {err.decode()[-300:]}")
        tmp.rename(out)
    return out


def _local_frames() -> list[tuple[dt.datetime, Path]]:
    """Frames already on disk: frames/HHMMSS.jpg from the recorder, plus backfill frames that
    became slide lines (their time is in data/events.jsonl)."""
    today = dt.date.today().isoformat()
    out = []
    for f in (ROOT / "frames").glob("*.jpg"):
        if f.stat().st_size > 1024 and f.stem[:6].isdigit():
            s = f.stem[:6]
            out.append((_parse(f"{s[:2]}:{s[2:4]}:{s[4:6]}", today), f))
    ev_file = ROOT / "data" / "events.jsonl"
    if ev_file.exists():
        for line in ev_file.read_text().splitlines():
            e = json.loads(line)
            if e.get("src", "").startswith("backfill/") and e["src"].endswith(".jpg"):
                f = ROOT / e["src"]
                if f.exists():
                    out.append((_parse(e["t"], e["date"]), f))
    return out


async def frame(t: str, date: str | None = None) -> Path:
    when = _parse(t, date)
    near = [(abs((w - when).total_seconds()), f) for w, f in _local_frames()]
    if near:
        gap, f = min(near)
        if gap <= 40:
            return f  # instant
    return await _cut("frame", t, date)


async def clip(t: str, date: str | None = None) -> Path:
    return await _cut("clip", t, date)


def warm():
    """Keep the current stream's signed URL and playlist fresh so the first request is quick."""
    try:
        urls = stream_urls()
        if urls:
            _segments(urls[0])
    except Exception as e:
        print(f"[media] warm-up failed: {e!r}")
