"""Cut a frame or a short video clip from the livestream at a given time, on request.

The live playlist keeps every segment since the stream started, each stamped with its clock time
(EXT-X-PROGRAM-DATE-TIME), so any moment of the day can be found and cut with ffmpeg in a few
seconds. Results are cached in data/media/.
"""
import asyncio
import bisect
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "media"
OUT.mkdir(parents=True, exist_ok=True)
STREAM_URL = os.getenv("STREAM_URL", "")
CLIP_SECONDS = 30

_hls = {"url": None, "at": 0.0}
_pl = {"segs": [], "at": 0.0}
_lock = asyncio.Lock()


def _hls_url() -> str:
    # the signed URL expires after a while; refresh it every 5 minutes
    if not _hls["url"] or time.time() - _hls["at"] > 300:
        out = subprocess.run(["yt-dlp", "-g", "-f", "live-1200/worst", STREAM_URL],
                             capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        if not out:
            raise RuntimeError("couldn't get the stream URL (is the stream still up?)")
        _hls.update(url=out[-1], at=time.time())
    return _hls["url"]


def _segments() -> list[tuple[dt.datetime, float, str]]:
    """(start time, duration, absolute URL) for every segment since the stream started."""
    if not _pl["segs"] or time.time() - _pl["at"] > 20:
        base = _hls_url()
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
        _pl.update(segs=segs, at=time.time())
    return _pl["segs"]


def _parse(t: str, date: str | None) -> dt.datetime:
    d = date or dt.date.today().isoformat()
    return dt.datetime.fromisoformat(f"{d}T{t}").astimezone()


async def _cut(kind: str, t: str, date: str | None) -> Path:
    when = _parse(t, date)
    out = OUT / f"{when:%Y%m%d-%H%M%S}.{'mp4' if kind == 'clip' else 'jpg'}"
    if out.exists() and out.stat().st_size > 0:
        return out
    async with _lock:
        segs = await asyncio.to_thread(_segments)
        starts = [x[0] for x in segs]
        if not segs or when < starts[0] - dt.timedelta(seconds=5):
            raise LookupError("that time is before the stream started")
        i = max(0, bisect.bisect_right(starts, when) - 1)
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
            s_ = f.stem[:6]
            out.append((_parse(f"{s_[:2]}:{s_[2:4]}:{s_[4:6]}", today), f))
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


def warm():
    """Keep the signed stream URL and playlist fresh so the first clip request is quick."""
    try:
        _segments()
    except Exception as e:
        print(f"[media] warm-up failed: {e!r}")


async def clip(t: str, date: str | None = None) -> Path:
    return await _cut("clip", t, date)
