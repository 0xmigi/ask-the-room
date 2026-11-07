"""Index the part of a live stream that happened before the recorder started.

X/Periscope live playlists keep every segment from the start (EXT-X-PLAYLIST-TYPE:EVENT) with
real clock times (EXT-X-PROGRAM-DATE-TIME). This splits that history across several ffmpeg
workers (one segment range each, so the download runs in parallel), cuts it into 60 s audio
chunks + 1 frame/min, and transcribes them while the download is still running. Lines land in
the same data/events.jsonl, stamped with the time they were actually said.

    uv run backfill.py                  # from stream start up to where record.sh began
    uv run backfill.py --until 14:00    # or up to a given local time
"""
import argparse
import asyncio
import datetime as dt
import json
import re
import subprocess

import httpx

import ingest
import providers

BF = ingest.ROOT / "backfill"
STATE = BF / "state.json"
WORKERS = 8
CONCURRENCY = 4  # Superlinked free tier rate-limits above this


def parse_playlist(url: str):
    """-> list of (start_time, duration) per segment, from EXT-X-PROGRAM-DATE-TIME + EXTINF."""
    m3u8 = httpx.get(url, timeout=30).text
    if "#EXT-X-MEDIA-SEQUENCE:0" not in m3u8:
        raise SystemExit("Playlist doesn't start at segment 0; the start of the stream isn't available")
    segs, pdt = [], None
    for line in m3u8.splitlines():
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            pdt = dt.datetime.fromisoformat(line.split(":", 1)[1].replace("Z", "+00:00")).astimezone()
        elif line.startswith("#EXTINF:"):
            dur = float(re.match(r"#EXTINF:([\d.]+)", line).group(1))
            if pdt is None:
                pdt = segs[-1][0] + dt.timedelta(seconds=segs[-1][1])
            segs.append((pdt, dur))
            pdt = None
    return segs


def live_recorder_start() -> dt.datetime | None:
    for e in ingest.EVENTS.read_text().splitlines() if ingest.EVENTS.exists() else []:
        e = json.loads(e)
        if e.get("src", "").startswith("audio/"):
            return dt.datetime.fromisoformat(f'{e["date"]}T{e["t"]}').astimezone()
    return None


async def process(client, sem, path, rel, when: dt.datetime, done: set, schedule):
    async with sem:
        t = when.strftime("%H:%M:%S")
        try:
            if path.suffix == ".wav":
                text = await ingest.transcribe(client, path)
                if not ingest.is_junk(text):
                    ingest.append_event({"date": when.date().isoformat(), "t": t, "kind": "speech",
                                         "speaker": ingest.speaker_for(t[:5], schedule),
                                         "text": text, "src": f"backfill/{rel}"})
                    print(f"[backfill] {t} speech: {text[:80]}")
            else:
                text = await ingest.describe(client, path)
                if text:
                    ingest.append_event({"date": when.date().isoformat(), "t": t, "kind": "slide",
                                         "text": text, "src": f"backfill/{rel}"})
                    print(f"[backfill] {t} slide: {text[:80]}")
            done.add(rel)
        except httpx.HTTPStatusError as e:
            print(f"[backfill] {rel}: HTTP {e.response.status_code} {e.response.text[:150]}")
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                done.add(rel)
            else:
                await asyncio.sleep(3)  # rate limited: back off, the main loop retries it
        except Exception as e:
            print(f"[backfill] {rel}: {e!r}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=ingest.os.getenv("STREAM_URL"))
    ap.add_argument("--until", help="local HH:MM to stop at (default: where record.sh started)")
    args = ap.parse_args()
    providers.check_key()

    hls = subprocess.run(["yt-dlp", "-g", "-f", "worst", args.url],
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    segs = parse_playlist(hls)
    start = segs[0][0]
    if args.until:
        h, m = map(int, args.until.split(":"))
        end = start.replace(hour=h, minute=m, second=0, microsecond=0)
    else:
        end = live_recorder_start() or dt.datetime.now().astimezone()
    segs = [s for s in segs if s[0] < end]
    print(f"[backfill] indexing {start:%H:%M} → {end:%H:%M} "
          f"({len(segs)} segments, {sum(d for _, d in segs) / 60:.0f} min) with {WORKERS} workers")

    try:
        done = set(json.loads(STATE.read_text())["done"])
    except Exception:
        done = set()

    # Split the segment list into WORKERS contiguous ranges; each ffmpeg reads only its range
    per = -(-len(segs) // WORKERS)
    workers = []
    for k in range(WORKERS):
        a, b = k * per, min((k + 1) * per, len(segs))
        if a >= b:
            break
        wdir = BF / f"w{k}"
        (wdir / "audio").mkdir(parents=True, exist_ok=True)
        (wdir / "frames").mkdir(parents=True, exist_ok=True)
        seconds = sum(d for _, d in segs[a:b])
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-live_start_index", str(a), "-i", hls,
            "-t", f"{seconds:.1f}",
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "segment", "-segment_time", "60", "-reset_timestamps", "1", str(wdir / "audio" / "%04d.wav"),
            "-map", "0:v:0?", "-an", "-vf", "fps=1/60", "-q:v", "3", "-start_number", "0",
            "-t", f"{seconds:.1f}",  # -t is per output: without this the frames ran on to the live edge
            str(wdir / "frames" / "%04d.jpg"),
        )
        workers.append((wdir, segs[a][0], proc))

    sem = asyncio.Semaphore(CONCURRENCY)
    schedule, tasks = ingest.load_schedule(), {}
    async with httpx.AsyncClient() as client:
        while True:
            all_finished = True
            for wdir, wstart, proc in workers:
                finished = proc.returncode is not None
                all_finished &= finished
                for kind in ("audio", "frames"):
                    files = sorted((wdir / kind).iterdir())
                    for p in files if finished else files[:-1]:  # newest file may still be writing
                        rel = f"{wdir.name}/{kind}/{p.name}"
                        if rel in done or (rel in tasks and not tasks[rel].done()) or p.stat().st_size < 1024:
                            continue
                        when = wstart + dt.timedelta(seconds=int(p.stem) * 60)
                        if when >= end:
                            continue
                        tasks[rel] = asyncio.create_task(process(client, sem, p, rel, when, done, schedule))
            await asyncio.sleep(2)
            STATE.write_text(json.dumps({"done": sorted(done)}))
            if all_finished and all(t.done() for t in tasks.values()) and all(r in done for r in tasks):
                break
    STATE.write_text(json.dumps({"done": sorted(done)}))
    print(f"[backfill] done: {len(done)} files processed. The server adds them to Cognee on its next sync.")


if __name__ == "__main__":
    asyncio.run(main())
