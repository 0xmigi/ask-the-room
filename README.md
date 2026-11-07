# Ask the Room

**A shared memory of a live stream that anyone in the room can ask questions.** It's like Granola, but for a whole room instead of one person, and it stays private.

It records a livestream's audio and slides as they happen. Anyone in the room can open a page on their phone and ask *"what did dmarz say about routing at kickoff?"* The answer comes back with the time it was said, e.g. **[10:17]**. It works on any stream: conferences, classes, meetups.

Built solo at **LOCALHOST: New York's Sovereign AI Hackathon**, Nov 7 2026.

## Why

Everyone misses half the talks at an event. Note-takers like Granola give one person a private memory. Ask the Room gives the **whole room** one memory. It also runs on hardware you control: the memory (Cognee) and the embeddings (Superlinked SIE) run on the laptop, and nothing gets stored in someone else's cloud.

## How it works

```
X livestream ──yt-dlp──► HLS ──ffmpeg (one process)
   ├─ audio: 60 s WAV chunks ──► Darkbloom whisper ──► "[13:42] Speaker said: …"
   └─ video: 1 frame / 30 s  ──► Darkbloom vision  ──► "[13:42] On screen: …"   (overlay & empty frames dropped)
                                         │
                                  data/events.jsonl
                                         │
             ┌───────────────────────────┴──────────────────────────┐
     Cognee memory (graph + vectors)                     SIE index (source lines)
     LLM = Darkbloom gpt-oss-20b                         Superlinked SIE, local
     embeddings = Superlinked SIE (local)                       │
             └───────────────────────────┬──────────────────────────┘
                                         ▼
                 Phone page ──► /api/ask ──► answer + [HH:MM] + the exact source lines
```

| Piece | What it does | Sponsor |
|---|---|---|
| `record.sh` | yt-dlp gets the live HLS URL; one ffmpeg process writes `audio/HHMMSS.wav` (60 s) and `frames/HHMMSS.jpg` (every 30 s). It restarts itself when the stream token expires. | |
| `ingest.py` | Sends each new chunk to **Darkbloom** speech-to-text and each frame to Darkbloom vision. Writes timestamped lines to `data/events.jsonl`, `transcript.md` and `slides.md`. Speakers are labelled from `schedule.json`. | **Darkbloom** |
| `memory.py` | Every 3 min: `cognee.add` + `cognee.cognify` the new lines into a knowledge graph. Answers use `cognee.search(HYBRID_COMPLETION)`. **Cognee** uses Darkbloom as its LLM and **Superlinked SIE** as its embedding engine (OpenAI-compatible `/v1/embeddings`, running locally on Apple Silicon). | **Cognee**, **Superlinked**, **Darkbloom** |
| `server.py` + `static/index.html` | Phone page: question box, live status, and answers with clickable timestamps and source lines. | |

## Run it

Needs macOS or Linux, `ffmpeg`, `yt-dlp`, `uv`, and a Darkbloom key from [console.darkbloom.dev](https://console.darkbloom.dev).

```bash
brew install ffmpeg yt-dlp uv
cp .env.example .env          # paste DARKBLOOM_API_KEY, set STREAM_URL
uv sync
./start.sh                    # starts SIE, server, ingester, recorder; prints the phone URL
```

Or run each part on its own, one terminal each:

```bash
SIE_AUTH_MODE=none uvx --python 3.12 --from "sie-server[local]" sie-server serve --host 127.0.0.1 --preload Snowflake/snowflake-arctic-embed-s
uv run server.py              # http://<laptop-ip>:8000
uv run ingest.py
./record.sh                   # or ./record.sh some-talk.mp4 to test offline
```

If Cognee misbehaves, set `MEMORY=simple` in `.env`. Answers then come straight from the Superlinked SIE index plus Darkbloom.

## Privacy

- Recordings, transcripts, the knowledge graph and the embeddings all stay in this folder on your machine (`audio/`, `frames/`, `data/`, `.cognee/`).
- The SIE embedding server listens only on `127.0.0.1`.
- Transcription, vision and answers go through Darkbloom, an OpenAI-compatible API, so any self-hosted endpoint can replace it by changing `DARKBLOOM_BASE_URL`.

## What's next

- Speaker diarization instead of labels from the schedule.
- A "camera on the table" mode for rooms with no stream.
- Pay-per-question unlock (1 USDC on Solana Pay) for public rooms.
