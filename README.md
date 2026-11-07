# Ask the Room

**A shared memory of a live stream that anyone in the room can ask questions.** It's like Granola, but for a whole room instead of one person, and it stays private.

It records a livestream's audio and slides as they happen. Anyone in the room can open a page on their phone and ask *"what did dmarz say about routing at kickoff?"* The answer comes back with the time it was said, e.g. **[10:17]**. It works on any stream: conferences, classes, meetups.

Built solo at **LOCALHOST: New York's Sovereign AI Hackathon**, Nov 7 2026.

## Why

Everyone misses half the talks at an event. Note-takers like Granola give one person a private memory. Ask the Room gives the **whole room** one memory. It also runs on hardware you control: the memory (Cognee) and the embeddings (Superlinked SIE) run on the laptop, and nothing gets stored in someone else's cloud. Only the raw audio and frames go out to be transcribed and read, and those calls are stateless.

## How it works

```
X livestream ──yt-dlp──► HLS ──ffmpeg (one process)
   ├─ audio: 60 s WAV chunks ──► Superlinked whisper-large-v3-turbo ──► "[13:42] Speaker said: …"
   └─ video: 1 frame / 30 s  ──► Superlinked Qwen3.8-27B vision    ──► "[13:42] On screen: …"   (overlay & empty frames dropped)
                                         │
                                  data/events.jsonl
                                         │
             ┌───────────────────────────┴──────────────────────────┐
     Cognee memory (graph + vectors)                     SIE index (source lines)
     LLM = Superlinked Qwen3.8-27B                       Superlinked SIE, local
     embeddings = Superlinked SIE (local)                       │
             └───────────────────────────┬──────────────────────────┘
                                         ▼
                 Phone page ──► /api/ask ──► answer + [HH:MM] + the exact source lines
```

| Piece | What it does | Sponsor |
|---|---|---|
| `record.sh` | yt-dlp gets the live HLS URL; one ffmpeg process writes `audio/HHMMSS.wav` (60 s) and `frames/HHMMSS.jpg` (every 30 s). It restarts itself when the stream token expires. | |
| `ingest.py` + `providers.py` | Sends each new chunk to **Superlinked Serverless** speech-to-text (`whisper-large-v3-turbo`) and each frame to Qwen3.8-27B vision. Writes timestamped lines to `data/events.jsonl`, `transcript.md` and `slides.md`. Speakers are labelled from `schedule.json`. `PROVIDER=darkbloom` switches all model calls to Darkbloom instead. | **Superlinked** (or **Darkbloom**) |
| `memory.py` | Every 3 min: `cognee.add` + `cognee.cognify` the new lines into a knowledge graph. Answers use `cognee.search(HYBRID_COMPLETION)`. **Cognee** uses Superlinked Qwen3.8-27B as its LLM and **Superlinked SIE** as its embedding engine (OpenAI-compatible `/v1/embeddings`, running locally on Apple Silicon). | **Cognee**, **Superlinked** |
| `speakers.py` | Works out **who was speaking when** from the stream itself: self-introductions ("I'm Kaido from Darkbloom"), host hand-offs ("next up is…"), thank-yous ("thank you so much Kaido" = Kaido just finished) and company names on slides. Lines get real speaker names, "who said X" questions work, and "what does Kaido look like" jumps to the middle of Kaido's talk. Names only come from the stream; unnamed presenters are labelled by company. | |
| `media.py` | Any moment can be shown on request: tap a timestamp in an answer for the frame and a 30 s clip, or ask "show me…". Cuts only the needed HLS segments, so it takes seconds, and remembers every stream link so earlier broadcasts keep working. | |
| `server.py` + `static/index.html` | Phone page: question box, live status, and answers with clickable timestamps and source lines. | |

## Plug your agent into the room

The memory is also an **MCP server**, so people building at the event can give their own agent
(Claude Code, Cursor, any MCP client) instant context on every talk, slide and speaker. Open the
chat page, tap **Connect your agent**, and paste the command, e.g.:

```bash
claude mcp add --transport http ask-the-room http://<laptop-ip>:8000/mcp --header "Authorization: Bearer <ROOM_KEY>"
```

| Tool | What it returns |
|---|---|
| `ask(question)` | An answer citing `[HH:MM]` times, the source lines, and frame/clip links for "show me" questions |
| `search(query, limit)` | Timestamped speech and slide lines with speaker names, best matches first |
| `recent(minutes)` | Everything from the last N minutes |
| `transcript(start, end)` | Raw lines between two times |
| `speakers_timeline()` | Who presented when |
| `moment(time)` | Links to the video frame and a 30 s clip |

**Access:** one shared memory per room, gated by a room key (`ROOM_KEY` in `.env`, generated on
first run and shown on the chat page). Being in the room, on the same network and seeing the page,
is what gets you in. Cognee itself supports per-user, per-dataset permissions if you need
separate rooms or read-only guests.

## Run it

Needs macOS or Linux, `ffmpeg`, `yt-dlp`, `uv`, and a Superlinked key from [console.superlinked.com](https://console.superlinked.com) (free credits, no card). Darkbloom works too: set `PROVIDER=darkbloom` and `DARKBLOOM_API_KEY`.

```bash
brew install ffmpeg yt-dlp uv
cp .env.example .env          # paste SIE_API_KEY, set STREAM_URL
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

If Cognee misbehaves, set `MEMORY=simple` in `.env`. Answers then come straight from the local SIE index plus the chat model.

## Privacy

- Recordings, transcripts, the knowledge graph and the embeddings all stay in this folder on your machine (`audio/`, `frames/`, `data/`, `.cognee/`).
- The SIE embedding server listens only on `127.0.0.1`.
- Transcription, vision and answers go through Superlinked Serverless (or Darkbloom). SIE is the same engine that runs locally, so the whole pipeline can move onto your own GPU by pointing `SUPERLINKED_BASE_URL` at your own `sie-server`.

## What's next

- Speaker diarization instead of labels from the schedule.
- A "camera on the table" mode for rooms with no stream.
- Pay-per-question unlock (1 USDC on Solana Pay) for public rooms.
