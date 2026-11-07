"""Model calls: transcription, vision, chat. PROVIDER in .env picks the backend.

- superlinked (default): Superlinked Serverless. whisper-large-v3-turbo + Qwen3.8-27B (vision + chat)
- darkbloom: Darkbloom. whisper-1 + Qwen3.6 VL (vision) + gpt-oss-20b (chat)
"""
import base64
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROVIDER = os.getenv("PROVIDER", "superlinked").lower()

if PROVIDER == "superlinked":
    BASE = os.getenv("SUPERLINKED_BASE_URL", "https://api.superlinked.com/v1").rstrip("/")
    KEY = os.getenv("SIE_API_KEY", "")
    TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "openai/whisper-large-v3-turbo")
    VISION_MODEL = os.getenv("VISION_MODEL", "Qwen/Qwen3.8-27B-FP8")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen3.8-27B-FP8")
elif PROVIDER == "darkbloom":
    BASE = os.getenv("DARKBLOOM_BASE_URL", "https://api.darkbloom.dev/v1").rstrip("/")
    KEY = os.getenv("DARKBLOOM_API_KEY", "")
    TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
    VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.6-35b-a3b-vl-mtp-mxfp8")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-oss-20b")
else:
    raise SystemExit(f"Unknown PROVIDER={PROVIDER!r} (use superlinked or darkbloom)")

HEADERS = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}


def check_key():
    if not KEY or "REPLACE_ME" in KEY:
        name = "SIE_API_KEY" if PROVIDER == "superlinked" else "DARKBLOOM_API_KEY"
        raise SystemExit(f"Set {name} in .env first (PROVIDER={PROVIDER})")


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


async def chat(client: httpx.AsyncClient, messages: list, max_tokens: int = 600) -> str:
    body = {"model": CHAT_MODEL if not _has_image(messages) else VISION_MODEL, "messages": messages}
    # Superlinked's examples use max_completion_tokens; Darkbloom follows the older max_tokens
    body["max_completion_tokens" if PROVIDER == "superlinked" else "max_tokens"] = max_tokens
    if PROVIDER == "superlinked":
        body["temperature"] = 0
    r = await client.post(f"{BASE}/chat/completions", headers=HEADERS, json=body, timeout=120)
    r.raise_for_status()
    choices = r.json().get("choices") or []
    return _strip_thinking(choices[0]["message"].get("content") if choices else "")


def _has_image(messages: list) -> bool:
    return any(isinstance(m.get("content"), list) and
               any(p.get("type") == "image_url" for p in m["content"]) for m in messages)


async def describe_image(client: httpx.AsyncClient, jpg: Path, prompt: str) -> str:
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    return await chat(client, [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}], max_tokens=300)


async def transcribe(client: httpx.AsyncClient, wav: Path) -> str:
    if PROVIDER == "superlinked":
        # SIE extract: audio goes in as base64 JSON; slashes in the model id are URL-encoded
        model = TRANSCRIBE_MODEL.replace("/", "%2F")
        r = await client.post(
            f"{BASE}/extract/{model}", headers=HEADERS, timeout=120,
            json={"items": [{"audio": {"data": base64.b64encode(wav.read_bytes()).decode(),
                                       "format": "wav"}}]},
        )
        r.raise_for_status()
        return _transcript_text(r.json())
    with wav.open("rb") as f:
        r = await client.post(
            f"{BASE}/audio/transcriptions", headers=HEADERS, timeout=120,
            files={"file": (wav.name, f, "audio/wav")}, data={"model": TRANSCRIBE_MODEL},
        )
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def _transcript_text(resp) -> str:
    """Pull the transcript out of an SIE extract response, whichever nesting it uses."""
    item = resp
    if isinstance(resp, dict) and isinstance(resp.get("items"), list) and resp["items"]:
        item = resp["items"][0]
    data = item.get("data", item) if isinstance(item, dict) else item
    if isinstance(data, dict):
        data = data.get("text", data.get("transcript", ""))
    if isinstance(data, list):
        data = " ".join(d.get("text", "") if isinstance(d, dict) else str(d) for d in data)
    return str(data or "").strip()
