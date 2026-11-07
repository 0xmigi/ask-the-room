#!/usr/bin/env bash
# Starts everything: Superlinked SIE (embeddings), the server + phone page, the ingester, the recorder.
# Ctrl+C stops all of it. Logs go to *.log in this folder.
cd "$(dirname "$0")"
[ -f .env ] || { echo "copy .env.example to .env and add your Darkbloom key"; exit 1; }
set -a; . ./.env; set +a
trap 'kill 0' EXIT

if ! curl -s -m 2 "${SIE_URL:-http://localhost:8080}/healthz" >/dev/null; then
  echo "starting Superlinked SIE on :8080 (first run downloads the model)"
  SIE_AUTH_MODE=none uvx --python 3.12 --from "sie-server[local]" sie-server serve \
    --host 127.0.0.1 --preload "${EMBED_MODEL}" > sie.log 2>&1 &
  until curl -s -m 2 "${SIE_URL:-http://localhost:8080}/healthz" >/dev/null; do sleep 2; done
fi

uv run server.py > server.log 2>&1 &
uv run ingest.py > ingest.log 2>&1 &
./record.sh > record.log 2>&1 &

IP=$(ipconfig getifaddr en0 2>/dev/null || echo localhost)
echo "Ask the Room is running → http://$IP:${PORT:-8000}  (open on your phone, same network)"
tail -f ingest.log
