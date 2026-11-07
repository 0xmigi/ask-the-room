#!/usr/bin/env bash
# Records a live stream into 60 s audio chunks + 1 frame every 30 s.
#   ./record.sh                       # uses STREAM_URL from .env
#   ./record.sh https://x.com/i/broadcasts/...
#   ./record.sh test.mp4              # a local file, played in real time (for testing)
# Restarts automatically when the HLS token expires or the connection drops.
set -u
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a

SRC="${1:-${STREAM_URL:-}}"
[ -z "$SRC" ] && { echo "usage: ./record.sh <stream-url-or-file>"; exit 1; }
mkdir -p audio frames data

record_once() {
  local input re_flag=()
  if [ -f "$SRC" ]; then
    input="$SRC"; re_flag=(-re)
  else
    # 480p is plenty for audio + reading slides, and a quarter of 1080p's bandwidth on busy Wi-Fi
    input="$(yt-dlp -g -f "live-1200/best[height<=480]/worst" "$SRC" 2>/dev/null | tail -1)"
    [ -z "$input" ] && { echo "[record] yt-dlp gave no URL (stream offline?)"; return 1; }
  fi
  echo "[record] $(date +%H:%M:%S) recording from ${input:0:80}..."
  ffmpeg -hide_banner -loglevel warning ${re_flag[@]+"${re_flag[@]}"} -i "$input" \
    -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le \
      -f segment -segment_time 60 -reset_timestamps 1 -strftime 1 "audio/%H%M%S.wav" \
    -map 0:v:0? -an -vf fps=1/30 -q:v 3 \
      -f image2 -strftime 1 "frames/%H%M%S.jpg"
}

# keep the Mac awake while recording
caffeinate -i -w $$ &

while true; do
  record_once
  [ -f "$SRC" ] && { echo "[record] file finished"; break; }
  echo "[record] ffmpeg stopped, retrying in 3 s"
  sleep 3
done
