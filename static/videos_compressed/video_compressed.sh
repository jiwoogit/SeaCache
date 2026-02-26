SRC="static/videos"
DST="static/videos_compressed"

find "$SRC" -type f -name "*.mp4" | while IFS= read -r in; do
  rel="${in#$SRC/}"
  out="$DST/$rel"
  mkdir -p "$(dirname "$out")"

  ffmpeg -nostdin -y -i "$in" \
    -vf "scale=-2:720" -r 30 \
    -c:v libx264 -crf 23 -preset slow \
    -c:a aac -b:a 128k \
    -movflags +faststart \
    "$out"
done
