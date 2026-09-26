#!/usr/bin/env bash
# Download a Piper voice (.onnx + .onnx.json) from the rhasspy/piper-voices Hugging Face repo
# into models/piper/. Usage: scripts/get_piper_voice.sh [voice]   (default en_US-lessac-medium)
set -euo pipefail

VOICE="${1:-en_US-lessac-medium}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/models/piper"
mkdir -p "$DEST"

# en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium
LANG_CODE="${VOICE%%-*}"
REST="${VOICE#*-}"
NAME="${REST%-*}"
QUALITY="${REST##*-}"
FAMILY="${LANG_CODE%%_*}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/$FAMILY/$LANG_CODE/$NAME/$QUALITY/$VOICE"

for EXT in .onnx .onnx.json; do
  OUT="$DEST/$VOICE$EXT"
  if [ -s "$OUT" ]; then
    echo "have $OUT"
    continue
  fi
  echo "fetching $BASE$EXT"
  curl -fL --retry 3 -o "$OUT.part" "$BASE$EXT?download=true"
  mv "$OUT.part" "$OUT"
done
ls -l "$DEST/$VOICE".onnx*
