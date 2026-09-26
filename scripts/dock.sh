#!/usr/bin/env bash
# Run a command inside the Ultralytics JetPack 6 container with the GPU, camera, I2C, sound and input
# devices, and this repo mounted at /askroom. TensorRT engines must be built and run in this same image.
#
#   scripts/dock.sh python3 -m core.detect --device 0 --seconds 15
#   scripts/dock.sh python3 -m core.detect --export          # rebuild the engine after changing prompts
#   scripts/dock.sh bash                                      # a shell
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${ASKROOM_IMAGE:-ultralytics/ultralytics:latest-jetson-jetpack6}"
devs=()
for d in /dev/video* /dev/i2c-7 /dev/snd /dev/input; do [ -e "$d" ] && devs+=(--device "$d"); done
tty=(); [ -t 0 ] && tty=(-it)
exec docker run --rm "${tty[@]}" --runtime=nvidia --ipc=host --network=host "${devs[@]}" \
  -v "$PWD":/askroom -w /askroom -e PYTHONPATH=/askroom "$IMAGE" "$@"
