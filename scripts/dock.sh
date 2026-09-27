#!/usr/bin/env bash
# Run a command inside the Ultralytics JetPack 6 container with the GPU, camera, I2C, sound and input
# devices, and this repo mounted at /askroom. TensorRT engines must be built and run in this same image.
#
#   scripts/dock.sh python3 -m core.detect --device 0 --seconds 15
#   scripts/dock.sh python3 -m core.detect --export          # rebuild the engine after changing prompts
#   scripts/dock.sh bash                                      # a shell
set -euo pipefail
# The host's time zone is passed in so spoken times ("put there at 8:05 PM") are local.
cd "$(dirname "$0")/.."
IMAGE="${ASKROOM_IMAGE:-askroom:latest}"   # docker/Dockerfile: the Ultralytics image + app packages
devs=()
for d in /dev/video* /dev/i2c-7 /dev/snd /dev/input; do [ -e "$d" ] && devs+=(--device "$d"); done
tty=(); [ -t 0 ] && tty=(-it)
envf=(); [ -f .env ] && envf=(--env-file .env)   # API keys (e.g. XAI_API_KEY); .env is gitignored
v4l=(); [ -d /dev/v4l ] && v4l=(-v /dev/v4l:/dev/v4l:ro)   # stable camera paths: main.py --camera /dev/v4l/by-id/...
# Host PulseAudio (a Bluetooth speaker lives there, not in ALSA): pass the socket through. Only the
# askroom:audio image (ALSA pulse plugin, /etc/asound.conf default -> pulse) can use it; askroom:latest ignores it.
pulse=(); PS="/run/user/$(id -u)/pulse/native"
[ -S "$PS" ] && pulse=(-v "$(dirname "$PS")":"$(dirname "$PS")" -e "PULSE_SERVER=unix:$PS" -e "PULSE_COOKIE=/askroom/.pulse-cookie")
[ -S "$PS" ] && [ -f "$HOME/.config/pulse/cookie" ] && cp -f "$HOME/.config/pulse/cookie" "$PWD/.pulse-cookie" 2>/dev/null
exec docker run --rm "${tty[@]}" --runtime=nvidia --ipc=host --network=host "${devs[@]}" \
  -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro \
  "${envf[@]}" "${v4l[@]}" "${pulse[@]}" -v "$PWD":/askroom -w /askroom -e PYTHONPATH=/askroom "$IMAGE" "$@"
