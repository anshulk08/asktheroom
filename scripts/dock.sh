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
IMAGE="${ASKROOM_IMAGE:-askroom:latest}"   # docker/Dockerfile: the Ultralytics image + app packages; e.g.
                                           # ASKROOM_IMAGE=askroom:demo to try a new build, latest to roll back
devs=()
# gpiochip: Blinka's board module (adafruit_servokit) imports Jetson.GPIO, which reads the GPIO chips.
for d in /dev/video* /dev/i2c-7 /dev/gpiochip* /dev/snd /dev/input; do [ -e "$d" ] && devs+=(--device "$d"); done
tty=(); [ -t 0 ] && tty=(-it)
envf=(); [ -f .env ] && envf=(--env-file .env)   # API keys (e.g. XAI_API_KEY); .env is gitignored
# Cameras: --device only passes the /dev/video* nodes that exist now, so a Brio replugged (or re-enumerated)
# while the container runs was unreachable. The host's /dev is mounted instead and every V4L2 node (char
# major 81) is allowed, so /dev/v4l/by-id/... finds the camera again. ASKROOM_DEV_BIND=0 turns this off.
cams=()
if [ "${ASKROOM_DEV_BIND:-1}" = 1 ]; then
  cams=(--device-cgroup-rule='c 81:* rmw' -v /dev:/dev)
elif [ -d /dev/v4l ]; then
  cams=(-v /dev/v4l:/dev/v4l:ro)                # stable camera paths only (no replug)
fi
exec docker run --rm "${tty[@]}" --runtime=nvidia --ipc=host --network=host "${devs[@]}" "${cams[@]}" \
  -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro \
  "${envf[@]}" -v "$PWD":/askroom -w /askroom -e PYTHONPATH=/askroom "$IMAGE" "$@"
