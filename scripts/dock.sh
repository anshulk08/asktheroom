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
# Docker masks /proc/device-tree, so Blinka (adafruit_servokit's `board`) and Jetson.GPIO can't tell which
# Jetson this is ("board not supported"); name it from the host's device tree. Orin Nano / NX (p3767):
# board.I2C() is then /dev/i2c-7, header pins 3/5.
blinka=()
if [ -r /proc/device-tree/compatible ] && tr -d '\0' < /proc/device-tree/compatible | grep -q p3767; then
  blinka=(-e BLINKA_FORCEBOARD="${BLINKA_FORCEBOARD:-JETSON_ORIN_NANO}" -e BLINKA_FORCECHIP="${BLINKA_FORCECHIP:-T234}"
          -e JETSON_MODEL_NAME="${JETSON_MODEL_NAME:-JETSON_ORIN_NANO}")
fi
# ${a[@]+"${a[@]}"}: an empty array is "unbound" under set -u in bash < 4.4 (macOS).
# ASKROOM_DOCKER_ARGS: extra `docker run` options, e.g. "-d --name askroom_room_app" (scripts/room_app.sh).
read -r -a extra <<< "${ASKROOM_DOCKER_ARGS:-}"
# Host PulseAudio (a Bluetooth speaker lives there, not in ALSA): pass the socket through. Only the
# askroom:audio image (ALSA pulse plugin, /etc/asound.conf default -> pulse) can use it; askroom:latest ignores it.
pulse=(); PS="/run/user/$(id -u)/pulse/native"
[ -S "$PS" ] && pulse=(-v "$(dirname "$PS")":"$(dirname "$PS")" -e "PULSE_SERVER=unix:$PS" -e "PULSE_COOKIE=/askroom/.pulse-cookie")
[ -S "$PS" ] && [ -f "$HOME/.config/pulse/cookie" ] && cp -f "$HOME/.config/pulse/cookie" "$PWD/.pulse-cookie" 2>/dev/null
exec docker run --rm ${tty[@]+"${tty[@]}"} ${extra[@]+"${extra[@]}"} --runtime=nvidia --ipc=host --network=host ${devs[@]+"${devs[@]}"} ${cams[@]+"${cams[@]}"} ${pulse[@]+"${pulse[@]}"} ${blinka[@]+"${blinka[@]}"} \
  -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro \
  ${envf[@]+"${envf[@]}"} -v "$PWD":/askroom -w /askroom -e PYTHONPATH=/askroom "$IMAGE" "$@"
