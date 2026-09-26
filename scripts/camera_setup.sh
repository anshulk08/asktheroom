#!/usr/bin/env bash
# Lock the overhead camera's settings (spec 3.1, task P3). Run before capture, and again after replugging:
# UVC settings reset when the camera is unplugged.
#
#   scripts/camera_setup.sh [exposure_100us] [device] [gain]
#   scripts/camera_setup.sh 333 /dev/v4l/by-id/usb-046d_0809_A1C0DC94-video-index0 0     # Pro 9000
#
# exposure is in units of 100 us. Keep it under 333 (33 ms) or the camera can't hold 30 fps; multiples of 83
# (8.3 ms) avoid flicker bands under 60 Hz lighting. Tune it once over the real table in venue light, so the
# table is well exposed without clipping, then pass that value here. Each setting is applied on its own and
# skipped when the camera doesn't have it (the icSpring has gamma and no focus; Logitech cameras have
# exposure_dynamic_framerate, which otherwise halves the fps in dim light, and some have lockable focus).
#
# Measured on the icSpring camera (32e6:9221): auto exposure drops it to 7.5-15 fps indoors; manual exposure
# holds 30 fps at 1280x720 MJPG. Its sensor is dim: in a well-lit room, 33 ms at gain 32 gave 33/255;
# gain 63 + gamma 160 gives ~144/255 with little extra grain. Logitech QuickCam Pro 9000 (046d:0809): holds
# 30 fps with exposure_dynamic_framerate off; its autofocus is not exposed on this kernel (it stays on).
set -uo pipefail

EXPOSURE="${1:-333}"          # 33 ms: the longest that holds 30 fps
DEV="${2:-/dev/video0}"
GAIN="${3:-63}"

set_ctrl() {                  # one control; a camera without it is not an error
  v4l2-ctl -d "$DEV" -c "$1" 2>/dev/null || echo "  (no ${1%%=*} on $DEV; skipped)"
}
for c in power_line_frequency=2 auto_exposure=1 exposure_dynamic_framerate=0 "exposure_time_absolute=$EXPOSURE" \
         "gain=$GAIN" gamma=160 white_balance_automatic=0 backlight_compensation=0 \
         focus_automatic_continuous=0 focus_auto=0; do
  set_ctrl "$c"
done
# white_balance_temperature only takes effect once automatic white balance is off
set_ctrl white_balance_temperature=4000

v4l2-ctl -d "$DEV" --list-ctrls | grep -E "auto_exposure|exposure_time_absolute|exposure_dynamic|gain|white_balance|power_line|focus" \
  | sed 's/^[[:space:]]*//' | cut -c1-100
