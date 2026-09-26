#!/usr/bin/env bash
# Lock the overhead camera's settings (spec 3.1, task P3). Run before capture, and again after replugging:
# UVC settings reset when the camera is unplugged.
#
#   scripts/camera_setup.sh [exposure_100us] [device]
#
# exposure is in units of 100 us. Keep it under 333 (33 ms) or the camera can't hold 30 fps; multiples of 83
# (8.3 ms) avoid flicker bands under 60 Hz lighting. Tune it once over the real table in venue light, so the
# table is well exposed without clipping, then pass that value here.
#
# Measured on the icSpring camera (32e6:9221): auto exposure drops it to 7.5-15 fps indoors; manual exposure
# holds 30 fps at 1280x720 MJPG. It has no focus control (fixed focus). Its sensor is dim: in a well-lit
# room, 33 ms at gain 32 gave 33/255; gain 63 + gamma 160 gives ~144/255 with little extra grain.
set -euo pipefail

EXPOSURE="${1:-333}"          # 33 ms: the longest that holds 30 fps
DEV="${2:-/dev/video0}"

v4l2-ctl -d "$DEV" \
  -c power_line_frequency=2 \
  -c auto_exposure=1 \
  -c exposure_time_absolute="$EXPOSURE" \
  -c gain=63 \
  -c gamma=160 \
  -c white_balance_automatic=0 \
  -c backlight_compensation=0
# white_balance_temperature only takes effect once automatic white balance is off
v4l2-ctl -d "$DEV" -c white_balance_temperature=4000

v4l2-ctl -d "$DEV" -C auto_exposure -C exposure_time_absolute -C gain -C white_balance_automatic \
  -C white_balance_temperature -C power_line_frequency
