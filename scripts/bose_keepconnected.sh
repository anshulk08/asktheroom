#!/usr/bin/env bash
# PROPOSAL for the Jetson host (user guru, outside the container); nothing runs it yet.
# Every INTERVAL s (60): if the Bose SoundLink Micro isn't connected, connect it; then make sure its card is on
# the A2DP profile and its sink is PulseAudio's default, so the rig's speech (askroom:audio -> pulse) reaches it.
# Covers the speaker coming back after a power cycle or an A2DP profile that dropped to "off" (rig, Sat 26 Sep).
# It can't switch the speaker on: tts.keepalive_s keeps it from switching itself off.
#
#   scripts/bose_keepconnected.sh --once                      # one check, then exit
#   setsid nohup scripts/bose_keepconnected.sh >> ~/bose_keepconnected.log 2>&1 < /dev/null &
set -u
MAC="${BOSE_MAC:-2C:41:A1:76:9C:F0}"
INTERVAL="${INTERVAL:-60}"
ID="${MAC//:/_}"
CARD="bluez_card.$ID"
SINK="bluez_sink.$ID.a2dp_sink"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

say() { echo "$(date '+%F %T') $*"; }

default_sink() { pactl info 2>/dev/null | sed -n 's/^Default Sink: //p'; }

check() {
    if ! bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
        say "not connected; connecting $MAC"
        timeout 20 bluetoothctl connect "$MAC" > /dev/null 2>&1 || say "connect failed (speaker off or out of range?)"
        sleep 3
    fi
    pactl list short cards 2>/dev/null | grep -q "$CARD" || return 0
    if ! pactl list short sinks 2>/dev/null | grep -q "$SINK"; then
        say "no A2DP sink; setting $CARD to a2dp_sink"
        pactl set-card-profile "$CARD" a2dp_sink || say "set-card-profile failed"
        sleep 1
    fi
    if pactl list short sinks 2>/dev/null | grep -q "$SINK" && [ "$(default_sink)" != "$SINK" ]; then
        say "default sink was $(default_sink); setting $SINK"
        pactl set-default-sink "$SINK" || say "set-default-sink failed"
    fi
}

while true; do
    check
    [ "${1:-}" = "--once" ] && exit 0
    sleep "$INTERVAL"
done
