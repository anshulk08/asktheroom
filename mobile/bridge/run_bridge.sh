#!/usr/bin/env bash
# Run the Ask the Room BLE bridge in the background on the Jetson host (no sudo needed).
#   mobile/bridge/run_bridge.sh start | stop | restart | status | log
# Logs: data/ble_bridge.log in the repo. Extra args after "start" go to ble_bridge.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LOG="$REPO/data/ble_bridge.log"
PIDF="$REPO/data/ble_bridge.pid"
mkdir -p "$REPO/data"

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

cmd="${1:-start}"; shift || true
case "$cmd" in
  start)
    if running; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
    setsid nohup /usr/bin/python3 -u "$HERE/ble_bridge.py" --repo "$REPO" --log "$LOG" "$@" \
      >/dev/null 2>&1 < /dev/null &
    echo $! > "$PIDF"
    sleep 2
    if running; then echo "started (pid $(cat "$PIDF")); log: $LOG"; tail -n 5 "$LOG"
    else echo "failed to start; last log lines:"; tail -n 20 "$LOG"; exit 1; fi ;;
  stop)
    if running; then kill "$(cat "$PIDF")"; sleep 1; echo stopped; else echo "not running"; fi
    rm -f "$PIDF" ;;
  restart) "$0" stop; "$0" start "$@" ;;
  status)
    if running; then echo "running (pid $(cat "$PIDF"))"; tail -n 3 "$LOG"; else echo "not running"; exit 1; fi ;;
  log) tail -n 50 -f "$LOG" ;;
  *) echo "usage: $0 start|stop|restart|status|log"; exit 2 ;;
esac
