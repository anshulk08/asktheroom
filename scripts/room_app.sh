#!/usr/bin/env bash
# Start, stop or restart the room app (main.py in the askroom container) safely: running it twice never
# launches a second app, and the log is appended, never truncated.
#
#   scripts/room_app.sh start [main.py args...]    # no-op if it is already running
#   scripts/room_app.sh stop
#   scripts/room_app.sh restart [main.py args...]
#   scripts/room_app.sh status                     # container, log file, /healthz
#
# One container, named $NAME (askroom_room_app). `start` refuses while any other container runs main.py
# (a second app would fight it for the Brio) and prints how to stop that one; it never stops it itself.
# A lock (flock, or a lock directory where flock is missing) makes two concurrent runs take turns.
# Each start writes to data/room/app-YYYYmmdd-HHMMSS.log (appending) and points data/room/app.log at it,
# which demo_check.py's memory check reads. Run from the checkout the app should use (e.g. ~/askroom_room).
# Env: ASKROOM_IMAGE (as scripts/dock.sh), ROOM_APP_NAME, ROOM_APP_PORT (for status; default 8000).
set -uo pipefail
cd "$(dirname "$0")/.."
NAME="${ROOM_APP_NAME:-askroom_room_app}"
PORT="${ROOM_APP_PORT:-8000}"
LOCK="${TMPDIR:-/tmp}/askroom_room_app.lock"
DOCKER="${DOCKER:-docker}"

lock() {
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK.f"
    flock -w 60 9 || { echo "another room_app.sh is still running (lock $LOCK.f)"; exit 1; }
  else
    local i=0
    until mkdir "$LOCK.d" 2>/dev/null; do
      i=$((i + 1)); [ "$i" -ge 120 ] && { echo "another room_app.sh is still running (lock $LOCK.d)"; exit 1; }
      sleep 0.5
    done
    trap 'rmdir "$LOCK.d" 2>/dev/null' EXIT
  fi
}

running() {  # our container is up
  [ -n "$("$DOCKER" ps -q --filter "name=^/${NAME}$")" ]
}

others() {   # other running containers whose command runs main.py: "name<TAB>command"
  "$DOCKER" ps --no-trunc --format '{{.Names}}\t{{.Command}}' | awk -F'\t' -v me="$NAME" '$1 != me && $2 ~ /main\.py/'
}

start() {
  if running; then
    echo "room app already running ($NAME); not starting another. 'scripts/room_app.sh restart' restarts it."
    return 0
  fi
  local o
  o="$(others)"
  if [ -n "$o" ]; then
    echo "another app container is running main.py and holds the camera:"
    echo "$o" | sed 's/^/  /'
    echo "stop it first (docker stop $(echo "$o" | head -1 | cut -f1)), then run this again."
    return 1
  fi
  mkdir -p data/room
  local log="data/room/app-$(date +%Y%m%d-%H%M%S).log"
  echo "=== room_app.sh start $(date '+%F %T') image ${ASKROOM_IMAGE:-askroom:latest} args: $*" >> "$log"
  ln -sfn "$(basename "$log")" data/room/app.log
  ASKROOM_DOCKER_ARGS="-d --name $NAME" scripts/dock.sh \
    bash -c 'exec python3 main.py "$@" >> "$0" 2>&1' "$log" "$@" < /dev/null > /dev/null || {
      echo "docker run failed; see $log"; return 1; }
  sleep 2
  if running; then
    echo "room app started ($NAME), log $log (data/room/app.log)"
  else
    echo "room app exited at once; last lines of $log:"; tail -20 "$log"; return 1
  fi
}

stop() {
  if ! running; then
    echo "room app not running ($NAME)"
    return 0
  fi
  "$DOCKER" stop -t 20 "$NAME" > /dev/null || { echo "docker stop $NAME failed"; return 1; }
  local i=0                  # --rm removes it in the background; the name must be free before a start
  while [ -n "$("$DOCKER" ps -aq --filter "name=^/${NAME}$")" ] && [ "$i" -lt 20 ]; do sleep 0.5; i=$((i + 1)); done
  echo "room app stopped ($NAME)"
}

status() {
  if running; then echo "running: $NAME"; else echo "not running: $NAME"; fi
  local o; o="$(others)"; [ -n "$o" ] && { echo "other app containers:"; echo "$o" | sed 's/^/  /'; }
  [ -e data/room/app.log ] && echo "log: data/room/$(readlink data/room/app.log 2>/dev/null || echo app.log)"
  if command -v curl >/dev/null 2>&1; then
    curl -s -m 3 "http://127.0.0.1:$PORT/healthz" > /dev/null && echo "dashboard: up on :$PORT" \
      || echo "dashboard: not answering on :$PORT"
  fi
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  start)   lock; start "$@" ;;
  stop)    lock; stop ;;
  restart) lock; stop && start "$@" ;;
  status)  status ;;
  *) echo "usage: scripts/room_app.sh start|stop|restart|status [main.py args...]"; exit 2 ;;
esac
