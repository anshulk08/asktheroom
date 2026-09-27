#!/usr/bin/env bash
# Start, stop or restart the room app (main.py in the askroom container) safely: running it twice never
# launches a second app, and the log is appended, never truncated.
#
#   scripts/room_app.sh start [main.py args...]    # no-op if it is already running
#   scripts/room_app.sh stop
#   scripts/room_app.sh restart [main.py args...]  # no args: the ones the last start used
#   scripts/room_app.sh status                     # container, args, log file, dashboard
#
# One container, named $NAME (askroom_room_app). `start` refuses while any other container runs main.py or
# demo_check.py, or while some process has the camera open (fuser, when installed and allowed to see it):
# a second app would fight it for the Brio. It prints how to stop that one and never stops it itself.
# A leftover stopped container of our own name is removed. A lock (flock, or a lock directory where flock
# is missing) makes two concurrent runs take turns.
# `start` saves its main.py args to data/room/app.args: `restart` with no args reuses them, and `status` and
# the start-up wait read --port from them. It reports "started" only once the dashboard answers
# (ROOM_APP_WAIT_S, 60 s), else prints the log's last lines. Each start appends to
# data/room/app-YYYYmmdd-HHMMSS.log and points data/room/app.log at it (demo_check.py check 15 reads it).
# Run from the checkout the app should use (e.g. ~/askroom_room).
# Env: ASKROOM_IMAGE (as scripts/dock.sh), ROOM_APP_NAME, ROOM_APP_PORT (when the args give no --port;
# default 8000), ROOM_APP_CAMERA (default: the first /dev/v4l/by-id/*-video-index0), ROOM_APP_WAIT_S.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME="${ROOM_APP_NAME:-askroom_room_app}"
LOCK="${TMPDIR:-/tmp}/askroom_room_app.lock"
ARGS_FILE=data/room/app.args
WAIT_S="${ROOM_APP_WAIT_S:-60}"
DOCKER="${DOCKER:-docker}"

lock() {
  if command -v flock >/dev/null 2>&1; then
    # Create the lock file only if missing and open it read-only: a lock file left by root (sudo) must not
    # block everyone else (flock works on a read-only descriptor).
    if [ ! -e "$LOCK.f" ] && ! : > "$LOCK.f"; then
      echo "can't create the lock file $LOCK.f (see the error above)"; exit 1
    fi
    if ! exec 9<"$LOCK.f"; then
      echo "can't open the lock file $LOCK.f: $(ls -l "$LOCK.f" 2>&1)"; exit 1
    fi
    flock -w 60 9 || { echo "another room_app.sh is still running (lock $LOCK.f)"; exit 1; }
  else
    local i=0 err
    until err="$(mkdir "$LOCK.d" 2>&1)"; do
      if [ ! -d "$LOCK.d" ]; then echo "can't create the lock directory: $err"; exit 1; fi
      i=$((i + 1)); [ "$i" -ge 120 ] && { echo "another room_app.sh is still running (lock $LOCK.d)"; exit 1; }
      sleep 0.5
    done
    trap 'rmdir "$LOCK.d" 2>/dev/null' EXIT
  fi
}

exists()  { [ -n "$("$DOCKER" ps -aq --filter "name=^/${NAME}$")" ]; }   # ours, running or not
running() { [ -n "$("$DOCKER" ps -q --filter "name=^/${NAME}$" --filter status=running)" ]; }

others() {   # other running containers running main.py or demo_check.py: "name<TAB>command"
  "$DOCKER" ps --no-trunc --format '{{.Names}}\t{{.Command}}' |
    awk -F'\t' -v me="$NAME" '$1 != me && $2 ~ /(main|demo_check)\.py/'
}

saved_args() {   # the last start's main.py args, one per line (bash 3.2: no mapfile)
  SAVED=()
  [ -f "$ARGS_FILE" ] || return 0
  local a
  while IFS= read -r a || [ -n "$a" ]; do SAVED+=("$a"); done < "$ARGS_FILE"
}

port() {   # --port N / --port=N from the saved args, else ROOM_APP_PORT, else 8000
  saved_args
  local prev="" a
  for a in ${SAVED[@]+"${SAVED[@]}"}; do
    case "$a" in --port=*) echo "${a#--port=}"; return ;; esac
    [ "$prev" = "--port" ] && { echo "$a"; return; }
    prev="$a"
  done
  echo "${ROOM_APP_PORT:-8000}"
}

healthy() { curl -s -m 2 -o /dev/null "http://127.0.0.1:$(port)/healthz"; }

camera_busy() {  # prints what holds the camera; returns 0 if busy
  local cam="${ROOM_APP_CAMERA:-$(ls /dev/v4l/by-id/*-video-index0 2>/dev/null | head -1)}"
  [ -n "$cam" ] && [ -e "$cam" ] || return 1
  if ! command -v fuser >/dev/null 2>&1; then
    echo "(fuser not installed: not checking whether $cam is open)" >&2
    return 1
  fi
  local dev who
  dev="$(readlink -f "$cam" 2>/dev/null || echo "$cam")"
  who="$(fuser "$dev" 2>/dev/null)" || return 1
  [ -n "${who// /}" ] || return 1
  echo "$cam ($dev) is open by pid(s)$who"
}

start() {
  if running; then
    echo "room app already running ($NAME); not starting another. 'scripts/room_app.sh restart' restarts it."
    return 0
  fi
  local o busy
  o="$(others)"
  if [ -n "$o" ]; then
    echo "another app container is running and holds the camera:"
    echo "$o" | sed 's/^/  /'
    echo "stop it first (docker stop $(echo "$o" | head -1 | cut -f1)), then run this again."
    return 1
  fi
  if busy="$(camera_busy)"; then
    echo "the camera is in use: $busy. Stop that process first, then run this again."
    return 1
  fi
  if exists; then                  # a stopped leftover of ours would block the name
    echo "removing a stopped leftover container $NAME"
    "$DOCKER" rm "$NAME" > /dev/null || { echo "docker rm $NAME failed"; return 1; }
  fi
  mkdir -p data/room
  if [ $# -gt 0 ]; then printf '%s\n' "$@" > "$ARGS_FILE"; elif [ ! -f "$ARGS_FILE" ]; then : > "$ARGS_FILE"; fi
  saved_args
  local stamp log
  stamp="$(date +%Y%m%d-%H%M%S)"
  log="data/room/app-$stamp.log"
  if [ -f data/room/app.log ] && [ ! -L data/room/app.log ]; then   # an old plain log: keep it
    mv data/room/app.log "data/room/app-$stamp-before.log"
    echo "kept the old data/room/app.log as data/room/app-$stamp-before.log"
  fi
  echo "=== room_app.sh start $(date '+%F %T') image ${ASKROOM_IMAGE:-askroom:latest} args: ${SAVED[*]:-}" >> "$log"
  ln -sfn "$(basename "$log")" data/room/app.log
  ASKROOM_DOCKER_ARGS="-d --name $NAME" scripts/dock.sh \
    bash -c 'exec python3 main.py "$@" >> "$0" 2>&1' "$log" ${SAVED[@]+"${SAVED[@]}"} < /dev/null > /dev/null || {
      echo "docker run failed; see $log"; return 1; }
  local t=0
  while [ "$t" -lt "$WAIT_S" ]; do
    if ! running; then
      echo "room app exited while starting; last lines of $log:"; tail -20 "$log"; return 1
    fi
    if healthy; then
      echo "room app started ($NAME) on :$(port) after ${t} s, log $log (data/room/app.log)"
      return 0
    fi
    sleep 1; t=$((t + 1))
  done
  echo "room app is running but its dashboard didn't answer on :$(port) within $WAIT_S s; last lines of $log:"
  tail -20 "$log"
  return 1
}

stop() {
  if ! running; then
    if exists; then "$DOCKER" rm "$NAME" > /dev/null && echo "removed a stopped leftover container $NAME"; fi
    echo "room app not running ($NAME)"
    return 0
  fi
  "$DOCKER" stop -t 20 "$NAME" > /dev/null || { echo "docker stop $NAME failed"; return 1; }
  local i=0                  # --rm removes it in the background; the name must be free before a start
  while exists && [ "$i" -lt 20 ]; do sleep 0.5; i=$((i + 1)); done
  echo "room app stopped ($NAME)"
}

status() {
  if running; then echo "running: $NAME"; else echo "not running: $NAME"; fi
  local o; o="$(others)"; [ -n "$o" ] && { echo "other app containers:"; echo "$o" | sed 's/^/  /'; }
  saved_args; echo "args: ${SAVED[*]:-(none)}"
  [ -e data/room/app.log ] && echo "log: data/room/$(readlink data/room/app.log 2>/dev/null || echo app.log)"
  if command -v curl >/dev/null 2>&1; then
    if healthy; then echo "dashboard: up on :$(port)"; else echo "dashboard: not answering on :$(port)"; fi
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
