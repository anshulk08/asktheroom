"""scripts/room_app.sh with a stub docker (honouring --filter), curl, fuser and flock: never two apps,
never a truncated log, restart keeps its args, "started" only once the dashboard answers."""
import os
import shutil
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform.startswith("win") or shutil.which("bash") is None, reason="bash")

# Containers live in $STUB_DIR/c/<name>/{status,cmd,rm}. ps honours -a, -q, --filter name=^/X$ and
# --filter status=running; run fails on a name in use; stop removes --rm containers.
DOCKER = r'''#!/usr/bin/env bash
d="$STUB_DIR"; echo "$*" >> "$d/calls"; mkdir -p "$d/c"
case "$1" in
  ps)
    shift; all=0; q=0; fmt=0; fname=""; fstatus=""
    while [ $# -gt 0 ]; do
      case "$1" in
        -a) all=1 ;; -q) q=1 ;; -aq|-qa) all=1; q=1 ;; --no-trunc) ;;
        --format) fmt=1; shift ;;
        --filter) shift; case "$1" in name=*) fname="${1#name=}" ;; status=*) fstatus="${1#status=}" ;; esac ;;
      esac; shift
    done
    for c in "$d"/c/*; do
      [ -d "$c" ] || continue; n="$(basename "$c")"; s="$(cat "$c/status")"
      [ "$all" = 1 ] || [ "$s" = running ] || continue
      [ -z "$fstatus" ] || [ "$s" = "$fstatus" ] || continue
      if [ -n "$fname" ]; then want="${fname#^/}"; want="${want%\$}"; [ "$n" = "$want" ] || continue; fi
      if [ "$q" = 1 ]; then echo "id-$n"; else printf '%s\t"%s"\n' "$n" "$(cat "$c/cmd")"; fi
    done ;;
  run)
    shift; name=""; rm=0; prev=""
    for a in "$@"; do [ "$prev" = --name ] && name="$a"; [ "$a" = --rm ] && rm=1; prev="$a"; done
    [ -d "$d/c/$name" ] && { echo "Conflict: name $name in use" >&2; exit 125; }
    mkdir -p "$d/c/$name"; echo "$*" > "$d/c/$name/cmd"; echo "$rm" > "$d/c/$name/rm"
    echo "${STUB_RUN_STATUS:-running}" > "$d/c/$name/status"; echo "id-$name" ;;
  stop)
    n="${@: -1}"; [ "$(cat "$d/c/$n/rm")" = 1 ] && rm -rf "$d/c/$n" || echo exited > "$d/c/$n/status" ;;
  rm)
    n="$2"; [ "$(cat "$d/c/$n/status")" = running ] && { echo "running" >&2; exit 1; }; rm -rf "$d/c/$n" ;;
esac
'''
CURL = r'''#!/usr/bin/env bash
echo "${@: -1}" >> "$STUB_DIR/curl"; [ -f "$STUB_DIR/healthy" ]
'''
FUSER = r'''#!/usr/bin/env bash
[ -f "$STUB_DIR/busy" ] && echo " 4242"
'''
FLOCK = r'''#!/usr/bin/env bash
# flock -w 60 9: the descriptor must be open (read-only is enough for a real flock)
{ true <&"$3"; } 2>/dev/null || { echo "flock: bad fd $3" >&2; exit 1; }
echo "flock $*" >> "$STUB_DIR/flock"
'''


def add_container(state, name, cmd, status="running", rm=1):
    c = state / "c" / name
    c.mkdir(parents=True)
    (c / "status").write_text(status + "\n")
    (c / "cmd").write_text(cmd + "\n")
    (c / "rm").write_text(f"{rm}\n")


@pytest.fixture
def rig(tmp_path):
    (tmp_path / "scripts").mkdir()
    for f in ("room_app.sh", "dock.sh"):
        shutil.copy(ROOT / "scripts" / f, tmp_path / "scripts" / f)
    stub = tmp_path / "bin"
    stub.mkdir()
    for name, body in (("docker", DOCKER), ("curl", CURL), ("fuser", FUSER)):
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    (state / "healthy").write_text("")
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(state), TMPDIR=str(tmp_path),
               ROOM_APP_WAIT_S="3", ROOM_APP_CAMERA=str(tmp_path / "no-camera"))

    def run(*args, **env_over):
        return subprocess.run(["bash", str(tmp_path / "scripts" / "room_app.sh"), *args],
                              env=dict(env, **env_over), capture_output=True, text=True, timeout=60)
    return tmp_path, state, run


def calls(state, verb):
    p = state / "calls"
    return [c for c in p.read_text().splitlines() if c.split()[0] == verb] if p.exists() else []


def test_start_twice_launches_one_app_and_appends_a_timestamped_log(rig):
    root, state, run = rig
    r = run("start", "--listen", "always")
    assert r.returncode == 0 and "room app started" in r.stdout, r.stdout + r.stderr
    r = run("start")
    assert r.returncode == 0 and "already running" in r.stdout
    assert len(calls(state, "run")) == 1
    cmd = calls(state, "run")[0]
    assert "-d --name askroom_room_app" in cmd and "python3 main.py" in cmd and "--listen always" in cmd
    link = root / "data" / "room" / "app.log"
    assert link.is_symlink() and os.readlink(link).startswith("app-")
    first = link.resolve()
    first.write_text(first.read_text() + "the app's own lines\n")
    assert run("restart").returncode == 0
    assert len(calls(state, "run")) == 2 and len(calls(state, "stop")) == 1
    assert "the app's own lines" in first.read_text()          # the old log is kept whole
    logs = "".join(p.read_text() for p in (root / "data" / "room").glob("app-*.log"))
    assert logs.count("=== room_app.sh start") == 2              # a restart in the same second appends too


def test_restart_reuses_the_saved_args_and_port(rig):
    root, state, run = rig
    assert run("start", "--port", "8080", "--no-voice").returncode == 0
    assert (root / "data" / "room" / "app.args").read_text() == "--port\n8080\n--no-voice\n"
    r = run("restart")
    assert r.returncode == 0 and "on :8080" in r.stdout, r.stdout
    assert "--port 8080 --no-voice" in calls(state, "run")[1]    # not a bare main.py on 8000 with voice on
    assert all(":8080/healthz" in u for u in (state / "curl").read_text().split())
    r = run("status")
    assert "args: --port 8080 --no-voice" in r.stdout and "dashboard: up on :8080" in r.stdout
    assert run("restart", "--port=9000").returncode == 0         # new args replace the saved ones
    assert (root / "data" / "room" / "app.args").read_text() == "--port=9000\n"
    assert "on :9000" in run("status").stdout


@pytest.mark.parametrize("cmd", ['bash -c python3 main.py --no-voice', 'python3 demo_check.py --live'])
def test_start_refuses_while_another_app_container_holds_the_camera(rig, cmd):
    root, state, run = rig
    add_container(state, "upbeat_curie", cmd)
    r = run("start")
    assert r.returncode == 1 and "upbeat_curie" in r.stdout and "docker stop upbeat_curie" in r.stdout
    assert not calls(state, "run") and not calls(state, "stop") and not calls(state, "rm")


def test_start_refuses_while_the_camera_is_open(rig):
    root, state, run = rig
    cam = root / "video0"
    cam.write_text("")
    (state / "busy").write_text("")
    r = run("start", ROOM_APP_CAMERA=str(cam))
    assert r.returncode == 1 and "camera is in use" in r.stdout and "4242" in r.stdout
    assert not calls(state, "run")


def test_a_stopped_leftover_of_ours_is_removed_not_a_blocker(rig):
    root, state, run = rig
    add_container(state, "askroom_room_app", "python3 main.py", status="exited", rm=0)
    r = run("start")
    assert r.returncode == 0 and "removing a stopped leftover" in r.stdout, r.stdout
    assert len(calls(state, "rm")) == 1 and len(calls(state, "run")) == 1


def test_started_only_once_the_dashboard_answers(rig):
    root, state, run = rig
    (state / "healthy").unlink()
    r = run("start")
    assert r.returncode == 1 and "didn't answer on :8000 within 3 s" in r.stdout and "=== room_app.sh" in r.stdout
    run("stop")
    r = run("start", STUB_RUN_STATUS="exited")                  # the app died at once
    assert r.returncode == 1 and "exited while starting" in r.stdout


def test_an_old_plain_app_log_is_moved_aside_not_deleted(rig):
    root, state, run = rig
    (root / "data" / "room").mkdir(parents=True)
    (root / "data" / "room" / "app.log").write_text("last night's log\n")
    r = run("start")
    assert r.returncode == 0 and "kept the old data/room/app.log" in r.stdout
    kept = list((root / "data" / "room").glob("app-*-before.log"))
    assert len(kept) == 1 and kept[0].read_text() == "last night's log\n"
    assert (root / "data" / "room" / "app.log").is_symlink()


def test_concurrent_starts_take_turns(rig):
    root, state, run = rig
    out = []
    ts = [threading.Thread(target=lambda: out.append(run("start"))) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls(state, "run")) == 1
    assert sorted(("already running" in r.stdout) for r in out) == [False, True]


def test_stop_when_not_running_is_fine(rig):
    root, state, run = rig
    r = run("stop")
    assert r.returncode == 0 and "not running" in r.stdout


def test_flock_path_opens_an_existing_lock_read_only_and_reports_an_unreadable_one(rig):
    """This Mac has no flock (and bash 3.2), so a stub flock checks the descriptor the script hands it."""
    root, state, run = rig
    (root / "bin" / "flock").write_text(FLOCK)
    (root / "bin" / "flock").chmod(0o755)
    lock = root / "askroom_room_app.lock.f"
    lock.write_text("")
    lock.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)       # like a root-owned 0644 file: not writable
    r = run("start")
    assert r.returncode == 0 and "room app started" in r.stdout, r.stdout + r.stderr
    assert (state / "flock").read_text().strip() == "flock -w 60 9"
    run("stop")
    lock.chmod(0)                                                # not even readable
    r = run("start")
    assert r.returncode == 1 and "can't open the lock file" in r.stdout
    lock.chmod(0o644)
