"""scripts/room_app.sh with a stub docker: never two apps, never a truncated log."""
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform.startswith("win") or shutil.which("bash") is None, reason="bash")

STUB = r'''#!/usr/bin/env bash
d="$STUB_DIR"; echo "$*" >> "$d/calls"
case "$1" in
  ps)
    if [[ "$*" == *--format* ]]; then
      cat "$d/others" 2>/dev/null
      [ -f "$d/running" ] && printf '%s\t%s\n' "$(cat "$d/running")" '"bash -c exec python3 main.py"'
    else
      [ -f "$d/running" ] && echo c0ffee
    fi
    exit 0 ;;
  run)
    prev=""; for a in "$@"; do [ "$prev" = "--name" ] && name="$a"; prev="$a"; done
    [ -f "$d/running" ] && { echo "name in use" >&2; exit 125; }
    echo "$name" > "$d/running"; echo c0ffee ;;
  stop) rm -f "$d/running" ;;
esac
'''


@pytest.fixture
def rig(tmp_path):
    (tmp_path / "scripts").mkdir()
    for f in ("room_app.sh", "dock.sh"):
        shutil.copy(ROOT / "scripts" / f, tmp_path / "scripts" / f)
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "docker").write_text(STUB)
    (stub / "docker").chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(state), TMPDIR=str(tmp_path))

    def run(*args):
        return subprocess.run(["bash", str(tmp_path / "scripts" / "room_app.sh"), *args], env=env,
                              capture_output=True, text=True, timeout=60)
    return tmp_path, state, run


def calls(state, verb):
    return [c for c in (state / "calls").read_text().splitlines() if c.split()[0] == verb]


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
    assert logs.count("=== room_app.sh start") == 2                # a restart in the same second appends too


def test_start_refuses_while_another_app_container_holds_the_camera(rig):
    root, state, run = rig
    (state / "others").write_text('upbeat_curie\t"bash -c python3 main.py --no-voice"\n')
    r = run("start")
    assert r.returncode == 1 and "upbeat_curie" in r.stdout and "docker stop upbeat_curie" in r.stdout
    assert not calls(state, "run") and not calls(state, "stop")   # it never stops someone else's app


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
