"""Guided recording of the open-world replay clips: the Mac speaks each step, the Jetson records raw
frames (eval/raw_record.py), and the spoken cues become the ground truth (truth.json), so nobody
annotates video afterwards. eval/score_clip.py replays and scores a clip.

    python -m eval.guided --list
    python -m eval.guided still --id still_1        # setup first (it is printed); then the clip runs

Physical props get fixed ids for every clip: A wallet, B keys, C phone, BOX box, NB notebook.
Timing: each step's t is when its cue was spoken (Mac time, converted to the Jetson's clock with a
measured offset) minus the first frame's time. Commands, questions and checkpoints are scheduled
relative to the first cue. A cue takes a second or two to say and a few seconds to act on: the scorer
allows for that with its own windows, not with shifted truth.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

JETSON = "guru@192.168.55.1"
DEVICE = "/dev/v4l/by-id/usb-046d_0809_A1C0DC94-video-index0"
PROPS = {"A": "wallet", "B": "small object", "C": "phone", "NB": "notebook", "BOX": "box"}   # B: keys or any small solid object; BOX: any open container
ALL_ON_TABLE = {p: {"state": "on_table"} for p in PROPS}

CLIPS = {
    "still": {
        "props": PROPS, "seconds": 32,
        "setup": "Notebook, wallet, a small object (B), the phone and the box spread out on the table, not touching. Hands away.",
        "steps": [{"at": 0, "say": "Recording. Don't touch anything for thirty seconds.", "event": "hands_out"}],
        "checkpoints": [{"at": 8, "expect": ALL_ON_TABLE}, {"at": 30, "expect": ALL_ON_TABLE}],
    },
    "hands": {
        "props": PROPS, "seconds": 32,
        "setup": "Same layout as the still clip. Hands away.",
        "steps": [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out"},
                  {"at": 5, "say": "Wave your hand over the table. Don't touch anything.", "event": "wave"},
                  {"at": 15, "say": "Rest your forearm on an empty part of the table.", "event": "rest_arm"},
                  {"at": 23, "say": "Hands away.", "event": "hands_out"}],
        "checkpoints": [{"at": 30, "expect": ALL_ON_TABLE}],
    },
    "place_name_pickup": {
        "props": {"A": "wallet", "NB": "notebook"}, "seconds": 28,
        "setup": "Only the notebook on the table. Hold the wallet, off the table.",
        "steps": [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out"},
                  {"at": 4, "say": "Put the wallet in the middle of the table, then hands away.", "event": "place",
                   "obj": "A"},
                  {"at": 13, "say": "Pick up the wallet and hold it above the table.", "event": "pickup", "obj": "A"},
                  {"at": 17, "say": "Put it down on the right side, then hands away.", "event": "putdown", "obj": "A"}],
        "commands": [{"at": 10, "text": "this is my brown wallet"}],
        "questions": [{"at": 25, "text": "where is my brown wallet?", "expect_prop": "A"}],
        "checkpoints": [{"at": 11, "expect": {"A": {"state": "on_table"}}},
                        {"at": 15, "expect": {"A": {"state": "held"}}},
                        {"at": 25, "expect": {"A": {"state": "on_table"}}}],
    },
    "shell": {
        "props": {"B": "small object", "BOX": "box", "NB": "notebook"}, "seconds": 40,
        "setup": "Box open side up, and the notebook, on the table, apart. Hold the small object (B), off the table.",
        "steps": [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out"},
                  {"at": 4, "say": "Put the small object in the middle, then hands away.", "event": "place", "obj": "B"},
                  {"at": 10, "say": "Slide the notebook over it, then hands away.", "event": "cover", "obj": "B",
                   "parent": "NB"},
                  {"at": 17, "say": "Lift the notebook off, put the object in the box, put the notebook aside, "
                                    "then hands away.", "event": "put_inside", "obj": "B", "parent": "BOX"},
                  {"at": 28, "say": "Slide the box to a new spot, then hands away.", "event": "move", "obj": "BOX"}],
        "commands": [{"at": 8, "text": "this is my test object"}],
        "questions": [{"at": 36, "text": "where is my test object?", "expect_prop": "B", "expect_parent": "BOX"}],
        "checkpoints": [{"at": 15, "expect": {"B": {"state": "under", "parent": "NB"}}},
                        {"at": 26, "expect": {"B": {"state": "inside", "parent": "BOX"}}},
                        {"at": 36, "expect": {"B": {"state": "inside", "parent": "BOX"}}}],
    },
    "exit": {
        "props": {"C": "phone", "NB": "notebook"}, "seconds": 18,
        "setup": "Notebook on the table, and the phone near the right edge. Hands away.",
        "steps": [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out"},
                  {"at": 4, "say": "Pick up the phone and carry it off the right edge, out of view.",
                   "event": "exit_edge", "obj": "C"},
                  {"at": 10, "say": "Hands away.", "event": "hands_out"}],
        "checkpoints": [{"at": 16, "expect": {"C": {"state": "gone"}}}],
    },
}


def truth_from(clip: dict, cue_mac: list[float], first_frame_wall: float, clock_offset_s: float) -> dict:
    """Ground truth in seconds since the first frame. cue_mac: Mac wall time each step's cue started;
    clock_offset_s: Jetson clock minus Mac clock; first_frame_wall: Jetson wall time of frame 0."""
    to_clip = lambda mac: mac + clock_offset_s - first_frame_wall        # noqa: E731
    zero = to_clip(cue_mac[0])
    steps = [dict({k: v for k, v in s.items() if k not in ("at", "say")}, t=round(to_clip(m), 3), note=s["say"])
             for s, m in zip(clip["steps"], cue_mac)]
    for s in steps:
        s.setdefault("obj", None)
        s.setdefault("parent", None)
    rel = lambda items: [dict({k: v for k, v in i.items() if k != "at"}, t=round(zero + i["at"], 3))  # noqa: E731
                         for i in items]
    return {"props": dict(clip["props"]), "steps": steps, "commands": rel(clip.get("commands", [])),
            "questions": rel(clip.get("questions", [])), "checkpoints": rel(clip.get("checkpoints", []))}


# ---------------------------------------------------------------- driving the rig

def ssh(cmd: str, timeout: float = 60) -> str:
    return subprocess.run(["ssh", JETSON, cmd], capture_output=True, text=True, timeout=timeout).stdout


def clock_offset() -> float:
    """Jetson clock minus Mac clock, from the fastest of five round trips."""
    best = None
    for _ in range(5):
        t0 = time.time()
        j = float(ssh("python3 -c 'import time; print(time.time())'").strip())
        t1 = time.time()
        if best is None or t1 - t0 < best[0]:
            best = (t1 - t0, j - (t0 + t1) / 2)
    return best[1]


def camera_controls() -> dict:
    out = ssh(f"v4l2-ctl -d {DEVICE} --list-ctrls")
    ctl = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and "value=" in line:
            ctl[parts[0]] = line.split("value=")[1].split()[0]
    return ctl


def say(text: str) -> subprocess.Popen:
    return subprocess.Popen(["say", "-r", "185", text])


def run_clip(name: str, clip_id: str, exposure: int = 333, gain: int = 96) -> Path:
    clip = CLIPS[name]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    ssh("C=$(docker ps -q --filter ancestor=askroom:latest); [ -n \"$C\" ] && docker stop $C >/dev/null; "
        f"cd ~/askroom && scripts/camera_setup.sh {exposure} {DEVICE} {gain} >/dev/null", timeout=90)
    offset = clock_offset()
    controls = camera_controls()
    out = f"data/clips/{clip_id}"
    ssh(f"rm -rf ~/askroom/{out}; mkdir -p ~/askroom/{out}")    # ours, so truth.json can be added after
    rec = subprocess.Popen(["ssh", JETSON, "cd ~/askroom && scripts/dock.sh python3 -m eval.raw_record "
                            f"--out {out} --device {DEVICE} --seconds {clip['seconds']} "
                            f"--controls {shlex.quote(json.dumps(controls))} --git {git} --clock-offset {offset:.6f}"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    say("Get ready.").wait()
    for _ in range(120):                                  # the container starts, the camera settles
        if ssh(f"test -f ~/askroom/{out}/READY && echo y").strip() == "y":
            break
        time.sleep(0.5)
    else:
        rec.kill()
        raise RuntimeError("the recorder never started")
    t_zero = time.time() + 0.5
    cues = []
    for s in clip["steps"]:
        time.sleep(max(0.0, t_zero + s["at"] - time.time()))
        cues.append(time.time())
        say(s["say"])
        print(f"  {cues[-1] - t_zero:5.1f} s  {s['say']}", flush=True)
    rec.wait(timeout=clip["seconds"] + 60)
    say("Done.")
    frames = json.loads(ssh(f"cat ~/askroom/{out}/frames.json"))
    truth = truth_from(clip, cues, frames["wall"][0], offset)
    subprocess.run(["ssh", JETSON, f"cat > ~/askroom/{out}/truth.json"], input=json.dumps(truth, indent=1),
                   text=True, check=True)
    local = Path("data/clips") / clip_id
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{JETSON}:askroom/{out}/", str(local) + "/"], check=True)
    n, dur = len(frames["wall"]), frames["t"][-1]
    print(f"{clip_id}: {n} frames, {dur:.1f} s ({(n - 1) / max(dur, 1e-6):.1f} fps), clock offset {offset:+.3f} s -> {local}")
    return local


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", nargs="?", choices=sorted(CLIPS))
    ap.add_argument("--id")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gain", type=int, default=96)
    a = ap.parse_args(argv)
    if a.list or not a.clip:
        for k, c in CLIPS.items():
            print(f"{k:18s} {c['seconds']:3d} s  setup: {c['setup']}")
        return 0
    run_clip(a.clip, a.id or f"{a.clip}_{time.strftime('%H%M%S')}", gain=a.gain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
