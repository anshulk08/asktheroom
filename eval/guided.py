"""Guided recording of the open-world replay clips: the Mac speaks each step, the Jetson records raw
frames (eval/raw_record.py), and the spoken cues become the ground truth (truth.json), so nobody
annotates video afterwards. eval/score_clip.py replays and scores a clip.

    python -m eval.guided --list
    python -m eval.guided still --id still_1        # setup first (it is printed); then the clip runs
    ASKROOM_RIG_DIR=askroom_room python -m eval.guided room_still --id room_still_1     # the room demo rig

Physical props get fixed ids for every clip: A wallet, B keys, C phone, BOX box, NB notebook (room demo
clips add PB, the pill bottle).

Room-demo clips (room_*, spec 0010) record the full corner-camera frame at the app's capture size (the
rig's config.local.yaml: room_memory on, 2560x1440; eval/raw_record.py) with the camera settings the app
left (no camera_setup.sh), and are scored for identity by eval/scorecard.py. They put each prop down with
a cue first (a place step), so the scorer knows which identity is which prop without annotation. Each
step's `seg` (still: nobody near the table; people: someone moving, sitting or reaching) splits the clip
for the phantom-birth rates; carry_to (from the table) and place_room (straight into the room, never on
the table) steps name the room zone (room_zones.json key, or 'floor') the prop goes to; block / unblock
steps: a person hides a resting prop and moves away again; a putdown with expect_same brings a carried or
removed prop back, and the scorer wants the identity it had before.

The recorder needs the camera, so the live app must be stopped first (the rig owner does that:
scripts/room_app.sh stop). This driver never stops it: it refuses while any container runs main.py
(--stop-app restores the old behaviour of stopping askroom:latest containers, for the table rig).
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

import os
JETSON = os.environ.get("ASKROOM_JETSON", "guru@10.90.84.178")            # Wi-Fi; guru@192.168.55.1 over USB
RIG_DIR = os.environ.get("ASKROOM_RIG_DIR", "askroom")      # the checkout on the Jetson: askroom_room for the room demo
DEVICE = os.environ.get("ASKROOM_CAMERA", "/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0")
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
    "blanket": {
        "props": {"B": "small object", "C": "phone", "NB": "notebook", "BL": "blanket (a cover the rig has no class for)"},
        "seconds": 30,
        "setup": "The AirPods case (B), the phone and the notebook spread out on the table. Hold the blanket, off the table.",
        "steps": [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out"},
                  {"at": 4, "say": "Lay the blanket over everything, then hands away.", "event": "cover", "obj": "B",
                   "parent": "BL"},
                  {"at": 4.01, "say": "", "event": "cover", "obj": "C", "parent": "BL"},
                  {"at": 4.02, "say": "", "event": "cover", "obj": "NB", "parent": "BL"},
                  {"at": 16, "say": "Lift the blanket off and take it away, then hands away.", "event": "uncover",
                   "obj": "B", "parent": "BL"},
                  {"at": 16.01, "say": "", "event": "uncover", "obj": "C", "parent": "BL"},
                  {"at": 16.02, "say": "", "event": "uncover", "obj": "NB", "parent": "BL"}],
        "questions": [{"at": 12, "text": "where is my phone?", "expect_prop": "C", "expect_parent": "BL"}],
        "checkpoints": [{"at": 13, "expect": {"B": {"state": "under", "parent": "BL"},
                                              "C": {"state": "under", "parent": "BL"}}},
                        {"at": 27, "expect": {"B": {"state": "on_table"}, "C": {"state": "on_table"},
                                              "NB": {"state": "on_table"}}}],
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


# ---------------------------------------------------------------- room demo clips (spec 0010)

DEMO = {"A": "wallet", "B": "keys", "C": "phone", "BOX": "box", "NB": "notebook", "PB": "pill bottle"}
STILL, PEOPLE = "still", "people"
PLACE_EVERY = 6.0           # s between place cues: ~2 s to say, ~3 s to put it down and pull the hand back


def _place_in(props: list, t0: float = 3.0) -> list:
    """Cues that put each prop down one by one (binds each identity to its prop), then hands away."""
    steps = [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out", "seg": STILL}]
    for i, p in enumerate(props):
        steps.append({"at": t0 + i * PLACE_EVERY, "say": f"Put the {DEMO[p]} on the table, then hands away.",
                      "event": "place", "obj": p, "seg": PEOPLE})
    return steps


def _end_of(steps: list) -> float:
    return steps[-1]["at"] + PLACE_EVERY


def _on(*props) -> dict:
    return {p: {"state": "on_table"} for p in props}


def _room_still() -> dict:
    props = ["A", "B", "C", "BOX", "NB", "PB"]
    steps = _place_in(props)
    t = _end_of(steps)
    steps.append({"at": t, "say": "Hands away. Nobody touch the table or walk past it for a minute.",
                  "event": "hands_out", "seg": STILL})
    return {"room": True, "props": {p: DEMO[p] for p in props}, "seconds": int(t + 64),
            "setup": "Empty coffee table. Hold the wallet, keys, phone, box, notebook and pill bottle, off the "
                     "table. Put each down when told, spread out, a hand-width apart.",
            "steps": steps, "checkpoints": [{"at": t + 30, "expect": _on(*props)}, {"at": t + 60, "expect": _on(*props)}]}


def _room_clutter() -> dict:
    props = ["A", "B", "C", "BOX", "NB", "PB"]
    steps = _place_in(props)
    t = _end_of(steps)
    steps.append({"at": t, "say": "Hands away. Nobody touch the table or walk past it for a minute.",
                  "event": "hands_out", "seg": STILL})
    return {"room": True, "props": {p: DEMO[p] for p in props}, "seconds": int(t + 64),
            "scene": "an open laptop and a pile of cables on the table from the start",
            "scene_objects": ["laptop", "cable pile"],
            "setup": "An open laptop and a loose pile of cables (chargers, a USB cable) on the coffee table. Hold "
                     "the six props, off the table. Put each down when told, among the clutter, not touching it.",
            "steps": steps, "checkpoints": [{"at": t + 30, "expect": _on(*props)}, {"at": t + 60, "expect": _on(*props)}]}


def _room_couch() -> dict:
    props = ["A", "B", "C", "BOX", "NB", "PB"]
    steps = _place_in(props)
    t = _end_of(steps)
    steps += [
        {"at": t, "say": "Sit down on the couch now, one or two of you.", "event": "sit", "seg": PEOPLE},
        {"at": t + 8, "say": "Put your feet up on the edge of the table, near the props but not touching them.",
         "event": "feet_up", "seg": PEOPLE},
        {"at": t + 28, "say": "Lean forward and rest your hands near the table edge. Don't touch the props.",
         "event": "hands_near", "seg": PEOPLE},
        {"at": t + 43, "say": "Feet down, sit back, and keep still.", "event": "sit", "seg": PEOPLE},
        {"at": t + 58, "say": "Stand up and walk past the table, without touching it.", "event": "walk",
         "seg": PEOPLE},
        {"at": t + 68, "say": "Everyone away from the table. Keep still.", "event": "hands_out", "seg": STILL}]
    return {"room": True, "props": {p: DEMO[p] for p in props}, "seconds": int(t + 90),
            "setup": "Empty coffee table, couch free. Hold the six props. Put each down when told, spread out, "
                     "a little in from the edge the couch faces. Then sit on the couch when told, in socks or "
                     "shoes, jeans are good (feet at the table edge are what goes wrong).",
            "steps": steps, "checkpoints": [{"at": t + 66, "expect": _on(*props)},
                                            {"at": t + 88, "expect": _on(*props)}]}


def _room_carry() -> dict:
    steps = [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out", "seg": STILL},
             {"at": 3, "say": "Put the wallet on the table, then hands away.", "event": "place", "obj": "A",
              "seg": PEOPLE}]
    t = 12.0
    for zone, say in (("couch", "the couch"), ("side_table", "the side table"), ("counter", "the kitchen counter")):
        steps += [{"at": t, "say": f"Pick up the wallet, carry it to {say}, put it down there where the camera "
                                   "can see it, and step away.", "event": "carry_to", "obj": "A", "zone": zone,
                   "seg": PEOPLE},
                  {"at": t + 25, "say": "Bring the wallet back to the table, put it down, then hands away.",
                   "event": "putdown", "obj": "A", "expect_same": True, "seg": PEOPLE}]
        t += 37
    steps.append({"at": t, "say": "Hands away. Nobody touch anything.", "event": "hands_out", "seg": STILL})
    return {"room": True, "props": {"A": "wallet", "C": "phone", "NB": "notebook", "BOX": "box"},
            "seconds": int(t + 15),
            "setup": "Phone, notebook and box on the coffee table, apart. Couch, side table and kitchen counter "
                     "clear enough to see a wallet on them. Hold the wallet, off the table.",
            "steps": steps,
            "checkpoints": [{"at": s["at"] + 10, "expect": _on("A")} for s in steps if s["event"] == "putdown"]}


def _room_move() -> dict:
    steps = _place_in(["A", "C", "NB"])
    t = _end_of(steps)
    steps += [
        {"at": t, "say": "Hands away.", "event": "hands_out", "seg": STILL},
        {"at": t + 8, "say": "Pick up the phone and hold it up above the table.", "event": "pickup", "obj": "C",
         "seg": PEOPLE},
        {"at": t + 13, "say": "Put it down on the other side of the table, then hands away.", "event": "putdown",
         "obj": "C", "seg": PEOPLE},
        {"at": t + 25, "say": "Pick up the wallet and hold it up.", "event": "pickup", "obj": "A", "seg": PEOPLE},
        {"at": t + 30, "say": "Put it down somewhere else on the table, then hands away.", "event": "putdown",
         "obj": "A", "seg": PEOPLE},
        {"at": t + 42, "say": "Slide the notebook to a new spot, then hands away.", "event": "move", "obj": "NB",
         "seg": PEOPLE},
        {"at": t + 54, "say": "Hands away. Keep still.", "event": "hands_out", "seg": STILL}]
    return {"room": True, "props": {"A": "wallet", "C": "phone", "NB": "notebook", "BOX": "box", "PB": "pill bottle"},
            "seconds": int(t + 75),
            "setup": "Box and pill bottle on the coffee table (they stay put). Hold the wallet, the phone and the "
                     "notebook; put each down when told.",
            "steps": steps,
            "checkpoints": [{"at": t + 6, "expect": _on("A", "C", "NB")}, {"at": t + 23, "expect": _on("C")},
                            {"at": t + 72, "expect": _on("A", "C", "NB")}]}


def _room_remove() -> dict:
    props = ["A", "B", "C", "PB", "NB", "BOX"]
    steps = _place_in(props)
    t = _end_of(steps)
    steps.append({"at": t, "say": "Hands away.", "event": "hands_out", "seg": STILL})
    for i, p in enumerate(props):
        steps.append({"at": t + 10 + 9 * i, "say": f"Take the {DEMO[p]} off the table and put it away, out of "
                                                   "sight.", "event": "remove", "obj": p, "seg": PEOPLE})
    end = t + 10 + 9 * len(props)
    steps.append({"at": end, "say": "Hands away. Nobody near the table.", "event": "hands_out", "seg": STILL})
    return {"room": True, "props": {p: DEMO[p] for p in props}, "seconds": int(end + 20),
            "setup": "Empty coffee table. Hold the six props. Put each down when told, then take each away when "
                     "told (a pocket, a bag, behind your back, off camera).",
            "steps": steps, "checkpoints": [{"at": t + 8, "expect": _on(*props)}]}


def _room_straight() -> dict:
    steps = _place_in(["A", "NB"])
    t = _end_of(steps)
    steps += [
        {"at": t, "say": "Hands away.", "event": "hands_out", "seg": STILL},
        {"at": t + 8, "say": "Put the phone straight onto the couch, where the camera can see it. Don't touch the "
                             "table.", "event": "place_room", "obj": "C", "zone": "couch", "seg": PEOPLE},
        {"at": t + 26, "say": "Put the keys on the floor beside the table. Don't touch the table.",
         "event": "place_room", "obj": "B", "zone": "floor", "seg": PEOPLE},
        {"at": t + 44, "say": "Everyone step away from the table and the couch. Keep still.", "event": "hands_out",
         "seg": STILL}]
    return {"room": True, "props": {"A": "wallet", "NB": "notebook", "C": "phone", "B": "keys"},
            "seconds": int(t + 66),
            "setup": "Empty coffee table, couch clear. Hold the wallet, the notebook, the phone and the keys. The "
                     "phone and keys never touch the table: carry them round it, not over it.",
            "steps": steps, "checkpoints": [{"at": t + 6, "expect": _on("A", "NB")},
                                            {"at": t + 64, "expect": _on("A", "NB")}]}


def _room_block() -> dict:
    steps = _place_in(["A", "C", "PB"])
    t = _end_of(steps)
    steps += [
        {"at": t, "say": "Hands away.", "event": "hands_out", "seg": STILL},
        {"at": t + 10, "say": "Sit or crouch right in front of the pill bottle, so the camera can't see it, and "
                              "keep still. Don't touch it.", "event": "block", "obj": "PB", "seg": PEOPLE},
        {"at": t + 26, "say": "Move away from the table.", "event": "unblock", "obj": "PB", "seg": PEOPLE},
        {"at": t + 34, "say": "Everyone away. Keep still.", "event": "hands_out", "seg": STILL}]
    return {"room": True, "props": {"A": "wallet", "C": "phone", "PB": "pill bottle"}, "seconds": int(t + 56),
            "setup": "Empty coffee table. Hold the wallet, the phone and the pill bottle. Put the pill bottle down "
                     "near the table edge by the couch, so someone sitting or crouching there hides it from the "
                     "camera.",
            "steps": steps, "checkpoints": [{"at": t + 8, "expect": _on("A", "C", "PB")},
                                            {"at": t + 54, "expect": _on("A", "C", "PB")}]}


def _room_keys_off() -> dict:
    steps = [{"at": 0, "say": "Recording. Hands away.", "event": "hands_out", "seg": STILL},
             {"at": 3, "say": "Put the keys on the table, then hands away.", "event": "place", "obj": "B",
              "seg": PEOPLE},
             {"at": 12, "say": "Pick up the keys and put them straight on the couch, the shortest way, not past the "
                              "side table or the counter. Then step away.", "event": "carry_to", "obj": "B",
              "zone": "couch", "seg": PEOPLE},
             {"at": 35, "say": "Bring the keys back to the table, put them down, then hands away.",
              "event": "putdown", "obj": "B", "expect_same": True, "seg": PEOPLE},
             {"at": 47, "say": "Pick up the keys and put them on the floor by the doorway, away from the couch, the "
                              "side table and the counter. Then step away.", "event": "carry_to", "obj": "B",
              "zone": "floor", "seg": PEOPLE},
             {"at": 70, "say": "Bring the keys back to the table, put them down, then hands away.",
              "event": "putdown", "obj": "B", "expect_same": True, "seg": PEOPLE},
             {"at": 82, "say": "Hands away. Nobody touch anything.", "event": "hands_out", "seg": STILL}]
    return {"room": True, "props": {"B": "keys", "C": "phone", "NB": "notebook"}, "seconds": 97,
            "setup": "Phone and notebook on the coffee table, apart. Couch clear. Hold the keys, off the table.",
            "steps": steps, "checkpoints": [{"at": 45, "expect": _on("B")}, {"at": 94, "expect": _on("B")}]}


def _room_return() -> dict:
    steps = _place_in(["A", "C", "NB"])
    t = _end_of(steps)
    steps += [
        {"at": t, "say": "Hands away.", "event": "hands_out", "seg": STILL},
        {"at": t + 8, "say": "Pick up the wallet and take it out of the room, out of the camera's view.",
         "event": "remove", "obj": "A", "seg": PEOPLE},
        {"at": t + 28, "say": "Bring the wallet back and put it down on a different spot on the table, then hands "
                              "away.", "event": "putdown", "obj": "A", "expect_same": True, "seg": PEOPLE},
        {"at": t + 40, "say": "Hands away. Keep still.", "event": "hands_out", "seg": STILL}]
    return {"room": True, "props": {"A": "wallet", "C": "phone", "NB": "notebook"}, "seconds": int(t + 60),
            "setup": "Empty coffee table. Hold the wallet, the phone and the notebook; put each down when told. "
                     "Carry the wallet right out of the room when told (behind a door or round a corner).",
            "steps": steps, "checkpoints": [{"at": t + 6, "expect": _on("A", "C", "NB")},
                                            {"at": t + 58, "expect": _on("A", "C", "NB")}]}


ROOM_CLIPS = {"room_still": _room_still(), "room_clutter": _room_clutter(), "room_couch": _room_couch(),
              "room_carry": _room_carry(), "room_move": _room_move(), "room_remove": _room_remove(),
              "room_straight": _room_straight(), "room_block": _room_block(), "room_keys_off": _room_keys_off(),
              "room_return": _room_return()}
CLIPS.update(ROOM_CLIPS)


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
    out = {"props": dict(clip["props"]), "steps": steps, "commands": rel(clip.get("commands", [])),
           "questions": rel(clip.get("questions", [])), "checkpoints": rel(clip.get("checkpoints", []))}
    for k in ("scene", "scene_objects"):
        if clip.get(k):
            out[k] = clip[k]
    return out


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


APP_RUNNING = ("docker ps --no-trunc --format '{{.Names}}\\t{{.Image}}\\t{{.Command}}' "
               "| awk -F'\\t' '$3 ~ /(main|demo_check)\\.py/'")


def app_running() -> str:
    """Containers on the Jetson running the app (main.py / demo_check.py): 'name<TAB>image<TAB>command' lines."""
    return ssh(APP_RUNNING).strip()


def run_clip(name: str, clip_id: str, exposure: int = 333, gain: int = 96, setup: bool = True,
             stop_app: bool = False) -> Path:
    clip = CLIPS[name]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rig = f"~/{RIG_DIR}"
    if stop_app:                                          # the table rig's old flow: stop askroom:latest
        ssh("C=$(docker ps -q --filter ancestor=askroom:latest); [ -n \"$C\" ] && docker stop $C >/dev/null",
            timeout=90)
    running = app_running()
    if running:
        raise RuntimeError("the app is running on the Jetson and holds the camera; its owner stops it first "
                           f"(cd {rig} && scripts/room_app.sh stop):\n{running}")
    if setup:
        ssh(f"cd {rig} && scripts/camera_setup.sh {exposure} {DEVICE} {gain} >/dev/null", timeout=90)
    offset = clock_offset()
    controls = camera_controls()
    out = f"data/clips/{clip_id}"
    ssh(f"rm -rf {rig}/{out}; mkdir -p {rig}/{out}")    # ours, so truth.json can be added after
    image = os.environ.get("ASKROOM_IMAGE")
    env = f"ASKROOM_IMAGE={shlex.quote(image)} " if image else ""
    rec = subprocess.Popen(["ssh", JETSON, f"cd {rig} && {env}scripts/dock.sh python3 -m eval.raw_record "
                            f"--out {out} --device {DEVICE} --seconds {clip['seconds']} "
                            f"--controls {shlex.quote(json.dumps(controls))} --git {git} --clock-offset {offset:.6f}"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    say("Get ready.").wait()
    for _ in range(120):                                  # the container starts, the camera settles
        if ssh(f"test -f {rig}/{out}/READY && echo y").strip() == "y":
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
        if s["say"]:
            say(s["say"])
        print(f"  {cues[-1] - t_zero:5.1f} s  {s['say']}", flush=True)
    rec.wait(timeout=clip["seconds"] + 60)
    say("Done.")
    frames = json.loads(ssh(f"cat {rig}/{out}/frames.json"))
    truth = truth_from(clip, cues, frames["wall"][0], offset)
    subprocess.run(["ssh", JETSON, f"cat > {rig}/{out}/truth.json"], input=json.dumps(truth, indent=1),
                   text=True, check=True)
    local = Path("data/clips") / clip_id
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{JETSON}:{RIG_DIR}/{out}/", str(local) + "/"], check=True)
    n, dur = len(frames["wall"]), frames["t"][-1]
    rec_meta = (json.loads((local / "meta.json").read_text()).get("record") or {}) if (local / "meta.json").exists() else {}
    size = "x".join(str(v) for v in rec_meta.get("size_px") or []) or "?"
    print(f"{clip_id}: {n} frames at {size}, {dur:.1f} s ({(n - 1) / max(dur, 1e-6):.1f} fps, "
          f"{rec_meta.get('dropped', 0)} dropped by the writer), clock offset {offset:+.3f} s -> {local}")
    return local


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", nargs="?", choices=sorted(CLIPS))
    ap.add_argument("--id")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gain", type=int, default=96)
    ap.add_argument("--no-setup", action="store_true", help="keep the camera's current settings (a tuned camera; "
                                                           "always so for room_* clips)")
    ap.add_argument("--stop-app", action="store_true",
                    help="stop askroom:latest containers first (the table rig's old flow; never the room app)")
    a = ap.parse_args(argv)
    if a.list or not a.clip:
        for k, c in CLIPS.items():
            print(f"{k:18s} {c['seconds']:3d} s  setup: {c['setup']}")
        return 0
    clip = CLIPS[a.clip]
    print(f"setup: {clip['setup']}")
    if _sys.stdin.isatty():
        input("Set the scene up as above, then press Enter to record (Ctrl-C cancels). ")
    run_clip(a.clip, a.id or f"{a.clip}_{time.strftime('%H%M%S')}", gain=a.gain,
             setup=not (a.no_setup or clip.get("room")), stop_app=a.stop_app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
