"""Replay a guided clip (eval/clip.py) through the pipeline production runs, and score it against the
clip's truth.json, so anyone can replay a recording and read PASS / FAIL without interpreting it.

    python -m eval.score_clip data/clips/<id>                       # needs ultralytics + the models
    python -m eval.score_clip data/clips/<id> --json out.json
    python -m eval.score_clip data/clips/<id> --config config.yaml  # today's thresholds, not the recorded ones
    python -m eval.score_clip data/clips/<id> --no-model            # Mac, no models: change proposer only
    python -m eval.score_clip data/clips/<id> --detect-model my.pt --proposals change   # Mac, a local .pt
    python -m eval.score_clip data/clips/<id> --hands-off --yoloe-model models/yoloe-26s-seg-pf.pt
                                          # Mac with only the YOLOE .pt: no fixed-class detector, so no hands

On the Jetson, run it inside the app's container (the TensorRT engines only load there):

    scripts/dock.sh python3 -m eval.score_clip data/clips/<id>

It loads the full detector and proposer, like the app, so stop the app first (8 GB shared RAM).

What is replayed, as main.build wires it: the config recorded in meta.json (or --config), table_cal from
meta.json in a temp file plus core.table.apply_saved_size, core.detect.Detector (configured backend and
proposer), core.hands.HandTracker, World(cfg, events, embed=core.embed.make_embedder(cfg)) on an
in-memory EventLog, and main.Room.perceive for every frame the live loop would have taken
(main.perception_max_fps by video time; --live-timing also drops the frames a slow detector misses).
Frames are video.mp4 with frames.json's t / wall; a clip recorded with room memory on holds full camera
frames (e.g. 2560x1440), and the replay puts them behind core.room_view.TableView exactly as
main.open_frames does (eval/clip.clip_view: the recorded table_view_rect, resized to frame_size_px), so the
table pipeline sees the live cut. When the replay config has room memory on and the clip recorded its
zones (meta.json room_zones, or --zones), main.make_room_memory runs it on the full frames, as the app does;
room things are named by Grok only with --grok (live calls, background thread, so not deterministic).
truth.commands and truth.questions go at their t
through Room.ask: the care layer and voice.pipeline.make_ask (both clocked by the clip's wall time, so
'put there 20 seconds ago' is measured on the clip) with voice.understand.Understander offline (the
rule parser) -> voice.answers, so "this is my X" binds an
alias exactly as it does live. They count as asked with the clicker; each record also says whether the
always-on mic's overheard filter would have dropped it (--overheard applies that filter).

Not reproduced: Grok (visual questions, open questions and the model interpreter need the network;
the rig offline answers the same way), audio (speech is injected as text), and mp4 compression (the live frame was not
re-encoded, so the change proposer sees slightly different pixels). Re-ID loads before the first
frame (live, it has no appearance evidence until its engine is ready).

Scoring. The detector's class names are not trusted as truth. Each physical prop (truth.props) is
mapped to the world entity that stands for it, over time:
  - a place / putdown / uncover / move step of the prop binds it to an entity that arrives (turns
    VISIBLE, or APPEARED / MOVED / PUT_BACK / UNCOVERED / TAKEN_OUT / CORRECTED) within resolve_s
    after the step (from pre_s before): the prop's current entity first, then the configured object
    the prop IS (only when its description is a configured name: 'wallet', 'notebook'; 'lego tub
    (container stand-in)' is matched by position), then any unclaimed entity (one near a hand first,
    then the earliest). A prop on the table from the start binds to its configured object if the
    world ever saw it; otherwise truth has no position for it, so it is a GUESS (reported): a
    configured object of its role ('container' -> box) or named in its description, else (a prop
    with no configured name only) the earliest unclaimed thing of the initial scene. Undeclared
    configured objects (a phone lying about) are never guessed for another type, and a named prop
    the detector never saw by name stays unmapped. Place each prop with a step when identity matters.
  - a place / putdown / move step does not bind an unrelated entity that the detector merely found
    again where it already lay (LOST_TRACK -> CORRECTED at the same spot, not held in between).
  - between its steps a prop resting on the table keeps its entity while that entity is visible where
    the prop is (within match_cm); another entity standing there instead takes the mapping over.
Steps: obj is the prop moved; parent the container / cover (put_inside obj into parent; cover: obj
goes under parent; uncover: parent lifted off obj). A step involves obj, parent and whatever is hidden
in them, from pre_s before to resolve_s after it (or until its entity arrived).

Births. Every thing the world admits within initial_s of the first processed frame, or later where a
detection already was in that window (clutter admitted late), is the initial scene (truth.scene
describes it): counted separately, never a false birth. A later thing born where an earlier identity
was lost is a rebirth (an identity change, prop 'scene'); where one is still visible, a duplicate.

Metrics, each with a PASS / FAIL line (Bars below; n/a when the clip has nothing to measure):
  false births / min         new things after the initial scene that are no prop and not born during a
                             place / putdown / exit_edge step (listed apart), duplicates included
  identity changes           a prop's entity changed while it rested untouched, or at a step that is
                             not a pickup / put (uncover, move), plus scene rebirths; changes at place /
                             pickup / putdown / put_inside are listed as excused
  missed placements          a place / putdown step no entity arrived for within resolve_s
  confirmation delay         step t -> the entity visible (median over placements). Timed from the
                             spoken cue, so it includes the person: 2.7-4.9 s from cue to hand off on
                             the rig clips, the world's own share ~0.4 s (TECHNICAL_DESIGN.md)
  untouched disappearances   PICKED_UP / COVERED / PUT_INSIDE / EXITED_VIEW / LOST_TRACK on a prop's
                             entity while no step involved the prop
  checkpoint states          state and (inside / under) parent of every expected prop at each checkpoint
  questions                  the answer points at the prop's entity (and, with expect_parent, the world
                             has it in that parent)
  hands at shell-game steps  a hand box within hand_near_cm of the props during pickup / put_inside /
                             cover steps
  <prop> detected            for each container / cover prop (or stand-in described as one): frames the
                             detector found that kind of object (the box, the notebook) where the prop is,
                             over the frames it should be on the table
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # allow `python eval/<name>.py` as well as `python -m eval.<name>`
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import bisect
import copy
import json
import logging
import math
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional

from eval.clip import Clip, load_clip

log = logging.getLogger("askroom.score_clip")

STATE = {"VISIBLE": "on_table", "HELD": "held", "INSIDE": "inside", "UNDER": "under", "GONE": "gone",
         "UNKNOWN": "unknown"}
AFTER = {"place": "on_table", "putdown": "on_table", "uncover": "on_table", "move": "on_table",
         "pickup": "held", "put_inside": "inside", "cover": "under", "exit_edge": "gone",
         "carry_to": "room", "remove": "gone"}      # carry_to: into truth step 'zone' (room demo, spec 0010)
RESOLVE = ("place", "putdown", "uncover", "move")       # the prop comes to rest in view: find its entity
MISSABLE = ("place", "putdown")
NEW_SPOT = ("place", "putdown", "move")     # an unrelated entity found again in place is no arrival for these
PUT = ("place", "pickup", "putdown", "put_inside")      # identity changes across these are excused
HAND_STEPS = ("pickup", "put_inside", "cover")
HIDDEN = ("inside", "under")
DISAPPEAR = {"PICKED_UP", "COVERED", "PUT_INSIDE", "EXITED_VIEW", "LOST_TRACK"}
ARRIVE = {"APPEARED", "MOVED", "PUT_BACK", "TAKEN_OUT", "UNCOVERED", "CORRECTED", "FOUND"}
THING = "thing:"


@dataclass
class Bars:
    """Matching windows and pass bars. Starting values: tune them from real clips like the thresholds."""
    resolve_s: float = 8.0              # after a step, its entity must arrive within this
    pre_s: float = 1.0                  # ... or this long before it (people act as they are told)
    match_cm: float = 8.0               # an entity this close to a resting prop's spot is that prop
    hand_near_cm: float = 10.0          # a hand box this close to a prop is handling it
    touch_s: float = 3.0                # a prop handled this recently may have moved: follow its entity
    initial_s: float = 3.0              # identities admitted this soon after the first frame are the scene
    false_births_per_min: float = 0.5
    identity_changes: int = 0
    missed_placements: int = 0
    confirm_median_s: float = 5.0        # from the spoken cue: the person takes 2.7-4.9 s to act; the world ~0.4 s
    false_disappearances: int = 0
    checkpoint_accuracy: float = 0.9
    question_accuracy: float = 1.0
    role_rate: float = 0.8


# ----- replay -------------------------------------------------------------------------------------------

@dataclass
class Sample:
    """The world after one processed frame (read through state_json) and what was detected in it."""
    t: float
    ents: dict                          # entity -> (status, parent, pos_cm or None)
    hands: list                         # hand boxes, table cm
    seen: dict                          # detected class -> [centre cm] ('thing' for proposals)
    zones: dict = field(default_factory=dict)   # entity -> room zone, for entities off the table
    names: dict = field(default_factory=dict)   # thing -> {"label", "aliases", "guess"} when it has any


@dataclass
class Trace:
    samples: list = field(default_factory=list)
    events: list = field(default_factory=list)      # {"t", "obj", "type", "parent", "to_cm", "from_cm"}
    asks: list = field(default_factory=list)        # commands and questions with their answers
    kinds: dict = field(default_factory=dict)       # entity -> kind, as the world reports it
    merged: dict = field(default_factory=dict)      # entity -> entity it was merged into
    objects: dict = field(default_factory=dict)     # the config's objects: name -> kind
    frames_decoded: int = 0
    calibrating: int = 0
    errors: int = 0
    wall_s: float = 0.0
    detect_ms: list = field(default_factory=list)
    max_fps: float = 0.0
    live_timing: bool = False
    detector: str = ""
    view: Optional[list] = None                     # [table_view_rect, out_size] the frames were cut to
    room: bool = False                              # room memory ran


def make_detector(cfg: dict, table):
    """The app's detector: config backend (YOLO engine / .pt) and proposer. Tests replace this."""
    import core.detect
    return core.detect.Detector(cfg, table)


class NoBoxes:
    """--no-model backend: the fixed-class detector finds nothing (no known objects, no hands)."""

    def infer(self, img) -> list:
        return []


def make_model_free_detector(cfg: dict, table):
    """--no-model, for a Mac without the models: no YOLO boxes, and the model-free change proposer
    (proposals.kind change) in place of the configured one. Only open-world things reach the world."""
    import core.detect
    cfg["proposals"] = dict(cfg.get("proposals") or {}, enabled=True, kind="change")
    return core.detect.Detector(cfg, table, backend=NoBoxes())


def prepare_config(clip: Clip, cfg: Optional[dict], workdir: str) -> dict:
    """The recorded config (or cfg), writing its paths into workdir: table_cal from meta.json, an
    in-memory event log, snapshots in workdir. Re-ID loads up front so every replay is the same."""
    from core.config import load_config
    base = cfg if cfg is not None else (clip.meta.get("config") or load_config())
    cfg = copy.deepcopy(base)
    cal = Path(workdir) / "table_cal.json"
    if clip.meta.get("table_cal"):
        cal.write_text(json.dumps(clip.meta["table_cal"]))
    snaps = Path(workdir) / "snapshots"
    snaps.mkdir(exist_ok=True)
    cfg["paths"] = dict(cfg.get("paths") or {}, table_cal=str(cal), events_db=":memory:", snapshots=str(snaps),
                        laser_cal=str(Path(workdir) / "laser_cal.json"))
    reid = cfg.get("reid") or {}
    if reid.get("enabled"):
        cfg["reid"] = dict(reid, background=False)
    rm = cfg.get("room_memory") or {}
    if rm.get("enabled"):           # the recorded zones, never whatever room_zones.json this checkout has
        zones = Path(workdir) / "room_zones.json"
        if clip.meta.get("room_zones"):
            zones.write_text(json.dumps(clip.meta["room_zones"]))
        cfg["room_memory"] = dict(rm, zones_path=str(zones))
    return cfg


def make_hands_off_detector(cfg: dict, table):
    """--hands-off: no fixed-class detector (so no hands and no prop labels, which the rig's conf_threshold
    0.99 turns off anyway), the configured proposer (YOLOE .pt on a Mac). Runs with only the YOLOE model."""
    import core.detect
    return core.detect.Detector(cfg, table, backend=NoBoxes())


def replay_clip(clip: Clip, cfg: Optional[dict] = None, detector=None, max_fps: Optional[float] = None,
                live_timing: bool = False, overheard: bool = False, no_model: bool = False,
                hands_off: bool = False, grok: bool = False) -> Trace:
    """Every frame the live perception loop would have taken, through Room.perceive; truth.commands and
    truth.questions through Room.ask at their t. Returns what the world believed after each frame.
    A clip recorded with room memory on is cut to the table view as live (clip.view()), and room memory
    runs on the full frames when the config has it on and the zones were recorded. grok: Grok names new
    things (table and room, auto_name) with live calls, as the app does online."""
    import core.table
    from core.embed import make_embedder
    from core.events import EventLog
    from core.hands import HandTracker
    from core.room_view import TableView
    from core.world import World
    from eval.clip import FrameSlot
    from main import Room, make_room_memory
    from voice.pipeline import make_ask
    from voice.understand import Understander

    with tempfile.TemporaryDirectory(prefix="askroom_clip_") as tmp:
        cfg = prepare_config(clip, cfg, tmp)
        core.table.apply_saved_size(cfg)            # one-tag mode: the saved tracked area (main.build)
        table = core.table.Table(cfg)
        events = EventLog(":memory:", cfg["paths"]["snapshots"])
        try:
            world = World(cfg, events, embed=make_embedder(cfg))
            if detector is None:
                detector = (make_model_free_detector(cfg, table) if no_model else
                            make_hands_off_detector(cfg, table) if hands_off else make_detector(cfg, table))
            namer = None
            if grok:                                    # main.build: auto names, and world.online from NetMonitor
                import core.auto_name
                world.online = True
                namer = core.auto_name.from_config(cfg, world, online=lambda: True)
            view = clip.view()
            slot = FrameSlot()
            frames = TableView(slot, view[0], view[1]) if view is not None else None
            hands = HandTracker(frame_size=tuple(cfg.get("frame_size_px") or (1280, 720)))
            interpret = Understander(cfg, online=lambda: False)       # offline: the rule parser
            clock = {"wall": clip.wall[0] if clip.wall else time.time()}
            ask = make_ask(cfg, world, events, net=None, interpret=interpret, visual=None,
                           clock=lambda: clock["wall"])                # 'ago' on clip time
            room = Room(cfg, world, events, table, frames, None, ask, detector=detector, hands=hands,
                        interpret=interpret)
            if view is not None and (cfg.get("room_memory") or {}).get("enabled") \
                    and getattr(detector, "backend", None) is not None:
                room.room_memory = make_room_memory(cfg, world, detector, view[0])
            if (cfg.get("care") or {}).get("enabled", True):          # main.build's attach_care, on clip time
                from voice.care import Care
                care = Care(cfg, world, events, room.base_ask, clock=lambda: clock["wall"], online=lambda: False)
                room.base_ask, room.care = care.ask, care
            fps = max_fps or float((cfg.get("main") or {}).get("perception_max_fps", 15))
            pc = cfg.get("proposals") or {}
            model = "none (hands off)" if hands_off else (cfg.get("detect") or {}).get("model")
            kind = pc.get("kind") if pc.get("enabled") else "off"
            pmodel = (pc.get(kind) or {}).get("model") if kind == "yoloe" else None
            trace = Trace(objects=dict(cfg.get("objects") or {}), max_fps=fps, live_timing=live_timing,
                          detector="model-free (no YOLO, change proposer)" if no_model else
                          f"{type(detector).__name__}: {model}, proposals {kind}"
                          + (f" ({pmodel})" if pmodel else "") + (", Grok names" if grok else ""),
                          view=[list(view[0]), list(view[1])] if view is not None else None,
                          room=room.room_memory is not None)
            try:
                _run(clip, room, world, interpret, trace, 1.0 / fps if fps > 0 else 0.0, clock, overheard,
                     slot if frames is not None else None, frames)
            finally:
                for x in (room.room_memory, namer):
                    if x is not None:
                        x.stop()
            return trace
        finally:
            events.close()


def _run(clip: Clip, room, world, interpret, trace: Trace, period: float, clock: dict, overheard: bool,
         slot=None, view=None) -> None:
    """The perception loop on video time. It is ready again max(period, the step's time with
    --live-timing) after it started a step, and then takes the NEWEST frame (the camera thread keeps
    only the latest), or waits for the next one. Speech goes in at its t, between steps. slot / view: the
    FrameSlot and TableView of a clip the app cut a table view from (Room.frames is the view)."""
    said = sorted([("command", c) for c in clip.truth["commands"]]
                  + [("question", q) for q in clip.truth["questions"]], key=lambda x: float(x[1]["t"]))
    k, ready, t0 = 0, -math.inf, time.perf_counter()

    def say_until(t: float) -> None:
        nonlocal k
        while k < len(said) and float(said[k][1]["t"]) <= t + 1e-9:
            trace.asks.append(_ask(room, world, interpret, *said[k], overheard))
            k += 1

    def step(f, start: float) -> float:
        clock["wall"] = f.wall + (start - f.t)
        say_until(start)
        s0 = time.perf_counter()
        try:
            if view is not None:
                slot.cur = f
                f = view.at(f.t)
            out = room.perceive(f)
        except Exception:
            log.exception("perception step failed at t=%.2f", f.t)
            trace.errors += 1
            return start + period
        spent = time.perf_counter() - s0
        if out is None:
            trace.calibrating += 1
        else:
            dets, evs = out
            if getattr(room.detector, "last_ms", None) is not None:
                trace.detect_ms.append(float(room.detector.last_ms))
            _sample(trace, world, f.t, dets, evs)
        return start + max(period, spent if trace.live_timing else 0.0)

    prev = None
    for f in clip.frames():
        trace.frames_decoded += 1
        if prev is not None and f.t > ready + 1e-9:      # prev is the newest frame when the loop is ready
            ready = step(prev, max(ready, prev.t))
        prev = f
    if prev is not None:
        step(prev, max(ready, prev.t))
    say_until(math.inf)
    trace.wall_s = time.perf_counter() - t0


def _sample(trace: Trace, world, t: float, dets, evs) -> None:
    st = world.state_json()
    ents, zones, names = {}, {}, {}
    for e in st["entities"]:
        ents[e["name"]] = (e["status"], e["parent"], tuple(e["pos_cm"]) if e["pos_cm"] is not None else None)
        trace.kinds[e["name"]] = e["kind"]
        if e.get("zone") not in (None, "table"):
            zones[e["name"]] = e["zone"]
        got = {k: e[k] for k in ("label", "aliases", "guess") if e.get(k)}
        if got:
            names[e["name"]] = got
    trace.merged = dict(st.get("merged") or {})
    seen: dict = defaultdict(list)
    for d in dets.items:
        seen[d.cls].append(tuple(d.center_cm))
    trace.samples.append(Sample(t=t, ents=ents, hands=[tuple(h.box_cm) for h in dets.hands], seen=dict(seen),
                                zones=zones, names=names))
    for ev in evs:
        trace.events.append({"t": ev.t, "obj": ev.obj, "type": str(ev.type), "parent": ev.parent,
                             "to_cm": list(ev.to_cm) if ev.to_cm is not None else None,
                             "from_cm": list(ev.from_cm) if ev.from_cm is not None else None})


def _ask(room, world, interpret, kind: str, item: dict, overheard: bool) -> dict:
    text = str(item["text"])
    heard = interpret(text, overheard=True).kind != "IGNORE"
    rec = {"t": float(item["t"]), "kind": kind, "text": text, "overheard_ignored": not heard}
    if overheard and not heard:
        return dict(rec, answer=None, point_at=None, action=None, intent="IGNORE")
    ans = room.ask(text, "voice")
    rec.update(answer=ans.text, point_at=ans.point_at, action=ans.action, intent=interpret(text).kind,
               merged=dict(world.state_json().get("merged") or {}))
    return rec


# ----- scoring ------------------------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _box_dist(p, box) -> float:
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return math.hypot(dx, dy)


def _survivor(name: Optional[str], merged: dict) -> Optional[str]:
    seen = set()
    while name in merged and name not in seen:
        seen.add(name)
        name = merged[name]
    return name


def _normalize_steps(steps: list, types: dict, objects: dict) -> list:
    """Copies with obj = the prop that is moved or hidden. cover / uncover written the other way round
    (obj the notebook, parent what goes under it) are turned around."""
    out = []
    for s in steps:
        s = dict(s, t=float(s["t"]))
        obj, par = s.get("obj"), s.get("parent")
        if s["event"] in ("cover", "uncover") and obj and par \
                and objects.get(types.get(obj)) == "cover" and objects.get(types.get(par)) != "cover":
            s["obj"], s["parent"] = par, obj
        out.append(s)
    return out


class _Scorer:
    def __init__(self, trace: Trace, truth: dict, bars: Bars):
        self.tr, self.bars = trace, bars
        self.types = dict(truth["props"])
        self.props = list(self.types)
        self.objects = trace.objects
        self.steps = _normalize_steps(truth["steps"], self.types, self.objects)
        self.samples = trace.samples
        self.ts = [s.t for s in trace.samples]
        self.prop_types = {n for n in (self.named(p) for p in self.props) if n}
        self._arrivals()
        self._births()
        self._expectations()
        self._windows()

    # -- lookups

    def at(self, t: float) -> Optional[Sample]:
        """The world as of t: the latest processed frame at or before it."""
        i = bisect.bisect_right(self.ts, t + 1e-9) - 1
        return self.samples[i] if i >= 0 else None

    def between(self, a: float, b: float) -> list:
        return self.samples[bisect.bisect_left(self.ts, a - 1e-9):bisect.bisect_right(self.ts, b + 1e-9)]

    def expected(self, p: str, t: float):
        """(state, parent) the truth says p is in at t, or None before it is on the table."""
        got = None
        for te, state, parent in self.expect[p]:
            if te <= t + 1e-9:
                got = (state, parent)
        return got

    def involved(self, p: str, t: float) -> bool:
        return any(a <= t <= b for a, b in self.win[p])

    def mapped(self, p: str, t: float) -> Optional[str]:
        got = None
        for m in self.mapping.get(p, ()):
            if m["t"] <= t + 1e-9:
                got = m["entity"]
        return got

    def as_prop(self, ent: Optional[str], t: float) -> Optional[str]:
        """A world parent as a prop id: None, 'hand', the prop mapped to it at t, or the raw name."""
        if ent is None:
            return None
        if ent.startswith("hand"):
            return "hand"
        for p in self.props:
            if self.mapped(p, t) == ent:
                return p
        return ent

    def named(self, p: str) -> Optional[str]:
        """The configured object a prop is, when its description IS a configured name ('wallet',
        'pill bottle'); None for 'small object' or 'lego tub (container stand-in)'."""
        key = str(self.types[p]).strip().lower().replace(" ", "_")
        return key if key in self.objects else None

    def allowed(self, p: str, ent: str) -> bool:
        """Any entity can stand for a prop by position (a lego tub the detector calls 'box' is the tub),
        except a configured object that is another prop by name."""
        return ent.startswith(THING) or ent == self.named(p) or ent not in self.prop_types

    # -- precomputation

    def _arrivals(self) -> None:
        self.arr = defaultdict(list)           # entity -> [(t, pos)]
        self.births = {}                        # thing -> (t, pos)
        prev: dict = {}
        for s in self.samples:
            for n, (st, _, pos) in s.ents.items():
                if st == "VISIBLE" and prev.get(n) != "VISIBLE" and pos is not None:
                    self.arr[n].append((s.t, pos))
            prev = {n: v[0] for n, v in s.ents.items()}
        for ev in self.tr.events:
            if ev["type"] in ARRIVE:
                pos = ev["to_cm"]
                if pos is None:
                    s = self.at(ev["t"])
                    v = s.ents.get(ev["obj"]) if s else None
                    pos = v[2] if v else None
                if pos is not None:
                    self.arr[ev["obj"]].append((ev["t"], tuple(pos)))
            if ev["type"] == "APPEARED" and ev["obj"].startswith(THING):
                self.births.setdefault(ev["obj"], (ev["t"], ev["to_cm"]))
        for n in self.arr:
            self.arr[n].sort(key=lambda x: x[0])
        self.ever_visible = set(self.arr)
        self.resighted = {n: {t for t, pos in arr if self._resighting(n, t, pos)} for n, arr in self.arr.items()}

    def _resighting(self, n: str, t: float, pos) -> bool:
        """n turns up within match_cm of where it was last VISIBLE, not held in between: the detector lost
        and found an object that never moved (a flicker, LOST_TRACK -> CORRECTED), no arrival of anything.
        A thing picked up and put back in its spot (HELD, PUT_BACK) did arrive."""
        i = bisect.bisect_left(self.ts, t - 1e-9) - 1
        while i >= 0:
            v = self.samples[i].ents.get(n)
            if v is None:
                return False
            if v[0] == "HELD":
                return False
            if v[0] == "VISIBLE" and v[2] is not None:
                return _dist(v[2], pos) <= self.bars.match_cm and not any(
                    ev["obj"] == n and ev["type"] in ("PICKED_UP", "PUT_BACK") and self.samples[i].t < ev["t"] <= t + 1e-9
                    for ev in self.tr.events)
            i -= 1
        return False

    def _births(self) -> None:
        """What each new thing was, before knowing which props it stands for:
          initial    admitted within initial_s of the first frame, or late where a detection already
                     was in that window (static clutter admitted late: the scene)
          rebirth    where an earlier identity was lost: the same object, a new identity (churn)
          duplicate  where an earlier identity is still visible: a second identity for one object
          new        anything else (a place step, a false birth, a hand fragment ...)"""
        first = self.ts[0] if self.ts else 0.0
        early = [c for s in self.between(first, first + self.bars.initial_s) for cs in s.seen.values() for c in cs]
        self.birth_kind, self.birth_of = {}, {}
        for n, (t, pos) in self.births.items():
            if t <= first + self.bars.initial_s + 1e-9:
                self.birth_kind[n] = "initial"
                continue
            if pos is None:
                self.birth_kind[n] = "new"
                continue
            before = self.at(t - 1e-6)
            owners = sorted((_dist(v[2], pos), m, v[0], v[2]) for m, v in (before.ents.items() if before else ())
                            if m != n and v[2] is not None and self.arr.get(m) and self.arr[m][0][0] < t
                            and _dist(v[2], pos) <= self.bars.match_cm)
            owner = owners[0] if owners else None
            # its own box was there from the start: an early detection nearer to it than to any neighbour
            if any(_dist(c, pos) <= self.bars.match_cm and (owner is None or _dist(c, pos) < _dist(c, owner[3]))
                   for c in early):
                self.birth_kind[n] = "initial"
            elif owner is not None:
                self.birth_kind[n] = "duplicate" if owner[2] == "VISIBLE" else "rebirth"
                self.birth_of[n] = owner[1]
            else:
                self.birth_kind[n] = "new"
        self.scene = {n for n, k in self.birth_kind.items() if k == "initial"}

    def _expectations(self) -> None:
        self.expect = {p: [] for p in self.props}
        first = {}
        for s in self.steps:
            for q in (s.get("obj"), s.get("parent")):
                if q in self.types and q not in first:
                    first[q] = s
        self.initial = {p: not (p in first and first[p]["event"] == "place" and first[p].get("obj") == p)
                        for p in self.props}
        for p in self.props:
            if self.initial[p]:
                self.expect[p].append((-math.inf, "on_table", None))
        for s in self.steps:
            p = s.get("obj")
            if p in self.types and s["event"] in AFTER:
                state = AFTER[s["event"]]
                self.expect[p].append((s["t"], state, s.get("parent") if state in HIDDEN else None))

    def _windows(self) -> None:
        self.win = {p: [] for p in self.props}
        self.step_props = []
        for s in self.steps:
            inv = {q for q in (s.get("obj"), s.get("parent")) if q in self.types}
            for q in self.props:
                e = self.expected(q, s["t"] - 1e-6)
                if e and e[0] in HIDDEN and e[1] in inv:
                    inv.add(q)
            w = [s["t"] - self.bars.pre_s, s["t"] + self.bars.resolve_s]
            for q in inv:
                self.win[q].append(w)       # the same list: resolving the step shortens it for every prop
            self.step_props.append((s, inv, w))

    # -- identity mapping

    def run_mapping(self) -> None:
        self.mapping = {p: [] for p in self.props}
        self.cur = {p: None for p in self.props}
        self.anchor = {p: None for p in self.props}
        self.changes, self.placements = [], []
        self.guessed: set = set()
        self.placed: set = set()                # entities a place / putdown step bound: placements
        k = 0
        for s in self.samples:
            while k < len(self.step_props) and self.step_props[k][0]["t"] <= s.t:
                self._step(*self.step_props[k])
                k += 1
            for p in self.props:
                self._rest(p, s)
        for sp in self.step_props[k:]:
            self._step(*sp)
        for p in self.props:
            self.mapping[p].sort(key=lambda m: m["t"])

    def _claimed(self, p: str) -> set:
        return {e for q, e in self.cur.items() if q != p and e is not None}

    def _bind(self, p: str, ent: str, t: float, pos, why: str, excused: Optional[bool] = None) -> None:
        old = self.cur[p]
        if old is not None and old != ent:
            self.changes.append({"t": round(t, 2), "prop": p, "from": old, "to": ent, "excused": bool(excused),
                                 "why": why})
        if old != ent:
            self.mapping[p].append({"t": round(t, 3), "entity": ent, "why": why})
        self.cur[p], self.anchor[p] = ent, pos

    def _step(self, s: dict, inv: set, w: list) -> None:
        p = s.get("obj")
        if s["event"] not in RESOLVE or p not in self.types:
            return
        ts = s["t"]
        nxt = min((x["t"] for x in self.steps if x["t"] > ts and p in (x.get("obj"), x.get("parent"))),
                  default=math.inf)
        a, b = ts - self.bars.pre_s, min(ts + self.bars.resolve_s, nxt)
        claimed = self._claimed(p)
        hand_boxes = [h for smp in self.between(a - 1.0, b) for h in smp.hands]
        cands = []
        for ent, arr in self.arr.items():
            if ent in claimed or not self.allowed(p, ent):
                continue
            rank = 0 if ent == self.cur[p] else (1 if ent == self.named(p) else 2)
            # an unrelated entity found again where it already lay is not the prop coming to rest there
            skip = self.resighted.get(ent, ()) if rank == 2 and s["event"] in NEW_SPOT else ()
            hits = [(t, pos) for t, pos in arr if a <= t <= b and t not in skip]
            if not hits:
                continue
            t_arr, pos = hits[0]
            near = any(_box_dist(pos, h) <= self.bars.hand_near_cm for h in hand_boxes)
            cands.append(((rank, 0 if near else 1, t_arr), ent, t_arr, pos))
        row = {"t": ts, "event": s["event"], "prop": p, "entity": None, "t_confirm": None, "delay_s": None,
               "missed": True, "counts": s["event"] in MISSABLE}
        if cands:
            _, ent, t_arr, pos = min(cands, key=lambda c: c[0])
            self._bind(p, ent, t_arr, pos, f"{s['event']} at {ts:g} s", excused=s["event"] in PUT)
            w[1] = min(w[1], max(t_arr, ts) + self.bars.pre_s)
            row.update(entity=ent, t_confirm=round(t_arr, 2), delay_s=round(t_arr - ts, 2), missed=False)
            if s["event"] in MISSABLE:
                self.placed.add(ent)
        self.placements.append(row)

    def _rest(self, p: str, s: Sample) -> None:
        """A prop resting on the table between its steps: keep, bind or take over its entity by position."""
        if self.involved(p, s.t):
            return
        exp = self.expected(p, s.t)
        if not exp or exp[0] != "on_table":
            return
        claimed = self._claimed(p)
        if self.cur[p] is None:
            if self.initial[p]:
                self._bind_initial(p, s, claimed)
            return
        v, a = s.ents.get(self.cur[p]), self.anchor[p]
        vis = v is not None and v[0] == "VISIBLE" and v[2] is not None
        if vis and (a is None or _dist(v[2], a) <= self.bars.match_cm):
            self.anchor[p] = v[2]
            return
        if a is not None and self._touched(p, a, s.t):     # the person moved it on after its step:
            if vis:                                         # its entity went with it, the spot is stale
                self.anchor[p] = v[2]
            return
        if a is not None:
            there = sorted((_dist(x[2], a), n) for n, x in s.ents.items()
                           if n != self.cur[p] and n not in claimed and x[0] == "VISIBLE" and x[2] is not None
                           and self.allowed(p, n) and _dist(x[2], a) <= self.bars.match_cm)
            if there:
                n = there[0][1]
                self._bind(p, n, s.t, s.ents[n][2], "another entity stands where the untouched prop is",
                           excused=False)
                return
        if vis:
            self.anchor[p] = v[2]

    def _touched(self, p: str, spot, t: float) -> bool:
        """A hand near the prop's spot, or its entity held, within touch_s before t."""
        ent = self.cur[p]
        for smp in self.between(t - self.bars.touch_s, t):
            v = smp.ents.get(ent)
            if v is not None and v[0] == "HELD":
                return True
            if any(_box_dist(spot, h) <= self.bars.hand_near_cm for h in smp.hands):
                return True
        return False

    def _bind_initial(self, p: str, s: Sample, claimed: set) -> None:
        """A prop on the table from the start: its configured object by name if the world ever saw it,
        else (truth has no positions) a GUESS: a configured object of its role or named in its
        description, else (only for a prop with no configured name) the earliest-born unclaimed thing
        of the initial scene. Other configured objects (an undeclared phone or wallet lying about) are
        never guessed; a named prop the detector never saw by name stays unmapped."""
        named = self.named(p)
        if named and named not in claimed and named in self.ever_visible:
            v = s.ents.get(named)
            if v and v[0] == "VISIBLE" and v[2] is not None:
                self._bind(p, named, self.arr[named][0][0], v[2], "on the table from the start (by name)")
            return
        role = self.role(p)
        words = set(re.findall(r"[a-z]+", str(self.types[p]).lower()))
        cands = []
        for n, v in s.ents.items():
            if n in claimed or v[0] != "VISIBLE" or v[2] is None or not self.allowed(p, n) or not self.arr.get(n):
                continue
            if n in self.objects:       # a configured object of the prop's role, or named in its description
                if not ((role and self.objects[n] == role) or n.replace("_", " ") in " ".join(words)):
                    continue
                rank = 0
            elif n in self.scene and not named:
                rank = 1
            else:
                continue
            cands.append((rank, self.arr[n][0][0], n))
        if cands:
            _, t0, n = min(cands)
            self._bind(p, n, t0, s.ents[n][2], "GUESS: on the table from the start; truth has no positions, so "
                                               "the earliest unclaimed identity of its kind (place it with a step "
                                               "to be sure)")
            self.guessed.add(p)

    # -- metrics

    def report(self, clip: Clip) -> dict:
        self.run_mapping()
        b, tr = self.bars, self.tr
        video_s = clip.duration_s or ((self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 0.0)
        births, initial, at_steps = self._classify_births()
        per_min = len(births) / (video_s / 60.0) if video_s > 0 else float(len(births))
        counted = [c for c in self.changes if not c["excused"]]
        places = [p for p in self.placements if p["counts"]]
        missed = [p for p in places if p["missed"]]
        delays = [p["delay_s"] for p in places if p["delay_s"] is not None]
        false_dis = self._false_disappearances()
        checks = self._checkpoints(clip)
        questions = self._questions(clip)
        commands = [self._said(a) for a in tr.asks if a["kind"] == "command"]
        hand_steps = self._hand_steps()
        roles = self._roles()
        handled = [bool(h["near"] if h["near"] is not None else h["hands_seen"]) for h in hand_steps]
        c_acc = sum(r["ok"] for r in checks) / len(checks) if checks else None
        q_acc = sum(q["correct"] for q in questions) / len(questions) if questions else None
        crit = [
            _crit("false births", f"{len(births)} ({per_min:.2f}/min, bar <= {b.false_births_per_min:g})",
                  per_min <= b.false_births_per_min),
            _crit("identity changes", f"{len(counted)}" + _list(counted, lambda c: f"{c['prop']}: {c['from']} -> "
                                                               f"{c['to']} at {c['t']:g} s")
                  + (f"; {len(self.changes) - len(counted)} excused" if len(self.changes) > len(counted) else ""),
                  len(counted) <= b.identity_changes),
            _crit("missed placements", f"{len(missed)} of {len(places)}", len(missed) <= b.missed_placements,
                  applies=bool(places)),
            _crit("confirmation delay", (f"median {median(delays):.2f} s, max {max(delays):.2f} s "
                                         f"(bar <= {b.confirm_median_s:g} s)") if delays else "no placements",
                  bool(delays) and median(delays) <= b.confirm_median_s, applies=bool(delays)),
            _crit("untouched disappearances", f"{len(false_dis)}" + _list(false_dis, lambda e: f"{e['type']} of "
                                                                          f"{e['prop']} at {e['t']:g} s"),
                  len(false_dis) <= b.false_disappearances),
            _crit("checkpoint states", f"{sum(r['ok'] for r in checks)}/{len(checks)}" + _list(
                [r for r in checks if not r["ok"]], lambda r: f"{r['prop']} at {r['t']:g} s: {r['got_state']}"
                f"/{r['got_parent']} not {r['state']}/{r['parent']}"),
                  c_acc is not None and c_acc >= b.checkpoint_accuracy, applies=bool(checks)),
            _crit("questions", f"{sum(q['correct'] for q in questions)}/{len(questions)}" + _list(
                [q for q in questions if not q["correct"]], lambda q: f"{q['text']!r} -> {q['point_at']} "
                f"({_want(q)})"),
                  q_acc is not None and q_acc >= b.question_accuracy, applies=bool(questions)),
            _crit("hands at shell-game steps", f"{sum(handled)}/{len(handled)}" + _list(
                [h for h, ok in zip(hand_steps, handled) if not ok], lambda h: f"{h['event']} at {h['t']:g} s: "
                f"{h['frames_with_hands']}/{h['frames']} frames with a hand"), all(handled), applies=bool(handled)),
        ]
        for p, r in roles.items():
            crit.append(_crit(f"{p} ({r['type']}) detected", f"{100 * r['rate']:.1f}% of {r['frames']} frames"
                              f" (bar >= {100 * b.role_rate:.0f}%)", r["rate"] >= b.role_rate,
                              applies=r["frames"] > 0))
        return {
            "clip": clip.name,
            "replay": {"frames_decoded": tr.frames_decoded, "frames_processed": len(self.samples),
                       "frames_calibrating": tr.calibrating, "errors": tr.errors, "video_s": round(video_s, 2),
                       "wall_s": round(tr.wall_s, 2),
                       "fps": round(len(self.samples) / tr.wall_s, 1) if tr.wall_s > 0 else None,
                       "max_fps": tr.max_fps, "live_timing": tr.live_timing, "detector": tr.detector,
                       "view": tr.view, "room_memory": tr.room,
                       "detect_ms_median": round(median(tr.detect_ms), 1) if tr.detect_ms else None,
                       "git": clip.meta.get("git"), "models": clip.meta.get("models"),
                       "clock_offset_s": clip.meta.get("clock_offset_s")},
            "props": self.types, "scene": clip.truth.get("scene"),
            "mapping": self.mapping,
            "false_births": births, "false_births_per_min": round(per_min, 3),
            "initial_scene": {"count": len(initial), "entities": initial},
            "births_at_steps": at_steps, "guessed": sorted(self.guessed),
            "identity_changes": counted, "identity_changes_excused": [c for c in self.changes if c["excused"]],
            "placements": self.placements, "missed_placements": len(missed),
            "confirm_delay_s": {"median": round(median(delays), 2), "max": round(max(delays), 2)} if delays else None,
            "false_disappearances": false_dis,
            "checkpoints": checks, "checkpoint_accuracy": c_acc,
            "questions": questions, "question_accuracy": q_acc, "commands": commands,
            "hand_steps": hand_steps, "roles": roles,
            "events": Counter(e["type"] for e in tr.events),
            "bars": asdict(b),
            "criteria": crit,
            "pass": all(c["result"] != "FAIL" for c in crit),
        }

    def _classify_births(self) -> tuple[list, list, list]:
        """(false births, the initial scene, births in a place / putdown / exit_edge step's window). The
        initial scene lists every identity admitted at the start, with the prop it stands for if any; of
        the rest, things mapped to a prop are that prop, and an unmapped rebirth is an identity change
        (prop 'scene') rather than a birth."""
        ever = {m["entity"]: p for p in self.props for m in self.mapping[p]}
        births, initial, at_steps = [], [], []
        for n, (t, pos) in sorted(self.births.items(), key=lambda x: x[1][0]):
            kind = self.birth_kind[n]
            prop = ever.get(n) or ever.get(_survivor(n, self.tr.merged))
            row = {"entity": n, "t": round(t, 2), "pos_cm": [round(float(v), 1) for v in pos] if pos else None,
                   "kind": kind, "of": self.birth_of.get(n), "during": self._during(t), "prop": prop}
            if kind == "initial" and n not in self.placed:
                initial.append(row)
            elif prop is not None:
                continue
            elif kind == "rebirth":
                self.changes.append({"t": round(t, 2), "prop": "scene", "from": self.birth_of[n], "to": n,
                                     "excused": False, "why": "an untracked object came back as a new identity"})
            elif kind == "new" and any(s["event"] in ("place", "putdown", "exit_edge")
                                       and s["t"] - self.bars.pre_s <= t <= s["t"] + self.bars.resolve_s
                                       for s in self.steps):
                at_steps.append(row)
            else:
                births.append(row)
        return births, initial, at_steps

    def _during(self, t: float) -> Optional[str]:
        live = [s for s in self.steps if s["t"] - self.bars.pre_s <= t <= s["t"] + self.bars.resolve_s]
        return f"{live[-1]['event']} at {live[-1]['t']:g} s" if live else None

    def _false_disappearances(self) -> list:
        out = []
        for ev in self.tr.events:
            if ev["type"] not in DISAPPEAR:
                continue
            for p in self.props:
                if self.mapped(p, ev["t"]) == ev["obj"] and not self.involved(p, ev["t"]):
                    out.append({"t": round(ev["t"], 2), "prop": p, "entity": ev["obj"], "type": ev["type"],
                                "parent": ev["parent"]})
        return out

    def _checkpoints(self, clip: Clip) -> list:
        rows = []
        for c in clip.truth["checkpoints"]:
            t = float(c["t"])
            s = self.at(t)
            for p, exp in (c.get("expect") or {}).items():
                ent = self.mapped(p, t)
                v = s.ents.get(ent) if (s is not None and ent) else None
                got_state = STATE.get(v[0], str(v[0]).lower()) if v else None
                got_parent = self.as_prop(v[1], t) if v else None
                want, want_parent = exp.get("state"), exp.get("parent")
                ok = got_state == want and (want not in HIDDEN or got_parent == want_parent)
                rows.append({"t": t, "prop": p, "entity": ent, "state": want, "parent": want_parent,
                             "got_state": got_state, "got_parent": got_parent, "ok": ok})
        return rows

    def _questions(self, clip: Clip) -> list:
        asked = {(a["t"], a["text"]): a for a in self.tr.asks if a["kind"] == "question"}
        out = []
        for q in clip.truth["questions"]:
            t = float(q["t"])
            a = asked.get((t, str(q["text"]))) or {}
            want = self.mapped(q.get("expect_prop"), t) if q.get("expect_prop") in self.types else None
            got = _survivor(a.get("point_at"), a.get("merged") or {})
            ent_ok = want is not None and got == want
            parent_ok = None
            if q.get("expect_parent"):
                s = self.at(t - 1e-6)
                v = s.ents.get(want) if (s is not None and want) else None
                parent_ok = v is not None and self.as_prop(v[1], t) == q["expect_parent"]
            out.append({"t": t, "text": q["text"], "answer": a.get("answer"), "intent": a.get("intent"),
                        "point_at": got, "expect_prop": q.get("expect_prop"), "expected_entity": want,
                        "expect_parent": q.get("expect_parent"), "parent_ok": parent_ok, "overheard_ignored": a.get("overheard_ignored"),
                        "correct": bool(ent_ok and parent_ok is not False)})
        return out

    def _said(self, a: dict) -> dict:
        got = _survivor(a.get("point_at"), a.get("merged") or {})
        return {"t": a["t"], "text": a["text"], "answer": a.get("answer"), "intent": a.get("intent"),
                "point_at": got, "prop": self.as_prop(got, a["t"]) if got else None,
                "overheard_ignored": a["overheard_ignored"]}

    def _hand_steps(self) -> list:
        out = []
        for i, (s, inv, _) in enumerate(self.step_props):
            if s["event"] not in HAND_STEPS:
                continue
            later = [x[0]["t"] for x in self.step_props[i + 1:] if x[0]["t"] > s["t"]]
            a, b = s["t"] - self.bars.pre_s, min(s["t"] + self.bars.resolve_s, later[0] if later else math.inf)
            smp = self.between(a, b)
            now = self.at(s["t"])
            spots = []
            for q in sorted(inv):
                ent = self.mapped(q, s["t"])
                v = now.ents.get(ent) if (now is not None and ent) else None
                if v and v[2] is not None:
                    spots.append(v[2])
            with_hands = [x for x in smp if x.hands]
            near = (any(_box_dist(p, h) <= self.bars.hand_near_cm for x in with_hands for h in x.hands for p in spots)
                    if spots else None)
            out.append({"t": s["t"], "event": s["event"], "props": sorted(inv), "frames": len(smp),
                        "frames_with_hands": len(with_hands), "hands_seen": bool(with_hands), "near": near})
        return out

    def role(self, p: str) -> Optional[str]:
        """container / cover for the box and notebook, or a stand-in described as one ('lego tub
        (container stand-in)'), else None."""
        named = self.named(p)
        if named:
            return self.objects[named] if self.objects[named] in ("container", "cover") else None
        words = set(re.findall(r"[a-z]+", str(self.types[p]).lower()))
        for kind in ("container", "cover"):
            if kind in words or words & {n for n, k in self.objects.items() if k == kind}:
                return kind
        return None

    def _roles(self) -> dict:
        """Per container / cover prop: the share of frames it should be on the table in which the detector
        found an object of that kind (the box for a container) where the prop is."""
        out = {}
        for p in self.props:
            kind = self.role(p)
            if kind is None:
                continue
            classes = [n for n, k in self.objects.items() if k == kind]
            want = [s for s in self.samples if (self.expected(p, s.t) or ("",))[0] == "on_table"]
            hit = 0
            for s in want:
                v = s.ents.get(self.mapped(p, s.t) or "")
                found = [c for n in classes for c in s.seen.get(n, ())]
                hit += bool(found) if v is None or v[2] is None else \
                    any(_dist(c, v[2]) <= self.bars.match_cm for c in found)
            out[p] = {"type": self.types[p], "kind": kind, "classes": classes, "frames": len(want),
                      "rate": hit / len(want) if want else 0.0,
                      "world_kind": [self.tr.kinds.get(m["entity"]) for m in self.mapping[p]]}
        return out


def _want(q: dict) -> str:
    """Why a question is wrong: the entity it should point at, or (the right one) the parent the world
    does not have it in."""
    if q["point_at"] == q["expected_entity"] and q.get("parent_ok") is False:
        return f"want {q['expected_entity']} in {q['expect_parent']}, the world has it elsewhere"
    return f"want {q['expected_entity']}"


def _crit(name: str, detail: str, ok: bool, applies: bool = True) -> dict:
    return {"name": name, "detail": detail, "result": ("PASS" if ok else "FAIL") if applies else "n/a"}


def _list(rows: list, fmt, n: int = 3) -> str:
    if not rows:
        return ""
    more = f", +{len(rows) - n} more" if len(rows) > n else ""
    return " (" + "; ".join(fmt(r) for r in rows[:n]) + more + ")"


def score(trace: Trace, clip: Clip, bars: Optional[Bars] = None) -> dict:
    return _Scorer(trace, clip.truth, bars or Bars()).report(clip)


def score_clip(clip: Clip, cfg: Optional[dict] = None, detector=None, bars: Optional[Bars] = None,
               **replay_kw) -> dict:
    """Replay the clip through the production pipeline, then score it."""
    return score(replay_clip(clip, cfg, detector=detector, **replay_kw), clip, bars)


# ----- report -------------------------------------------------------------------------------------------

def print_report(r: dict) -> None:
    rp = r["replay"]
    print(f"clip {r['clip']}: {rp['video_s']} s of video, {rp['frames_decoded']} frames, {rp['frames_processed']} "
          f"processed (cap {rp['max_fps']:g} fps{', live timing' if rp['live_timing'] else ''}), "
          f"{rp['errors']} errors, {rp['wall_s']} s to replay"
          + (f", detector median {rp['detect_ms_median']} ms" if rp["detect_ms_median"] is not None else ""))
    print(f"  detector: {rp['detector']}")
    if rp.get("view"):
        print(f"  table view: {rp['view'][0]} of the full frame, resized to {rp['view'][1][0]}x{rp['view'][1][1]}"
              f"; room memory {'on' if rp.get('room_memory') else 'off'}")
    if r.get("scene"):
        print(f"  scene: {r['scene']}")
    print(f"  initial scene: {r['initial_scene']['count']} identities admitted at the start "
          f"({', '.join(e['entity'] for e in r['initial_scene']['entities']) or '-'})")
    if rp["frames_calibrating"]:
        print(f"  {rp['frames_calibrating']} frames spent calibrating the table (no table_cal in meta.json)")
    print("props -> world entities:")
    for p, what in r["props"].items():
        chain = ", ".join(f"{m['entity']} from {m['t']:g} s" for m in r["mapping"][p]) or "never found"
        print(f"  {p:5s} {what:12s} {chain}" + (" (guessed)" if p in r["guessed"] else ""))
    for s in r["placements"]:
        print(f"  {s['event']:8s} {s['prop']:5s} at {s['t']:6.2f} s -> "
              + (f"{s['entity']} after {s['delay_s']:.2f} s" if not s["missed"] else "nothing arrived"))
    for c in r["commands"]:
        print(f"  said {c['t']:6.2f} s {c['text']!r}: {c['answer']!r}"
              + (" [the always-on mic alone would drop this]" if c["overheard_ignored"] else ""))
    for q in r["questions"]:
        print(f"  asked {q['t']:5.2f} s {q['text']!r}: {q['answer']!r} -> {q['point_at']} "
              f"({'right' if q['correct'] else 'WRONG, ' + _want(q)})")
    for c in r["criteria"]:
        print(f"{c['result']:5s} {c['name']:28s} {c['detail']}")
    print(f"OVERALL: {'PASS' if r['pass'] else 'FAIL'}")


def add_replay_args(ap: argparse.ArgumentParser) -> None:
    """The replay options score_clip and eval.scorecard share."""
    ap.add_argument("--config", help="replay with this config.yaml instead of the one recorded in meta.json")
    ap.add_argument("--fps", type=float, help="perception cap (default: the config's main.perception_max_fps)")
    ap.add_argument("--live-timing", action="store_true",
                    help="also drop the frames the live loop would miss while the detector is busy")
    ap.add_argument("--overheard", action="store_true",
                    help="speech goes through the always-on mic's filter (default: asked with the clicker)")
    ap.add_argument("--no-model", action="store_true",
                    help="no YOLO / YOLOE (a Mac without the models): only the model-free change proposer")
    ap.add_argument("--hands-off", action="store_true",
                    help="no fixed-class detector (no hands, no prop labels): only the configured proposer, "
                         "e.g. YOLOE .pt on a Mac (with --yoloe-model)")
    ap.add_argument("--detect-model", help="detector weights instead of the config's detect.model (e.g. a .pt "
                                           "on the Mac; the .engine only loads in the Jetson container)")
    ap.add_argument("--proposals", choices=["config", "change", "yoloe", "off"], default="config",
                    help="proposer instead of the recorded one (change needs no model)")
    ap.add_argument("--yoloe-model", help="YOLOE weights instead of proposals.yoloe.model (the rig's reduced "
                                          ".engine only loads on the Jetson; models/yoloe-26s-seg-pf.pt on a Mac)")
    ap.add_argument("--zones", help="room_zones.json for a room clip recorded without its zones in meta.json")
    ap.add_argument("--grok", action="store_true",
                    help="name new things with Grok as the app does online (needs XAI_API_KEY; live calls on a "
                         "background thread, so two replays can differ)")


def replay_config(clip: Clip, a: argparse.Namespace) -> dict:
    """The replay config from the command line: the recorded one (or --config) with the overrides."""
    from core.config import load_config
    cfg = copy.deepcopy(load_config(a.config) if a.config else (clip.meta.get("config") or load_config()))
    if a.detect_model:
        cfg["detect"] = dict(cfg.get("detect") or {}, model=a.detect_model)
    if a.proposals != "config":
        cfg["proposals"] = dict(cfg.get("proposals") or {}, enabled=a.proposals != "off",
                                **({"kind": a.proposals} if a.proposals != "off" else {}))
    if a.yoloe_model:
        pc = cfg.get("proposals") or {}
        cfg["proposals"] = dict(pc, yoloe=dict(pc.get("yoloe") or {}, model=a.yoloe_model))
    return cfg


def replay_from_args(clip: Clip, a: argparse.Namespace) -> Trace:
    if a.zones:
        clip.meta["room_zones"] = json.loads(Path(a.zones).read_text())
    return replay_clip(clip, replay_config(clip, a), max_fps=a.fps, live_timing=a.live_timing,
                       overheard=a.overheard, no_model=a.no_model, hands_off=a.hands_off, grok=a.grok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="clip directory (video.mp4, frames.json, meta.json, truth.json)")
    ap.add_argument("--json", help="also write the full report here")
    add_replay_args(ap)
    ap.add_argument("--initial-s", type=float, default=Bars.initial_s,
                    help="identities admitted this soon after the first frame are the initial scene")
    ap.add_argument("--resolve-s", type=float, default=Bars.resolve_s,
                    help="a step's entity must arrive within this many seconds")
    ap.add_argument("--match-cm", type=float, default=Bars.match_cm,
                    help="an entity this close to a resting prop's spot is that prop")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    clip = load_clip(a.clip)
    if not clip.video.exists():
        print(f"no video.mp4 in {a.clip}")
        return 2
    trace = replay_from_args(clip, a)
    r = score(trace, clip, Bars(initial_s=a.initial_s, resolve_s=a.resolve_s, match_cm=a.match_cm))
    print_report(r)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, default=str))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
