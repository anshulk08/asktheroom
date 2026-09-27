"""Speed on far zones (spec 0010, P0-3): while a handoff is open the room pass visits a zone every
room_every_n_hot frames; an arrival (its spot changed) confirms on one visit; verification jobs go to Grok
before open naming and static clutter is not sent at all while a handoff is open; and the World decides a
thing track the moment its Grok name lands (RoomNamer.on_named), not on the zone's next visit. Fake backend,
proposer and Grok; no hardware, no network."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import core.crops
from core.config import Config, load_config
from core.proposals import Proposal
from core.room import RoomMemory, RoomNamer
from core.room_types import RoomConfig, RoomTrack, ZoneVisit
from core.room_zones import Zone, Zones
from core.types import EventType, Frame, Status
from core.world import World
from tests.synth import Scene
from tests.test_room_driver import COUCH, SHELF, TABLE_RECT, TO_OBJ, FakeBackend, StubWorld, frame
from tests.test_room_things import REMOTE, FakeProposer, Namer
from tests.test_room_things_world import named_thing_leaves, namer_for
from tests.test_room_tracker import KEYS, Run

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
CHANGE = [(90, 90, 150, 140)]              # a change blob over KEYS
BOX = Proposal((100, 100, 140, 140), 0.5)  # shelf crop px -> full (1700, 200, 1740, 240)


class HintWorld(StubWorld):
    """Records room_update visits (returning one event each) and answers room_handoff_hints from `hints`."""

    def __init__(self, hints=(), fail_synthetic=False):
        super().__init__()
        self.hints, self.fail_synthetic = list(hints), fail_synthetic

    def room_handoff_hints(self, t):
        return list(self.hints)

    def room_update(self, visit):
        if self.fail_synthetic and visit.crop is None:
            raise RuntimeError("world broke")
        return super().room_update(visit)


def make(zones=(SHELF,), world=None, props=(), namer=None, **cfg):
    cfg.setdefault("room_every_n", 5)
    backend = FakeBackend()
    world = world if world is not None else HintWorld()
    zs = Zones("v", (W, H), {z.name: z for z in zones})
    rm = RoomMemory(RoomConfig(enabled=True, **cfg), zs, backend, TO_OBJ, world, TABLE_RECT,
                    proposer=FakeProposer(props) if props else None, namer=namer)
    return rm, backend, world


def arrival_frames(box_full=(1700, 200, 1740, 240)):
    """Frame 1 with a bright patch where the object will be, frame 2 plain: the zone's second visit sees a
    change blob over the spot (a first visit has no previous crop to differ from)."""
    f1, f2 = frame(1), frame(2)
    x1, y1, x2, y2 = box_full
    f1.img[y1:y2, x1:x2] = 255
    return f1, f2


def things(rm):
    return [t for t in rm.tracker.tracks() if t.cls == "thing"]


# ---------------------------------------------------------------- 1. hot cadence

def test_hot_mode_visits_a_zone_every_frame_while_a_handoff_is_open():
    rm, backend, world = make(zones=(SHELF, COUCH), room_every_n_hot=1)
    for i in range(1, 16):
        rm.step(frame(i))
    assert len(backend.shapes) == 3 and [v.frame_idx for v in world.visits] == [5, 10, 15]
    world.hints = [REMOTE]                                  # something just left the table
    for i in range(16, 31):
        rm.step(frame(i))
    assert len(backend.shapes) == 18                        # one zone per frame now
    assert [v.frame_idx for v in world.visits[3:]] == list(range(16, 31))
    assert [v.zone for v in world.visits[3:6]] == ["couch", "shelf", "couch"]     # still round-robin
    world.hints = []
    for i in range(31, 46):
        rm.step(frame(i))
    assert len(backend.shapes) == 21                        # back to every fifth frame


def test_hot_cadence_is_configurable():
    rm, backend, world = make(world=HintWorld([REMOTE]), room_every_n_hot=2)
    for i in range(1, 11):
        rm.step(frame(i))
    assert len(backend.shapes) == 5 and [v.frame_idx for v in world.visits] == [2, 4, 6, 8, 10]


def test_hot_defaults_to_every_second_frame_and_is_in_config_yaml(monkeypatch):
    """Every frame halved the table's fps on the rig (a foot at the table edge kept it hot for 2 min):
    every second frame, for hot_max_s only, with a dwell gate (core/room_world.room_hot)."""
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    assert RoomConfig().room_every_n_hot == 2
    rc = RoomConfig.from_dict(load_config().get("room_memory"))
    assert (rc.room_every_n_hot, rc.names_per_minute, rc.confirm_visits_arrival) == (2, 60, 1)
    assert (rc.hot_max_s, rc.handoff_min_dwell_s) == (30.0, 4.0)


def test_hot_is_never_slower_than_cold():
    rm, backend, world = make(world=HintWorld([REMOTE]), room_every_n=1, room_every_n_hot=2)
    for i in range(1, 5):
        rm.step(frame(i))
    assert len(backend.shapes) == 4


def test_a_world_that_cannot_say_keeps_the_cold_cadence():
    rm, backend, _ = make(world=StubWorld())
    for i in range(1, 11):
        rm.step(frame(i))
    assert len(backend.shapes) == 2


def test_no_frame_means_no_visit_in_hot_mode_too():
    rm, backend, world = make(world=HintWorld([REMOTE]))
    assert rm.step(None) == [] and rm.step(frame(1, img=False)) == []
    assert backend.shapes == []


# ---------------------------------------------------------------- 2. arrivals confirm on one visit

def test_an_arrival_confirms_on_its_first_visit_a_static_track_on_its_second():
    r = Run()
    v = r.visit(("thing", KEYS), changes=CHANGE)
    [tr] = r.tr.tracks()
    assert tr.changed and tr.confirmed and v.confirmed == [tr]
    r2 = Run()
    v1 = r2.visit(("thing", KEYS))
    assert v1.confirmed == [] and not r2.tr.tracks()[0].confirmed
    v2 = r2.visit(("thing", KEYS))
    assert len(v2.confirmed) == 1


def test_a_track_placed_under_a_hand_confirms_when_the_change_shows():
    r = Run()
    assert r.visit(("keys", KEYS)).confirmed == []          # the hand still there: no change evidence yet
    v = r.visit(("keys", KEYS), changes=CHANGE)
    [tr] = r.tr.tracks()
    assert tr.changed and tr.confirmed and v.confirmed == [tr] and tr.hits == 2


def test_confirm_visits_arrival_is_configurable():
    r = Run(confirm_visits_arrival=2)
    assert r.visit(("thing", KEYS), changes=CHANGE).confirmed == []
    assert len(r.visit(("thing", KEYS)).confirmed) == 1
    r = Run(confirm_visits=3, confirm_visits_arrival=1)
    assert len(r.visit(("thing", KEYS), changes=CHANGE).confirmed) == 1


def test_an_arrival_confirms_on_the_first_visit_that_shows_the_change_in_the_driver():
    rm, _, world = make(props=[BOX], room_every_n=1)
    f1, f2 = arrival_frames()
    assert rm.step(f1) and things(rm)[0].confirmed is False
    rm.step(f2)
    [th] = things(rm)
    assert th.changed and th.confirmed and [x.tid for x in world.visits[-1].confirmed] == [th.tid]


# ---------------------------------------------------------------- 3. Grok: verification first, arrivals only

def track(i, **kw):
    return RoomTrack(tid=f"r:{i}", zone="shelf", cls="thing", box_px=(0, 0, 10, 10), first_seen=0.0,
                     first_wall=0.0, last_seen=0.0, last_wall=0.0, confirmed=True, **kw)


def img():
    return np.zeros((20, 20, 3), np.uint8)


def test_verification_jobs_go_before_open_naming_newest_first():
    namer = RoomNamer(Namer(), per_minute=100, start=False, clock=lambda: 0.0,
                      verify_fn=lambda im, hints: dict(hints[0]))
    trs = [track(i) for i in range(1, 6)]
    namer.submit(trs[0], img())                    # open naming, oldest
    namer.submit(trs[1], img())
    namer.submit(trs[2], img(), hints=[REMOTE])    # a verification job
    namer.submit(trs[3], img())                    # open naming, newest
    namer.submit(trs[4], img(), hints=[REMOTE])    # a newer verification job
    order = []
    while namer.step(0.0):
        order.append([t.tid for t in trs if t.guess is not None and t.tid not in order][0])
    assert order == ["r:5", "r:3", "r:4", "r:2", "r:1"]


def test_the_default_name_cap_is_sixty_a_minute():
    assert RoomConfig().names_per_minute == 60


def test_static_clutter_is_not_sent_to_grok_while_a_handoff_is_open():
    namer = RoomNamer(Namer(), start=False, clock=lambda: 0.0, verify_fn=lambda im, h: dict(h[0]))
    rm, _, _ = make(world=HintWorld([REMOTE]), props=[BOX], namer=namer, room_every_n=1)
    rm.step(frame(1))
    rm.step(frame(2))                                       # plain frames: nothing changed there
    [th] = things(rm)
    assert th.confirmed and not th.changed
    assert namer.pending() == 0 and not th.name_asked
    rm.step(frame(3))
    assert namer.pending() == 0                             # never, however often it is seen


def test_an_arrival_is_sent_to_grok_while_a_handoff_is_open():
    namer = RoomNamer(Namer(), start=False, clock=lambda: 0.0, verify_fn=lambda im, h: dict(h[0]))
    rm, _, _ = make(world=HintWorld([REMOTE]), props=[BOX], namer=namer, room_every_n=1)
    f1, f2 = arrival_frames()
    rm.step(f1)
    rm.step(f2)
    [th] = things(rm)
    assert th.changed and th.name_asked and namer.pending() == 1


def test_everything_is_named_when_the_world_cannot_say():
    fn = Namer()
    namer = RoomNamer(fn, start=False, clock=lambda: 0.0)
    rm, _, _ = make(world=StubWorld(), props=[BOX], namer=namer, room_every_n=1)
    rm.step(frame(1))
    rm.step(frame(2))
    [th] = things(rm)
    assert not th.changed and th.name_asked and namer.pending() == 1


# ---------------------------------------------------------------- 4. decide the moment the name lands

def confirmed_arrival(world, **kw):
    """A RoomMemory whose one thing track arrived, confirmed and was queued for verification."""
    namer = RoomNamer(Namer(), start=False, clock=lambda: 0.0, verify_fn=lambda im, h: dict(h[0]))
    rm, _, world = make(world=world, props=[BOX], namer=namer, room_every_n=1, **kw)
    f1, f2 = arrival_frames()
    rm.step(f1)
    rm.step(f2)
    [th] = things(rm)
    assert th.name_asked and namer.pending() == 1
    return rm, namer, th


def test_a_landed_name_is_decided_at_once_with_a_one_track_visit():
    rm, namer, th = confirmed_arrival(HintWorld([REMOTE]))
    world = rm.world
    n = len(world.visits)
    assert namer.step(0.0) and th.guess == REMOTE
    assert len(world.visits) == n + 1
    v = world.visits[-1]
    assert isinstance(v, ZoneVisit) and v.confirmed == [th] and v.missed == [] and v.dropped == []
    assert (v.zone, v.say, v.t, v.wall, v.crop) == ("shelf", "the bookshelf", th.last_seen, th.last_wall, None)
    assert v.frame_idx == 2                                  # the last frame the driver processed
    assert rm.tracker.tracks() == [th]                       # the tracker was not touched


def test_the_events_of_a_synthetic_visit_come_out_of_the_next_step():
    rm, namer, th = confirmed_arrival(HintWorld([REMOTE]))
    namer.step(0.0)
    out = rm.step(frame(3))
    assert out[0] == "event3" and len(out) == 2             # the synthetic visit's event first, then frame 3's
    assert rm.step(frame(4)) == ["event5"]                   # drained once


def test_a_synthetic_visit_needs_the_namer_wired_to_the_driver():
    namer = RoomNamer(Namer(), start=False, clock=lambda: 0.0)
    assert namer.on_named is None
    tr = track(1)
    namer.submit(tr, img())
    assert namer.step(0.0) and tr.guess is not None         # no callback: naming alone


def test_on_named_failures_are_logged_never_raised(caplog):
    rm, namer, th = confirmed_arrival(HintWorld([REMOTE], fail_synthetic=True))
    with caplog.at_level(logging.WARNING, logger="core.room"):
        assert namer.step(0.0)
    assert th.guess == REMOTE
    assert any("deciding" in r.getMessage() and th.tid in r.getMessage() for r in caplog.records)
    assert rm.step(frame(3)) == ["event3"]                   # nothing queued by the failed one


def test_a_track_the_tracker_dropped_meanwhile_is_not_decided():
    rm, namer, th = confirmed_arrival(HintWorld([REMOTE]))
    rm.tracker._tracks["shelf"] = []                         # gone (absent) before Grok answered
    n = len(rm.world.visits)
    assert namer.step(0.0) and th.guess == REMOTE
    assert len(rm.world.visits) == n


# ---------------------------------------------------------------- end to end: the handoff lands with the name

COUCH_ZONE = Zone("couch", "the couch", [(0, 900), (300, 900), (300, 1070), (0, 1070)])
COUCH_BOX = Proposal((100, 50, 140, 90), 0.5)               # couch crop px -> full (100, 950, 140, 990)


@pytest.fixture
def cfg():
    return Config.load(ROOT / "config.yaml")


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0, render=True)


@pytest.fixture
def world(cfg):
    return World(cfg)


@pytest.fixture(autouse=True)
def no_crop_store():
    old = core.crops.active()
    core.crops.set_active(None)
    yield
    core.crops.set_active(old)


class RoomRig:
    """The real World and a RoomMemory on the scene's clock, a couch zone, a fake proposer and a namer
    whose verify_fn answers the hint (Grok: 'yes, that is the remote')."""

    def __init__(self, scene, world, verify=None):
        self.scene, self.world = scene, world
        self.asked = []

        def verify_fn(im, hints):
            self.asked.append(list(hints))
            return dict(hints[0]) if verify is None else verify(im, hints)

        self.namer = RoomNamer(Namer(), start=False, clock=lambda: scene.t, verify_fn=verify_fn)
        zones = Zones("v", (W, H), {"couch": COUCH_ZONE})
        self.rm = RoomMemory(RoomConfig(enabled=True, room_every_n=1), zones, FakeBackend(), TO_OBJ, world,
                             TABLE_RECT, proposer=FakeProposer([COUCH_BOX]), namer=self.namer)
        self.idx = 10_000

    def visit(self, patch=False, dt=0.5):
        self.scene.t += dt
        self.idx += 1
        im = np.full((H, W, 3), 128, np.uint8)
        if patch:
            im[950:990, 100:140] = 255
        return self.rm.step(Frame(t=self.scene.t, wall=self.scene.t, img=im, idx=self.idx))


def test_the_remote_is_on_the_couch_the_moment_grok_says_so(scene, world):
    named_thing_leaves(scene, world, namer_for(world, REMOTE))       # thing:1 (a remote) left the table
    rig = RoomRig(scene, world)
    assert rig.visit(patch=True) == []                                # first sighting: no change evidence yet
    assert rig.visit() == []                                          # the change shows: confirmed, queued
    [th] = things(rig.rm)
    assert th.changed and th.confirmed and th.name_asked and rig.namer.pending() == 1
    assert world.get("thing:1").zone == "table"                       # nothing decided before Grok answers
    assert rig.namer.step()                                           # Grok: it is the remote
    assert rig.asked == [[REMOTE]] and th.guess == REMOTE
    ent = world.get("thing:1")
    assert (ent.status, ent.zone) == (Status.VISIBLE, "couch")        # decided at once, no visit in between
    assert (th.role, th.entity) == ("assoc", "thing:1")
    p = world.place("thing:1", now=scene.t)
    assert p.kind == "room" and p.zone == "couch" and p.tentative and p.box_px == (100, 950, 140, 990)
    assert "thing:1" not in world._departures
    seen_t = th.last_seen
    out = rig.visit()                                                 # the FOUND comes out of the next step
    assert [(e.type, e.obj) for e in out] == [(EventType.FOUND, "thing:1")]
    assert out[0].t == seen_t < scene.t                               # dated at the sighting, not at the answer
    assert rig.visit() == []                                          # its own track: refresh only


def test_a_name_that_does_not_match_settles_the_track_at_once_too(scene, world):
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    rig = RoomRig(scene, world, verify=lambda im, h: {"name": "shoe", "also": [], "confidence": 0.9})
    rig.visit(patch=True)
    rig.visit()
    assert rig.namer.step()
    [th] = things(rig.rm)
    assert th.role == "ignored" and world.get("thing:1").zone == "table" and "thing:1" in world._departures
    assert rig.visit() == []


# ---------------------------------------------------------------- 5. the verify call's provider

def test_the_verify_call_uses_the_auto_namers_provider_at_its_lowest_reasoning_effort(monkeypatch):
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    from core.auto_name import AutoNamer
    from core.room import make_verify_fn
    raw = load_config()
    namer = AutoNamer(raw, StubWorld(), online=lambda: False, start=False)
    assert namer.provider.c.reasoning_effort == "none"              # visual_memory.reasoning_effort
    assert namer.provider.c.timeout_s == raw["visual_memory"]["timeout_s"]
    assert callable(make_verify_fn(namer))
