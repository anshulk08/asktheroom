"""Handoff guard (spec 0009 M0, Acquire): a room track takes a departed prop only if something moved at
its spot when it appeared (a hand, or pixels that changed). A detector that starts calling a static cushion
'phone' while the phone is away has no such evidence, so it stays a conflict sighting."""
from pathlib import Path

import numpy as np
import pytest

from core.config import Config
from core.room import RoomMemory, RoomTracker
from core.room_types import RoomConfig, RoomObservation, RoomTrack
from core.room_zones import Zone, Zones
from core.world import World
from tests.synth import Scene
from tests.test_room_world import SHELF_BOX, appear, depart, seen, types
from core.types import EventType, Frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0)


@pytest.fixture
def world(cfg):
    return World(cfg)


def obs(box, t, cls='keys'):
    return RoomObservation(zone='bookshelf', cls=cls, conf=0.9, box_px=box, t=t, wall=t, frame_idx=0)


# ----- tracker: evidence recorded when a track starts --------------------------------------------------

def test_a_track_that_appears_where_pixels_changed_has_arrival_evidence():
    tr = RoomTracker(RoomConfig.from_dict({}))
    tr.visit('bookshelf', 'the bookshelf', [], [], [], 1.0, 1.0, 1, activity=[])
    v = tr.visit('bookshelf', 'the bookshelf', [obs(SHELF_BOX, 2.0)], [], [], 2.0, 2.0, 2,
                 activity=[(1490, 190, 1570, 250)])
    assert tr.tracks('bookshelf')[0].arrival_evidence is True


def test_a_hand_there_a_few_seconds_earlier_is_evidence_too():
    tr = RoomTracker(RoomConfig.from_dict({}))
    tr.visit('bookshelf', 'the bookshelf', [], [], [], 1.0, 1.0, 1, activity=[(1480, 180, 1600, 260)])
    tr.visit('bookshelf', 'the bookshelf', [obs(SHELF_BOX, 4.0)], [], [], 4.0, 4.0, 2, activity=[])
    assert tr.tracks('bookshelf')[0].arrival_evidence is True


def test_a_label_on_static_scenery_has_no_arrival_evidence():
    tr = RoomTracker(RoomConfig.from_dict({}))
    tr.visit('bookshelf', 'the bookshelf', [], [], [], 1.0, 1.0, 1, activity=[(100, 100, 200, 200)])  # elsewhere
    tr.visit('bookshelf', 'the bookshelf', [obs(SHELF_BOX, 2.0)], [], [], 2.0, 2.0, 2, activity=[])
    assert tr.tracks('bookshelf')[0].arrival_evidence is False


def test_old_activity_outside_the_window_is_not_evidence():
    tr = RoomTracker(RoomConfig.from_dict({'arrival_window_s': 10}))
    tr.visit('bookshelf', 'the bookshelf', [], [], [], 1.0, 1.0, 1, activity=[SHELF_BOX])
    tr.visit('bookshelf', 'the bookshelf', [obs(SHELF_BOX, 30.0)], [], [], 30.0, 30.0, 2, activity=[])
    assert tr.tracks('bookshelf')[0].arrival_evidence is False


def test_without_activity_data_evidence_is_unknown():
    tr = RoomTracker(RoomConfig.from_dict({}))
    tr.visit('bookshelf', 'the bookshelf', [obs(SHELF_BOX, 2.0)], [], [], 2.0, 2.0, 2)
    assert tr.tracks('bookshelf')[0].arrival_evidence is None


# ----- world: only a track with arrival evidence takes the departed prop --------------------------------

def test_a_track_with_no_arrival_evidence_does_not_take_the_departed_keys(scene, world):
    depart(scene, world)
    trk = appear(scene, world, 'r:1')
    trk.arrival_evidence = False
    assert seen(scene, world, trk) == []
    assert trk.role == 'conflict' and world.get('keys').zone == 'table'


def test_a_track_with_arrival_evidence_takes_the_departed_keys(scene, world):
    depart(scene, world)
    trk = appear(scene, world, 'r:1')
    trk.arrival_evidence = True
    assert types(seen(scene, world, trk)) == [EventType.FOUND]


def test_the_guard_can_be_turned_off(scene, cfg):
    w = World(cfg)
    w.room_cfg = RoomConfig.from_dict({'arrival_evidence': False})
    depart(scene, w)
    trk = appear(scene, w, 'r:1')
    trk.arrival_evidence = False
    assert types(seen(scene, w, trk)) == [EventType.FOUND]


# ----- driver: pixels that changed at the spot are the evidence -----------------------------------------

class Backend:
    def __init__(self):
        self.box = None

    def infer(self, img):
        return [] if self.box is None else [("keys", 0.9, self.box)]


def driver(world):
    zones = Zones("v", (1920, 1080), {"bookshelf": Zone("bookshelf", "the bookshelf",
                                                        [(1400, 100), (1800, 100), (1800, 400), (1400, 400)])})
    rc = RoomConfig.from_dict({"room_every_n": 1})
    return RoomMemory(rc, zones, Backend(), {"keys": "keys"}, world, (400, 200, 1200, 800))


def frame(img, t):
    return Frame(t=t, wall=t, img=img, idx=int(t * 10))


def test_driver_marks_a_real_arrival_and_a_static_misread(world):
    rm = driver(world)
    img = np.full((1080, 1920, 3), 90, np.uint8)
    rm.step(frame(img, 1.0))                                   # the empty shelf
    shown = img.copy()
    shown[210:250, 1510:1570] = 220                            # something put on it
    rm.backend.box = (110, 110, 170, 150)                      # crop px (zone bbox starts at 1400, 100)
    rm.step(frame(shown, 2.0))
    assert rm.tracker.tracks('bookshelf')[0].arrival_evidence is True

    rm2 = driver(world)
    rm2.step(frame(img, 1.0))
    rm2.backend.box = (110, 110, 170, 150)                     # a label on the unchanged shelf
    rm2.step(frame(img, 2.0))
    assert rm2.tracker.tracks('bookshelf')[0].arrival_evidence is False


# ----- the same guard on a Grok-named thing carried to a zone -----------------------------------------

def test_a_thing_track_with_no_arrival_evidence_is_not_handed_the_thing(cfg):
    from tests.test_room_things_world import REMOTE, named_thing_leaves, namer_for, room_thing
    import core.crops
    scene = Scene(cfg, fps=10, t0=1000.0, render=True)    # the namer's close-up is a frame crop
    w = World(cfg)
    old = core.crops.active()
    core.crops.set_active(None)
    try:
        named_thing_leaves(scene, w, namer_for(w, REMOTE))
    finally:
        core.crops.set_active(old)
    trk = room_thing(scene, w, 'r:1')
    trk.arrival_evidence = False                   # a static couch patch YOLOE boxes and Grok calls a remote
    seen(scene, w, trk, zone='couch')
    assert w.get('thing:1').zone == 'table' and trk.entity != 'thing:1'
    trk2 = room_thing(scene, w, 'r:2')
    trk2.arrival_evidence = True
    assert types(seen(scene, w, trk2, zone='couch')) == [EventType.FOUND]
