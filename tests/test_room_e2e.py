"""End to end (spec 0009 M0): the table pipeline (a synthetic Scene) and the room pass (the real RoomMemory
with a fake backend and synthetic 1080p frames) feed one real World, and the offline answers speak it."""
from __future__ import annotations

import numpy as np
import pytest

from core.config import Config, load_config
from core.detect import class_list
from core.room import RoomMemory
from core.room_types import RoomConfig
from core.room_zones import Zone, Zones
from core.types import EventType, Frame, Intent, Status
from core.world import World
from tests.synth import Scene
from voice.answers import answer

W, H = 1920, 1080
TABLE_RECT = (360, 202, 1560, 877)
SHELF = Zone("shelf", "the bookshelf", [(1600, 100), (1900, 100), (1900, 400), (1600, 400)])
KEYS_IN_CROP = ("keys", 0.9, (100, 100, 140, 130))        # crop px of the shelf zone
HAND_OVER_KEYS = ("hand", 0.9, (80, 80, 160, 150))


class Backend:
    def __init__(self):
        self.raw: list = []

    def infer(self, img):
        return list(self.raw)


class Rig:
    """A real World fed by a table Scene and a RoomMemory on the same monotonic clock (wall == t here)."""

    def __init__(self):
        self.raw_cfg = load_config()
        self.world = World(self.raw_cfg)
        self.scene = Scene(Config.from_dict(self.raw_cfg), fps=10, t0=1000.0)
        self.backend = Backend()
        zones = Zones("v", (W, H), {"shelf": SHELF})
        self.room = RoomMemory(RoomConfig(enabled=True, room_every_n=1), zones, self.backend,
                               class_list(self.raw_cfg)[1], self.world, TABLE_RECT)
        self.idx = 10_000

    def carry_keys_off_left_edge(self):
        s, w = self.scene, self.world
        s.place("keys", 40, 30)
        s.run(w, 1.0)
        s.hand(1, 40, 30)
        s.run(w, 0.3)
        s.remove("keys")
        s.run(w, 1.0)
        s.hand(1, 5, 30)
        s.run(w, 0.3)
        s.hand_off(1)
        return s.run(w, 1.0)

    def room_visit(self, *raw, dt=0.5):
        self.backend.raw = list(raw)
        self.scene.t += dt
        self.idx += 1
        f = Frame(t=self.scene.t, wall=self.scene.t, img=np.full((H, W, 3), 128, np.uint8), idx=self.idx)
        return self.room.step(f)

    def where(self, obj="keys", later=0.5):
        return answer(Intent("WHERE", obj, f"where are my {obj}"), self.world, self.world.events, self.raw_cfg,
                      now=self.scene.t + later).text


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    return Rig()


def test_demo_moment_keys_carried_to_the_shelf(rig):
    ev = rig.carry_keys_off_left_edge()
    assert EventType.EXITED_VIEW in [e.type for e in ev]
    assert rig.room_visit(KEYS_IN_CROP) == []                         # first sighting: not confirmed yet
    found = rig.room_visit(KEYS_IN_CROP)
    assert [e.type for e in found] == [EventType.FOUND]
    keys = rig.world.get("keys")
    assert keys.zone == "shelf" and keys.pos_cm is None and rig.world.resolve("keys")[0] is None
    text = rig.where()
    assert text.startswith("Your keys are on the bookshelf.") and "appeared there" in text


def test_hand_over_the_shelf_is_not_absence_but_an_empty_shelf_is(rig):
    rig.carry_keys_off_left_edge()
    rig.room_visit(KEYS_IN_CROP)
    rig.room_visit(KEYS_IN_CROP)
    for _ in range(4):                                                 # someone reaching in: blocked visits
        rig.room_visit(HAND_OVER_KEYS)
    assert rig.world.get("keys").status == Status.VISIBLE
    assert "can't see" not in rig.where()
    for _ in range(3):                                                 # a clear, empty shelf
        rig.room_visit()
    assert rig.world.get("keys").status == Status.UNKNOWN and rig.world.get("keys").zone == "shelf"
    assert rig.where().endswith("I can't see them there now.")


def test_keys_brought_back_to_the_table_are_on_the_table(rig):
    rig.carry_keys_off_left_edge()
    rig.room_visit(KEYS_IN_CROP)
    rig.room_visit(KEYS_IN_CROP)
    rig.scene.place("keys", 50, 30)
    ev = rig.scene.run(rig.world, 1.0)
    assert EventType.MOVED in [e.type for e in ev]
    assert rig.world.get("keys").zone == "table"
    for _ in range(3):                                                 # the shelf is empty now: no effect
        rig.room_visit()
    assert rig.world.get("keys").status == Status.VISIBLE
    assert "on the table" in rig.where()


def test_decoy_already_on_the_shelf_never_becomes_the_keys(rig):
    rig.room_visit(KEYS_IN_CROP)                                       # a key ring on the shelf first
    rig.room_visit(KEYS_IN_CROP)
    rig.carry_keys_off_left_edge()                                     # then your keys leave the table
    rig.room_visit(KEYS_IN_CROP)
    rig.room_visit(KEYS_IN_CROP)
    assert rig.world.get("keys").zone == "table"                      # still GONE off the table edge
    assert "I also see keys on the bookshelf." in rig.where()


def _pick_up_and_hold(rig):
    s, w = rig.scene, rig.world
    s.place("keys", 40, 30)
    s.run(w, 1.0)
    s.hand(1, 40, 30)
    s.run(w, 0.3)
    s.remove("keys")
    s.run(w, 1.0)
    assert w.get("keys").status == Status.HELD


def test_zone_sees_the_keys_before_the_table_decides_they_left(rig):
    """Review finding: the table emits EXITED_VIEW ~0.5-2 s after the keys really left. A shelf next to the
    table sees them inside that window; the departure is dated at the holding hand's last sighting."""
    _pick_up_and_hold(rig)
    s, w = rig.scene, rig.world
    s.hand(1, 5, 30)
    s.run(w, 0.3)
    s.hand_off(1)                                                      # the hand leaves the table view
    assert rig.room_visit(KEYS_IN_CROP, dt=0.1) == []                  # the shelf sees them first
    assert EventType.EXITED_VIEW in [e.type for e in s.run(w, 1.0)]    # then the table decides
    assert [e.type for e in rig.room_visit(KEYS_IN_CROP)] == [EventType.FOUND]
    assert rig.world.get("keys").zone == "shelf"
    assert rig.where().startswith("Your keys are on the bookshelf.")


def test_a_key_ring_seen_on_the_shelf_while_the_keys_are_in_hand_stays_a_conflict(rig):
    _pick_up_and_hold(rig)
    rig.room_visit(KEYS_IN_CROP)                                       # while the keys are still held
    s, w = rig.scene, rig.world
    s.hand(1, 5, 30)
    s.run(w, 0.3)
    s.hand_off(1)
    s.run(w, 1.0)
    rig.room_visit(KEYS_IN_CROP)
    rig.room_visit(KEYS_IN_CROP)
    assert rig.world.get("keys").zone == "table"
    assert "I also see keys on the bookshelf." in rig.where()
