"""Decoys (spec 0010 P0-4): a second remote-like object in the room must never be taken for the remote.

The closest thing to production without the rig: one real World fed by a rendered table Scene with a real
AutoNamer (fake Grok names the table thing), and the real RoomMemory (RoomTracker, a proposer, a RoomNamer
with a fake Grok stepped by hand) on synthetic 1080p frames. Room objects are bright rectangles drawn on a
grey frame: the proposer boxes them, and drawing one after a zone's first visit is the frame difference
that makes its track 'changed' (it arrived), exactly as on the rig. Grok's answers are given per call."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.config import Config, load_config
from core.detect import class_list
from core.proposals import Proposal
from core.room import RoomMemory, RoomNamer
from core.room_types import RoomConfig
from core.room_zones import Zone, Zones
from core.types import EventType, Frame, Intent, Status
from core.world import World
from tests.synth import Scene
from tests.test_room_e2e import Backend
from tests.test_room_things_world import namer_for
from voice.answers import answer

W, H = 1920, 1080
TABLE_RECT = (360, 202, 1560, 877)
ZONES = {
    "couch": Zone("couch", "the couch", [(0, 700), (320, 700), (320, 1040), (0, 1040)]),
    "side_table": Zone("side_table", "the side table", [(1600, 700), (1900, 700), (1900, 1040), (1600, 1040)]),
    "counter": Zone("counter", "the counter", [(1600, 60), (1900, 60), (1900, 400), (1600, 400)]),
}
COUCH_SPOT = (100, 800, 160, 840)          # full-frame px
COUCH_SPOT_2 = (220, 920, 280, 960)
SIDE_SPOT = (1700, 800, 1760, 840)
COUNTER_SPOT = (1700, 150, 1760, 190)
REMOTE = {"name": "remote control", "also": ["remote"], "confidence": 0.7}
PHONE = {"name": "phone", "also": ["smartphone"], "confidence": 0.8}
BRIGHT = 200                               # drawn objects (inside lum_lo..lum_hi) on a 128 background


class BrightProposer:
    """Proposes every bright blob of a zone crop (the objects the rig draws); hands and known props are
    not drawn, so nothing is excluded."""

    def propose(self, img, known, hands):
        grey = img[..., 0] if img.ndim == 3 else img
        mask = (grey > (BRIGHT + 128) // 2).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            out.append(Proposal(box_px=(x, y, x + w, y + h), conf=0.9))
        return out

    def reset(self):
        pass

    def set_roi(self, polygon):
        pass


class ThingRig:
    """A real World, its table Scene (rendered, Grok-named things) and a RoomMemory with things on one clock."""

    def __init__(self, table_replies=(REMOTE,)):
        self.raw_cfg = load_config()
        self.world = World(self.raw_cfg)
        self.scene = Scene(Config.from_dict(self.raw_cfg), fps=10, t0=1000.0, render=True)
        self.table_namer = namer_for(self.world, *table_replies)
        self.reply = None
        self.asked: list = []                  # the hints of every room Grok call
        self.namer = RoomNamer(self._grok, per_minute=100, verify_fn=self._grok_verify, start=False)
        self.room = RoomMemory(RoomConfig(enabled=True, room_every_n=1), Zones("v", (W, H), dict(ZONES)),
                               Backend(), class_list(self.raw_cfg)[1], self.world, TABLE_RECT,
                               proposer=BrightProposer(), namer=self.namer)
        self.objects: dict[str, tuple] = {}    # key -> full-frame box drawn in the room
        self.idx = 10_000

    # -- the table side

    def thing_named_on_table(self, key="remote", at=(40, 30)) -> str:
        """An unnamed object put on the table becomes a thing and Grok names it; returns the thing."""
        before = set(self.world.entities)
        self.scene.thing(key, *at)
        self.scene.run(self.world, 1.5)
        new = [n for n in self.world.entities if n not in before and n.startswith("thing:")]
        assert len(new) == 1, new
        assert self.table_namer.step() is True
        assert self.world.thing_guess(new[0]) is not None
        return new[0]

    def remote_leaves(self, key="remote", at=(40, 30)) -> str:
        """The named thing is picked up and carried out by the left edge (a table departure)."""
        thing = self.thing_named_on_table(key, at)
        s, w = self.scene, self.world
        s.hand(1, *at)
        s.run(w, 0.3)
        s.remove(key)
        s.run(w, 1.0)
        s.hand(1, 5, at[1])
        s.run(w, 0.3)
        s.hand_off(1)
        events = s.run(w, 1.0)
        assert EventType.EXITED_VIEW in [e.type for e in events if e.obj == thing]
        assert thing in w._departures
        return thing

    # -- the room side

    def put(self, key: str, box: tuple) -> None:
        self.objects[key] = box

    def take(self, key: str) -> None:
        self.objects.pop(key, None)

    def _frame(self) -> np.ndarray:
        img = np.full((H, W, 3), 128, np.uint8)
        for x1, y1, x2, y2 in self.objects.values():
            img[y1:y2, x1:x2] = BRIGHT
        return img

    def visit(self, dt=0.3) -> list:
        """One processed zone (round-robin, as the driver does), dt after the last frame."""
        self.scene.t += dt
        self.idx += 1
        return self.room.step(Frame(t=self.scene.t, wall=self.scene.t, img=self._frame(), idx=self.idx))

    def round(self, dt=0.3) -> list:
        """Every zone once."""
        out = []
        for _ in ZONES:
            out += self.visit(dt)
        return out

    def track(self, zone: str):
        (trk,) = self.room.tracker.tracks(zone)
        return trk

    # -- Grok for room crops

    def _grok(self, img):
        return self._answer(None)

    def _grok_verify(self, img, hints):
        return self._answer(hints)

    def _answer(self, hints):
        self.asked.append(hints)
        return dict(self.reply) if self.reply else None

    def grok(self, reply: dict) -> None:
        """Grok answers the newest room crop waiting (the worker takes the freshest arrival first)."""
        self.reply = reply
        assert self.namer.step() is True

    # -- asking

    def where(self, obj="remote", later=0.5) -> str:
        return answer(Intent("WHERE", obj, f"where is my {obj}"), self.world, self.world.events, self.raw_cfg,
                      now=self.scene.t + later).text


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    return ThingRig()


def found(events):
    return [(e.type, e.obj) for e in events if e.type == EventType.FOUND]


# ----- 1. a decoy already on the couch; the remote goes to the side table --------------------------

def test_a_decoy_already_on_the_couch_never_gets_the_remote_carried_to_the_side_table(rig):
    rig.put("decoy", COUCH_SPOT)                    # a remote-like thing sat there before anything happened
    rig.round()
    rig.round()                                     # confirmed; nothing left the table, so Grok isn't asked
    decoy = rig.track("couch")
    assert decoy.confirmed and not decoy.changed and rig.asked == []
    thing = rig.remote_leaves()
    rig.put("remote", SIDE_SPOT)                    # carried to the side table
    rig.round()                                     # hot: the arrival confirms on its first visit and is asked;
    rig.grok(REMOTE)                                # the static decoy is never asked (P0-3). Grok confirms it
    assert decoy.guess is None and rig.asked == [[REMOTE]]
    assert found(rig.round()) == [(EventType.FOUND, thing)]
    ent = rig.world.get(thing)
    assert (ent.status, ent.zone) == (Status.VISIBLE, "side_table")
    assert rig.track("side_table").entity == thing and decoy.role != "assoc"
    text = rig.where()
    assert "side table" in text and "couch" not in text and "also" not in text, text
    rig.world.room_cfg.thing_name_wait_s = 5.0
    for _ in range(4):                              # the decoy's wait runs out: given up, never found
        assert found(rig.round(dt=0.5)) == []
    assert decoy.role == "ignored" and rig.world.get(thing).zone == "side_table"
    assert "side table" in rig.where() and "couch" not in rig.where()


def test_a_decoy_arriving_while_the_remote_stays_on_the_table_is_not_a_handoff(rig):
    thing = rig.thing_named_on_table()
    rig.round()                                     # backgrounds
    rig.put("decoy", COUCH_SPOT)                    # a new object arrives on the couch
    rig.round()
    rig.round()
    decoy = rig.track("couch")
    assert decoy.confirmed and decoy.changed
    assert rig.asked == [] and decoy.guess is None  # no departure: not even asked
    decoy.guess = dict(REMOTE)                      # and had Grok called it a remote control anyway ...
    assert found(rig.round()) == []                 # ... nothing left the table: no handoff
    assert rig.world.get(thing).zone == "table" and decoy.role != "assoc"
    text = rig.where()
    assert "on the table" in text and "couch" not in text, text


# ----- 2. one departure, two arrivals -------------------------------------------------------------

def test_of_two_verified_arrivals_the_first_decided_gets_the_remote_and_the_departure_is_used_once(rig):
    rig.round()                                     # backgrounds
    thing = rig.remote_leaves()
    rig.put("a", COUCH_SPOT)
    rig.put("b", COUNTER_SPOT)
    rig.round()                                     # both appear (changed)
    rig.round()                                     # both confirmed and sent to Grok
    couch, counter = rig.track("couch"), rig.track("counter")
    assert rig.namer.pending() == 2
    rig.grok(REMOTE)                                # Grok answers the freshest crop first: the counter
    assert counter.guess == REMOTE and couch.guess is None
    assert found(rig.round()) == [(EventType.FOUND, thing)]
    assert rig.world.get(thing).zone == "counter" and thing not in rig.world._departures
    rig.grok(REMOTE)                                # the couch one is a remote control too, says Grok
    assert couch.guess == REMOTE
    for _ in range(3):
        assert found(rig.round()) == []             # the departure is spent: no second FOUND
    assert couch.role != "assoc" and rig.world.get(thing).zone == "counter"
    assert "counter" in rig.where() and "couch" not in rig.where()


def test_a_second_verified_arrival_after_the_handoff_gets_nothing(rig):
    rig.round()
    thing = rig.remote_leaves()
    rig.put("a", COUCH_SPOT)
    rig.round()
    rig.round()
    rig.grok(REMOTE)
    assert found(rig.round()) == [(EventType.FOUND, thing)]
    rig.put("b", COUNTER_SPOT)
    rig.round()
    rig.round()
    counter = rig.track("counter")
    assert counter.confirmed and counter.changed and counter.guess is None and rig.namer.pending() == 0
    counter.guess = dict(REMOTE)                    # even verified, the departure is gone
    assert found(rig.round()) == []
    assert rig.world.get(thing).zone == "couch" and counter.role != "assoc"


# ----- 3. a mismatch does not spend the departure ---------------------------------------------------

def test_a_decoy_grok_calls_a_phone_leaves_the_departure_for_the_real_remote(rig):
    rig.round()
    thing = rig.remote_leaves()
    rig.put("decoy", COUCH_SPOT)
    rig.round()
    rig.round()
    rig.grok(PHONE)                                 # "is it the remote?" - no, a phone
    decoy = rig.track("couch")
    assert decoy.guess == PHONE
    assert found(rig.round()) == []
    assert decoy.role == "ignored" and thing in rig.world._departures
    rig.put("remote", COUNTER_SPOT)                 # the real remote arrives on the counter
    rig.round()
    rig.round()
    rig.grok(REMOTE)
    assert found(rig.round()) == [(EventType.FOUND, thing)]
    assert rig.world.get(thing).zone == "counter" and decoy.role == "ignored"
    text = rig.where()
    assert "counter" in text and "couch" not in text, text


# ----- 4. the answer names the place, never the decoy -------------------------------------------------

def test_the_answer_names_the_side_table_and_never_the_couch_decoy(rig):
    rig.put("decoy", COUCH_SPOT)
    rig.round()
    rig.round()
    thing = rig.remote_leaves()
    rig.put("remote", SIDE_SPOT)
    rig.round()
    rig.grok(REMOTE)                                # the arrival, named; the static decoy is never asked (P0-3)
    assert found(rig.round()) == [(EventType.FOUND, thing)]
    text = rig.where()
    assert text.startswith("Your remote, I think, is on the side table."), text
    assert "couch" not in text and "I also see" not in text
    assert rig.world.place(thing, now=rig.scene.t).conflicts == []
