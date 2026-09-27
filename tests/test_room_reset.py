"""A spoken reset without a restart (spec 0010 P0-4): RoomMemory.reset drops the tracker's tracks, the namer's
queued and in-flight Grok jobs and every zone's previous crop, and the World's room side; Room.perceive
runs it on the perception thread when RESET was asked. After it, the next run passes like a fresh start."""
from __future__ import annotations

import logging

import numpy as np
import pytest

import main
from core.events import EventLog
from core.room import RoomNamer, RoomTracker
from core.room_types import RoomConfig, RoomObservation, RoomTrack
from core.types import EventType, Status
from core.world import World
from tests.test_room_decoy import COUCH_SPOT, COUNTER_SPOT, REMOTE, SIDE_SPOT, ThingRig, found
from tests.test_room_e2e import KEYS_IN_CROP, Rig
from tests.test_room_main import CFG, Det, FullFrames, Table, frame


class Hands:
    def update(self, hands, t):
        return hands

    def reset(self):
        pass


def obs(t, box=(100, 100, 140, 130)):
    return RoomObservation(zone="shelf", cls="keys", conf=0.9, box_px=box, t=t, wall=t, frame_idx=int(t))


def track(tid="r:1"):
    return RoomTrack(tid=tid, zone="couch", cls="thing", box_px=(0, 0, 10, 10), first_seen=0, first_wall=0,
                     last_seen=0, last_wall=0)


# ----- the parts -------------------------------------------------------------------------------------

def test_tracker_reset_drops_every_track_and_keeps_ids_increasing():
    tr = RoomTracker(RoomConfig())
    tr.visit("shelf", "the shelf", [obs(1.0)], [], [], 1.0, 1.0, 1)
    v = tr.visit("shelf", "the shelf", [obs(1.5)], [], [], 1.5, 1.5, 2)
    assert [t.tid for t in v.confirmed] == ["r:1"]
    tr.reset()
    assert tr.tracks() == []
    v = tr.visit("shelf", "the shelf", [obs(2.0)], [], [], 2.0, 2.0, 3)
    assert v.confirmed == [] and [t.tid for t in tr.tracks()] == ["r:2"]     # new, unconfirmed, a new id


def test_namer_reset_drops_the_queued_jobs():
    calls = []
    namer = RoomNamer(lambda img: calls.append(1) or dict(REMOTE), per_minute=100, start=False)
    old = track("r:1")
    namer.submit(old, np.zeros((8, 8, 3), np.uint8))
    assert namer.pending() == 1
    namer.reset()
    assert namer.pending() == 0 and namer.step() is False and old.guess is None and calls == []
    new = track("r:2")
    namer.submit(new, np.zeros((8, 8, 3), np.uint8))          # after the reset: named as usual
    assert namer.step() is True and new.guess == REMOTE


def test_a_guess_answered_during_a_reset_never_lands_and_is_not_retried():
    """The worker may be inside the Grok call when the reset comes: its answer is for a track that no
    longer exists, so it is thrown away, and a None from it is not queued again either."""
    namer = RoomNamer(None, per_minute=100, start=False)
    replies = [dict(REMOTE), None]

    def name_fn(img):
        namer.reset()                                          # reset lands while Grok is thinking
        return replies.pop(0)

    namer.name_fn = name_fn
    for tid in ("r:1", "r:2"):
        old = track(tid)
        namer.submit(old, np.zeros((8, 8, 3), np.uint8))
        assert namer.step() is True
        assert old.guess is None and namer.pending() == 0


# ----- the memory ------------------------------------------------------------------------------------

def test_memory_reset_clears_backgrounds_tracks_jobs_and_the_worlds_room_side(monkeypatch):
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    rig = ThingRig()
    rig.round()
    thing = rig.remote_leaves()
    rig.put("remote", SIDE_SPOT)
    rig.put("clutter", COUNTER_SPOT)
    rig.round()
    rig.round()                                                # both confirmed and queued for Grok
    assert rig.namer.pending() == 2 and rig.room._prev and rig.room.tracker.tracks()
    cfg = rig.world.room_cfg
    rig.world.room_cfg.thing_name_wait_s = 7.0
    rig.room.reset()
    assert rig.room._prev == {} and rig.room.tracker.tracks() == [] and rig.namer.pending() == 0
    assert rig.namer.step() is False
    assert (rig.world._room, rig.world._departures, rig.world._conflicts, rig.world._thing_tracks,
            rig.world._pending_things) == ({}, {}, {}, {}, {})
    assert rig.world.room_cfg is cfg and cfg.thing_name_wait_s == 7.0     # the config is not re-read
    assert rig.world.get(thing).zone == "table"                # the entities are World.reset's business


def test_props_handoff_reset_handoff_is_a_fresh_start(monkeypatch):
    """The app's order: World.reset on the asking thread, then RoomMemory.reset on the perception thread.
    One visit of the old tracks can slip in between (here: the shelf track, an 'assoc' one, re-decided
    against the fresh world as a conflict); the room reset wipes that too."""
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    rig = Rig()
    rig.carry_keys_off_left_edge()
    rig.room_visit(KEYS_IN_CROP)
    assert [e.type for e in rig.room_visit(KEYS_IN_CROP)] == [EventType.FOUND]
    assert rig.world.get("keys").zone == "shelf"
    rig.world.reset()
    rig.room_visit(KEYS_IN_CROP)                               # the visit in between
    assert rig.world.room_json()["conflicts"] != []           # a leftover a reset must not keep
    rig.room.reset()
    assert rig.world.room_json() == {"conflicts": []} and rig.room.tracker.tracks() == []
    assert "also" not in rig.where()
    rig.carry_keys_off_left_edge()                             # the second run
    assert rig.room_visit(KEYS_IN_CROP) == []
    assert [e.type for e in rig.room_visit(KEYS_IN_CROP)] == [EventType.FOUND]
    keys = rig.world.get("keys")
    assert (keys.status, keys.zone) == (Status.VISIBLE, "shelf")
    (trk,) = rig.room.tracker.tracks()
    assert trk.tid != "r:1" and trk.entity == "keys"
    text = rig.where()
    assert text.startswith("Your keys are on the bookshelf.") and "also" not in text


def test_things_handoff_reset_handoff_is_a_fresh_start(monkeypatch):
    """A Grok-named thing handed to the side table, a clutter crop still waiting for Grok, then a reset:
    the next thing carried to the couch is found there, nothing from before lands anywhere."""
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    rig = ThingRig(table_replies=(REMOTE, REMOTE))
    rig.round()
    first = rig.remote_leaves()
    rig.put("remote", SIDE_SPOT)
    rig.round()
    rig.round()
    rig.grok(REMOTE)
    assert found(rig.round()) == [(EventType.FOUND, first)]
    rig.put("clutter", COUNTER_SPOT)
    rig.round()
    rig.round()                                                # confirmed; hints are [] now: not asked
    rig.world.reset()                                          # the asking thread ...
    rig.take("remote")                                         # the demo brings the object back
    rig.take("clutter")
    rig.round()                                                # ... one round with the old tracks slips in
    rig.room.reset()                                           # ... then the perception thread
    assert rig.namer.step() is False and rig.room.tracker.tracks() == []
    assert first not in rig.world.entities
    rig.round()                                                # fresh backgrounds (zones are visited 3x a second)
    second = rig.remote_leaves(key="remote2")
    assert second != first
    rig.put("remote2", COUCH_SPOT)
    rig.round()
    rig.round()
    assert rig.namer.pending() == 1                            # only the new arrival is asked
    rig.grok(REMOTE)
    assert found(rig.round()) == [(EventType.FOUND, second)]
    assert rig.world.get(second).zone == "couch" and rig.world.room_json()["conflicts"] == []
    text = rig.where()
    assert "couch" in text and "side table" not in text and "counter" not in text, text


# ----- the wiring ------------------------------------------------------------------------------------

class Memory:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def step(self, full):
        self.calls.append("step")
        return []

    def reset(self):
        self.calls.append("reset")
        if self.fail:
            raise RuntimeError("no")


def make_room(tmp_path, memory):
    events = EventLog(":memory:", str(tmp_path))
    room = main.Room(CFG, World(CFG, events), events, Table(), FullFrames(), None, None, detector=Det(),
                     hands=Hands())
    room.room_memory = memory
    room.detector.reset_proposals = lambda: None
    return room


def test_reset_asked_resets_room_memory_once_on_the_perception_thread_before_its_next_step(tmp_path):
    mem = Memory()
    room = make_room(tmp_path, mem)
    room.perceive(frame(1))
    room._clear_ev.set()                                       # what Room.ask does on RESET
    room.perceive(frame(2))
    room.perceive(frame(3))
    assert mem.calls == ["step", "reset", "step", "step"]


def test_a_failing_room_reset_is_logged_and_perception_goes_on(tmp_path, caplog):
    room = make_room(tmp_path, Memory(fail=True))
    room._clear_ev.set()
    with caplog.at_level(logging.ERROR, logger="askroom.main"):
        dets, evs = room.perceive(frame(1))
    assert len(dets.items) == 1 and any("room memory reset" in r.getMessage() for r in caplog.records)
    assert not room._clear_ev.is_set()


def test_room_ask_reset_reaches_room_memory(tmp_path):
    """Through Room.ask ('reset the table'): the world resets at once, the room memory on the next perceive."""
    from tests.test_main import make_ask, no_model
    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    mem = Memory()
    room = main.Room(CFG, world, events, Table(), FullFrames(), None, make_ask(CFG, world, events, net=None,
                                                                            other=no_model),
                     detector=Det(), hands=Hands())
    room.room_memory = mem
    room.detector.reset_proposals = lambda: None
    room.perceive(frame(1))
    world.entities["keys"].pos_cm = (10.0, 10.0)
    room.ask("reset the table", "voice")
    assert world.entities["keys"].pos_cm is None and mem.calls == ["step"]
    room.perceive(frame(2))
    assert mem.calls == ["step", "reset", "step"]
