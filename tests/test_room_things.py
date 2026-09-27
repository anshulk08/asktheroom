"""Unnamed things in room zones (core/room.py, spec 0009): YOLOE proposals in the zone pass become cls 'thing'
observations, and each confirmed thing track is named once by Grok in the background (RoomNamer). A fake
backend (props), a fake proposer (crop px), a stub world and a fake name_fn. No hardware, no network."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from core.proposals import Proposal
from core.room import RoomMemory, RoomNamer
from core.room_types import RoomConfig, RoomTrack
from core.room_zones import Zone, Zones, view_version
from core.types import Frame

W, H = 1920, 1080
TABLE_RECT = (360, 202, 1560, 877)
TO_OBJ = {"keys": "keys", "wallet": "wallet", "hand": "hand"}
GUESS = {"name": "remote control", "also": ["remote"], "confidence": 0.7}


class FakeBackend:
    def __init__(self, raw=()):
        self.raw = list(raw)

    def infer(self, img):
        return list(self.raw)


class FakeProposer:
    """propose() returns self.out (crop px) and records what it was given."""

    def __init__(self, out=(), fail=False):
        self.out, self.fail, self.calls = list(out), fail, []

    def propose(self, img, known, hands):
        self.calls.append((img.shape, list(known), list(hands)))
        if self.fail:
            raise RuntimeError("yoloe went wrong")
        return list(self.out)


class StubWorld:
    def __init__(self):
        self.visits = []

    def room_update(self, visit):
        self.visits.append(visit)
        return []


class Namer:
    """A fake name_fn: records the crops it was sent."""

    def __init__(self, result=GUESS, fail=False):
        self.result, self.fail, self.imgs, self.ctxs = result, fail, [], []

    def __call__(self, img, ctx=None):
        self.imgs.append(img)
        self.ctxs.append(ctx)
        if self.fail:
            raise RuntimeError("grok timed out")
        return dict(self.result) if self.result is not None else None


def rect_zone(name, x1, y1, x2, y2):
    return Zone(name, f"the {name}", [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


SHELF = rect_zone("shelf", 1600, 100, 1900, 400)


def make(zones=(SHELF,), raw=(), props=(), namer=None, **cfg):
    cfg.setdefault("room_every_n", 1)
    backend, world, proposer = FakeBackend(raw), StubWorld(), FakeProposer(props)
    zs = Zones("v", (W, H), {z.name: z for z in zones})
    rm = RoomMemory(RoomConfig(enabled=True, **cfg), zs, backend, TO_OBJ, world, TABLE_RECT,
                    proposer=proposer, namer=namer)
    return rm, proposer, world


def frame(i):
    return Frame(t=float(i), wall=1000.0 + i, img=np.full((H, W, 3), 128, np.uint8), idx=i)


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def things(rm):
    return [t for t in rm.tracker.tracks() if t.cls == "thing"]


# ---------------------------------------------------------------- thing observations

def test_proposals_become_thing_tracks_in_full_px():
    rm, prop, _ = make(raw=[("keys", 0.9, (200, 200, 240, 240)), ("hand", 0.9, (250, 10, 290, 50))],
                       props=[Proposal((10, 20, 50, 60), 0.4)])
    rm.step(frame(1))
    [th] = things(rm)
    assert th.box_px == (1610, 120, 1650, 160) and not th.confirmed
    [(shape, known, hands)] = prop.calls
    assert shape == (301, 301, 3)
    assert known == [(200, 200, 240, 240)] and hands == [(250, 10, 290, 50)]     # crop px
    assert [t.cls for t in rm.tracker.tracks()] == ["keys", "thing"]


def test_thing_observation_carries_the_proposal_score():
    rm, _, world = make(props=[Proposal((10, 20, 50, 60), 0.4321)])
    captured = []
    real = rm.tracker.visit

    def spy(zone, say, obs, *a, **kw):
        captured.extend(obs)
        return real(zone, say, obs, *a, **kw)

    rm.tracker.visit = spy
    rm.step(frame(1))
    [o] = captured
    assert o.cls == "thing" and o.conf == 0.432 and o.zone == "shelf" and o.t == 1.0 and o.frame_idx == 1


def test_occluded_proposals_are_skipped():
    rm, _, _ = make(props=[Proposal((10, 20, 50, 60), 0.5, occluded=True), Proposal((100, 100, 140, 140), 0.5)])
    rm.step(frame(1))
    assert [t.box_px for t in things(rm)] == [(1700, 200, 1740, 240)]


def test_a_proposal_on_a_prop_observation_is_the_prop():
    rm, _, _ = make(raw=[("keys", 0.9, (10, 20, 50, 60))],
                    props=[Proposal((11, 21, 51, 61), 0.6), Proposal((100, 100, 140, 140), 0.5)])
    rm.step(frame(1))
    assert [(t.cls, t.box_px) for t in rm.tracker.tracks()] == [("keys", (1610, 120, 1650, 160)),
                                                                ("thing", (1700, 200, 1740, 240))]


def test_a_proposal_that_is_the_hand_is_dropped():
    rm, _, _ = make(raw=[("hand", 0.9, (10, 20, 50, 60))], props=[Proposal((11, 21, 51, 61), 0.6)])
    rm.step(frame(1))
    assert things(rm) == []


def test_polygon_and_table_rect_filters_apply_to_things():
    tri = Zone("shelf", "the shelf", [(1600, 100), (1900, 100), (1600, 400)])
    rm, _, _ = make(zones=(tri,), props=[Proposal((240, 240, 280, 280), 0.5), Proposal((10, 10, 30, 30), 0.5)])
    rm.step(frame(1))
    assert [t.box_px for t in things(rm)] == [(1610, 110, 1630, 130)]
    desk = rect_zone("desk", 1400, 700, 1800, 1000)
    rm, _, _ = make(zones=(desk,), props=[Proposal((10, 10, 30, 30), 0.5), Proposal((310, 210, 330, 230), 0.5)])
    rm.step(frame(1))
    assert [t.box_px for t in things(rm)] == [(1710, 910, 1730, 930)]


def test_things_follow_a_resized_zone_crop():
    big = rect_zone("wall", 0, 0, 1000, 500)
    rm, prop, _ = make(zones=(big,), raw=[("keys", 0.9, (100, 200, 120, 220))],
                       props=[Proposal((100, 100, 150, 150), 0.5)], max_crop_px=500)
    rm.step(frame(1))
    assert prop.calls[0][0] == (250, 500, 3) and prop.calls[0][1] == [(100, 200, 120, 220)]
    assert [t.box_px for t in things(rm)] == [(200, 200, 300, 300)]


def test_a_failing_proposer_is_logged_once_and_props_go_on(caplog):
    rm, prop, world = make(raw=[("keys", 0.9, (10, 20, 50, 60))])
    prop.fail = True
    with caplog.at_level(logging.WARNING, logger="core.room"):
        rm.step(frame(1))
        rm.step(frame(2))
    assert [t.cls for t in rm.tracker.tracks()] == ["keys"] and len(world.visits) == 2
    assert len(prop.calls) == 2
    assert sum("proposer" in r.getMessage() for r in caplog.records) == 1        # rate-limited


def test_things_off_never_calls_the_proposer():
    rm, prop, _ = make(raw=[("keys", 0.9, (10, 20, 50, 60))], props=[Proposal((100, 100, 140, 140), 0.5)],
                       things=False)
    rm.step(frame(1))
    assert prop.calls == [] and [t.cls for t in rm.tracker.tracks()] == ["keys"]


# ---------------------------------------------------------------- naming

def test_a_confirmed_thing_is_named_once_in_the_background():
    fn = Namer()
    namer = RoomNamer(fn)
    try:
        rm, _, world = make(props=[Proposal((100, 100, 140, 140), 0.5)], namer=namer)
        rm.step(frame(1))
        [th] = things(rm)
        assert not th.name_asked                                   # unconfirmed: not asked yet
        rm.step(frame(2))
        assert th.confirmed and th.name_asked
        assert wait_for(lambda: th.guess is not None)
        assert th.guess == GUESS
        rm.step(frame(3))
        rm.step(frame(4))
        time.sleep(0.1)
        assert len(fn.imgs) == 1                                   # once per track
        assert fn.imgs[0].shape == (80, 80, 3)                     # 40x40 box + 50% per side (NAME_MARGIN), native px
    finally:
        namer.stop()


def test_open_naming_also_sends_the_marked_spot():
    fn = Namer()
    namer = RoomNamer(fn, start=False, clock=lambda: 0.0)
    big = rect_zone("wall", 0, 0, 1000, 500)
    rm, _, _ = make(zones=(big,), props=[Proposal((100, 100, 150, 150), 0.5)], namer=namer, max_crop_px=500)
    rm.step(frame(1))
    rm.step(frame(2))
    assert namer.step(0.0)
    ctx = fn.ctxs[0]
    red = (ctx[..., 2] > 200) & (ctx[..., 1] < 50) & (ctx[..., 0] < 50)
    assert ctx.shape[0] >= 240 and red.any()


def test_a_named_crop_is_a_copy_of_the_native_zone_crop():
    fn = Namer()
    namer = RoomNamer(fn, start=False, clock=lambda: 0.0)
    big = rect_zone("wall", 0, 0, 1000, 500)
    rm, _, _ = make(zones=(big,), props=[Proposal((100, 100, 150, 150), 0.5)], namer=namer, max_crop_px=500)
    f1, f2 = frame(1), frame(2)
    rm.step(f1)
    rm.step(f2)
    f2.img[:] = 0
    assert namer.step(0.0)
    img = fn.imgs[0]
    assert img.shape == (200, 200, 3) and img.min() == 128        # 100x100 full px + 50%, copied before f2 changed


def test_props_are_never_sent_to_grok():
    fn = Namer()
    namer = RoomNamer(fn, start=False, clock=lambda: 0.0)
    rm, _, _ = make(raw=[("keys", 0.9, (10, 20, 50, 60))], namer=namer)
    rm.step(frame(1))
    rm.step(frame(2))
    assert namer.pending() == 0 and not namer.step(0.0)


def test_perception_never_waits_on_grok():
    gate = threading.Event()

    def slow(img, ctx=None):
        gate.wait(5)
        return dict(GUESS)

    namer = RoomNamer(slow)
    try:
        rm, _, _ = make(props=[Proposal((100, 100, 140, 140), 0.5)], namer=namer)
        t0 = time.monotonic()
        for i in range(1, 6):
            rm.step(frame(i))
        assert time.monotonic() - t0 < 1.0
        [th] = things(rm)
        assert th.guess is None
        gate.set()
        assert wait_for(lambda: th.guess is not None)
    finally:
        gate.set()
        namer.stop()


def track(i=1):
    return RoomTrack(tid=f"r:{i}", zone="shelf", cls="thing", box_px=(0, 0, 10, 10), first_seen=0.0,
                     first_wall=0.0, last_seen=0.0, last_wall=0.0, confirmed=True)


def img():
    return np.zeros((20, 20, 3), np.uint8)


def test_offline_jobs_wait_for_online():
    fn, online = Namer(), [False]
    namer = RoomNamer(fn, online=lambda: online[0])
    try:
        tr = track()
        namer.submit(tr, img())
        time.sleep(0.3)
        assert fn.imgs == [] and namer.pending() == 1
        online[0] = True
        assert wait_for(lambda: tr.guess is not None)
        assert len(fn.imgs) == 1
    finally:
        namer.stop()


def test_per_minute_cap():
    fn = Namer()
    namer = RoomNamer(fn, per_minute=2, start=False, clock=lambda: 0.0)
    trs = [track(i) for i in range(3)]
    for tr in trs:
        namer.submit(tr, img())
    assert namer.step(0.0) and namer.step(1.0)
    assert not namer.step(30.0) and len(fn.imgs) == 2
    assert namer.step(60.5)
    assert [t.guess is not None for t in trs] == [True, True, True]


def test_a_failed_call_is_retried_once_after_5_s_then_given_up():
    for fn in (Namer(fail=True), Namer(result=None)):
        namer = RoomNamer(fn, start=False, clock=lambda: 0.0)
        tr = track()
        namer.submit(tr, img())
        assert namer.step(0.0) and len(fn.imgs) == 1
        assert not namer.step(4.0) and len(fn.imgs) == 1
        assert namer.step(5.0) and len(fn.imgs) == 2
        assert not namer.step(100.0) and len(fn.imgs) == 2
        assert tr.guess is None and namer.pending() == 0


def test_a_retry_that_succeeds_names_the_track():
    calls = []

    def flaky(im):
        calls.append(im)
        if len(calls) == 1:
            raise RuntimeError("grok timed out")
        return dict(GUESS)

    namer = RoomNamer(flaky, start=False, clock=lambda: 0.0)
    tr = track()
    namer.submit(tr, img())
    namer.step(0.0)
    namer.step(5.0)
    assert tr.guess == GUESS


def test_the_queue_keeps_the_newest_eight():
    namer = RoomNamer(Namer(), per_minute=100, start=False, clock=lambda: 0.0)
    trs = [track(i) for i in range(10)]
    for tr in trs:
        namer.submit(tr, img())
    assert namer.pending() == 8
    while namer.step(0.0):
        pass
    assert [t.guess is not None for t in trs] == [False, False] + [True] * 8


def test_stop_ends_the_worker():
    namer = RoomNamer(Namer())
    namer.stop()
    assert not namer._thread.is_alive()


# ---------------------------------------------------------------- from_config

def zones_cfg(tmp_path, **rm):
    p = tmp_path / "zones.json"
    Zones(view_version((W, H), 100, TABLE_RECT), (W, H), {"shelf": SHELF}).save(p)
    d = {"enabled": True, "zones_path": str(p), "room_every_n": 1}
    d.update(rm)
    return {"objects": {"keys": {}, "wallet": {}}, "conf_threshold": {"default": 0.35}, "room_memory": d}


def test_from_config_without_a_proposer_is_props_only(tmp_path):
    rm = RoomMemory.from_config(zones_cfg(tmp_path), StubWorld(), FakeBackend(), TABLE_RECT)
    assert rm.proposer is None and rm.namer is None
    rm.stop()


def test_from_config_with_a_proposer_and_name_fn_names_things(tmp_path):
    fn, prop = Namer(), FakeProposer([Proposal((100, 100, 140, 140), 0.5)])
    rm = RoomMemory.from_config(zones_cfg(tmp_path, names_per_minute=3), StubWorld(), FakeBackend(), TABLE_RECT,
                                proposer=prop, name_fn=fn, online=lambda: True)
    try:
        assert rm.proposer is prop and isinstance(rm.namer, RoomNamer) and rm.namer.per_minute == 3
        rm.step(frame(1))
        rm.step(frame(2))
        [th] = things(rm)
        assert wait_for(lambda: th.guess is not None)
    finally:
        rm.stop()
    assert not rm.namer._thread.is_alive()


def test_from_config_things_off_drops_the_proposer_and_namer(tmp_path):
    rm = RoomMemory.from_config(zones_cfg(tmp_path, things=False), StubWorld(), FakeBackend(), TABLE_RECT,
                                proposer=FakeProposer(), name_fn=Namer())
    assert rm.proposer is None and rm.namer is None


def test_from_config_proposer_without_name_fn_tracks_unnamed_things(tmp_path):
    prop = FakeProposer([Proposal((100, 100, 140, 140), 0.5)])
    rm = RoomMemory.from_config(zones_cfg(tmp_path), StubWorld(), FakeBackend(), TABLE_RECT, proposer=prop)
    rm.step(frame(1))
    rm.step(frame(2))
    [th] = things(rm)
    assert th.confirmed and not th.name_asked and rm.namer is None
    rm.stop()


# ---------------------------------------------------------------- main.make_room_memory

class YModel:
    """Stands in for a loaded YOLOE model: never predicts here."""


class Det:
    backend = "the prop model"

    def __init__(self, proposer=None):
        self.proposer = proposer


def main_cfg(tmp_path, auto_name=True, **rm):
    cfg = zones_cfg(tmp_path, **rm)
    cfg["proposals"] = {"enabled": True, "kind": "yoloe",
                        "yoloe": {"model": "models/x.pt", "conf": 0.2, "ignore_px": [[0, 0, 10, 10]]}}
    cfg["auto_name"] = {"enabled": auto_name}
    return cfg


class FakeAutoNamer:
    made = []

    def __init__(self, cfg, world, online=None, start=True):
        self.online, self.start = online, start
        FakeAutoNamer.made.append(self)

    def _ask(self, img):
        return dict(GUESS)


def yoloe_det():
    from core.proposals import YOLOEProposer
    return Det(YOLOEProposer({"model": "models/x.pt"}, model=YModel()))


def test_make_room_memory_shares_the_detectors_yoloe_model_and_names_with_grok(tmp_path, monkeypatch):
    import core.auto_name
    import main
    FakeAutoNamer.made = []
    monkeypatch.setattr(core.auto_name, "AutoNamer", FakeAutoNamer)
    det = yoloe_det()
    world = StubWorld()
    world.online = False
    rm = main.make_room_memory(main_cfg(tmp_path), world, det, TABLE_RECT)
    try:
        assert rm.backend == "the prop model"
        assert rm.proposer is not det.proposer and rm.proposer.model is det.proposer.model
        assert rm.proposer._roi is None and rm.proposer.cfg.ignore_px == [] and rm.proposer.cfg.conf == 0.2
        [an] = FakeAutoNamer.made
        assert an.start is False and rm.namer.name_fn.__self__ is an
        assert rm.namer.online() is False
        world.online = True
        assert rm.namer.online() is True
    finally:
        rm.stop()


def test_make_room_memory_is_props_only_without_a_yoloe_proposer(tmp_path):
    import main
    from core.proposals import ChangeProposer
    for det in (Det(), Det(ChangeProposer())):
        rm = main.make_room_memory(main_cfg(tmp_path), StubWorld(), det, TABLE_RECT)
        assert rm is not None and rm.proposer is None and rm.namer is None


def test_make_room_memory_things_unnamed_when_auto_name_is_off(tmp_path):
    import main
    rm = main.make_room_memory(main_cfg(tmp_path, auto_name=False), StubWorld(), yoloe_det(), TABLE_RECT)
    assert rm.proposer is not None and rm.namer is None


def test_make_room_memory_things_off_builds_nothing(tmp_path, monkeypatch):
    import core.auto_name
    import main
    FakeAutoNamer.made = []
    monkeypatch.setattr(core.auto_name, "AutoNamer", FakeAutoNamer)
    rm = main.make_room_memory(main_cfg(tmp_path, things=False), StubWorld(), yoloe_det(), TABLE_RECT)
    assert rm.proposer is None and rm.namer is None and FakeAutoNamer.made == []


def test_make_room_memory_falls_back_to_props_when_the_namer_fails(tmp_path, monkeypatch, caplog):
    import core.auto_name
    import main

    def broken(*a, **kw):
        raise RuntimeError("no api key")

    monkeypatch.setattr(core.auto_name, "AutoNamer", broken)
    with caplog.at_level(logging.ERROR, logger="askroom.main"):
        rm = main.make_room_memory(main_cfg(tmp_path), StubWorld(), yoloe_det(), TABLE_RECT)
    assert rm is not None and rm.proposer is None and rm.namer is None
    assert any("room things" in r.getMessage() for r in caplog.records)


# ----- naming only while a handoff is possible; "is it one of these?" ------------------------------

class HintWorld(StubWorld):
    def __init__(self, hints):
        super().__init__()
        self.hints = hints

    def room_handoff_hints(self, t):
        return list(self.hints)


REMOTE = {"name": "remote control", "also": ["remote"], "confidence": 0.7}


def _confirm_one(rm, box_full=(1700, 200, 1740, 240)):
    """Two visits of the thing's zone; the first frame is bright where the thing is, so the second visit
    sees the change: it arrived (only arrivals are sent to Grok while a handoff is open, spec 0010 P0-3)."""
    f1, f2 = frame(1), frame(2)
    x1, y1, x2, y2 = box_full
    f1.img[y1:y2, x1:x2] = 255
    rm.step(f1)
    rm.step(f2)


def test_room_clutter_is_not_named_while_no_handoff_is_possible():
    fn = Namer()
    namer = RoomNamer(fn, start=False, clock=lambda: 0.0)
    rm, _, _ = make(props=[Proposal((100, 100, 140, 140), 0.5)], namer=namer)
    rm.world = HintWorld([])
    _confirm_one(rm)
    [th] = things(rm)
    assert th.confirmed and th.changed
    assert namer.pending() == 0 and not any(t.name_asked for t in rm.tracker.tracks())
    rm.world.hints = [REMOTE]                                # something just left the table
    rm.step(frame(3))
    assert namer.pending() == 1


def test_a_candidate_is_verified_by_name_not_named_openly():
    asked = []

    def verify(img, hints):
        asked.append(hints)
        return dict(hints[0])

    fn = Namer()
    namer = RoomNamer(fn, start=False, clock=lambda: 0.0, verify_fn=verify)
    rm, _, _ = make(props=[Proposal((100, 100, 140, 140), 0.5)], namer=namer)
    rm.world = HintWorld([REMOTE])
    _confirm_one(rm)
    assert namer.step(0.0)
    [th] = [t for t in rm.tracker.tracks() if t.cls == "thing"]
    assert asked == [[REMOTE]] and th.guess == REMOTE and fn.imgs == []


def test_the_newest_job_goes_first():
    order = []
    namer = RoomNamer(lambda img: order.append(img.shape[0]) or {"name": "x", "also": [], "confidence": 0.9},
                      start=False, clock=lambda: 0.0)
    from core.room_types import RoomTrack
    for n in (10, 20, 30):
        namer.submit(RoomTrack(f"r:{n}", "couch", "thing", (0, 0, 1, 1), 0, 0, 0, 0), np.zeros((n, n, 3), np.uint8))
    namer.step(0.0)
    assert order == [30]


def test_verify_fn_maps_a_match_to_the_hint_and_none_to_its_own_name():
    import json

    from core.room import make_verify_fn

    class Reply:
        def __init__(self, d):
            self.text = json.dumps(d)

    class Provider:
        def __init__(self, d):
            self.d, self.calls = d, []

        def narrate(self, system, parts, schema):
            self.calls.append(parts)
            return Reply(self.d)

    class N:
        pass

    n = N()
    n.c = type("C", (), {"crop_px": 384, "jpeg_quality": 90, "min_confidence": 0.5})()
    img = np.full((60, 60, 3), 128, np.uint8)
    n.provider = Provider({"match": "remote control", "name": "tv remote", "confidence": 0.8})
    assert make_verify_fn(n)(img, [REMOTE]) == REMOTE
    assert "remote control" in n.provider.calls[0][0][1]
    n.provider = Provider({"match": "none", "name": "phone", "confidence": 0.7})
    assert make_verify_fn(n)(img, [REMOTE])["name"] == "phone"
    n.provider = Provider({"match": "remote control", "name": "remote", "confidence": 0.2})
    assert make_verify_fn(n)(img, [REMOTE])["name"] == "remote"     # too unsure to call it a match


def test_verify_needs_grok_s_own_description_to_fit():
    import json

    from core.room import make_verify_fn

    class Reply:
        def __init__(self, d):
            self.text = json.dumps(d)

    class Provider:
        def __init__(self, d):
            self.d = d

        def narrate(self, system, parts, schema):
            return Reply(self.d)

    class N:
        pass

    n = N()
    n.c = type("C", (), {"crop_px": 384, "jpeg_quality": 90, "min_confidence": 0.5})()
    img = np.full((60, 60, 3), 128, np.uint8)
    n.provider = Provider({"match": "remote control", "name": "computer keyboard", "confidence": 0.9})
    assert make_verify_fn(n)(img, [REMOTE])["name"] == "computer keyboard"      # a yes that describes a keyboard
    n.provider = Provider({"match": "remote control", "name": "tv remote", "confidence": 0.65})
    assert make_verify_fn(n)(img, [REMOTE]) != REMOTE                            # under 0.7
    n.provider = Provider({"match": "remote control", "name": "tv remote", "confidence": 0.8})
    assert make_verify_fn(n)(img, [REMOTE]) == REMOTE


def test_a_candidate_is_sent_boxed_in_red_with_its_surroundings():
    from core.room import MARK_MIN_SIDE

    asked = []

    def verify(img, hints):
        asked.append(img)
        return dict(hints[0])

    namer = RoomNamer(Namer(), start=False, clock=lambda: 0.0, verify_fn=verify)
    big = rect_zone("wall", 0, 0, 1000, 500)
    rm, _, _ = make(zones=(big,), props=[Proposal((300, 200, 320, 210), 0.5)], namer=namer, max_crop_px=1000)
    rm.world = HintWorld([REMOTE])
    _confirm_one(rm, box_full=(300, 200, 320, 210))
    assert namer.step(0.0)
    img = asked[0]
    assert min(img.shape[:2]) >= MARK_MIN_SIDE                       # a 20x10 box gets a 240 px patch
    red = (img[..., 2] > 200) & (img[..., 1] < 50) & (img[..., 0] < 50)
    assert red.sum() > 0                                             # the box is drawn
