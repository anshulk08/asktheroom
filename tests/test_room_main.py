"""Room memory wiring (spec 0009 M0, plan Task E): Room.perceive runs RoomMemory.step on the full frame, main
opens the camera through the table view when room_memory.enabled, and demo_check checks the zones file."""
import logging
import sys
import types

import numpy as np
import pytest

import demo_check as dc
import main
from core.config import load_config
from core.events import EventLog
from core.types import Detection, Detections, Event, EventType, Frame
from core.world import World

CFG = load_config()
RECT = (360, 202, 1560, 877)


class Det:
    backend = "the prop model"

    def detect(self, f):
        d = Detection("wallet", 0.9, (100, 100, 150, 150), (60.0, 15.0), (58, 13, 62, 17))
        return Detections(t=f.t, frame_idx=f.idx, items=[d], hands=[])


class Hands:
    def update(self, hands, t):
        return hands


class Table:
    ok = True


class FullFrames:
    """The TableView side the perception thread uses: full_at(t) gives the full frame captured at t."""

    def __init__(self):
        self.asked = []

    def full_at(self, t):
        self.asked.append(t)
        return Frame(t=t, wall=1000.0 + t, img=np.zeros((1080, 1920, 3), np.uint8), idx=int(t * 10))


class Memory:
    def __init__(self, out=None, fail=False):
        self.steps, self.out, self.fail = [], out or [], fail

    def step(self, full):
        self.steps.append(full)
        if self.fail:
            raise RuntimeError("zone crop went wrong")
        return list(self.out)


def room_event(t):
    return Event(t=t, wall=1000.0 + t, obj="keys", type=EventType.FOUND)


def make_room(tmp_path, frames, memory=None):
    events = EventLog(":memory:", str(tmp_path))
    room = main.Room(CFG, World(CFG, events), events, Table(), frames, None, None, detector=Det(), hands=Hands())
    room.room_memory = memory
    return room


def frame(i):
    return Frame(t=i / 10, wall=1000.0 + i / 10, img=np.zeros((720, 1280, 3), np.uint8), idx=i)


# ---------------------------------------------------------------- Room.perceive

def test_room_memory_is_off_unless_build_attaches_it(tmp_path):
    assert make_room(tmp_path, FullFrames()).room_memory is None
    events = EventLog(":memory:", str(tmp_path))
    assert main.Room(CFG, World(CFG, events), events, Table(), None, None, None).room_memory is None


def test_perceive_steps_room_memory_on_the_full_frame_and_appends_its_events(tmp_path):
    frames, ev = FullFrames(), room_event(0.3)
    mem = Memory(out=[ev])
    room = make_room(tmp_path, frames, mem)
    table_only = make_room(tmp_path / "b", FullFrames())
    for i in range(1, 11):
        dets, evs = room.perceive(frame(i))
        _, table_evs = table_only.perceive(frame(i))
        assert evs == table_evs + [ev]                                  # table events first, room after
    assert frames.asked == [i / 10 for i in range(1, 11)]               # the full frame of each table frame
    assert [f.img.shape for f in mem.steps] == [(1080, 1920, 3)] * 10
    assert str(room.world.get("wallet").status) == "VISIBLE"


def test_a_failing_room_step_is_logged_once_and_table_perception_goes_on(tmp_path, caplog, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    mem = Memory(fail=True)
    room = make_room(tmp_path, FullFrames(), mem)
    table_only = make_room(tmp_path / "b", FullFrames())
    with caplog.at_level(logging.ERROR, logger="askroom.main"):
        for i in range(1, 6):
            dets, evs = room.perceive(frame(i))
            assert evs == table_only.perceive(frame(i))[1] and len(dets.items) == 1
        assert len(mem.steps) == 5
        assert sum("room memory" in r.getMessage() for r in caplog.records) == 1       # rate-limited
        now[0] += 11
        for i in range(6, 11):
            room.perceive(frame(i))
        assert sum("room memory" in r.getMessage() for r in caplog.records) == 2
    assert str(room.world.get("wallet").status) == "VISIBLE"


def test_without_room_memory_perceive_is_unchanged(tmp_path):
    frames = FullFrames()
    room = make_room(tmp_path, frames)
    dets, evs = room.perceive(frame(1))
    assert frames.asked == [] and len(dets.items) == 1 and isinstance(evs, list)


def test_no_full_frame_no_room_step(tmp_path):
    class NoFull:                                   # a plain FrameBuffer / VideoFileSource: no full_at
        pass

    class Empty(FullFrames):
        def full_at(self, t):
            return None

    for frames in (NoFull(), Empty()):
        mem = Memory(out=[room_event(0.1)])
        dets, evs = make_room(tmp_path, frames, mem).perceive(frame(1))
        assert mem.steps == [] and room_event(0.1) not in evs


def test_room_memory_waits_for_the_table_calibration(tmp_path):
    mem = Memory()
    room = make_room(tmp_path, FullFrames(), mem)
    room.table = types.SimpleNamespace(ok=False, calibrate=lambda img: False)
    assert room.perceive(frame(1)) is None and mem.steps == []


# ---------------------------------------------------------------- main.build helpers

def room_cfg(**rm):
    return dict(CFG, room_memory=dict(CFG.get("room_memory") or {}, **rm))


@pytest.fixture
def fake_capture(monkeypatch):
    """core.capture.FrameBuffer / open_camera recorders and a stand-in core.room_view (Task A's TableView)."""
    import core.capture
    made = {"fb": [], "opened": [], "views": []}

    class FB:
        def __init__(self, source, **kw):
            self.source, self.kw = source, kw
            made["fb"].append(self)

    class TV:
        def __init__(self, source, rect, out_size=(1280, 720)):
            self.source, self.rect, self.out_size = source, rect, out_size
            made["views"].append(self)

    def default_rect(size_px, zoom=100, ref_zoom=160, out_size=(1280, 720)):
        return (1, 2, 3, 4)

    mod = types.ModuleType("core.room_view")
    mod.TableView, mod.default_rect = TV, default_rect
    monkeypatch.setitem(sys.modules, "core.room_view", mod)
    monkeypatch.setattr(core.capture, "FrameBuffer", FB)
    monkeypatch.setattr(core.capture, "open_camera", lambda src, w, h: made["opened"].append((src, w, h)) or "cap")
    return made


def test_room_memory_off_opens_the_camera_as_always(fake_capture):
    frames, rect = main.open_frames(room_cfg(enabled=False), "/dev/video0")
    assert rect is None and frames is fake_capture["fb"][0]
    assert frames.source == "/dev/video0" and frames.kw == {}             # FrameBuffer(camera), nothing else
    assert fake_capture["views"] == []


def test_room_memory_on_opens_the_full_frame_behind_the_table_view(fake_capture):
    frames, rect = main.open_frames(room_cfg(enabled=True, table_view_rect=list(RECT), ring_s=0.5), 2)
    fb = fake_capture["fb"][0]
    assert rect == RECT and frames is fake_capture["views"][0]
    assert frames.source is fb and frames.rect == RECT and tuple(frames.out_size) == (1280, 720)
    assert fb.source == 2 and fb.kw["ring_s"] == 0.5
    assert fb.kw["opener"](2) == "cap" and fake_capture["opened"] == [(2, 1920, 1080)]     # 1080p, not 720p


def test_room_memory_without_a_measured_rect_uses_the_default_and_says_so(fake_capture, caplog):
    with caplog.at_level(logging.WARNING, logger="askroom.main"):
        frames, rect = main.open_frames(room_cfg(enabled=True, table_view_rect=None), 0)
    assert rect == (1, 2, 3, 4) and frames.rect == (1, 2, 3, 4)
    assert any("table_view_rect" in r.getMessage() for r in caplog.records)


def test_room_memory_is_built_from_config_with_the_detector_backend(monkeypatch):
    got = []
    mod = types.ModuleType("core.room")

    class RoomMemory:
        @staticmethod
        def from_config(cfg, world, backend, table_rect):
            got.append((cfg, world, backend, table_rect))
            return "memory"

    mod.RoomMemory = RoomMemory
    monkeypatch.setitem(sys.modules, "core.room", mod)
    cfg = room_cfg(enabled=True)
    assert main.make_room_memory(cfg, "world", Det(), RECT) == "memory"
    assert got == [(cfg, "world", "the prop model", RECT)]


def test_a_room_memory_that_fails_to_start_leaves_the_table_alone(monkeypatch, caplog):
    mod = types.ModuleType("core.room")

    class RoomMemory:
        @staticmethod
        def from_config(cfg, world, backend, table_rect):
            raise ValueError("bad zones file")

    mod.RoomMemory = RoomMemory
    monkeypatch.setitem(sys.modules, "core.room", mod)
    with caplog.at_level(logging.ERROR, logger="askroom.main"):
        assert main.make_room_memory(room_cfg(enabled=True), "world", Det(), RECT) is None
    assert any("room memory" in r.getMessage() for r in caplog.records)


def test_fake_runs_never_enable_room_memory(monkeypatch):
    import net
    import voice.understand
    monkeypatch.setattr(net.NetMonitor, "probe", lambda self: False)
    monkeypatch.setattr(voice.understand.Qwen, "health", lambda self, timeout=1.0: False)
    room, _ = main.build(room_cfg(enabled=True, table_view_rect=list(RECT)), fake=True, with_voice=False)
    try:
        assert room.room_memory is None and not hasattr(room.frames, "full_at")
    finally:
        room.shutdown()


# ---------------------------------------------------------------- demo_check

def zones_rig(tmp_path, zones=None, view=None, **rm):
    from core.room_zones import Zone, Zones, view_version
    path = tmp_path / "room_zones.json"
    rm = dict(dict(enabled=True, table_view_rect=list(RECT), zones_path=str(path)), **rm)
    cfg = room_cfg(**rm)
    if zones is not None:
        v = view or view_version((1920, 1080), 100, RECT)
        Zones(v, (1920, 1080), {n: Zone(n, say, [(0, 0), (100, 0), (100, 100)]) for n, say in zones.items()}
              ).save(path)
    return dc.Rig(cfg, fake=False, manual=False)


def test_room_memory_check_sits_before_room_pointing():
    names = [n for n, _ in dc.CHECKS]
    assert names.index("room memory") == names.index("room") - 1 and names[8] == "clock"


def test_room_memory_check_skips_when_off(tmp_path):
    rig = zones_rig(tmp_path, enabled=False)
    try:
        ok, msg = dc.check_room_memory(rig)
    finally:
        rig.close()
    assert ok is None and "room_memory.enabled" in msg


@pytest.mark.parametrize("case, want", [
    ("no file", "room_zones.json"),
    ("no zones", "no zones"),
    ("other view", "redraw"),
    ("no rect", "table_view_rect"),
])
def test_room_memory_check_fails_with_the_reason(tmp_path, case, want):
    kw = {"no file": {}, "no zones": {"zones": {}}, "other view": {"zones": {"shelf": "the bookshelf"},
          "view": "0123456789ab"}, "no rect": {"zones": {"shelf": "the bookshelf"}, "table_view_rect": None}}[case]
    rig = zones_rig(tmp_path, **kw)
    try:
        ok, msg = dc.check_room_memory(rig)
    finally:
        rig.close()
    assert ok is False and want in msg, msg


def test_room_memory_check_passes_and_names_the_zones(tmp_path):
    rig = zones_rig(tmp_path, zones={"shelf": "the bookshelf", "couch": "the couch"})
    try:
        ok, msg = dc.check_room_memory(rig)
        assert ok is True, msg
        assert "shelf (the bookshelf)" in msg and "couch (the couch)" in msg
        assert rig._parts == {}                                         # files only: no camera opened
    finally:
        rig.close()
