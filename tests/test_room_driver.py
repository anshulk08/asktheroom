"""RoomMemory driver and zone CLI (core/room.py, spec 0009 M0). A fake backend returns fixed raw boxes in crop
coordinates; a stub world records room_update calls. No hardware."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from core import room
from core.room import RoomMemory
from core.room_types import RoomConfig
from core.room_zones import Zone, Zones, view_version
from core.types import Frame

W, H = 1920, 1080
TABLE_RECT = (360, 202, 1560, 877)
TO_OBJ = {"keys": "keys", "key ring": "keys", "wallet": "wallet", "hand": "hand"}


class FakeBackend:
    """infer(img) returns self.raw (crop px) and records the image shapes it saw."""

    def __init__(self, raw=()):
        self.raw = list(raw)
        self.shapes = []

    def infer(self, img):
        self.shapes.append(img.shape)
        return list(self.raw)


class StubWorld:
    def __init__(self):
        self.visits = []

    def room_update(self, visit):
        self.visits.append(visit)
        return [f"event{len(self.visits)}"]


def rect_zone(name, x1, y1, x2, y2, say=None):
    return Zone(name, say or f"the {name}", [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def make(zones, raw=(), **cfg):
    cfg.setdefault("room_every_n", 1)
    backend, world = FakeBackend(raw), StubWorld()
    zs = Zones("v", (W, H), {z.name: z for z in zones})
    rm = RoomMemory(RoomConfig(enabled=True, **cfg), zs, backend, TO_OBJ, world, TABLE_RECT)
    return rm, backend, world


def frame(i, val=128, img=True):
    return Frame(t=float(i), wall=1000.0 + i, img=np.full((H, W, 3), val, np.uint8) if img else None, idx=i)


SHELF = rect_zone("shelf", 1600, 100, 1900, 400, "the bookshelf")
COUCH = rect_zone("couch", 0, 900, 300, 1070)


def test_round_robin_every_n_steps():
    rm, backend, world = make([SHELF, COUCH], room_every_n=5)
    out = [rm.step(frame(i)) for i in range(1, 16)]
    assert [len(o) for o in out] == [0, 0, 0, 0, 1] * 3
    assert [v.zone for v in world.visits] == ["shelf", "couch", "shelf"]
    assert [v.say for v in world.visits] == ["the bookshelf", "the couch", "the bookshelf"]
    assert [v.frame_idx for v in world.visits] == [5, 10, 15]
    assert world.visits[1].t == 10.0 and world.visits[1].wall == 1010.0
    assert out[4] == ["event1"]
    assert backend.shapes == [(301, 301, 3), (171, 301, 3), (301, 301, 3)]    # bbox is end-exclusive


def test_crop_offset_maps_back_to_full_px():
    rm, _, world = make([SHELF], raw=[("key ring", 0.9, (10, 20, 50, 60))])
    rm.step(frame(1))
    rm.step(frame(2))
    [tr] = rm.tracker.tracks("shelf")
    assert tr.cls == "keys" and tr.box_px == (1610, 120, 1650, 160)
    assert [x.tid for x in world.visits[1].confirmed] == [tr.tid]
    assert world.visits[1].crop.shape == (301, 301, 3)


def test_big_zone_is_resized_and_mapped_back():
    big = rect_zone("wall", 0, 0, 1000, 500)
    rm, backend, _ = make([big], raw=[("wallet", 0.9, (100, 100, 150, 150))], max_crop_px=500)
    rm.step(frame(1))
    assert backend.shapes == [(250, 500, 3)]                 # 1001x501 bbox scaled by 500/1001
    [tr] = rm.tracker.tracks("wall")
    assert tr.box_px == (200, 200, 300, 300)


def test_hand_overlapping_prop_dropped_and_hand_is_a_blocker():
    rm, backend, world = make([SHELF], raw=[("keys", 0.9, (10, 20, 50, 60))])
    rm.step(frame(1))
    rm.step(frame(2))
    [tr] = rm.tracker.tracks()
    backend.raw = [("hand", 0.9, (8, 18, 52, 62)), ("keys", 0.8, (10, 20, 50, 60))]
    rm.step(frame(3))
    v = world.visits[-1]
    assert v.confirmed == [] and v.missed == [] and tr.misses == 0     # blocked: nothing changes
    assert rm.tracker.tracks() == [tr]


def test_hand_over_a_new_prop_makes_no_track():
    rm, _, _ = make([SHELF], raw=[("hand", 0.9, (8, 18, 52, 62)), ("keys", 0.8, (10, 20, 50, 60))])
    rm.step(frame(1))
    assert rm.tracker.tracks() == []


def test_low_conf_dropped():
    rm, _, _ = make([SHELF], raw=[("keys", 0.3, (10, 20, 50, 60)), ("wallet", 0.5, (100, 100, 150, 150))])
    rm.step(frame(1))
    assert [x.cls for x in rm.tracker.tracks()] == ["wallet"]


def test_unknown_label_ignored():
    rm, _, _ = make([SHELF], raw=[("banana", 0.9, (10, 20, 50, 60))])
    rm.step(frame(1))
    assert rm.tracker.tracks() == []


def test_outside_polygon_dropped():
    tri = Zone("shelf", "the shelf", [(1600, 100), (1900, 100), (1600, 400)])
    # centre (1860, 360) is in the bbox but outside the triangle; (1620, 120) inside
    rm, _, _ = make([tri], raw=[("keys", 0.9, (240, 240, 280, 280)), ("wallet", 0.9, (10, 10, 30, 30))])
    rm.step(frame(1))
    assert [x.cls for x in rm.tracker.tracks()] == ["wallet"]


def test_inside_table_rect_dropped():
    z = rect_zone("desk", 1400, 700, 1800, 1000)
    # keys centre (1420, 720) is inside the table view; wallet centre (1720, 920) is not
    rm, _, _ = make([z], raw=[("keys", 0.9, (10, 10, 30, 30)), ("wallet", 0.9, (310, 210, 330, 230))])
    rm.step(frame(1))
    [tr] = rm.tracker.tracks()
    assert tr.cls == "wallet" and tr.box_px == (1710, 910, 1730, 930)


def test_same_object_under_two_prompts_is_one_observation():
    rm, _, _ = make([SHELF], raw=[("keys", 0.9, (10, 20, 50, 60)), ("key ring", 0.7, (11, 21, 51, 61))])
    rm.step(frame(1))
    assert len(rm.tracker.tracks()) == 1


def test_none_image_and_empty_crop_return_nothing():
    rm, backend, world = make([SHELF])
    assert rm.step(None) == []
    assert rm.step(frame(1, img=False)) == []
    assert backend.shapes == [] and world.visits == []
    off = rect_zone("off", 2000, 1200, 2100, 1300)
    rm, backend, world = make([off])
    assert rm.step(frame(1)) == []
    assert backend.shapes == [] and world.visits == []
    rm, backend, world = make([SHELF])
    tiny = Frame(t=1.0, wall=1.0, img=np.zeros((50, 50, 3), np.uint8), idx=1)   # zone off this frame
    assert rm.step(tiny) == [] and world.visits == []


def test_change_blob_over_the_spot_makes_the_visit_invalid():
    rm, backend, world = make([SHELF], raw=[("keys", 0.9, (100, 100, 140, 130))])
    rm.step(frame(1))
    rm.step(frame(2))
    [tr] = rm.tracker.tracks()
    backend.raw = []
    person = frame(3)
    person.img[80:400, 1650:1850] = 200                   # something big walks in front of the keys
    rm.step(person)
    assert world.visits[-1].missed == [] and tr.misses == 0
    still = frame(4)
    still.img[80:400, 1650:1850] = 200                    # no change since, keys not seen: a valid miss
    rm.step(still)
    assert [x.tid for x in world.visits[-1].missed] == [tr.tid] and tr.misses == 1


def test_small_change_does_not_block():
    rm, backend, world = make([SHELF], raw=[("keys", 0.9, (100, 100, 140, 130))])
    rm.step(frame(1))
    rm.step(frame(2))
    backend.raw = []
    f = frame(3)
    f.img[210:220, 1710:1720] = 255                        # the keys were picked up: a small change
    rm.step(f)
    assert len(world.visits[-1].missed) == 1


def test_dark_box_makes_the_visit_invalid():
    rm, backend, world = make([SHELF], raw=[("keys", 0.9, (100, 100, 140, 130))])
    rm.step(frame(1, val=5))
    rm.step(frame(2, val=5))
    backend.raw = []
    rm.step(frame(3, val=5))
    assert world.visits[-1].missed == []


def test_three_valid_empty_visits_miss_then_drop():
    rm, backend, world = make([SHELF], raw=[("keys", 0.9, (100, 100, 140, 130))])
    rm.step(frame(1))
    rm.step(frame(2))
    backend.raw = []
    for i in (3, 4, 5):
        rm.step(frame(i))
    assert [len(v.missed) for v in world.visits[2:]] == [1, 1, 1]
    assert [len(v.dropped) for v in world.visits[2:]] == [0, 0, 1]


# ---------------------------------------------------------------------------------------------
# from_config

def write_zones(path, view, zones=(SHELF,)):
    Zones(view, (W, H), {z.name: z for z in zones}).save(path)


def cfg_for(path, **rm):
    d = {"enabled": True, "zones_path": str(path)}
    d.update(rm)
    return {"objects": {"keys": {}, "wallet": {}}, "prompts": {"keys": ["keys", "key ring"]},
            "conf_threshold": {"default": 0.35}, "room_memory": d}


def test_from_config_ok(tmp_path):
    p = tmp_path / "zones.json"
    write_zones(p, view_version((W, H), 100, TABLE_RECT))
    rm = RoomMemory.from_config(cfg_for(p), StubWorld(), FakeBackend(), TABLE_RECT)
    assert isinstance(rm, RoomMemory)
    assert list(rm.zones.zones) == ["shelf"] and rm.to_obj["key ring"] == "keys" and rm.cfg.enabled


def test_from_config_none_when_disabled(tmp_path, caplog):
    p = tmp_path / "zones.json"
    write_zones(p, view_version((W, H), 100, TABLE_RECT))
    with caplog.at_level("INFO", logger="core.room"):
        assert RoomMemory.from_config(cfg_for(p, enabled=False), StubWorld(), FakeBackend(), TABLE_RECT) is None
        assert RoomMemory.from_config({}, StubWorld(), FakeBackend(), TABLE_RECT) is None
    assert "enabled" in caplog.text


def test_from_config_none_when_file_missing(tmp_path, caplog):
    with caplog.at_level("INFO", logger="core.room"):
        rm = RoomMemory.from_config(cfg_for(tmp_path / "nope.json"), StubWorld(), FakeBackend(), TABLE_RECT)
    assert rm is None and "nope.json" in caplog.text


def test_from_config_none_when_no_zones(tmp_path, caplog):
    p = tmp_path / "zones.json"
    write_zones(p, view_version((W, H), 100, TABLE_RECT), zones=())
    with caplog.at_level("INFO", logger="core.room"):
        assert RoomMemory.from_config(cfg_for(p), StubWorld(), FakeBackend(), TABLE_RECT) is None
    assert "no zones" in caplog.text


def test_from_config_none_on_view_mismatch(tmp_path, caplog):
    p = tmp_path / "zones.json"
    write_zones(p, view_version((W, H), 100, TABLE_RECT))
    with caplog.at_level("INFO", logger="core.room"):
        other_rect = (350, 202, 1550, 877)
        assert RoomMemory.from_config(cfg_for(p), StubWorld(), FakeBackend(), other_rect) is None
        assert RoomMemory.from_config(cfg_for(p, zoom=120), StubWorld(), FakeBackend(), TABLE_RECT) is None
    assert "view" in caplog.text


def test_from_config_none_on_bad_file_or_no_rect(tmp_path):
    p = tmp_path / "zones.json"
    p.write_text("{not json")
    assert RoomMemory.from_config(cfg_for(p), StubWorld(), FakeBackend(), TABLE_RECT) is None
    write_zones(p, view_version((W, H), 100, TABLE_RECT))
    assert RoomMemory.from_config(cfg_for(p), StubWorld(), FakeBackend(), None) is None


# ---------------------------------------------------------------------------------------------
# CLI

def test_cli_zone_round_trip_and_list(tmp_path, capsys):
    from core.config import load_config
    from core.room_view import default_rect
    rc = RoomConfig.from_dict(load_config().get("room_memory"))
    p = tmp_path / "zones.json"
    assert room.main(["--zones", str(p), "--zone", "bookshelf", "--say", "the bookshelf",
                      "--poly", "1600,100", "1900,100", "1900,400", "1600,400"]) == 0
    assert room.main(["--zones", str(p), "--zone", "couch", "--poly", "0,900", "300,900", "300,1070"]) == 0
    d = json.loads(p.read_text())
    rect = rc.table_view_rect or default_rect(rc.capture_size, rc.zoom, rc.ref_zoom)
    assert d["view"] == view_version(rc.capture_size, rc.zoom, rect)
    zs = Zones.load(p)
    assert list(zs.zones) == ["bookshelf", "couch"]
    assert zs.zones["bookshelf"].say == "the bookshelf" and zs.zones["couch"].say == "the couch"
    assert zs.zones["bookshelf"].poly == [(1600, 100), (1900, 100), (1900, 400), (1600, 400)]
    # replace, then list
    assert room.main(["--zones", str(p), "--zone", "couch", "--say", "the sofa",
                      "--poly", "0,800", "300,800", "300,1070"]) == 0
    capsys.readouterr()
    assert room.main(["--zones", str(p), "--list"]) == 0
    out = capsys.readouterr().out
    assert "bookshelf" in out and "'the bookshelf'" in out and "'the sofa'" in out and "matches" in out
    # the file from_config loads with the same view
    cfg = cfg_for(p, capture_size=list(rc.capture_size), zoom=rc.zoom)
    assert RoomMemory.from_config(cfg, StubWorld(), FakeBackend(), rect) is not None
    assert room.main(["--zones", str(p), "--delete-zone", "couch"]) == 0
    assert list(Zones.load(p).zones) == ["bookshelf"]
    assert room.main(["--zones", str(p), "--delete-zone", "couch"]) == 1


def test_cli_list_missing_file_and_bad_poly(tmp_path):
    p = tmp_path / "zones.json"
    assert room.main(["--zones", str(p), "--list"]) == 1
    with pytest.raises(SystemExit):
        room.main(["--zones", str(p), "--zone", "x", "--poly", "1,1", "2,2"])
    assert not p.exists()


def test_cli_refuses_to_add_to_a_file_from_another_view(tmp_path):
    p = tmp_path / "zones.json"
    write_zones(p, "0123456789ab")
    assert room.main(["--zones", str(p), "--zone", "couch", "--poly", "0,900", "300,900", "300,1070"]) == 1
    assert list(Zones.load(p).zones) == ["shelf"]


def test_cli_show_draws_zones(tmp_path):
    p = tmp_path / "zones.json"
    room.main(["--zones", str(p), "--zone", "bookshelf", "--poly", "1600,100", "1900,100", "1900,400"])
    src, out = tmp_path / "full.jpg", tmp_path / "shown.jpg"
    cv2.imwrite(str(src), np.zeros((H, W, 3), np.uint8))
    assert room.main(["--zones", str(p), "--show", str(src), "--out", str(out)]) == 0
    img = cv2.imread(str(out))
    assert img.shape == (H, W, 3) and img.max() > 0
