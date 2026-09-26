"""core/room_zones.py (spec 0009 M0): zone polygons, the zones file and the view version."""
import json

import pytest

from core.room_zones import Zone, Zones, view_version

SQUARE = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]


def test_contains_inside_edge_and_outside():
    z = Zone("shelf", "the bookshelf", SQUARE)
    assert z.contains((150, 150))
    assert z.contains((100, 150))          # on an edge
    assert z.contains((200, 200))          # on a vertex
    assert not z.contains((99.5, 150))
    assert not z.contains((250, 150))


def test_contains_concave_polygon():
    # an L: the notch at the top right is outside
    z = Zone("couch", "the couch", [(0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)])
    assert z.contains((25, 75))
    assert z.contains((75, 25))
    assert not z.contains((75, 75))


def test_degenerate_polygon_contains_nothing():
    assert not Zone("x", "x", [(0, 0), (10, 10)]).contains((5, 5))


def test_bbox_is_int_inclusive_exclusive_and_clipped_at_zero():
    assert Zone("shelf", "the bookshelf", SQUARE).bbox() == (100, 100, 201, 201)
    assert Zone("a", "a", [(10.4, 20.6), (30.2, 20.6), (30.2, 40.9)]).bbox() == (10, 20, 31, 41)
    b = Zone("edge", "the edge", [(-5, -10), (50, -10), (50, 40)]).bbox()
    assert b == (0, 0, 51, 41)
    assert all(isinstance(v, int) for v in b)


def make_zones():
    return Zones(view="abc123def456", size_px=(1920, 1080), zones={
        "shelf": Zone("shelf", "the bookshelf", SQUARE),
        "couch": Zone("couch", "the couch", [(300, 300), (500, 300), (500, 400), (300, 400)]),
    })


def test_at_finds_the_zone_or_none():
    zs = make_zones()
    assert zs.at((150, 150)).name == "shelf"
    assert zs.at((400, 350)).say == "the couch"
    assert zs.at((250, 250)) is None


def test_json_round_trip(tmp_path):
    zs = make_zones()
    d = zs.to_dict()
    json.dumps(d)                          # plain JSON
    assert Zones.from_dict(d) == zs
    p = tmp_path / "sub" / "room_zones.json"
    zs.save(p)
    back = Zones.load(p)
    assert back == zs
    assert back.size_px == (1920, 1080)
    assert back.zones["shelf"].poly == SQUARE
    assert isinstance(back.zones["shelf"].poly[0], tuple)
    assert list(back.zones) == ["shelf", "couch"]


def test_load_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Zones.load(tmp_path / "nope.json")


def test_view_version_is_stable_and_changes_with_each_input():
    v = view_version((1920, 1080), 100, (360, 202, 1560, 877))
    assert len(v) == 12 and int(v, 16) >= 0
    assert v == view_version([1920, 1080], 100, [360, 202, 1560, 877])     # list or tuple: same
    assert v != view_version((1280, 720), 100, (360, 202, 1560, 877))
    assert v != view_version((1920, 1080), 160, (360, 202, 1560, 877))
    assert v != view_version((1920, 1080), 100, (361, 202, 1560, 877))
    assert v != view_version((1920, 1080), 100, None)
