"""core/viewframe.py: camera-frame table cm -> the user's frame for each seat, the tabletop crop, spoken
sides and areas, and the saved seat."""
import json
import math

import pytest

from core.viewframe import FRONTS, View, apply_saved, front_of, set_front

W, H = 100.0, 60.0
# Camera-frame corners (x left -> right across the table view, y far -> near the camera).
TL, TR, BR, BL = (5, 5), (95, 5), (95, 55), (5, 55)

# Worked by hand: where each camera corner lands for a user at that side, as (x = their left -> right,
# y = far -> near). Sitting at the camera's right end, the image's top is on their right.
EXPECT = {
    "bottom": ((100, 60), {TL: (5, 5), TR: (95, 5), BR: (95, 55), BL: (5, 55)}),
    "right": ((60, 100), {TL: (55, 5), TR: (55, 95), BR: (5, 95), BL: (5, 5)}),
    "top": ((100, 60), {TL: (95, 55), TR: (5, 55), BR: (5, 5), BL: (95, 5)}),
    "left": ((60, 100), {TL: (5, 95), TR: (5, 5), BR: (55, 5), BL: (55, 95)}),
}


def close(a, b, tol=1e-6):
    return all(abs(u - v) <= tol for u, v in zip(a, b))


@pytest.mark.parametrize("front", FRONTS)
def test_corners_land_where_worked_out(front):
    size, pts = EXPECT[front]
    v = View.make(front, (W, H))
    assert v.size == size
    for cam, want in pts.items():
        assert close(v.to_view(cam), want), (front, cam, v.to_view(cam))


@pytest.mark.parametrize("front", FRONTS)
def test_the_seat_side_is_nearest(front):
    """The camera edge the user sits at is 'bottom' (nearest them) and the opposite one is 'top'."""
    v = View.make(front, (W, H))
    opposite = {"bottom": "top", "top": "bottom", "left": "right", "right": "left"}
    assert v.edge(front) == "bottom"
    assert v.edge(opposite[front]) == "top"
    assert v.off_table(front) == "the side of the table nearest you"
    assert v.off_table(opposite[front]) == "the far side of the table"
    assert v.edge(None) is None and v.off_table(None) == "the table"


@pytest.mark.parametrize("front", FRONTS)
def test_edges_agree_with_positions(front):
    """A point just inside a camera edge lands just inside the viewer edge it maps to."""
    v = View.make(front, (W, H))
    w, h = v.size
    mids = {"left": (1, 30), "right": (99, 30), "top": (50, 1), "bottom": (50, 59)}
    for cam, p in mids.items():
        x, y = v.to_view(p)
        near = {"left": x, "right": w - x, "top": y, "bottom": h - y}
        assert min(near, key=near.get) == v.edge(cam), (front, cam)


@pytest.mark.parametrize("front,left_right", [("bottom", ("left", "right")), ("right", ("bottom", "top")),
                                              ("top", ("right", "left")), ("left", ("top", "bottom"))])
def test_your_left_and_right(front, left_right):
    v = View.make(front, (W, H))
    assert v.off_table(left_right[0]) == "the table on your left"
    assert v.off_table(left_right[1]) == "the table on your right"


def test_area_words_from_the_couch():
    """Front right (the rig's couch): the image's top-right corner is on the user's right, near them."""
    v = View.make("right", (W, H))
    assert v.area((90, 5)) == "on your right, near you"
    assert v.area((10, 5)) == "at the far right"
    assert v.area((10, 55)) == "at the far left"
    assert v.area((90, 55)) == "on your left, near you"
    assert v.area((50, 30)) == "in the middle"
    assert v.area((90, 30)) == "on the side nearest you"
    assert v.area((10, 30)) == "on the far side"
    assert v.area((50, 5)) == "on your right"
    assert v.area(None) == "somewhere on the table"
    assert v.area_word((90, 5)) == "near right"
    assert v.area_word((50, 5)) == "your right"
    assert v.area_word((10, 30)) == "far side"
    assert v.area_word((50, 30)) == "middle"
    assert v.area_word(None) is None


@pytest.mark.parametrize("front,want", [("bottom", "at the far right"), ("right", "on your right, near you"),
                                        ("top", "on your left, near you"), ("left", "at the far left")])
def test_area_of_the_image_top_right_for_each_seat(front, want):
    assert View.make(front, (W, H)).area((90, 5)) == want


def rect(cx, cy, w, h, deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in
            [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]]


@pytest.mark.parametrize("deg", [0, 8, -12, 30, 80, -75])
def test_outline_is_squared_and_cropped(deg):
    """A tabletop drawn at an angle becomes an upright rectangle filling the viewer frame; a steep one
    (turned more than 45 deg) keeps its long side across the camera's y axis."""
    poly = rect(50, 30, 80, 40, deg)
    v = View.make("bottom", (W, H), poly)
    long_across = abs(((deg + 45) % 180) - 45) > 45
    assert close(v.size, (40, 80) if long_across else (80, 40), 0.11)
    got = sorted((round(x, 1) + 0.0, round(y, 1) + 0.0) for x, y in map(v.to_view, poly))
    w, h = v.size
    assert close([c for p in got for c in p], [c for p in sorted([(0, 0), (0, h), (w, 0), (w, h)]) for c in p], 0.11)
    assert v.outline and v.to_json()["outline"] is True


def test_outline_with_the_seat():
    """Cropped and seated: the outline's camera-right edge is nearest a user at the right."""
    poly = [(20, 10), (90, 10), (90, 50), (20, 50)]
    v = View.make("right", (W, H), poly)
    assert v.size == (40, 70)
    assert close(v.to_view((90, 30)), (20, 70))      # middle of the near edge
    assert close(v.to_view((20, 30)), (20, 0))       # middle of the far edge
    assert close(v.to_view((55, 50)), (0, 35))       # the camera-bottom edge is on their left


def test_from_cfg_and_json():
    cfg = {"table": {"size_cm": [W, H]}, "viewer": {"front": "Right"},
           "table_area": {"polygon_cm": []}}
    v = View.from_cfg(cfg)
    assert v.front == "right" and v.size == (60, 100) and not v.outline
    j = v.to_json()
    assert j["front"] == "right" and j["table"] == [60, 100]
    (a, b, c), (d, e, f) = j["m"]
    assert close((a * 90 + b * 5 + c, d * 90 + e * 5 + f), (55, 90))
    assert front_of({}) == "bottom" and front_of({"viewer": {"front": "sideways"}}) == "bottom"


def test_set_front_saves_and_apply_saved_loads(tmp_path):
    path = tmp_path / "data" / "viewer.json"
    cfg = {"paths": {"viewer": str(path)}, "viewer": {"front": "bottom", "sides": {"right": "couch"}}}
    assert set_front(cfg, "RIGHT") == "right"
    assert cfg["viewer"] == {"front": "right", "sides": {"right": "couch"}}
    assert json.loads(path.read_text())["front"] == "right"
    assert not list(path.parent.glob("*.tmp"))
    fresh = {"paths": {"viewer": str(path)}, "viewer": {"front": "bottom"}}
    assert apply_saved(fresh)["viewer"]["front"] == "right"
    with pytest.raises(ValueError):
        set_front(cfg, "under")
    assert cfg["viewer"]["front"] == "right"


def test_apply_saved_ignores_a_bad_file(tmp_path):
    path = tmp_path / "viewer.json"
    cfg = {"paths": {"viewer": str(path)}, "viewer": {"front": "left"}}
    assert apply_saved(cfg)["viewer"]["front"] == "left"            # no file
    path.write_text("not json")
    assert apply_saved(cfg)["viewer"]["front"] == "left"
    path.write_text(json.dumps({"front": "diagonal"}))
    assert apply_saved(cfg)["viewer"]["front"] == "left"


def test_set_front_none_restores_the_configured_seat(tmp_path):
    path = tmp_path / "viewer.json"
    path.write_text(json.dumps({"front": "left"}))
    cfg = apply_saved({"paths": {"viewer": str(path)}, "viewer": {"front": "right"}})
    assert cfg["viewer"]["front"] == "left" and cfg["viewer"]["default_front"] == "right"
    assert set_front(cfg, None) == "right"
    assert cfg["viewer"]["front"] == "right" and not path.exists()
    assert set_front(cfg, None) == "right"                   # nothing saved: still fine
