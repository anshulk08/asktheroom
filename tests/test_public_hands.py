"""scripts/finetune/public_hands.py: polygon/mask -> tight box -> YOLO line with the 'hand' class index,
and the MAT v5 reader on a hand-built polygons.mat."""
import struct
import zlib

import cv2
import numpy as np
import pytest

from core.config import load_config
from scripts.finetune.common import class_names, trial_of
from scripts.finetune.public_hands import (HAND_FIELDS, fit_long_side, frame_order, frame_polygons, hand_lines,
                                           mask_box, polygon_box, read_mat, spread, stem_for, worn_glasses)

W, H = 1280, 720


def hand_id():
    return class_names(load_config()).index("hand")


def test_hand_is_class_8():
    assert hand_id() == 8


def test_polygon_box_is_tight_and_clipped():
    poly = np.array([[100.2, 50.0], [180.0, 60.5], [150.0, 140.7], [110.0, 120.0]])
    assert polygon_box(poly, W, H) == (100, 50, 181, 141)
    assert polygon_box(np.array([[-5, -5], [40, 0], [30, 30]]), W, H) == (0, 0, 41, 31)
    assert polygon_box(np.array([[1270, 700], [1300, 710], [1290, 760]]), W, H) == (1270, 700, 1280, 720)


@pytest.mark.parametrize("poly", [np.zeros((0, 2)), np.array([[1, 1], [5, 5]]),
                                  np.array([[10, 10], [11, 10], [10, 11]])])
def test_degenerate_polygons_give_no_box(poly):
    assert polygon_box(poly, W, H) is None


def test_mask_box_matches_filled_polygon():
    poly = np.array([[300, 200], [420, 210], [400, 330], [310, 300]], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly.reshape(-1, 1, 2)], 1)
    assert mask_box(m.astype(bool)) == polygon_box(poly, W, H) == (300, 200, 421, 331)
    assert mask_box(np.zeros((H, W), bool)) is None


def test_hand_lines_use_hand_index_and_skip_empty_hands():
    polys = [np.zeros((0, 2)), np.array([[640, 360], [767, 360], [767, 503], [640, 503]]), None,
             np.array([[0, 0], [128, 0], [128, 72]])]
    lines = hand_lines(polys, W, H, hand_id())
    assert len(lines) == 2
    c, cx, cy, bw, bh = lines[0].split()
    assert int(c) == 8
    assert (float(cx), float(cy), float(bw), float(bh)) == pytest.approx((0.55, 0.6, 0.1, 0.2))
    assert all(ln.split()[0] == "8" for ln in lines)


def test_scaled_lines_are_resolution_independent():
    poly = [np.array([[640, 360], [767, 360], [767, 503], [640, 503]])]
    big = hand_lines(poly, W, H, 8)
    img, s = fit_long_side(np.zeros((H, W, 3), np.uint8), 640)
    assert img.shape[:2] == (360, 640) and s == 0.5
    assert hand_lines(poly, 640, 360, 8, s) == big
    assert fit_long_side(np.zeros((H, W, 3), np.uint8), 1280)[1] == 1.0


def test_stems_are_one_train_py_trial():
    stems = {stem_for(vi, f"frame_{fr:04d}.jpg") for vi in (0, 47) for fr in (11, 2699)}
    assert stem_for(47, "frame_2699.jpg") == "pubhand_472699"
    assert {trial_of(s) for s in stems} == {"pubhand"}


def test_spread_and_frame_order():
    assert spread(100, 4) == [12, 37, 62, 87]
    assert spread(3, 10) == [0, 1, 2]
    o = frame_order(100, seed=3)
    assert sorted(o) == list(range(100)) and o[:15] == spread(100, 15)
    assert frame_order(100, seed=3) == o            # reruns pick the same frames


def test_worn_glasses_only_near_the_top():
    assert worn_glasses("glasses", (600, 40, 700, 80), 720)
    assert not worn_glasses("glasses", (600, 400, 700, 440), 720)
    assert not worn_glasses("phone", (600, 40, 700, 80), 720)


# ---- a hand-built MAT v5 file shaped like EgoHands' polygons.mat (1 x 2 struct of Nx2 doubles)

def _el(t: int, payload: bytes) -> bytes:
    return struct.pack("<II", t, len(payload)) + payload + b"\0" * ((-len(payload)) % 8)


def _matrix(name: str, cls: int, dims, body: bytes) -> bytes:
    head = (_el(6, struct.pack("<II", cls, 0)) + _el(5, struct.pack(f"<{len(dims)}i", *dims))
            + _el(1, name.encode()))
    return _el(14, head + body)


def _double(a: np.ndarray) -> bytes:
    a = np.asarray(a, np.float64)
    return _matrix("", 6, a.shape, _el(9, a.tobytes(order="F")))


def test_read_mat_struct_of_polygons():
    frames = [{"myleft": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), "myright": np.zeros((0, 0)),
               "yourleft": np.zeros((0, 0)), "yourright": np.zeros((0, 0))},
              {"myleft": np.zeros((0, 0)), "myright": np.zeros((0, 0)), "yourleft": np.zeros((0, 0)),
               "yourright": np.array([[10.0, 20.0], [30.0, 20.0], [30.0, 50.0], [10.0, 45.0]])}]
    flen = 32
    names = b"".join(f.encode().ljust(flen, b"\0") for f in HAND_FIELDS)
    body = _el(5, struct.pack("<i", flen)) + _el(1, names)
    body += b"".join(_double(fr[f]) for fr in frames for f in HAND_FIELDS)
    var = _matrix("polygons", 2, (1, 2), body)
    header = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\0" * 8 + struct.pack("<H", 0x0100) + b"IM"
    z = zlib.compress(var)                              # a compressed element, like the real files
    data = header + struct.pack("<II", 15, len(z)) + z

    polys = frame_polygons(read_mat(data))
    assert len(polys) == 2
    np.testing.assert_array_equal(polys[0][0], frames[0]["myleft"])
    np.testing.assert_array_equal(polys[1][3], frames[1]["yourright"])
    lines = hand_lines(polys[1], W, H, 8)
    assert len(lines) == 1 and lines[0].startswith("8 ")
    assert polygon_box(polys[1][3], W, H) == (10, 20, 31, 51)
