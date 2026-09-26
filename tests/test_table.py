"""core/table.py on synthetic images: markers drawn at known table positions under a perspective view."""
import json

import cv2
import numpy as np
import pytest

from core.config import load_config
from core.table import Table

CFG = load_config()
MARKERS_CM = {int(k): v for k, v in CFG["table"]["markers"].items()}      # 0..3 at the table corners
MARKER_CM = 6.0


def true_h() -> np.ndarray:
    """A plausible camera: table cm -> px with some tilt (not a pure scale)."""
    src = np.float32([[-5, -5], [95, -5], [95, 65], [-5, 65]])
    dst = np.float32([[120, 60], [1170, 90], [1210, 690], [80, 650]])
    return cv2.getPerspectiveTransform(src, dst)


def render(h: np.ndarray, ids=(0, 1, 2, 3), noise=0) -> np.ndarray:
    """Draw each marker as a flat square on the table plane, then photograph it through h."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ppc = 20                                                 # table texture resolution, px per cm
    plane = np.full((75 * ppc, 105 * ppc), 200, np.uint8)     # table cm -5..100 x -5..70
    for mid in ids:
        x, y = MARKERS_CM[mid]
        side = int(MARKER_CM * ppc)
        m = cv2.aruco.generateImageMarker(d, mid, side)
        x0, y0 = int((x + 5) * ppc - side / 2), int((y + 5) * ppc - side / 2)
        plane[y0 - 20:y0 + side + 20, x0 - 20:x0 + side + 20] = 255     # white quiet zone
        plane[y0:y0 + side, x0:x0 + side] = m
    to_cm = np.array([[1 / ppc, 0, -5], [0, 1 / ppc, -5], [0, 0, 1]])  # plane px -> table cm
    img = cv2.warpPerspective(plane, h @ to_cm, (1280, 720), flags=cv2.INTER_AREA, borderValue=90)
    if noise:
        img = cv2.add(img, np.random.default_rng(0).integers(0, noise, img.shape, dtype=np.uint8))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def project(h, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), h).reshape(-1, 2)


@pytest.fixture
def table(tmp_path):
    return Table(CFG, cal_path=str(tmp_path / "table_cal.json"))


def test_calibrate_finds_all_four_markers(table):
    assert table.calibrate(render(true_h()))
    assert table.ok and sorted(table.found) == [0, 1, 2, 3]


def test_marker_centres_map_to_their_table_positions(table):
    table.calibrate(render(true_h()))
    got = table.px_to_cm(project(true_h(), list(MARKERS_CM.values())))
    assert got == pytest.approx(np.float32(list(MARKERS_CM.values())), abs=0.3)


def test_points_across_the_table_are_within_1_cm(table):
    table.calibrate(render(true_h(), noise=12))
    grid = [[x, y] for x in range(0, 91, 15) for y in range(0, 61, 15)]
    err = np.linalg.norm(table.px_to_cm(project(true_h(), grid)) - np.float32(grid), axis=1)
    assert err.max() < 1.0


def test_px_to_cm_to_px_round_trips_within_1_px(table):
    table.calibrate(render(true_h()))
    px = np.float32([[200, 150], [640, 360], [1100, 600]])
    assert table.cm_to_px(table.px_to_cm(px)) == pytest.approx(px, abs=1.0)


def test_missing_a_marker_fails_and_keeps_the_previous_calibration(table):
    assert table.calibrate(render(true_h()))
    before = table.H.copy()
    assert not table.calibrate(render(true_h(), ids=(0, 1, 2)))
    assert table.ok and np.allclose(table.H, before)
    assert table.found == [0, 1, 2]


def test_calibration_is_saved_and_loaded(tmp_path):
    path = str(tmp_path / "table_cal.json")
    Table(CFG, cal_path=path).calibrate(render(true_h()))
    saved = json.loads(open(path).read())
    assert set(saved) >= {"H", "markers_px", "t"}
    t2 = Table(CFG, cal_path=path)
    assert t2.ok
    assert t2.px_to_cm([[640, 360]]) == pytest.approx(Table(CFG, cal_path=path).px_to_cm([[640, 360]]))


def test_uncalibrated_table_refuses_to_convert(tmp_path):
    t = Table(CFG, cal_path=str(tmp_path / "none.json"))
    assert not t.ok
    with pytest.raises(RuntimeError):
        t.px_to_cm([[0, 0]])


def test_in_bounds_uses_the_table_size(table):
    assert table.in_bounds((10, 10)) and table.in_bounds((90, 60))
    assert not table.in_bounds((-3, 10)) and not table.in_bounds((50, 70))
    assert table.size_cm == (90, 60)
