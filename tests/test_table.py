"""core/table.py on synthetic images: markers drawn at known table positions under a perspective view."""
import json

import cv2
import numpy as np
import pytest

from core.config import load_config
from core.table import Table

CFG = load_config()
CFG = dict(CFG, table_tag=dict(CFG.get("table_tag") or {}, enabled=False))    # these tests: four-marker mode
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


def test_failed_calibration_warns_once_until_markers_change(table, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="core.table")
    for _ in range(30):
        table.calibrate(render(true_h(), ids=(0, 1)))
    assert len(caplog.records) == 1
    table.calibrate(render(true_h(), ids=(0, 1, 2)))
    assert len(caplog.records) == 2


# ---------------------------------------------------------------- one AprilTag, no measuring

TAG_CM = 16.0
# OpenCV before 4.10 decodes a turned tag in only about half of these scenes (measured: 4.8.0 on the Jetson
# host 8/16, 4.11.0 in the app's container 16/16, 5.0.0 on the laptop 16/16). The app runs in the container.
NEW_CV = tuple(int(v) for v in cv2.__version__.split(".")[:2]) >= (4, 10)
needs_new_cv = pytest.mark.skipif(not NEW_CV, reason=f"OpenCV {cv2.__version__}: tag detection unreliable before 4.10")


def tag_cfg(tmp_path, **kw):
    tt = {"enabled": True, "family": "apriltag_36h11", "id": 0, "size_cm": TAG_CM, "frames": 10, "max_cm": 250}
    tt.update(kw)
    return dict(CFG, table_tag=tt, paths=dict(CFG.get("paths") or {}, table_cal=str(tmp_path / "table_cal.json")))


def render_tag(h, at=(40.0, 30.0), deg=15.0, noise=0, seed=0, present=True):
    """One 36h11 tag (black square TAG_CM) whose top-left corner sits at `at` (true plane cm), rotated deg."""
    ppc = 20
    plane = np.full((75 * ppc, 105 * ppc), 200, np.uint8)
    if present:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        side = int(TAG_CM * ppc)
        m = cv2.aruco.generateImageMarker(d, 0, side)
        quiet = cv2.copyMakeBorder(m, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = np.array([[c, -s], [s, c]])
        src = np.float32([[-40, -40], [side + 40, -40], [side + 40, side + 40], [-40, side + 40]])
        dst = (src / ppc) @ R.T + np.array(at)                                     # true cm
        dst_px = (dst + 5) * ppc
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [side + 80, 0], [side + 80, side + 80], [0, side + 80]]),
                                        np.float32(dst_px))
        warped = cv2.warpPerspective(quiet, M, (plane.shape[1], plane.shape[0]), borderValue=0)
        mask = cv2.warpPerspective(np.full_like(quiet, 255), M, (plane.shape[1], plane.shape[0]))
        plane[mask > 0] = warped[mask > 0]
    to_cm = np.array([[1 / ppc, 0, -5], [0, 1 / ppc, -5], [0, 0, 1]])
    img = cv2.warpPerspective(plane, h @ to_cm, (1280, 720), flags=cv2.INTER_AREA, borderValue=90)
    if noise:
        img = cv2.add(img, np.random.default_rng(seed).integers(0, noise, img.shape, dtype=np.uint8))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.ndim == 3 else cv2.COLOR_GRAY2BGR)


def calibrate_tag(t, h, n=10, **kw):
    ok = False
    for i in range(n):
        ok = t.calibrate(render_tag(h, noise=10, seed=i, **kw))
    return ok


@needs_new_cv
def test_one_tag_calibrates_the_table_without_measuring(tmp_path):
    t = Table(tag_cfg(tmp_path))
    assert not t.ok
    assert calibrate_tag(t, true_h()) and t.ok
    # distances between points across the view come out right in cm (the tag's printed size is the scale)
    grid = np.float32([[x, y] for x in range(5, 86, 20) for y in range(5, 56, 15)])      # true plane cm
    got = t.px_to_cm(project(true_h(), grid))
    i, j = np.triu_indices(len(grid), 1)
    err = np.abs(np.linalg.norm(got[i] - got[j], axis=1) - np.linalg.norm(grid[i] - grid[j], axis=1))
    assert err.max() < 1.5, err.max()


@needs_new_cv
def test_the_tracked_area_is_what_the_camera_sees(tmp_path):
    t = Table(tag_cfg(tmp_path))
    calibrate_tag(t, true_h())
    w, h = t.size_cm
    frame = t.px_to_cm([[0, 0], [1279, 0], [1279, 719], [0, 719]])
    assert frame.min() > -0.5 and frame[:, 0].max() < w + 0.5 and frame[:, 1].max() < h + 0.5
    # axes follow the camera image however the tag was dropped: the area is tight around the view
    # (true visible area from the synthetic camera, in cm^2) and x runs left -> right across the image
    true_foot = project(np.linalg.inv(true_h()), [[0, 0], [1279, 0], [1279, 719], [0, 719]])
    assert cv2.contourArea(np.float32(true_foot)) / (w * h) > 0.85
    left, right = t.px_to_cm([[100, 360], [1180, 360]])
    assert right[0] - left[0] > 50 and abs(right[1] - left[1]) < 10


@needs_new_cv
def test_axes_ignore_how_the_tag_was_turned(tmp_path):
    a, b = Table(tag_cfg(tmp_path / "a")), Table(tag_cfg(tmp_path / "b"))
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    calibrate_tag(a, true_h(), deg=0.0)
    calibrate_tag(b, true_h(), deg=40.0, at=(30.0, 20.0))
    assert abs(a.size_cm[0] - b.size_cm[0]) < 3 and abs(a.size_cm[1] - b.size_cm[1]) < 3


@needs_new_cv
def test_one_tag_needs_several_frames_and_keeps_the_old_calibration_without_it(tmp_path):
    t = Table(tag_cfg(tmp_path, frames=5))
    assert not any(t.calibrate(render_tag(true_h(), noise=10, seed=i)) for i in range(4))   # still averaging
    assert t.calibrate(render_tag(true_h(), noise=10, seed=4)) and t.ok
    before = t.H.copy()
    assert not t.calibrate(render_tag(true_h(), present=False))
    assert t.ok and np.allclose(t.H, before)


@needs_new_cv
def test_tag_calibration_and_the_area_size_are_saved_and_applied_at_startup(tmp_path):
    from core.table import apply_saved_size
    cfg = tag_cfg(tmp_path)
    t = Table(cfg)
    calibrate_tag(t, true_h())
    t2 = Table(cfg)
    assert t2.ok and t2.size_cm == t.size_cm
    fresh = tag_cfg(tmp_path)
    fresh["table"] = dict(fresh["table"], size_cm=[90, 60])
    apply_saved_size(fresh)
    assert tuple(fresh["table"]["size_cm"]) == t.size_cm
    four = dict(fresh, table_tag=dict(fresh["table_tag"], enabled=False), table=dict(fresh["table"], size_cm=[90, 60]))
    apply_saved_size(four)
    assert four["table"]["size_cm"] == [90, 60]              # four-marker mode keeps the configured size
