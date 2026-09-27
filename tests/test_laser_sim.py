import math

import numpy as np
import pytest

from act.calibrate import aim_stats, calibrate, measure_latency
from act.laser import MAX_MISSES, LaserFit, dot_px_diff
from act.sim import SimRig
from core.config import load_config


def rand_targets(n: int, seed: int, margin: float = 5.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(margin, 90 - margin, n), rng.uniform(margin, 60 - margin, n)])


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    """One calibrated sim rig shared by the pointing tests."""
    rig = SimRig(load_config(), seed=3)
    path = str(tmp_path_factory.mktemp("cal") / "laser_cal.json")
    laser = rig.make_laser(path)
    report = calibrate(laser, grid=4)
    return rig, laser, report, path


def test_find_dot_20_random_positions():
    rig = SimRig(seed=1)
    laser = rig.make_laser("")
    for x, y in rand_targets(20, seed=2, margin=3):
        laser.move_to(*rig.geom.pulses_for((x, y)))
        dot = laser.find_dot()
        assert dot is not None
        err = math.dist(dot, rig.true_dot_cm())
        assert err < 1.0, (x, y, dot, err)


def test_find_dot_none_when_blocked():
    rig = SimRig(seed=1)
    laser = rig.make_laser("")
    laser.move_to(*rig.geom.pulses_for((45, 30)))
    rig.blocked = True
    assert laser.find_dot() is None


def test_find_dot_none_when_dot_falls_off_table():
    """Warm wood grain at a bright moment passes a loose red HSV mask; that must not become a dot."""
    rig = SimRig(seed=1)
    laser = rig.make_laser("")
    for i in range(12):
        laser.move_to(*rig.geom.pulses_for([(94, 28), (69, 64), (24, -6)][i % 3]))
        rig.clock.sleep(0.37)
        assert laser.find_dot() is None, i


def test_hsv_fallback_when_off_frame_missing(monkeypatch):
    rig = SimRig(seed=2)
    laser = rig.make_laser("")
    real_grab = laser._grab
    monkeypatch.setattr(laser, "_grab", lambda on: real_grab(on) if on else None)
    hits = 0
    for x, y in rand_targets(20, seed=0, margin=3):
        laser.move_to(*rig.geom.pulses_for((x, y)))
        dot = laser.find_dot()
        hits += dot is not None and math.dist(dot, rig.true_dot_cm()) < 1.0
    assert hits >= 17


def test_latency_guard_beats_naive_fixed_sleep():
    """A fixed 60 ms sleep after switching the laser returns frames that still show the old state
    (sim pipeline lag 80 ms), so the off/on difference is empty; the Frame.t guard works."""
    rig = SimRig(seed=4, latency_s=0.08)
    laser = rig.make_laser("")
    laser.move_to(*rig.geom.pulses_for((50, 25)))
    rig.act.laser(True)
    rig.clock.sleep(0.5)                       # dot has been visible for a while

    rig.act.laser(False)
    rig.clock.sleep(0.06)
    naive_off = rig.frames.latest()
    rig.act.laser(True)
    rig.clock.sleep(0.06)
    naive_on = rig.frames.latest()
    assert dot_px_diff(naive_off.img, naive_on.img) is None          # stale: dot in both frames
    assert naive_off.img[..., 2].max() == 255                          # the "off" frame shows the dot

    dot = laser.find_dot()
    assert dot is not None and math.dist(dot, rig.true_dot_cm()) < 1.0
    off, on = laser.last_frames
    assert off.t > on.t - 1 and dot_px_diff(off.img, on.img) is not None


def test_measure_latency_recovers_sim_latency():
    rig = SimRig(seed=5, latency_s=0.08)
    laser = rig.make_laser("")
    laser.move_to(*rig.geom.pulses_for((45, 30)))
    m = measure_latency(laser, trials=3)
    assert m["seen"] == 3
    assert 0.08 <= m["recommended_s"] <= 0.08 + 1 / 30 + 0.01


def test_calibrate_fit_error(cal):
    rig, laser, report, path = cal
    assert report["fit_error_cm"]["max"] < 1.5
    assert report["n_points"] >= 9
    assert not rig.act.laser_on                            # calibration leaves the laser off
    back = LaserFit.load(path)
    assert np.allclose(back.predict((45, 30)), laser.fit.predict((45, 30)))
    assert back.table_px_to_cm is not None and np.allclose(back.table_px_to_cm, laser.fit.table_px_to_cm)
    assert back.servo_limits == laser.fit.servo_limits


def test_aim_20_targets_median_under_3cm(cal):
    rig, laser, _, _ = cal
    st = aim_stats(laser, rand_targets(20, seed=11), truth=rig.true_dot_cm)
    assert st["true_cm"]["median"] <= 3.0                  # spec H5 pass criterion
    assert st["measured_cm"]["median"] <= 3.0
    assert st["true_cm"]["n_under_1cm"] >= 15


def test_aim_leaves_laser_on_then_auto_off(cal):
    rig, laser, _, _ = cal
    err = laser.aim((30, 20))
    assert err < 1.5 and rig.act.laser_on and laser.state["on"]
    rig.clock.sleep(10.5)
    assert rig.act.state_at(rig.clock.now())[2] is False
    assert rig.frames.latest().img[..., 2].max() < 250     # no dot in the picture any more


def test_aim_object_shiny_offset(cal):
    rig, laser, _, _ = cal
    obj = (20.0, 45.0)
    laser.aim_object("phone", obj)
    dot = np.asarray(rig.true_dot_cm())
    toward_centre = (np.array([45.0, 30.0]) - obj) / np.linalg.norm(np.array([45.0, 30.0]) - obj)
    assert np.linalg.norm(dot - (obj + 3 * toward_centre)) < 1.5
    assert laser.state["target"] == "phone"
    laser.aim_object("wallet", obj)                        # not shiny: straight at it
    assert math.dist(rig.true_dot_cm(), obj) < 1.5


def test_sweep_edge_and_circle_stay_within_limits(cal):
    rig, laser, _, _ = cal
    (plo, phi), (tlo, thi) = rig.act.limits()
    for edge, axis, want in [("left", 0, 4.0), ("right", 0, 86.0), ("top", 1, 4.0), ("bottom", 1, 56.0)]:
        n0 = len(rig.act.writes)
        laser.sweep_edge(edge)
        w = rig.act.writes[n0:]
        assert w and all(plo <= p <= phi and tlo <= t <= thi for _, p, t in w)
        hits = np.array([rig.geom.hit(p, t) for p, t in rig.act.pose[n0 + 15:]])   # after the transit
        assert abs(np.median(hits[:, axis]) - want) < 2.0, edge
        assert hits[:, axis].min() > -0.5 and hits[:, axis].max() < (90.5 if axis == 0 else 60.5)
        assert np.ptp(hits[:, 1 - axis]) > 40 and rig.act.laser_on       # runs along the edge
    n0 = len(rig.act.writes)
    laser.circle((45, 30), r_cm=5)
    w = rig.act.writes[n0:]
    assert all(plo <= p <= phi and tlo <= t <= thi for _, p, t in w)
    r = [math.dist(rig.geom.hit(p, t), (45, 30)) for _, p, t in w[len(w) // 4:]]
    assert 3.0 < np.median(r) < 7.0
    laser.circle((1, 1), r_cm=20)                          # partly off the table: still clamped
    assert all(plo <= p <= phi and tlo <= t <= thi for _, p, t in rig.act.writes)
    with pytest.raises(ValueError):
        laser.sweep_edge("up")
    laser.off()
    assert not rig.act.laser_on


def test_aim_requires_calibration():
    rig = SimRig(seed=0)
    with pytest.raises(RuntimeError):
        rig.make_laser("").aim((10, 10))


def test_red_sleeve_does_not_hide_the_dot():
    """A blob bigger than max_area (a red sleeve moving between the off and on frames) is skipped, not
    taken as 'no dot'."""
    off = np.full((200, 300, 3), 90, np.uint8)
    on = off.copy()
    on[20:120, 20:120, 2] = 250                          # sleeve: 10000 px of red rise
    on[150:156, 200:206, 2] = 255                        # the dot
    x, y = dot_px_diff(off, on, max_area=3000)
    assert abs(x - 202.5) < 1 and abs(y - 152.5) < 1


def test_glint_elsewhere_is_ignored(cal):
    """A brighter second dot 40 cm away (a reflection) wins an ungated search, but not the aim's."""
    rig, laser, _, _ = cal
    target = (30.0, 20.0)
    rig.ghost_cm = (70.0, 45.0)
    try:
        laser.move_to(*laser.fit.predict(target))
        assert math.dist(laser.find_dot(), rig.ghost_cm) < 2        # the distractor works
        err = laser.aim(target)
        assert err < 2.0 and math.dist(rig.true_dot_cm(), target) < 2.0
    finally:
        rig.ghost_cm = None
        laser.off()


def test_dot_further_than_first_dot_cm_is_rejected(cal):
    """Inside the search box but 17 cm from the target: not the dot."""
    rig, laser, _, _ = cal
    target = (40.0, 30.0)
    laser.move_to(*laser.fit.predict(target))
    rig.blocked, rig.ghost_cm = True, (52.0, 42.0)
    try:
        assert laser.find_dot() is not None
        assert laser.find_dot(near_cm=target) is None
        assert laser.find_dot(near_cm=target, radius_cm=20) is not None
    finally:
        rig.blocked, rig.ghost_cm = False, None
        laser.off()


def test_aim_stops_after_three_misses_and_holds_open_loop_pose(cal):
    rig, laser, _, _ = cal
    target = (60.0, 35.0)
    rig.blocked = True
    try:
        t0 = rig.clock.now()
        assert math.isinf(laser.aim(target))
        assert laser.last_aim["tries"] == MAX_MISSES and laser.last_aim["reason"] == "not_seen"
        assert rig.clock.now() - t0 < 4.0                            # was 8 blinks (~5 s)
        assert np.allclose((rig.act.pan, rig.act.tilt), laser._clamp(*laser.fit.predict(target)))
        assert not rig.act.laser_on and laser.state["on"] is False   # no unconfirmed dot left on
        assert laser.last_aim["wide"]                                # the whole picture was searched once
    finally:
        rig.blocked = False
        laser.off()


class RefitTable:
    """The same camera and table after a refit that turned the table frame 3 deg and moved it (4, -2.5) cm."""

    def __init__(self, old, deg=3.0, t=(4.0, -2.5)):
        a = math.radians(deg)
        self.old, self.R, self.t = old, np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]), np.array(t)

    def to_new(self, cm):
        return np.asarray(cm, dtype=np.float64) @ self.R.T + self.t

    def px_to_cm(self, pts):
        return self.to_new(self.old.px_to_cm(pts))

    def cm_to_px(self, pts):
        return self.old.cm_to_px((np.asarray(pts, dtype=np.float64).reshape(-1, 2) - self.t) @ self.R)


def test_table_refit_is_remapped_through_the_camera(cal):
    rig, laser, _, _ = cal
    old_table, old_fit = laser.table, laser.fit
    laser.table = RefitTable(old_table)
    try:
        assert laser.refit_moved_cm() > 4.0
        for p in [(40.0, 25.0), (15.0, 50.0), (75.0, 10.0)]:
            q = tuple(laser.table.to_new([p])[0])                 # the same spot in the new table's cm
            laser.aim(q, mode="open")
            assert math.dist(rig.true_dot_cm(), p) < 1.5, p       # open loop already lands
            laser.aim(q)
            assert math.dist(rig.true_dot_cm(), p) < 2.0, p
        laser.fit = LaserFit.from_dict(dict(old_fit.to_dict(), table_px_to_cm=None))   # an old laser_cal.json
        laser.aim(tuple(laser.table.to_new([(40.0, 25.0)])[0]), mode="open")
        assert math.dist(rig.true_dot_cm(), (40.0, 25.0)) > 2.0    # without the stored frame it points off (< 0.8 with)
        assert laser.refit_moved_cm() is None
    finally:
        laser.table, laser.fit = old_table, old_fit
        laser.off()
    assert laser.refit_moved_cm() < 1e-6


@pytest.mark.parametrize("deg", [7.0, 9.0])
def test_a_knocked_head_is_found_by_the_whole_picture_search(deg):
    """The head turned after calibration: the dot lands ~15-20 cm off, outside the 15 cm search box.
    One whole-picture search finds it and the loop corrects onto the target."""
    rig = SimRig(load_config(), seed=3)
    laser = rig.make_laser("")
    calibrate(laser)
    rig.geom.off = (rig.geom.off[0] + deg, rig.geom.off[1])
    target = (45.0, 30.0)
    laser.aim(target, mode="open")
    assert math.dist(rig.true_dot_cm(), target) > 12.0                 # open loop is far off
    err = laser.aim(target)
    assert laser.last_aim["wide"] and laser.last_aim["reason"] == "within_tol", laser.last_aim
    assert err < 2.0 and math.dist(rig.true_dot_cm(), target) < 2.5 and rig.act.laser_on


def test_whole_picture_search_needs_exactly_one_dot(cal):
    rig, laser, _, _ = cal
    laser.move_to(*laser.fit.predict((30.0, 20.0)))
    try:
        assert laser.find_dot_wide() is not None
        rig.ghost_cm = (70.0, 45.0)                                    # two dots: ambiguous
        assert laser.find_dot_wide() is None
    finally:
        rig.ghost_cm = None
        laser.off()


# ----- eye safety in the table path (review of ws/laser)

def test_table_aims_move_dark_and_stay_lit_only_within_tol(cal, monkeypatch):
    from tests.test_room_map import lit_moves
    rig, laser, _, _ = cal
    t0 = rig.clock.now()
    laser.aim((20.0, 20.0))
    laser.aim((70.0, 45.0))                                   # a slew across the table
    assert lit_moves(rig.act, t0) == [] and laser.last_aim["reason"] == "within_tol" and rig.act.laser_on
    looks = iter([(30.0, 30.0)] + [None] * 10)               # seen once, then lost (a hand in the beam)
    monkeypatch.setattr(laser, "find_dot", lambda *a, **k: next(looks))
    laser.aim((36.0, 30.0))                                   # 6 cm off the seen dot: keeps looking
    assert laser.last_aim["reason"] == "lost" and rig.act.laser_on is False and laser.state["on"] is False
    laser.off()


def test_table_aims_obey_the_check_and_the_budget(cal, monkeypatch):
    rig, laser, _, _ = cal
    laser.aim((40.0, 30.0), check=lambda: "a hand near the target")
    assert laser.last_aim["reason"] == "unsafe" and rig.act.laser_on is False
    monkeypatch.setattr(laser, "max_on_s", 0.2)
    monkeypatch.setattr(laser, "tol_cm", 0.001)
    laser.aim((40.0, 30.0))
    assert laser.last_aim["reason"] == "budget" and rig.act.laser_on is False


def test_a_trace_reaches_its_start_dark_and_stops_at_the_budget(cal, monkeypatch):
    rig, laser, _, _ = cal
    laser.off()
    rig.act.move(1200, 1200)
    t0 = rig.clock.now()
    monkeypatch.setattr(laser, "max_on_s", 0.3)
    laser.circle((45.0, 30.0))
    first_lit = next(t for t, on in rig.act.laser_log if t >= t0 and on)
    reach = [w for w in rig.act.writes if t0 <= w[0] < first_lit]
    assert reach                                              # the head reached the start before lighting
    assert rig.act.laser_on is False and laser.state["on"] is False and laser.last_aim["lit_s"] < 0.5


@pytest.mark.parametrize("color", ["red", "green"])
def test_the_dot_detector_finds_the_configured_laser_colour_only(color):
    """The turret's module may be green: a red-only detector scored a green dot negative and never saw it."""
    import numpy as np
    from act.laser import dot_px_diff, dot_px_hsv
    off = np.full((120, 160, 3), 90, np.uint8)
    on = off.copy()
    bgr = {"red": (40, 60, 255), "green": (60, 255, 40)}
    on[50:54, 70:74] = bgr[color]
    other = "green" if color == "red" else "red"
    assert dot_px_diff(off, on, color=color) == pytest.approx((71.5, 51.5), abs=0.6)
    assert dot_px_diff(off, on, color=other) is None
    assert dot_px_hsv(on, color=color) is not None


def test_an_unknown_laser_colour_is_refused():
    from act.laser import Laser
    with pytest.raises(ValueError):
        Laser(object(), None, None, "", cfg={"laser_room": {"color": "blue"}})


def test_the_dot_finder_never_looks_in_the_ignored_boxes(cal):
    """The turret's laser module glows at the frame's bottom edge: that is never the dot."""
    rig, laser, _, _ = cal
    laser.aim((45.0, 30.0))
    d = laser.find_dot_px(1)
    assert d is not None
    laser.ignore_px = [(d[0] - 40, d[1] - 40, d[0] + 40, d[1] + 40)]
    try:
        assert laser.find_dot_px(1) is None
    finally:
        laser.ignore_px = []
        laser.off()
