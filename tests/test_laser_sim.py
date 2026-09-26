import math

import numpy as np
import pytest

from act.calibrate import aim_stats, calibrate, measure_latency
from act.laser import LaserFit, dot_px_diff
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
