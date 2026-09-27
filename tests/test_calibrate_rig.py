"""act.calibrate --rig on the simulated rig: the jog, the whole session, the sticky-note ruler check."""
import math

import numpy as np
import pytest

from act.calibrate import Region, calibrate, jog_limits, main, rig_session, spot_check
from act.sim import HeadGeometry, SimRig
from core.config import load_config


def keys_to(start, corners, step=10):
    """Key presses that jog from start to each corner (step us per press) and record it."""
    keys, (p, t) = [], start
    for cp, ct in corners:
        n = int(round((cp - p) / step))
        keys += ["d" if n > 0 else "a"] * abs(n)
        p += n * step
        n = int(round((ct - t) / step))
        keys += ["s" if n > 0 else "w"] * abs(n)
        t += n * step
        keys.append("enter")
    return keys


def edge_points(w, h):
    """JOG_POINTS in table cm: corners and edge middles, clockwise from top-left."""
    return [(0, 0), (w / 2, 0), (w, 0), (w, h / 2), (w, h), (w / 2, h), (0, h), (0, h / 2)]


def wide_rig(seed=3, geom=None):
    rig = SimRig(load_config(), seed=seed, geom=geom)
    rig.act._limits = ((600.0, 2400.0), (600.0, 2400.0))      # config's full travel, as on the rig today
    return rig


def test_jog_limits_cover_the_table_and_the_laser_timeout_still_applies():
    rig = wide_rig()
    w, h = rig.table.size_cm
    corners = [rig.geom.pulses_for(c) for c in edge_points(w, h)]
    keys = keys_to((1500, 1500), corners)
    idle_at, seen = len(keys) // 2, {}

    def read_key():
        if len(seen) == 0 and read_key.i == idle_at:           # the person stops for 11 s
            rig.clock.sleep(11)
            rig.act.poll()
            seen["idle_laser"] = rig.act.laser_on
        k = keys[read_key.i]
        read_key.i += 1
        return k
    read_key.i = 0
    lim = jog_limits(rig.act, read_key, out=lambda s: None)
    assert seen["idle_laser"] is False                         # auto-off happened while idle
    assert not rig.act.laser_on
    c = np.array(corners)
    for ax in (0, 1):
        assert lim[ax][0] <= c[:, ax].min() and lim[ax][1] >= c[:, ax].max()
        assert lim[ax][1] - lim[ax][0] < 1.2 * np.ptp(c[:, ax]) + 30
    steps = np.diff([p for _, p, _ in rig.act.writes])
    assert np.abs(steps).max() <= 50                           # no press moves a servo more than 50 us


def test_jog_quit_returns_none_with_the_laser_off():
    rig = wide_rig()
    assert jog_limits(rig.act, iter(["d", "l", "l", "q"]).__next__, out=lambda s: None) is None
    assert not rig.act.laser_on


def test_rig_session_jog_to_gate_pass(tmp_path):
    rig = wide_rig()
    w, h = rig.table.size_cm
    corners = [rig.geom.pulses_for(c) for c in edge_points(w, h)]
    laser = rig.make_laser("")
    laser.cal_path = str(tmp_path / "laser_cal.json")
    lines = []
    rep = rig_session(laser, Region((w, h)), read_key=iter(keys_to((1500, 1500), corners)).__next__,
                      aims=6, out=lines.append)
    assert rep["pass"], lines
    assert rep["latency"]["seen"] >= 3 and 0.08 <= rep["camera_latency_s"] < 0.15
    assert rep["fit_error_cm"]["median"] < 1.5 and rep["centre_cm"] < 3.0
    assert rep["held_out"]["measured_cm"]["n"] == 6
    assert (tmp_path / "laser_cal.json").exists() and not rig.act.laser_on
    text = "\n".join(lines)
    assert "F6 gate" in text and "PASS" in text and "servo_limits:" in text and "camera_latency_s:" in text


def test_region_outline_keeps_the_fit_on_the_tabletop():
    rig = SimRig(load_config(), seed=3)
    laser = rig.make_laser("")
    poly = [(0, 0), (50, 0), (50, 60), (0, 60)]             # only the left part is 'tabletop'
    region = Region((90, 60), poly)
    assert region.inside((25, 30)) and not region.inside((70, 30)) and region.inside((51, 30), -2)
    assert region.centre == pytest.approx((25, 30))
    assert all(region.inside(p, 6) for p in region.sample(10, inset=6))
    rep = calibrate(laser, region=region)
    assert rep["n_points"] >= 6
    assert rep["fit_error_cm"]["median"] < 1.5
    assert laser.aim((25, 30)) < 2.0 and math.dist(rig.true_dot_cm(), (25, 30)) < 2.0


def test_spot_check_finds_each_note_and_scores_the_ruler(cal_rig):
    rig, laser = cal_rig
    spots = [(20.0, 15.0), (70.0, 15.0), (60.0, 40.0), (25.0, 48.0)]
    measured = []

    def ask(prompt):
        if "sticky note" in prompt:
            rig.notes.append(spots[len(rig.notes)])
            return ""
        d = math.dist(rig.true_dot_cm(), rig.notes[-1])
        measured.append(d)
        return f"{d:.1f}"
    lines = []
    rep = spot_check(laser, ask, n=len(spots), out=lines.append)
    assert rep["pass"], lines
    got = [v for s in rep["spots"] for v in (s["x_cm"], s["y_cm"])]
    assert got == pytest.approx([v for p in spots for v in p], abs=1.0)
    assert max(measured) < 2.0 and not rig.act.laser_on
    assert all(0 < s["surface_red"] < 215 for s in rep["spots"])      # light-blue notes: headroom for the dot
    assert "| spot |" in lines[-1] and "PASS" in lines[-1]


def test_spot_check_asks_again_when_no_note_appears(cal_rig):
    rig, laser = cal_rig
    rig.notes.clear()
    answers = iter(["", "q"])                                # Enter without putting a note down, then quit
    lines = []
    rep = spot_check(laser, lambda p: next(answers), n=3, out=lines.append)
    assert rep["spots"] == [] and not rep["pass"]
    assert any("no new note" in s for s in lines)


@pytest.fixture
def cal_rig():
    rig = SimRig(load_config(), seed=3)
    laser = rig.make_laser("")
    calibrate(laser)
    return rig, laser


def test_rig_refuses_the_fake_actuator(capsys):
    assert main(["--rig"]) == 2                              # config.yaml: actuator fake
    assert "pca9685" in capsys.readouterr().err
    assert main([]) == 2


def test_a_surface_near_saturation_gets_the_exposure_hint(cal_rig, monkeypatch):
    import act.calibrate as ac
    rig, laser = cal_rig
    monkeypatch.setattr(ac, "surface_red", lambda laser, cm, img, r_cm=2.0: 240.0)   # a white box top
    answers = iter(["", "1.0"])

    def ask(prompt):
        if "sticky note" in prompt:
            rig.notes.append((60.0, 40.0))
        return next(answers)
    lines = []
    rep = spot_check(laser, ask, n=1, out=lines.append)
    assert rep["spots"][0]["surface_red"] == 240
    assert any("near saturation" in s and "camera_setup.sh" in s for s in lines)


def test_jog_limits_cover_the_bowed_edges_of_a_low_head():
    """Head 40 cm up, 10 cm off the near edge: the near edge's middle needs pulses well outside the four
    corners' box. The 8-point jog covers every point along every edge."""
    rig = wide_rig(geom=HeadGeometry(pos=(45.0, -10.0, 40.0)))
    w, h = rig.table.size_cm
    pts = [rig.geom.pulses_for(c) for c in edge_points(w, h)]
    lim = jog_limits(rig.act, iter(keys_to((1500, 1500), pts)).__next__, out=lambda s: None)
    corners = np.array([pts[i] for i in (0, 2, 4, 6)])
    along = np.array([rig.geom.pulses_for((x, y)) for x in np.linspace(0, w, 13) for y in (0, h)] +
                     [rig.geom.pulses_for((x, y)) for y in np.linspace(0, h, 9) for x in (0, w)])
    c_lo, c_hi = corners.min(axis=0), corners.max(axis=0)
    assert ((along < c_lo - 0.05 * (c_hi - c_lo)) | (along > c_hi + 0.05 * (c_hi - c_lo))).any()  # corners alone clip
    for ax in (0, 1):
        assert lim[ax][0] - 5 <= along[:, ax].min() and along[:, ax].max() <= lim[ax][1] + 5, ax   # +-5: 10 us steps
