"""act.calibrate --rig on the simulated rig: the jog, the whole session, the sticky-note ruler check."""
import math

import numpy as np
import pytest

from act.calibrate import Region, calibrate, jog_limits, main, rig_session, spot_check
from act.sim import SimRig
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


def wide_rig(seed=3):
    rig = SimRig(load_config(), seed=seed)
    rig.act._limits = ((600.0, 2400.0), (600.0, 2400.0))      # config's full travel, as on the rig today
    return rig


def test_jog_limits_cover_the_table_and_the_laser_timeout_still_applies():
    rig = wide_rig()
    w, h = rig.table.size_cm
    corners = [rig.geom.pulses_for(c) for c in [(0, 0), (w, 0), (w, h), (0, h)]]
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
    corners = [rig.geom.pulses_for(c) for c in [(0, 0), (w, 0), (w, h), (0, h)]]
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
