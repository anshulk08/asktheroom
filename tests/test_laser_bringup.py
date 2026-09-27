import subprocess
import sys
import time

import pytest

from act.actuator import FakeActuator, SimClock
from scripts import laser_bringup as lb

CFG = {"servo_limits": {"pan": [1000, 2000], "tilt": [1100, 1900]}, "laser_timeout_s": 10}


def rig():
    clock = SimClock()
    act = FakeActuator(CFG, clock=clock)
    return act, clock


def guard(act, clock, allowed=True, max_on_s=4.0):
    return lb.LaserGuard(act, allowed, max_on_s, now=clock.now, timer=None, out=lambda s: None)


def lit(act) -> bool:
    return any(on for _, on in act.laser_log)


@pytest.mark.parametrize("laser, eye_safe", [(False, False), (True, False), (False, True)])
def test_laser_never_on_without_both_flags(laser, eye_safe):
    act, clock = rig()
    msgs = []
    g = lb.LaserGuard(act, laser and eye_safe, now=clock.now, timer=None, out=msgs.append)
    lb.jog(act, ["l", "d", "l", "l", "q"], g, out=lambda s: None)
    assert not lit(act) and not act.laser_on
    assert any("refused" in m for m in msgs)


def test_laser_on_with_both_flags_then_off_after_q():
    act, clock = rig()
    lb.jog(act, ["l", "d", "q", "l"], guard(act, clock), out=lambda s: None)
    assert lit(act)
    assert act.laser_log[-1][1] is False and not act.laser_on


def test_laser_off_after_exception():
    act, clock = rig()

    def keys():
        yield "l"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        lb.jog(act, keys(), guard(act, clock), out=lambda s: None)
    assert lit(act) and not act.laser_on and act.laser_log[-1][1] is False


def test_laser_off_after_keyboard_interrupt():
    act, clock = rig()

    def keys():
        yield "l"
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        lb.jog(act, keys(), guard(act, clock), out=lambda s: None)
    assert not act.laser_on


def test_on_time_cap_turns_it_off():
    act, clock = rig()
    g = guard(act, clock, max_on_s=4.0)
    assert act.laser_timeout_s == 4.0            # the actuator's own watchdog is tightened too

    def keys():
        yield "l"
        clock.sleep(3.0)
        yield "p"
        assert act.laser_on                      # still inside the cap
        clock.sleep(1.5)
        yield "p"
        assert not act.laser_on                  # cut at 4 s, while still jogging
        yield "q"

    lb.jog(act, keys(), g, out=lambda s: None)
    assert [on for _, on in act.laser_log].count(True) == 1


def test_on_time_cap_real_timer():
    act = FakeActuator(CFG, realtime=False)
    g = lb.LaserGuard(act, True, max_on_s=0.05, out=lambda s: None)
    assert g.on() and act.laser_on
    deadline = time.monotonic() + 2
    while act.laser_on and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not act.laser_on and not g.is_on


def test_guard_rejects_bad_cap():
    act, clock = rig()
    with pytest.raises(ValueError):
        lb.LaserGuard(act, True, max_on_s=0, timer=None)


def test_jog_moves_within_limits_and_clamps():
    act, clock = rig()
    g = guard(act, clock, allowed=False)
    (pan, tilt), marks = lb.jog(act, ["+", "+"] + ["d"] * 30 + ["]"] + ["w"] * 30 + ["["] + ["q"], g,
                                out=lambda s: None)
    assert (pan, tilt) == (2000, 1100)
    assert all(1000 <= p <= 2000 and 1100 <= t <= 1900 for _, p, t in act.writes)
    assert marks == [(2000, 1500), (2000, 1100)]
    (pan, tilt), _ = lb.jog(act, ["a"] * 30 + ["s"] * 30 + ["q"], g, out=lambda s: None)
    assert (pan, tilt) == (1000, 1900)


def test_jog_step_size_and_centre():
    act, clock = rig()
    g = guard(act, clock, allowed=False)
    (pan, _), _ = lb.jog(act, ["-", "-", "-", "d", "q"], g, out=lambda s: None)
    assert pan == 1505                           # smallest step, 5 us
    (pan, tilt), _ = lb.jog(act, ["d", "s", "c", "q"], g, out=lambda s: None)
    assert (pan, tilt) == (1500, 1500)


def test_limits_yaml():
    act, _ = rig()
    assert "no limits marked" in lb.limits_yaml(act, (1500, 1500), [])
    y = lb.limits_yaml(act, (1500, 1500), [(1200, 1500), (1800, 1500)])
    assert "pan: [1200, 1800]" in y and "tilt: [1100, 1900]" in y and y.startswith("servo_limits:")


def test_range_keeps_laser_off_and_visits_both_extremes():
    act, _ = rig()
    lb.sweep(act, steps=7, pause_s=0, sleep=lambda s: None, out=lambda s: None)
    assert not lit(act) and not act.laser_on
    pans = {round(p) for _, p, _ in act.writes}
    tilts = {round(t) for _, _, t in act.writes}
    assert {1000, 2000} <= pans and {1100, 1900} <= tilts
    assert all(1000 <= p <= 2000 and 1100 <= t <= 1900 for _, p, t in act.writes)


def test_line_keys():
    assert list(lb.line_keys(["ddd\n", "left\n", "\n", "Q\n"])) == ["d", "d", "d", "left", "q"]


I2CDETECT = """     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- --
40: 40 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: 70 -- -- -- -- -- -- --
"""


def test_parse_i2cdetect():
    assert lb.parse_i2cdetect(I2CDETECT, 0x40)
    assert lb.parse_i2cdetect(I2CDETECT, 0x70)
    assert not lb.parse_i2cdetect(I2CDETECT, 0x41)
    assert lb.parse_i2cdetect(I2CDETECT.replace("40: 40", "40: UU"), 0x40)


def test_check_missing_device():
    ok, msg = lb.i2c_probe(exists=lambda p: False)
    assert not ok and "/dev/i2c-7 missing" in msg


def test_check_no_smbus2_no_i2cdetect(monkeypatch):
    monkeypatch.setitem(sys.modules, "smbus2", None)          # import smbus2 -> ImportError

    def run(*a, **k):
        raise FileNotFoundError("i2cdetect")

    ok, msg = lb.i2c_probe(exists=lambda p: True, run=run)
    assert not ok and "i2cdetect failed" in msg


def test_check_i2cdetect_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "smbus2", None)
    run = lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=I2CDETECT)  # noqa: E731
    assert lb.i2c_probe(exists=lambda p: True, run=run)[0]


def test_cmd_check_reports_driver_and_exit_code():
    out = []
    cfg = dict(CFG, actuator="pca9685")
    rc = lb.cmd_check(cfg, out=out.append, probe=lambda bus, addr: (False, "not found"))
    assert rc == 1
    text = "\n".join(out)
    assert "FAIL not found" in text and "FakeActuator" in text
    assert "fell back" in text or "config says actuator: fake" in text
    assert lb.cmd_check(dict(CFG, actuator="fake"), out=out.append, probe=lambda b, a: (True, "ok")) == 0


def test_check_probes_the_turret_serial_port_not_i2c():
    from scripts.laser_bringup import cmd_check, serial_probe
    out, i2c = [], []
    cfg = {"actuator": "turret", "turret": {"port": "/dev/ttyACM9"}}
    rc = cmd_check(dict(cfg, actuator="fake"), out=out.append, probe=lambda b, a: i2c.append(1) or (True, "i2c"),
                   sprobe=lambda p: (False, p))
    assert i2c == [1] and rc == 0
    out.clear()
    rc = cmd_check(cfg, out=out.append, probe=lambda b, a: (True, "i2c"), sprobe=lambda p: (False, f"no {p}"))
    assert rc == 1 and out[0] == "FAIL no /dev/ttyACM9"
    assert serial_probe("/dev/x", exists=lambda p: False)[0] is False
    assert serial_probe("/dev/x", exists=lambda p: True, access=lambda p, m: False)[0] is False
    assert serial_probe("/dev/x", exists=lambda p: True, access=lambda p, m: True)[0] is True


def test_the_test_point_refuses_without_the_eye_safe_confirmation(capsys):
    from scripts.laser_testpoint import main
    assert main(["0", "-22"]) == 2
    assert "eye-safe-confirmed" in capsys.readouterr().err
