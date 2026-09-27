import sys
import time
import types

import pytest

from act.actuator import (STEP_S, Actuator, FakeActuator, SimClock, make_actuator)

CFG = {"servo_limits": {"pan": [600, 2400], "tilt": [700, 2300]}, "laser_timeout_s": 10,
       "laser_channel": 15}


def fake(**kw) -> FakeActuator:
    return FakeActuator(dict(CFG, **kw), realtime=False)


def in_limits(a, pan, tilt) -> bool:
    (plo, phi), (tlo, thi) = a.limits()
    return plo <= pan <= phi and tlo <= tilt <= thi


def test_protocol_and_limits():
    a = fake()
    assert isinstance(a, Actuator)
    assert a.limits() == ((600.0, 2400.0), (700.0, 2300.0))
    assert (a.pan, a.tilt) == (1500.0, 1500.0)


def test_clamps_to_limits():
    a = fake()
    a.move(5000, -100, duration_s=0.2)
    assert (a.pan, a.tilt) == (2400.0, 700.0)
    a.move(-1e9, 1e9, duration_s=0)
    assert (a.pan, a.tilt) == (600.0, 2300.0)
    assert all(in_limits(a, p, t) for _, p, t in a.writes)


def test_easing_path_smooth_within_limits_ends_at_target():
    a = fake()
    a.move(2000, 1000, duration_s=0.3)
    w = a.writes
    assert len(w) == round(0.3 / STEP_S)
    pans = [p for _, p, _ in w]
    tilts = [t for _, _, t in w]
    assert pans == sorted(pans) and tilts == sorted(tilts, reverse=True)      # monotone
    assert (pans[-1], tilts[-1]) == (2000.0, 1000.0)
    steps = [b - a_ for a_, b in zip(pans, pans[1:])]
    assert steps[0] < max(steps) and steps[-1] < max(steps)                   # eases in and out
    assert all(in_limits(a, p, t) for _, p, t in w)
    ts = [t for t, _, _ in w]
    assert all(b > a_ for a_, b in zip(ts, ts[1:]))                           # ~20 ms apart


def test_nonrealtime_does_not_sleep():
    a = fake()
    t0, c0 = time.perf_counter(), a.clock.now()
    a.move(2400, 2300, duration_s=5.0)
    assert time.perf_counter() - t0 < 0.5
    assert a.clock.now() - c0 >= 5.0                     # simulated time advanced


def test_fake_logs_calls_with_timestamps():
    a = fake()
    a.move(1000, 1000, 0.1)
    a.laser(True)
    a.laser(False)
    names = [c[1] for c in a.calls]
    assert names == ["move", "laser", "laser"]
    assert a.calls[0][2:] == (1000, 1000, 0.1)
    assert a.calls[1][0] >= a.calls[0][0] + 0.1          # timestamps follow the clock
    assert [on for _, on in a.laser_log] == [True, False]
    assert a.laser_on is False


def test_laser_auto_off_after_timeout():
    a = fake(laser_timeout_s=3)
    a.laser(True)
    a.clock.sleep(2.9)
    a.poll()
    assert a.laser_on
    a.laser(True)                                        # re-aim resets the timer
    a.clock.sleep(2.9)
    a.poll()
    assert a.laser_on
    a.clock.sleep(0.2)
    a.poll()
    assert not a.laser_on
    assert a.calls[-1][1] == "auto_off"
    t_off, on = a.laser_log[-1]
    assert not on and t_off == pytest.approx(a.calls[-2][0] + 3)    # recorded at the deadline


def test_laser_timeout_never_exceeds_the_apps_max_on():
    """The watchdog is the backstop for laser_room.max_on_s: a crash must not leave 10 s of beam."""
    assert fake(laser_timeout_s=10).laser_timeout_s == 4.0
    assert fake(laser_timeout_s=10, laser_room={"max_on_s": 2.5}).laser_timeout_s == 2.5
    assert fake(laser_timeout_s=1).laser_timeout_s == 1.0
    a = fake(laser_timeout_s=10)
    a.laser(True)
    a.clock.sleep(4.0)
    a.poll()
    assert not a.laser_on


def test_laser_auto_off_realtime_watchdog():
    a = FakeActuator(dict(CFG, laser_timeout_s=0.15), realtime=True)
    a.laser(True)
    time.sleep(0.45)
    assert not a.laser_on
    a.close()


def test_the_watchdog_survives_a_failing_laser_off():
    """One serial error must not end the only real-time guard on the laser."""
    a = FakeActuator(dict(CFG, laser_timeout_s=0.15), realtime=True)
    fails = []
    hw_laser = a._hw_laser

    def flaky(on, t):
        if not on and not fails:
            fails.append(t)
            raise OSError("serial write failed")
        hw_laser(on, t)

    a._hw_laser = flaky
    a.laser(True)
    deadline = time.monotonic() + 2.0
    while a.laser_on and time.monotonic() < deadline:
        time.sleep(0.02)
    assert fails and not a.laser_on                     # the retry after the failure switched it off
    assert a._watchdog.is_alive()
    a.close()


def test_close_turns_laser_off():
    a = fake()
    a.laser(True)
    a.close()
    assert not a.laser_on and a.laser_log[-1][1] is False
    with pytest.raises(IOError):
        a.laser(True)


def test_make_actuator_default_fake_and_unknown():
    a = make_actuator(dict(CFG), clock=SimClock())
    assert type(a) is FakeActuator
    with pytest.raises(ValueError):
        make_actuator(dict(CFG, actuator="warp"))
    with pytest.raises(NotImplementedError):
        make_actuator(dict(CFG, actuator="bus"))


def test_hardware_libs_are_lazy():
    assert "adafruit_servokit" not in sys.modules and "serial" not in sys.modules


class _Chan:
    def __init__(self):
        self.duty_cycle = None


class _Servo:
    def __init__(self):
        self.range, self.fraction = None, None

    def set_pulse_width_range(self, lo, hi):
        self.range = (lo, hi)


class _Kit:
    def __init__(self, channels, address=0x40):
        self.servo = [_Servo() for _ in range(channels)]
        self._pca = types.SimpleNamespace(channels=[_Chan() for _ in range(channels)])


def test_pca9685_pulses_and_laser_channel(monkeypatch):
    monkeypatch.setitem(sys.modules, "adafruit_servokit", types.SimpleNamespace(ServoKit=_Kit))
    a = make_actuator(dict(CFG, actuator="pca9685"), clock=SimClock())
    kit = a._kit
    assert kit.servo[0].range == (600, 2400) and kit.servo[1].range == (700, 2300)
    a.move(1500, 9999, duration_s=0.1)
    assert kit.servo[0].fraction == pytest.approx(0.5) and kit.servo[1].fraction == pytest.approx(1.0)
    a.laser(True)
    assert kit._pca.channels[15].duty_cycle == 0xFFFF
    a.close()
    assert kit._pca.channels[15].duty_cycle == 0


def test_serial_protocol(monkeypatch):
    sent = []

    class _Ser:
        def __init__(self, *a, **kw):
            pass

        def write(self, b):
            sent.append(b)

        def readline(self):
            return b"OK\r\n"

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=_Ser))
    a = make_actuator(dict(CFG, actuator="serial"), clock=SimClock())
    a.move(3000, 1000.4, duration_s=0.04)
    a.laser(True)
    assert sent[-2] == b"P2400,1000,0\n" and sent[-1] == b"P2400,1000,1\n"
    a.close()
    assert sent[-1] == b"P2400,1000,0\n"


def test_a_driver_that_cannot_start_falls_back_to_fake_with_the_reason(monkeypatch):
    from act.actuator import make_actuator_or_fake
    monkeypatch.setitem(sys.modules, "adafruit_servokit", None)
    a, why = make_actuator_or_fake(dict(CFG, actuator="pca9685"), clock=SimClock())
    assert type(a) is FakeActuator and "pca9685" in why and "adafruit_servokit" in why
    a, why = make_actuator_or_fake(dict(CFG, actuator="fake"), clock=SimClock())
    assert type(a) is FakeActuator and why is None


def test_set_limits_only_narrows_and_reprograms_the_pca9685(monkeypatch):
    monkeypatch.setitem(sys.modules, "adafruit_servokit", types.SimpleNamespace(ServoKit=_Kit))
    a = make_actuator(dict(CFG, actuator="pca9685"), clock=SimClock())
    a.set_limits(((1200, 1800), (500, 1600)))
    assert a.limits() == ((1200.0, 1800.0), (700.0, 1600.0))             # tilt can't go below config's 700
    assert a._kit.servo[0].range == (1200, 1800) and a._kit.servo[1].range == (700, 1600)
    a.move(3000, 0, duration_s=0)
    assert (a.pan, a.tilt) == (1800.0, 700.0)
    with pytest.raises(ValueError):
        a.set_limits(((2500, 2600), (800, 900)))
    a.close()
