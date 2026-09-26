"""Pan-tilt head + laser actuators (spec H3). Owner: A.

Pan/tilt are servo pulse widths in microseconds. Every actuator clamps to cfg servo_limits, eases
moves in ~20 ms smoothstep steps, and switches the laser off after cfg laser_timeout_s, on close()
and at interpreter exit. Hardware libraries are imported inside the driver classes only.
"""
from __future__ import annotations

import atexit
import logging
import math
import threading
import time
import weakref
from typing import Optional, Protocol, runtime_checkable

log = logging.getLogger(__name__)

Limits = tuple[tuple[float, float], tuple[float, float]]
STEP_S = 0.02          # easing step
SETTLE_S = 0.05        # extra wait after the last step (hobby servos have no feedback)


# ---------------------------------------------------------------- clocks

class Clock:
    """Real time. All timing in act/ goes through a clock so the simulator can fake it."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, s: float) -> None:
        if s > 0:
            time.sleep(s)


class SimClock(Clock):
    """Simulated monotonic time: sleep() advances instantly. Not thread-safe (tests only)."""

    def __init__(self, t0: float = 1000.0):
        self.t = float(t0)

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        if s > 0:
            self.t += s


# ---------------------------------------------------------------- protocol

@runtime_checkable
class Actuator(Protocol):
    def move(self, pan: float, tilt: float, duration_s: float = 0.3) -> None:
        """Ease to (pan, tilt) pulses in µs, clamped to limits; blocks until settled."""
        ...

    def laser(self, on: bool) -> None: ...

    def limits(self) -> Limits:
        """((pan_lo, pan_hi), (tilt_lo, tilt_hi)) in µs."""
        ...


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


class ActuatorError(IOError):
    pass


_live: "weakref.WeakSet[BaseActuator]" = weakref.WeakSet()


@atexit.register
def _close_all() -> None:
    for a in list(_live):
        try:
            a.close()
        except Exception:  # noqa: BLE001 - best effort at exit
            pass


# ---------------------------------------------------------------- shared behaviour

class BaseActuator:
    """Clamping, easing, locking and laser safety. Subclasses write the hardware.

    `lock` is an RLock: a caller (e.g. search mode) may hold it across several move()/laser() calls.
    """

    def __init__(self, cfg: dict, clock: Optional[Clock] = None, watchdog: bool = True):
        lim = cfg.get("servo_limits") or {}
        self._limits: Limits = (tuple(map(float, lim.get("pan", (600, 2400)))),   # type: ignore[assignment]
                                tuple(map(float, lim.get("tilt", (600, 2400)))))
        for lo, hi in self._limits:
            if not lo < hi:
                raise ValueError(f"bad servo_limits {self._limits}")
        self.clock = clock or Clock()
        self.lock = threading.RLock()
        self.laser_timeout_s = float(cfg.get("laser_timeout_s", 10))
        (plo, phi), (tlo, thi) = self._limits
        self.pan = (plo + phi) / 2.0     # assumed until the first write
        self.tilt = (tlo + thi) / 2.0
        self.laser_on = False
        self._laser_deadline: Optional[float] = None
        self._closed = False
        _live.add(self)
        self._stop = threading.Event()
        if watchdog:
            threading.Thread(target=_watchdog_loop, args=(weakref.ref(self), self._stop),
                             name="laser-watchdog", daemon=True).start()

    # -- hardware hooks (t = clock time of the write; hardware ignores it, the sim records it)
    def _hw_pulses(self, pan: float, tilt: float, t: float) -> None:
        raise NotImplementedError

    def _hw_laser(self, on: bool, t: float) -> None:
        raise NotImplementedError

    def _hw_close(self) -> None:
        pass

    # -- protocol
    def limits(self) -> Limits:
        return self._limits

    def set_limits(self, limits: Limits) -> None:
        """New servo limits (act.calibrate --rig's jog measures them). Only narrower than the current
        ones: a jog can't widen what config allows."""
        (plo, phi), (tlo, thi) = self._limits
        new = ((max(plo, float(limits[0][0])), min(phi, float(limits[0][1]))),
               (max(tlo, float(limits[1][0])), min(thi, float(limits[1][1]))))
        for lo, hi in new:
            if not lo < hi:
                raise ValueError(f"bad servo_limits {limits} (current {self._limits})")
        with self.lock:
            self._limits = new                               # type: ignore[assignment]
            self._on_limits()

    def _on_limits(self) -> None:
        pass

    def clamp(self, pan: float, tilt: float) -> tuple[float, float]:
        (plo, phi), (tlo, thi) = self._limits
        return clamp(float(pan), plo, phi), clamp(float(tilt), tlo, thi)

    def move(self, pan: float, tilt: float, duration_s: float = 0.3) -> None:
        with self.lock:
            self._check_timeout()
            self._on_call("move", pan, tilt, duration_s)
            p1, t1 = self.clamp(pan, tilt)
            p0, t0 = self.pan, self.tilt
            n = max(1, int(math.ceil(max(0.0, duration_s) / STEP_S)))
            for i in range(1, n + 1):
                e = smoothstep(i / n)
                self._emit(p0 + (p1 - p0) * e, t0 + (t1 - t0) * e)
                self.clock.sleep(duration_s / n if duration_s > 0 else 0.0)
            self.clock.sleep(SETTLE_S)

    def laser(self, on: bool) -> None:
        with self.lock:
            self._check_timeout()
            self._on_call("laser", bool(on))
            if on and self._closed:
                raise ActuatorError("actuator closed")
            self._set_laser(bool(on), self.clock.now())
            self._laser_deadline = self.clock.now() + self.laser_timeout_s if on else None

    def poll(self) -> None:
        """Apply the laser timeout now (the watchdog thread does this in real time)."""
        with self.lock:
            self._check_timeout()

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        with self.lock:
            try:
                self._set_laser(False, self.clock.now())
            finally:
                self._closed = True
                self._laser_deadline = None
                self._hw_close()

    # -- internals
    def _on_call(self, name: str, *args) -> None:
        """Hook for FakeActuator's call log."""

    def _emit(self, pan: float, tilt: float) -> None:
        pan, tilt = self.clamp(pan, tilt)            # never emit outside limits
        self._hw_pulses(pan, tilt, self.clock.now())
        self.pan, self.tilt = pan, tilt

    def _set_laser(self, on: bool, t: float) -> None:
        self._hw_laser(on, t)
        self.laser_on = on

    def _check_timeout(self) -> None:
        d = self._laser_deadline
        if d is not None and self.laser_on and self.clock.now() >= d:
            log.info("laser auto-off after %.1f s", self.laser_timeout_s)
            self._set_laser(False, d)
            self._laser_deadline = None
            self._on_call("auto_off")


def _watchdog_loop(ref: "weakref.ref[BaseActuator]", stop: threading.Event) -> None:
    """Real-time laser timeout. Holds only a weak reference so the actuator can be collected."""
    while not stop.wait(0.1):
        a = ref()
        if a is None:
            return
        if a._laser_deadline is not None and a.clock.now() >= a._laser_deadline:
            if a.lock.acquire(timeout=0.5):
                try:
                    a._check_timeout()
                finally:
                    a.lock.release()
            else:  # somebody holds the lock for too long: safety wins
                a._set_laser(False, a.clock.now())
                a._laser_deadline = None
        del a


# ---------------------------------------------------------------- drivers

class PCA9685Actuator(BaseActuator):
    """Two hobby servos + laser transistor on a PCA9685 (adafruit ServoKit)."""

    def __init__(self, cfg: dict, clock: Optional[Clock] = None):
        from adafruit_servokit import ServoKit  # lazy: Jetson only

        super().__init__(cfg, clock)
        self._kit = ServoKit(channels=16, address=int(cfg.get("pca9685_address", 0x40)))
        self._pan_ch = int(cfg.get("pan_channel", 0))
        self._tilt_ch = int(cfg.get("tilt_channel", 1))
        self._laser_ch = int(cfg.get("laser_channel", 15))
        # Map fraction 0..1 onto exactly the configured limits, so no pulse can fall outside them.
        for ch, (lo, hi) in ((self._pan_ch, self._limits[0]), (self._tilt_ch, self._limits[1])):
            self._kit.servo[ch].set_pulse_width_range(int(lo), int(hi))
        self._pca = self._kit._pca  # ServoKit has no public raw-channel API
        self._hw_laser(False, 0.0)

    def _on_limits(self) -> None:
        for ch, (lo, hi) in ((self._pan_ch, self._limits[0]), (self._tilt_ch, self._limits[1])):
            self._kit.servo[ch].set_pulse_width_range(int(lo), int(hi))

    def _hw_pulses(self, pan: float, tilt: float, t: float) -> None:
        (plo, phi), (tlo, thi) = self._limits
        self._kit.servo[self._pan_ch].fraction = (pan - plo) / (phi - plo)
        self._kit.servo[self._tilt_ch].fraction = (tilt - tlo) / (thi - tlo)

    def _hw_laser(self, on: bool, t: float) -> None:
        self._pca.channels[self._laser_ch].duty_cycle = 0xFFFF if on else 0

    def _hw_close(self) -> None:
        for ch in (self._pan_ch, self._tilt_ch):
            try:
                self._kit.servo[ch].fraction = None   # stop driving (servo goes limp, no buzz)
            except Exception:  # noqa: BLE001
                pass


class SerialActuator(BaseActuator):
    """ESP32 / stepper board over USB serial: 'P<pan>,<tilt>,<laser>\\n' -> 'OK'."""

    def __init__(self, cfg: dict, clock: Optional[Clock] = None):
        import serial  # lazy: pyserial

        super().__init__(cfg, clock)
        self._ser = serial.Serial(cfg.get("serial_port", "/dev/ttyUSB0"),
                                  int(cfg.get("serial_baud", 115200)),
                                  timeout=float(cfg.get("serial_timeout_s", 0.1)))
        self._send(self.pan, self.tilt, False)

    def _send(self, pan: float, tilt: float, on: bool) -> None:
        (plo, phi), (tlo, thi) = self._limits
        p = int(clamp(round(pan), math.ceil(plo), math.floor(phi)))
        q = int(clamp(round(tilt), math.ceil(tlo), math.floor(thi)))
        line = f"P{p},{q},{int(on)}\n".encode()
        for _ in range(2):
            self._ser.write(line)
            reply = self._ser.readline().strip()
            if reply == b"OK":
                return
            log.warning("serial actuator replied %r to %r", reply, line)
        raise ActuatorError(f"no OK from serial actuator for {line!r}")

    def _hw_pulses(self, pan: float, tilt: float, t: float) -> None:
        self._send(pan, tilt, self.laser_on)

    def _hw_laser(self, on: bool, t: float) -> None:
        self._send(self.pan, self.tilt, on)

    def _hw_close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass


class BusServoActuator(BaseActuator):
    """Feetech STS / Dynamixel bus servos via a USB adapter. Not implemented for the hackathon."""

    def __init__(self, cfg: dict, clock: Optional[Clock] = None):
        raise NotImplementedError(
            "BusServoActuator is a stub: use actuator: pca9685 (hobby servos) or serial (ESP32). "
            "For Feetech STS, map µs to 0-4095 steps and write register 42 (goal position).")


class FakeActuator(BaseActuator):
    """Logs every call with timestamps and tracks state. realtime=False: simulated clock, no sleeping."""

    def __init__(self, cfg: Optional[dict] = None, clock: Optional[Clock] = None, realtime: bool = True):
        clock = clock or (Clock() if realtime else SimClock())
        super().__init__(cfg or {}, clock, watchdog=isinstance(clock, Clock) and not isinstance(clock, SimClock))
        self.calls: list[tuple] = []     # (t, name, *args)
        self.writes: list[tuple[float, float, float]] = []   # (t, pan, tilt) every emitted step
        self.laser_log: list[tuple[float, bool]] = []

    def _on_call(self, name: str, *args) -> None:
        self.calls.append((self.clock.now(), name, *args))

    def _hw_pulses(self, pan: float, tilt: float, t: float) -> None:
        self.writes.append((t, pan, tilt))

    def _hw_laser(self, on: bool, t: float) -> None:
        self.laser_log.append((t, on))

    def close(self) -> None:
        if not self._closed:
            self._on_call("close")
        super().close()


ACTUATORS = {"pca9685": PCA9685Actuator, "serial": SerialActuator, "bus": BusServoActuator,
             "fake": FakeActuator}


def make_actuator(cfg: dict, clock: Optional[Clock] = None) -> BaseActuator:
    """Pick the driver by cfg['actuator'] (pca9685|serial|bus|fake, default fake)."""
    kind = str(cfg.get("actuator", "fake")).lower()
    if kind not in ACTUATORS:
        raise ValueError(f"unknown actuator {kind!r}; expected one of {sorted(ACTUATORS)}")
    return ACTUATORS[kind](cfg, clock)


def make_actuator_or_fake(cfg: dict, clock: Optional[Clock] = None) -> tuple[BaseActuator, Optional[str]]:
    """make_actuator, but a driver that can't start (adafruit_servokit missing from the image, no board
    on I2C, serial port gone) gives a FakeActuator and the reason instead of stopping the app: the
    caller then runs with the laser disabled and answers by voice only."""
    try:
        return make_actuator(cfg, clock), None
    except Exception as e:  # noqa: BLE001 - ImportError, OSError, ValueError, NotImplementedError ...
        why = f"actuator {cfg.get('actuator')!r} failed to start: {type(e).__name__}: {e}"
        log.error("%s. LASER DISABLED: answers are spoken only. Fix it (docker/Dockerfile installs "
                  "adafruit-circuitpython-servokit; check the board with i2cdetect -y -r 7) and restart.", why)
        return FakeActuator(cfg, clock), why
