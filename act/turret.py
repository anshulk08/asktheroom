"""Serial client for firmware/turret (Arduino Uno + two MKS SERVO42D steppers). Owner: A.

The firmware speaks one command per line at 115200 baud, in degrees at the axis (+ pan right,
+ tilt up): 'A <pan> <tilt>' aims and later sends 'DONE', 'L 0|1' switches the laser, 'P' reports
'POS <pan> <tilt> <moving> <laser>'; the full list is at the top of firmware/turret/turret.ino.

Opening the port resets the Uno, which prints 'READY' and calls wherever the mount points 0,0.
The firmware switches the laser off after 1 s without a command, so while the laser is on this
client sends 'P' every HEARTBEAT_S: the laser stays on only while this process is alive.

The port is opened exclusively, and a 'READY' after that first one means the Uno reset mid-run
(USB glitch, its watchdog, another program opening the port): it re-zeroed wherever it stood, so
the pose is unknown and the laser and moves are refused until this process restarts.

Most code uses it through act.actuator.TurretActuator (actuator: turret). By hand:
    python -m act.turret /dev/ttyACM0          type raw commands (A 10 5, L 1, P, H ...)
    python -m act.turret /dev/ttyACM0 demo     pan 90 right and left, tilt 45 up
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

HEARTBEAT_S = 0.5
LASER_PIN = 8           # the Uno pin the laser module's signal is wired to: firmware/turret/turret.ino LASER_PIN,
                        # firmware/README.md and docs/LASER_SAFETY.md must all say this (tests/test_turret.py)
TILT_MAX_DEG = -10.0    # turret.ino LASER_TILT_MAX_DEG: the firmware never lights the beam above this tilt
REPLY_TIMEOUT_S = 2.0
_NEED_KNOWN_ZERO = ("A", "R", "Z")      # refused after a board reset, with "L 1"


class TurretError(IOError):
    pass


class Turret:
    def __init__(self, port: str, baud: int = 115200, boot_timeout_s: float = 5.0, ser=None):
        """ser: an already-open serial-like object (tests); otherwise pyserial opens `port`."""
        if ser is None:
            import serial  # lazy: pyserial, only where hardware is used

            # exclusive: a second opener (another askroom, a serial monitor) would reset the Uno under us
            ser = serial.Serial(port, baud, timeout=0.1, exclusive=True)
        self._ser = ser
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._laser_on = False
        self._closed = False
        self._booted = False
        self.reset_seen = False     # the board rebooted after the open: pose unknown, laser locked out
        self._wait_for_line("READY", boot_timeout_s)
        self._booted = True
        self._heartbeat = threading.Thread(target=self._heartbeat_loop, name="turret-heartbeat",
                                           daemon=True)
        self._heartbeat.start()

    # ---- commands
    def aim(self, pan_deg: float, tilt_deg: float, wait: bool = False,
            timeout_s: float = 10.0) -> tuple[float, float]:
        """Point at absolute angles. Returns the (pan, tilt) the firmware targeted after its limits."""
        reply = self._command(f"A {pan_deg:.3f} {tilt_deg:.3f}", "OK A")
        if wait:
            self.wait_done(timeout_s)
        _, _, pan, tilt = reply.split()
        return float(pan), float(tilt)

    def nudge(self, dpan_deg: float, dtilt_deg: float, wait: bool = False,
              timeout_s: float = 10.0) -> tuple[float, float]:
        reply = self._command(f"R {dpan_deg:.3f} {dtilt_deg:.3f}", "OK A")
        if wait:
            self.wait_done(timeout_s)
        _, _, pan, tilt = reply.split()
        return float(pan), float(tilt)

    def laser(self, on: bool) -> None:
        """Raises TurretError for on=True after a board reset; on=False always goes through."""
        self._command(f"L {1 if on else 0}", "OK L")
        self._laser_on = bool(on)

    def brightness(self, level: int) -> None:
        self._command(f"B {int(level)}", "OK B")

    def position(self) -> tuple[float, float, bool, bool]:
        """(pan_deg, tilt_deg, moving, laser_on)."""
        _, pan, tilt, moving, laser = self._command("P", "POS").split()
        return float(pan), float(tilt), moving == "1", laser == "1"

    def speed(self, deg_per_s: float) -> str:
        return self._command(f"V {deg_per_s}", "OK V")

    def accel(self, deg_per_s2: float) -> str:
        return self._command(f"C {deg_per_s2}", "OK V")

    def stop(self) -> None:
        self._command("S", "OK S")

    def estop(self) -> None:
        """Stop at once, laser off, motors released."""
        self._command("X", "OK X")
        self._laser_on = False

    def zero(self) -> None:
        """Call the current position 0,0."""
        self._command("Z", "OK Z")

    def release(self) -> None:
        self._command("E 0", "OK E")

    def wait_done(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._done.wait(0.05):
                return
            with self._lock:
                self._drain()
            if self._done.is_set():
                return
            if not self.position()[2]:
                return
        raise TurretError("move did not finish in time")

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._laser_on:
                self.laser(False)
        finally:
            self._closed = True
            self._ser.close()

    def __enter__(self) -> "Turret":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- internals
    def _command(self, line: str, expect: str) -> str:
        if self._closed:
            raise TurretError("turret closed")
        with self._lock:
            self._drain()
            self._refuse_after_reset(line)
            if expect == "OK A":
                self._done.clear()
            self._ser.write((line + "\n").encode())
            deadline = time.monotonic() + REPLY_TIMEOUT_S
            while time.monotonic() < deadline:
                reply = self._readline()
                if reply is None:
                    continue
                if reply.startswith(expect):
                    # A reset seen mid-exchange: the fresh board may have taken the command.
                    self._refuse_after_reset(line, sent=True)
                    return reply
                if reply.startswith("ERR"):
                    raise TurretError(f"{line!r}: {reply}")
            self._refuse_after_reset(line, sent=True)
            raise TurretError(f"no reply to {line!r}")

    def _refuse_after_reset(self, line: str, sent: bool = False) -> None:
        """After a board reset its 0,0 is wherever the head happened to be, so a lit laser or an aim
        could land anywhere (eyes included). Only a process restart (re-homing by hand) clears it."""
        if not self.reset_seen:
            return
        if line.startswith("L 1"):
            if sent:                            # make sure the fresh board did not light it
                self._ser.write(b"L 0\n")
                self._laser_on = False
            raise TurretError("board reset: laser locked out until restart")
        if line.split()[0] in _NEED_KNOWN_ZERO:
            raise TurretError(f"board reset: pose unknown, {line!r} refused until restart")

    def _readline(self) -> Optional[str]:
        raw = self._ser.readline()
        if not raw:
            return None
        line = raw.decode(errors="replace").strip()
        if line == "DONE":
            self._done.set()
            return None
        if line in ("LASER TIMEOUT", "LASER MAX ON", "LASER TILT"):     # the firmware turned it off by itself
            self._laser_on = False
            return None
        if line.startswith("READY") and self._booted:
            if not self.reset_seen:
                log.error("TURRET BOARD RESET mid-run (%r): its 0,0 is now wherever the head stood. "
                          "Laser and moves refused until askroom restarts.", line)
            self.reset_seen = True
            self._laser_on = False              # the Uno drives the laser LOW from reset
            return None
        return line or None

    def _drain(self) -> None:
        while self._ser.in_waiting:
            self._readline()

    def _wait_for_line(self, prefix: str, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self._readline()
            if line and line.startswith(prefix):
                return line
        raise TurretError(f"no {prefix!r} from the board; is firmware/turret flashed?")

    def _heartbeat_loop(self) -> None:
        while not self._closed:
            time.sleep(HEARTBEAT_S)
            if self._laser_on and not self._closed:
                try:
                    self.position()
                except (TurretError, OSError):
                    pass


def _interactive(t: Turret) -> None:
    print("Connected. Type commands (H for help, Ctrl-D to quit).")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        with t._lock:
            t._drain()
            t._ser.write((line + "\n").encode())
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                raw = t._ser.readline()
                if raw:
                    print(raw.decode(errors="replace").rstrip())


def _demo(t: Turret) -> None:
    for pan, tilt in [(90, 0), (0, 0), (-90, 0), (0, 0), (0, 45), (0, 0)]:
        start = time.monotonic()
        target = t.aim(pan, tilt, wait=True)
        print(f"aim {target} done in {time.monotonic() - start:.2f} s -> {t.position()}")
        time.sleep(1.5)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with Turret(sys.argv[1]) as turret:
        if len(sys.argv) > 2 and sys.argv[2] == "demo":
            _demo(turret)
        else:
            _interactive(turret)
