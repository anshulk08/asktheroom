"""TurretActuator + act.turret against an in-memory emulator of firmware/turret's serial protocol."""
import sys
import time
import types
from collections import deque

import pytest

import act.turret as turret_mod
from act.actuator import ActuatorError, SimClock, TurretActuator, make_actuator, make_actuator_or_fake
from act.laser import Laser
from act.turret import Turret, TurretError


class FakeBoard:
    """Line-level stand-in for the Uno running firmware/turret: moves finish instantly."""
    LIMITS = ((-170.0, 170.0), (-45.0, 90.0))

    def __init__(self, boot: bytes = b"READY turret\n"):
        self.out = deque([boot] if boot else [])
        self.pan = self.tilt = 0.0
        self.laser = 0
        self.lines: list[str] = []
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self.out)

    def readline(self) -> bytes:
        return self.out.popleft() if self.out else b""

    def close(self) -> None:
        self.closed = True

    def _say(self, s: str) -> None:
        self.out.append((s + "\n").encode())

    def write(self, data: bytes) -> None:
        for line in data.decode().splitlines():
            self.lines.append(line)
            op, *args = line.split()
            nums = [float(a) for a in args]
            if op in ("A", "R"):
                if op == "R":
                    nums = [self.pan + nums[0], self.tilt + nums[1]]
                (plo, phi), (tlo, thi) = self.LIMITS
                self.pan, self.tilt = min(phi, max(plo, nums[0])), min(thi, max(tlo, nums[1]))
                self._say(f"OK A {self.pan:.2f} {self.tilt:.2f}")
                self._say("DONE")
            elif op == "L":
                self.laser = int(nums[0] != 0)
                self._say(f"OK L {self.laser}")
            elif op == "P":
                self._say(f"POS {self.pan:.2f} {self.tilt:.2f} 0 {self.laser}")
            elif op in ("V", "C"):
                self._say("OK V 120.0 120.0 C 600.0")
            elif op == "X":
                self.laser = 0
                self._say("OK X")
            else:
                self._say(f"ERR unknown {op}")

    def commands(self, op: str) -> list[str]:
        return [ln for ln in self.lines if ln.split()[0] == op]


CFG = {"actuator": "turret", "servo_limits": {"pan": [1050, 1950], "tilt": [1050, 1950]},
       "laser_timeout_s": 10, "turret": {"port": "/dev/fake", "center_us": 1500, "us_per_deg": 10}}


def turret_act(board=None, **turret_cfg):
    board = board or FakeBoard()
    cfg = dict(CFG, turret=dict(CFG["turret"], **turret_cfg))
    a = TurretActuator(cfg, clock=SimClock(), turret=Turret("/dev/fake", ser=board))
    return a, board


def test_make_actuator_opens_the_port_through_pyserial(monkeypatch):
    board = FakeBoard()
    opened = []

    def Serial(port, baud, timeout):
        opened.append((port, baud))
        return board

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=Serial))
    a = make_actuator(CFG, clock=SimClock())
    assert isinstance(a, TurretActuator)
    assert opened == [("/dev/fake", 115200)]
    assert (a.pan, a.tilt) == (1500.0, 1500.0)       # the firmware's 0,0 at power-up
    a.close()


def test_pulses_map_to_degrees():
    a, board = turret_act()
    a.move(1600, 1450)
    assert board.commands("A")[-1] == "A 10.000 -5.000"
    assert (board.pan, board.tilt) == (10.0, -5.0)
    assert (a.pan, a.tilt) == (1600.0, 1450.0)


def test_servo_limits_clamp_before_sending():
    a, board = turret_act()
    a.move(3000, 0)
    assert (board.pan, board.tilt) == (45.0, -45.0)
    assert (a.pan, a.tilt) == (1950.0, 1050.0)


def test_firmware_clamp_is_what_the_actuator_reports():
    board = FakeBoard()
    cfg = dict(CFG, servo_limits={"pan": [0, 3000], "tilt": [0, 3000]})
    a = TurretActuator(cfg, clock=SimClock(), turret=Turret("/dev/fake", ser=board))
    a.move(1500, 0)                                   # asks for -150 deg tilt
    assert board.tilt == -45.0
    assert a.tilt == pytest.approx(1500 - 45 * 10)


def test_us_per_deg_scales():
    a, board = turret_act(us_per_deg=20)
    a.move(1700, 1500)
    assert board.pan == 10.0


def test_aim_deg_and_point_at():
    a, board = turret_act()
    assert a.aim_deg(30, 15) == (30.0, 15.0)
    assert a.point_at(1.0, 0.0, 1.0) == pytest.approx((45.0, 0.0))
    assert a.point_at(0.0, 0.5, 1.0) == pytest.approx((0.0, 26.57), abs=0.01)
    with pytest.raises(ActuatorError):
        a.point_at(0.0, 0.0, -1.0)                    # behind: pan 180, outside servo_limits


def test_point_at_corrects_for_the_laser_offset():
    a, board = turret_act(laser_offset_m=[0, 0.05, 0])
    a.point_at(0.0, 0.0, 1.0)
    assert board.tilt == pytest.approx(-2.87, abs=0.01)   # aims down to cancel 5 cm of parallax


def test_laser_and_timeout():
    a, board = turret_act()
    a.laser(True)
    assert board.laser == 1
    a.clock.sleep(11)
    a.poll()
    assert board.laser == 0 and not a.laser_on


def test_close_turns_the_laser_off_parks_and_closes_the_port():
    a, board = turret_act()
    a.move(1800, 1600)
    a.laser(True)
    a.close()
    assert board.laser == 0
    assert (board.pan, board.tilt) == (0.0, 0.0)
    assert board.closed


def test_no_park_when_disabled():
    a, board = turret_act(park_on_close=False)
    a.move(1800, 1600)
    a.close()
    assert board.pan == 30.0 and board.closed


def test_laser_move_to_lands_on_target_through_anti_backlash():
    a, board = turret_act()
    las = Laser(a, frames=None, table=None, cal_path=None, cfg={"laser_anti_backlash_us": 25})
    las.move_to(1700, 1400)
    moves = board.commands("A")
    assert moves[-2] == "A 17.500 -12.500"            # approach from below by 2.5 deg
    assert (board.pan, board.tilt) == (20.0, -10.0)


def test_speed_and_accel_are_sent_when_configured():
    a, board = turret_act(max_speed_deg_s=200, accel_deg_s2=900)
    assert board.commands("V") == ["V 200.0"]
    assert board.commands("C") == ["C 900.0"]


def test_heartbeat_keeps_the_laser_alive(monkeypatch):
    monkeypatch.setattr(turret_mod, "HEARTBEAT_S", 0.02)
    t = Turret("/dev/fake", ser=FakeBoard())
    board = t._ser
    t.laser(True)
    time.sleep(0.2)
    assert len(board.commands("P")) >= 3
    t.close()
    assert board.laser == 0


def test_client_errors():
    t = Turret("/dev/fake", ser=FakeBoard())
    with pytest.raises(TurretError, match="ERR"):
        t._command("Q", "OK")
    with pytest.raises(TurretError, match="READY"):
        Turret("/dev/fake", boot_timeout_s=0.1, ser=FakeBoard(boot=b""))


def test_missing_board_falls_back_to_fake(monkeypatch):
    def Serial(*a, **k):
        raise OSError("no such port")

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=Serial))
    a, why = make_actuator_or_fake(CFG, clock=SimClock())
    assert type(a).__name__ == "FakeActuator"
    assert "turret" in why and "no such port" in why


@pytest.mark.parametrize("why", ["LASER TIMEOUT", "LASER MAX ON"])
def test_the_firmware_turning_the_laser_off_by_itself_is_noted(why):
    """firmware/turret: off after 1 s without a command, and 5 s after it was lit whatever the host sends."""
    board = FakeBoard()
    t = Turret("/dev/fake", ser=board)
    t.laser(True)
    assert t._laser_on
    board._say(why)
    t.position()                                  # any exchange reads the unsolicited line first
    assert not t._laser_on
    t.close()


def test_the_firmware_keeps_its_laser_safety():
    """The sketch the user flashes: LOW from reset, silence and max-on cut-offs, off on ERR, a watchdog."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "firmware" / "turret" / "turret.ino").read_text()
    assert "LASER_TIMEOUT_MS = 1000" in src and "LASER_MAX_ON_MS = 5000" in src
    assert "wdt_enable(WDTO_500MS)" in src and "wdt_reset();" in src
    setup = src[src.index("void setup()"):]
    assert setup.index("digitalWrite(LASER_PIN, LOW)") < setup.index("pinMode(LASER_PIN, OUTPUT)")
    assert src.count('Serial.println(F("ERR') + src.count('Serial.print(F("ERR') <= src.count("setLaser(false);")
