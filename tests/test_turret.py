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
    LIMITS = ((-90.0, 90.0), (-90.0, 90.0))    # turret.ino MIN_DEG / MAX_DEG (travel)
    LASER_TILT_MAX = -10.0                      # turret.ino LASER_TILT_MAX_DEG

    def __init__(self, boot: bytes = b"READY turret\n"):
        self.out = deque([boot] if boot else [])
        self.pan, self.tilt = 0.0, -20.0     # aimed down already: only the tilt-rule test starts level
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
                if nums[0] != 0 and self.tilt > self.LASER_TILT_MAX:
                    self.laser = 0
                    self._say("ERR L above the laser tilt limit")
                    continue
                self.laser = int(nums[0] != 0)
                self._say(f"OK L {self.laser}")
            elif op == "P":
                self._say(f"POS {self.pan:.2f} {self.tilt:.2f} 0 {self.laser}")
            elif op in ("V", "C"):
                self._say("OK V 120.0 120.0 C 600.0")
            elif op == "E":
                self.enabled = int(nums[0] != 0) if nums else 1
                self._say(f"OK E {self.enabled}")
            elif op == "Z":
                self.pan = self.tilt = 0.0
                self._say("OK Z")
            elif op == "X":
                self.laser = 0
                self._say("OK X")
            else:
                self._say(f"ERR unknown {op}")

    def reset(self) -> None:
        """The Uno rebooting mid-run: laser LOW, wherever it stands is 0,0 again."""
        self.laser, self.pan, self.tilt = 0, 0.0, 0.0
        self._say("READY turret")

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

    def Serial(port, baud, timeout, exclusive=False):
        opened.append((port, baud, exclusive))
        return board

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=Serial))
    a = make_actuator(CFG, clock=SimClock())
    assert isinstance(a, TurretActuator)
    assert opened == [("/dev/fake", 115200, True)]   # exclusive: a second opener would reset the Uno
    assert (a.pan, a.tilt) == (1500.0, 1500.0)       # the firmware's 0,0 at power-up
    a.close()


def test_pulses_map_to_degrees():
    a, board = turret_act()
    a.move(1600, 1350)
    assert board.commands("A")[-1] == "A 10.000 -15.000"
    assert (board.pan, board.tilt) == (10.0, -15.0)
    assert (a.pan, a.tilt) == (1600.0, 1350.0)


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
    assert board.tilt == -90.0
    assert a.tilt == pytest.approx(1500 - 90 * 10)


def test_us_per_deg_scales():
    a, board = turret_act(us_per_deg=20)
    a.move(1700, 1500)
    assert board.pan == 10.0


def test_aim_deg_and_point_at():
    a, board = turret_act()
    assert a.aim_deg(30, -15) == (30.0, -15.0)                        # below level: the tilt cap is -10
    assert a.point_at(1.0, -0.5, 1.0) == pytest.approx((45.0, -19.47), abs=0.01)
    assert a.point_at(0.0, -0.5, 1.0) == pytest.approx((0.0, -26.57), abs=0.01)
    with pytest.raises(ActuatorError):
        a.point_at(0.0, 0.0, -1.0)                    # behind: pan 180, outside servo_limits


def test_point_at_corrects_for_the_laser_offset():
    a, board = turret_act(laser_offset_m=[0, 0.05, 0])
    a.point_at(0.0, -0.5, 1.0)                        # 5 cm of parallax: a bit lower than the straight -26.57
    assert board.tilt < -26.57


def test_laser_and_timeout():
    a, board = turret_act()
    assert a.laser_timeout_s == 4.0                   # CFG's 10 s capped at laser_room.max_on_s
    a.laser(True)
    assert board.laser == 1
    a.clock.sleep(4.1)
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


class ErrLaserBoard(FakeBoard):
    """'L' fails: the board answers ERR, or the port raises mid-write."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def write(self, data: bytes) -> None:
        if data.startswith(b"L"):
            if self.mode == "raise":
                raise OSError("write failed")
            self.lines.append(data.decode().strip())
            self._say("ERR laser")
            return
        super().write(data)


@pytest.mark.parametrize("mode", ["err", "raise"])
def test_close_does_not_park_when_the_laser_off_fails(mode):
    """Swinging the head with the laser possibly lit could sweep it across someone's eyes."""
    a, board = turret_act(ErrLaserBoard(mode))
    a.move(1800, 1350)
    with pytest.raises(IOError):
        a.close()
    assert (board.pan, board.tilt) == (30.0, -15.0)   # not parked
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


def test_a_board_reset_mid_run_locks_out_the_laser_and_moves(caplog):
    """An unexpected READY: the Uno re-zeroed wherever it stood, so no aim can be trusted until restart."""
    board = FakeBoard()
    t = Turret("/dev/fake", ser=board)
    t.laser(True)
    board.reset()                                 # USB glitch / watchdog reset / another opener
    t.position()                                  # any exchange reads the unsolicited line first
    assert t.reset_seen and not t._laser_on
    assert "RESET" in caplog.text
    with pytest.raises(TurretError, match="laser locked out"):
        t.laser(True)
    with pytest.raises(TurretError, match="board reset"):
        t.aim(10, 0)
    with pytest.raises(TurretError, match="board reset"):
        t.nudge(1, 0)
    assert board.commands("A") == [] and board.commands("R") == []
    assert board.laser == 0
    t.laser(False)                                # switching off always works
    t.close()
    assert board.closed


def test_a_reset_during_the_laser_on_exchange_switches_it_back_off():
    class ResetOnLaserBoard(FakeBoard):
        def write(self, data: bytes) -> None:
            if data == b"L 1\n":
                self.reset()                      # the fresh board then takes the L 1 ...
                self.tilt = -20.0                 # ... (aimed down: the host's lockout, not the tilt rule)
            super().write(data)

    board = ResetOnLaserBoard()
    t = Turret("/dev/fake", ser=board)
    with pytest.raises(TurretError, match="laser locked out"):
        t.laser(True)
    assert board.laser == 0 and board.lines[-1] == "L 0" and not t._laser_on
    t.close()


def test_the_turret_actuator_refuses_after_a_board_reset():
    a, board = turret_act()
    board.reset()
    a._turret.position()
    with pytest.raises(TurretError):
        a.laser(True)
    with pytest.raises(TurretError):
        a.move(1600, 1500)
    assert board.laser == 0 and board.commands("A") == []
    a.close()                                     # L 0 goes through; the park is refused and logged
    assert board.closed


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


def test_the_laser_pin_is_one_fact_in_the_firmware_and_the_docs():
    """The module is wired to D8. Every cut-off in the firmware switches LASER_PIN: if it named another pin,
    they'd all switch an empty pin while the laser stayed lit."""
    import re
    from pathlib import Path
    from act.turret import LASER_PIN
    root = Path(__file__).resolve().parents[1]
    ino = (root / "firmware" / "turret" / "turret.ino").read_text()
    assert int(re.search(r"const uint8_t LASER_PIN = (\d+);", ino).group(1)) == LASER_PIN
    for doc in ("firmware/README.md", "docs/LASER_SAFETY.md"):
        text = (root / doc).read_text()
        pins = set(re.findall(r"\bD(\d+)\b", text))
        laser_lines = [ln for ln in text.splitlines() if "laser" in ln.lower() and re.search(r"\bD\d+\b", ln)]
        assert laser_lines and all(f"D{LASER_PIN}" in ln for ln in laser_lines), (doc, laser_lines, pins)


def test_the_beam_never_tilts_above_level_or_the_configured_cap():
    from act.pointing import FIRMWARE_LIMITS_DEG
    from act.turret import TILT_MAX_DEG
    from pathlib import Path
    ino = (Path(__file__).resolve().parents[1] / "firmware" / "turret" / "turret.ino").read_text()
    assert "const float LASER_TILT_MAX_DEG = -10;" in ino and FIRMWARE_LIMITS_DEG[1][1] == TILT_MAX_DEG == -10.0
    assert "if (on && tiltTooHigh())" in ino and 'println(F("LASER TILT"))' in ino
    a, _ = turret_act()                                               # servo_limits tilt up to +45 deg
    assert a.limits_deg()[1][1] == pytest.approx(-10.0)               # capped at tilt_max_deg
    a.move(1500, 1950)
    assert a.us_to_deg(a.tilt) <= -10.0 + 1e-9
    b, _ = turret_act(tilt_max_deg=5)                                 # a config above level is still level
    assert b.limits_deg()[1][1] <= -10.0


def test_the_firmware_never_lights_the_beam_above_its_tilt_limit_but_parks_level_dark():
    board = FakeBoard()
    t = Turret("/dev/fake", ser=board)
    t.aim(0.0, 0.0)                                   # level, dark: allowed (the park)
    with pytest.raises(TurretError):
        t.laser(True)
    assert board.laser == 0
    t.aim(0.0, -20.0)
    t.laser(True)
    assert board.laser == 1
    t.laser(False)
    t.close()


def test_rehome_without_the_battery_releases_then_zeroes_through_the_apps_port():
    """Switching the stepper battery rebooted the Jetson twice: re-home with E 0 / Z / E 1 instead."""
    import main
    from act.laser import Laser
    a, board = turret_act()
    laser = Laser(a, None, None, "", cfg=dict(CFG))
    room = main.Room({"frame_size_px": [1280, 720]}, type("W", (), {"laser": {}})(), None, None, None, laser, None)
    room.laser_locked = "the laser's zero moved"
    assert room.laser_rehome("release")["ok"] and board.enabled == 0 and board.laser == 0
    assert room.laser_locked.startswith("re-homing")
    board.pan, board.tilt = 12.0, -7.0                    # the user turns the free head level by hand
    refused = room.laser_rehome("zero")                   # no confirmation: nothing zeroed, still locked
    assert not refused["ok"] and (board.pan, board.tilt) == (12.0, -7.0) and room.laser_locked
    r = room.laser_rehome("zero", level_confirmed=True)
    assert r["ok"] and r["locked"] is None and (board.pan, board.tilt) == (0.0, 0.0) and board.enabled == 1
    assert not room.laser_rehome("spin")["ok"]


def test_the_rehome_endpoint_answers_only_this_machine():
    from fastapi.testclient import TestClient
    from core.fakeworld import demo_world
    from server.app import create_app
    calls = []
    app = create_app({"server": {}}, demo_world(), None,
                     rehome_fn=lambda step, ok, dp=0.0, dt=0.0: calls.append((step, ok)) or {"ok": True})
    far = TestClient(app, client=("10.90.84.50", 5000))
    assert far.post("/laser/rehome", json={"step": "release"}).status_code == 403 and calls == []
    near = TestClient(app, client=("127.0.0.1", 5000))
    assert near.post("/laser/rehome", json={"step": "zero", "level_confirmed": True}).json() == {"ok": True}
    assert calls == [("zero", True)]
    near.post("/laser/rehome", json={"step": "zero", "level_confirmed": "false"})      # a string is no confirmation
    assert calls[-1] == ("zero", False)
    r = near.post("/laser/rehome", json={"step": "release"}, headers={"X-Forwarded-For": "10.0.0.9"})
    assert r.status_code == 403 and len(calls) == 2                                   # a tunnel posing as local


def test_rehome_jogs_the_head_dark_within_limits_then_zeroes_without_releasing():
    """Closed-loop drivers undo a hand move on re-enable: jog by the motors instead, then zero there."""
    import main
    from act.laser import Laser
    a, board = turret_act()
    laser = Laser(a, None, None, "", cfg=dict(CFG))
    room = main.Room({"frame_size_px": [1280, 720]}, type("W", (), {"laser": {}})(), None, None, None, laser, None)
    board.pan, board.tilt = 3.0, -20.0                     # the head points down after a bad zero
    assert not room.laser_rehome("jog", dpan=0, dtilt=15)["ok"]           # more than 10 deg at once
    assert not room.laser_rehome("jog", dpan="up", dtilt=0)["ok"]
    for _ in range(2):
        r = room.laser_rehome("jog", dpan=-1.5, dtilt=10)
        assert r["ok"] and room.laser_locked and board.laser == 0
    assert (board.pan, board.tilt) == (0.0, 0.0)                         # 2 x (-1.5, +10) from (3, -20)
    assert not room.laser_rehome("jog", dpan=0, dtilt=10.5)["ok"]         # over the step limit
    board.tilt = 5.0
    assert not room.laser_rehome("jog", dpan=0, dtilt=6)["ok"]            # would leave the range above +10
    r = room.laser_rehome("zero", level_confirmed=True)
    assert r["ok"] and room.laser_locked is None and (board.pan, board.tilt) == (0.0, 0.0)


def test_the_rehome_endpoint_passes_jog_steps_through():
    from fastapi.testclient import TestClient
    from core.fakeworld import demo_world
    from server.app import create_app
    calls = []
    app = create_app({"server": {}}, demo_world(), None,
                     rehome_fn=lambda step, ok, dp=0.0, dt=0.0: calls.append((step, ok, dp, dt)) or {"ok": True})
    near = TestClient(app, client=("127.0.0.1", 5000))
    near.post("/laser/rehome", json={"step": "jog", "dpan": -2, "dtilt": 5})
    assert calls == [("jog", False, -2, 5)]
