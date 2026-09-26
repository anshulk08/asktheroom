import math

import numpy as np
import pytest

import demo_check as dc
from core.config import load_config

CFG = load_config()


class Answers:
    """Scripted replies for Rig.ask."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


@pytest.fixture
def fake_rig():
    rig = dc.Rig(CFG, fake=True, manual=False)
    yield rig
    rig.close()


def test_fake_run_passes_everything(capsys):
    assert dc.main(["--fake", "--skip-manual"]) == 0
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 8 and "[SKIP] 8" in out and "[SKIP] 10" in out and "all checks passed" in out


def test_missing_camera_fails_every_check_that_needs_it(monkeypatch, capsys):
    def no_camera(self):
        raise RuntimeError("can't open camera 7")
    monkeypatch.setattr(dc.Rig, "_make_frames", no_camera)
    monkeypatch.setattr(dc.Rig, "_make_table", lambda self: self.part("frames"))
    assert dc.main(["--skip-manual", "--only", "1", "2", "3", "7"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len([ln for ln in lines if ln.startswith("[FAIL]") and "can't open camera 7" in ln]) == 4


def test_part_failure_is_cached(fake_rig):
    calls = []

    def boom():
        calls.append(1)
        raise OSError("no /dev/i2c-1")
    fake_rig._make_thing = boom
    for _ in range(2):
        with pytest.raises(OSError):
            fake_rig.part("thing")
    assert calls == [1]
    assert dc.run_check(fake_rig, lambda r: r.part("thing")) == (False, "OSError: no /dev/i2c-1")


def test_marker_drift_fails(fake_rig):
    ok, msg = dc.check_markers(fake_rig)
    assert ok, msg
    table = fake_rig.part("table")
    table._set(np.array([[1, 0, 2.0], [0, 1, 0], [0, 0, 1]]) @ table.H)      # table shifted 2 cm
    ok, msg = dc.check_markers(fake_rig)
    assert not ok and "2.00 cm" in msg and "recalibrate" in msg


def test_laser_uncalibrated_fails(fake_rig):
    laser = fake_rig.part("laser")
    ok, msg = dc.check_laser(fake_rig)
    assert ok, msg
    laser.fit = None
    ok, msg = dc.check_laser(fake_rig)
    assert not ok and "act.calibrate" in msg


class StubNet:
    host = "https://api.x.ai"

    def __init__(self, result):
        self.result, self.online, self.cbs = result, False, []

    def on_change(self, cb):
        self.cbs.append(cb)

    def check_once(self):
        changed, self.online = self.result != self.online, self.result
        for cb in self.cbs if changed else []:
            cb(self.online)
        return self.online


@pytest.mark.parametrize("state", [True, False])
def test_network_agrees(fake_rig, state):
    ok, msg = dc.check_network(fake_rig, StubNet(state), probe=lambda url: state)
    assert ok and ("online" if state else "offline") in msg


def test_network_disagreement_fails(fake_rig):
    ok, msg = dc.check_network(fake_rig, StubNet(True), probe=lambda url: False)
    assert not ok and "disagrees" in msg


def test_network_dashboard_not_updated_fails(fake_rig):
    net = StubNet(True)
    net.on_change = lambda cb: None                  # callback never wired
    ok, msg = dc.check_network(fake_rig, net, probe=lambda url: True)
    assert not ok and "dashboard" in msg


def audio_rig(signal, *replies):
    rig = dc.Rig(CFG, fake=True, manual=bool(replies), ask=Answers(*replies))
    rig.record = lambda s: signal(int(s * 16000))
    played = []
    rig.play = lambda pcm, rate: played.append((len(pcm), rate))
    return rig, played


def tone(n, amp=0.2):
    return (amp * np.sin(2 * np.pi * 220 * np.arange(n) / 16000)).astype(np.float32)


def test_audio_pass_and_tone_played():
    rig, played = audio_rig(tone, "", "y")
    ok, msg = dc.check_audio(rig)
    assert ok and "heard" in msg and played and played[0][1] == 22050


@pytest.mark.parametrize("signal, replies, why", [
    (lambda n: np.zeros(n, np.float32), (), "silence"),
    (lambda n: tone(n, 1.2).clip(-1, 1), (), "clipping"),
    (lambda n: tone(n, 0.001), ("", "y"), "too quiet"),
    (tone, ("", "n"), "not heard"),
])
def test_audio_failures(signal, replies, why):
    rig, _ = audio_rig(signal, *replies)
    ok, msg = dc.check_audio(rig)
    assert not ok and why in msg


def test_kill_switch_prompts(fake_rig):
    assert dc.check_kill_switch(fake_rig)[0] is None           # --fake: skipped
    fake_rig.part("laser")                                     # build the sim laser, then act real
    fake_rig.fake, fake_rig.manual = False, True
    for reply, want in (("y", True), ("", False)):
        fake_rig._ask = Answers("", reply)
        ok, _ = dc.check_kill_switch(fake_rig)
        assert ok is want
        assert fake_rig.part("laser").state["on"] is False     # always left off


def test_world_reports_objects_away_from_home(fake_rig):
    fake_rig.home_cm = dict(fake_rig.home_cm, keys=(5.0, 5.0))
    ok, msg = dc.check_world(fake_rig)
    assert not ok and "keys" in msg and "from home" in msg


def test_helpers():
    assert dc.dbfs(np.ones(100, np.float32)) == pytest.approx(0.0)
    assert dc.dbfs(np.zeros(100, np.float32)) == -math.inf
    s = dc.line(3, "markers", False, "why", color=False)
    assert s.startswith("[FAIL] 3 markers") and s.endswith("why")
    assert dc.line(8, "kill", None, "x", color=False).startswith("[SKIP]")
    locked, detail = dc.exposure_mode(0)
    if not dc.sys.platform.startswith("linux"):
        assert locked is None and "Linux" in detail


# ---------------------------------------------------------------- one-tag calibration

def tag_rig(tmp_path, tag_img, cal_img):
    """A rig whose table was calibrated from one AprilTag (cal_img); the camera now shows tag_img."""
    import sys
    from types import SimpleNamespace

    from core.table import Table
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = dict(CFG, table_tag=dict(CFG.get("table_tag") or {}, enabled=True, frames=3),
               paths=dict(CFG["paths"], table_cal=str(tmp_path / "table_cal.json")))
    t = Table(cfg)
    for _ in range(3):
        t.calibrate(cal_img)
    assert t.ok
    frames = SimpleNamespace(latest=lambda: SimpleNamespace(img=tag_img))
    return SimpleNamespace(part=lambda name: {"table": t, "frames": frames}[name])


@pytest.mark.skipif(tuple(int(v) for v in __import__("cv2").__version__.split(".")[:2]) < (4, 10),
                    reason="tag detection unreliable before OpenCV 4.10 (the app's container has 4.11)")
def test_one_tag_markers_check_passes_with_the_tag_removed_and_catches_drift(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import test_table as tt
    h = tt.true_h()
    cal = tt.render_tag(h)
    ok, msg = dc.check_markers(tag_rig(tmp_path / "a", tt.render_tag(h, present=False), cal))
    assert ok and "not in view" in msg                            # tag picked up after calibrating: fine
    ok, msg = dc.check_markers(tag_rig(tmp_path / "b", cal, cal))
    assert ok and "drift" in msg                                  # still in place: no drift
    moved = tt.render_tag(h @ np.array([[1, 0, 6.0], [0, 1, 0], [0, 0, 1]]))     # camera knocked 6 cm
    ok, msg = dc.check_markers(tag_rig(tmp_path / "c", moved, cal))
    assert not ok and "recalibrate" in msg


def test_clock_check_passes_on_a_set_clock_and_keeps_its_number():
    rig = dc.Rig(load_config(), fake=True)
    try:
        ok, msg = dc.check_clock(rig)
    finally:
        rig.close()
    assert ok is True and dc.CHECKS[8][0] == "clock" and len(dc.CHECKS) == 10


def test_room_check_skips_when_off_and_rehits_the_sim_map():
    cfg = load_config()
    rig = dc.Rig(cfg, fake=True)
    try:
        assert dc.check_room(rig)[0] is None
        rig.cfg = dict(cfg, room=dict(cfg["room"], enabled=True))
        ok, msg = dc.check_room(rig)
        assert ok, msg
        assert "re-hit" in msg and dc.CHECKS[-1][0] == "room"
        laser, _ = rig.part("room")
        laser.room_map.px[:, 0] += 80                    # the camera moved since the sweep
        laser.room_map._index()
        ok, msg = dc.check_room(rig)
        assert not ok and "sweep again" in msg
    finally:
        rig.close()
