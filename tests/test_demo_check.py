import math

import numpy as np
import pytest

import demo_check as dc
from core.config import load_config

CFG = load_config()


@pytest.fixture(autouse=True)
def _no_pulse(monkeypatch):
    """The askroom:audio container sets PULSE_SERVER; tests that want PulseAudio set it themselves."""
    monkeypatch.delenv("PULSE_SERVER", raising=False)


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


def test_fake_run_passes_everything(capsys, monkeypatch):
    monkeypatch.setattr(dc, "_xai_key", lambda: "")          # no real Grok call from the tests
    assert dc.main(["--fake", "--skip-manual"]) == 0
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 9 and "[SKIP] 8" in out and "[SKIP] 10" in out and "all checks passed" in out


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
    assert ok is True and dc.CHECKS[8][0] == "clock" and len(dc.CHECKS) == 16


def test_room_check_skips_when_off_and_rehits_the_sim_map():
    cfg = load_config()
    rig = dc.Rig(cfg, fake=True)
    try:
        assert dc.check_room(rig)[0] is None
        rig.cfg = dict(cfg, room=dict(cfg["room"], enabled=True))
        ok, msg = dc.check_room(rig)
        assert ok, msg
        assert "re-hit" in msg and dc.CHECKS[10][0] == "room"
        laser, _ = rig.part("room")
        laser.room_map.px[:, 0] += 80                    # the camera moved since the sweep
        laser.room_map._index()
        ok, msg = dc.check_room(rig)
        assert not ok and "sweep again" in msg
    finally:
        rig.close()


def test_the_mic_check_records_the_named_mic_like_the_voice_loop(monkeypatch):
    """sd.rec with the raw stt.input_device string failed on a shared name or a 48 kHz-only mic while the
    voice loop worked; a named mic that isn't there must fail the check, not pass on the default mic."""
    import numpy as np

    import voice.stt as stt
    cfg = dict(CFG, stt=dict(CFG.get("stt") or {}, input_device="PnP"))
    rig = dc.Rig(cfg, fake=False, manual=False)
    got = []
    monkeypatch.setattr(stt, "record_seconds", lambda s, spec: got.append((s, spec)) or np.zeros(int(s * 16000), np.float32))
    assert len(rig.record(0.5)) == 8000 and got == [(0.5, "PnP")]

    def missing(s, spec):
        raise RuntimeError(f"stt.input_device {spec!r}: no input device has that name")

    monkeypatch.setattr(stt, "record_seconds", missing)
    with pytest.raises(RuntimeError, match="no input device"):
        rig.record(0.5)
    rig.close()


class BlockingSoundDevice:
    """sounddevice whose streams never finish and whose stop() never returns (a wedged ALSA device)."""

    def __init__(self):
        self.gate, self.rec_device = __import__("threading").Event(), None

    def rec(self, n, samplerate, channels, dtype, device=None):
        self.rec_device = device
        return np.zeros((n, channels), np.float32)

    def get_stream(self):
        return type("S", (), {"active": True})()

    def stop(self):
        self.gate.wait()

    def wait(self):
        self.gate.wait()


def test_a_blocking_audio_device_cannot_hang_the_run(monkeypatch, capsys):
    import sys
    import time
    import voice.tts
    sd = BlockingSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(voice.tts, "play_pcm", lambda pcm, rate, device=None: sd.gate.wait())
    monkeypatch.setattr(voice.tts, "resolve_output_device", lambda spec: None)
    import voice.stt                                          # the mic path is voice.stt.record_seconds
    monkeypatch.setattr(voice.stt, "record_seconds", lambda seconds, spec=None: sd.gate.wait())
    monkeypatch.setattr(dc, "DEFAULT_DEADLINE_S", 1.0)
    monkeypatch.setattr(dc, "AUDIO_GRACE_S", 0.2)
    t0 = time.monotonic()
    try:
        assert dc.main(["--skip-manual", "--only", "5", "9"]) == 1
    finally:
        sd.gate.set()
    assert time.monotonic() - t0 < 5.0
    out = capsys.readouterr().out
    assert "[FAIL] 5 audio" in out and "timed out after 1 s" in out and "[PASS] 9 clock" in out


def test_tone_plays_on_the_configured_speaker_with_a_deadline(monkeypatch):
    import voice.tts
    gate = __import__("threading").Event()
    monkeypatch.setattr(dc, "AUDIO_GRACE_S", 0.1)
    rig = dc.Rig(dict(CFG, tts={"output_device": "Jabra"}), manual=False)
    try:
        played = []
        monkeypatch.setattr(voice.tts, "resolve_output_device", lambda spec: 7 if spec == "Jabra" else None)
        monkeypatch.setattr(voice.tts, "play_pcm", lambda pcm, rate, device=None: played.append(device))
        rig.play(np.zeros(2205, np.int16), 22050)
        assert played == [7]                                 # the configured speaker, not the default (HDMI)
        monkeypatch.setattr(voice.tts, "play_pcm", lambda pcm, rate, device=None: gate.wait())
        with pytest.raises(TimeoutError, match="test tone"):
            rig.play(np.zeros(2205, np.int16), 22050)
    finally:
        gate.set()
        rig.close()

def test_a_hung_part_fails_its_check_and_later_ones_fast(fake_rig):
    import threading
    gate = threading.Event()
    fake_rig._make_thing = lambda: gate.wait() or 1
    try:
        ok, msg = dc.run_check_with_deadline(fake_rig, "x", lambda r: r.part("thing"), timeout=0.3)
        assert ok is False and "timed out after 0 s" in msg and "wait" in msg
        ok, msg = dc.run_check_with_deadline(fake_rig, "y", lambda r: r.part("thing"), timeout=5)
        assert ok is False and "hung while starting" in msg
    finally:
        gate.set()


def test_a_timed_out_prompt_does_not_swallow_the_next_answer():
    import os
    r, w = os.pipe()
    rig = dc.Rig(CFG, fake=True, manual=True)
    rig.prompts = dc.PromptReader(os.fdopen(r))
    rig._ask = rig.prompts.ask
    try:
        ok, msg = dc.run_check_with_deadline(rig, "a", lambda rg: (True, rg.ask("first? ")), timeout=0.3)
        assert ok is False and "timed out" in msg
        os.write(w, b"yes\n")
        ok, msg = dc.run_check_with_deadline(rig, "b", lambda rg: (True, rg.ask("second? ")), timeout=3)
        assert (ok, msg) == (True, "yes")
    finally:
        os.close(w)
        rig.close()


def test_laser_off_now_does_not_wait_for_a_held_lock(fake_rig):
    import threading
    import time
    laser = fake_rig.part("laser")
    laser.act.laser(True)
    held, release = threading.Event(), threading.Event()

    def hog():
        with laser.act.lock:
            held.set()
            release.wait(10)
    threading.Thread(target=hog, daemon=True).start()
    held.wait(2)
    t0 = time.monotonic()
    try:
        fake_rig.laser_off_now()
        assert time.monotonic() - t0 < 1.0
        assert laser.act.laser_log[-1][1] is False                  # the hardware was told off
    finally:
        release.set()



# ---------------------------------------------------------------- room checks (spec 0010 P1-3)

class Resp:
    def __init__(self, status=200, js=None, content=b""):
        self.status_code, self._js, self.content = status, js, content

    def json(self):
        return self._js


def room_rig(**cfg_over):
    cfg = dict(CFG, room_memory=dict(CFG.get("room_memory") or {}, enabled=True), **cfg_over)
    return dc.Rig(cfg, manual=False)


def test_room_app_up_stalled_and_down():
    import time
    rig = room_rig()
    try:
        state = {"state": {"t": time.time(), "fps": 11.5, "room": {"keys": {"zone": "couch"}, "conflicts": []}}}
        urls = []

        def up(url, timeout):
            urls.append(url)
            return Resp(js=state) if url.endswith("/state") else Resp(content=b"\xff\xd8jpeg")
        ok, msg = dc.check_room_app(rig, get=up)
        assert ok and "11.5 fps" in msg and "room memory up (1 object" in msg, msg
        assert urls[0] == "http://127.0.0.1:8000/state"
        stale = {"state": {"t": time.time() - 60, "fps": 11.5}}
        ok, msg = dc.check_room_app(rig, get=lambda url, timeout: Resp(js=stale))
        assert not ok and "stalled" in msg
        ok, msg = dc.check_room_app(rig, get=lambda url, timeout: Resp(js=state) if url.endswith("/state")
                                    else Resp(404))
        assert not ok and "room memory isn't running" in msg

        def down(url, timeout):
            raise ConnectionError("refused")
        ok, msg = dc.check_room_app(rig, get=down)
        assert not ok and "scripts/room_app.sh start" in msg
    finally:
        rig.close()


def test_grok_reachable_refused_and_offline(monkeypatch):
    rig = room_rig()
    try:
        monkeypatch.setenv("XAI_API_KEY", "sk-SECRET")
        seen = {}

        def ok_get(url, headers, timeout):
            seen.update(url=url, auth=headers["Authorization"])
            return Resp(200)
        ok, msg = dc.check_grok(rig, get=ok_get)
        assert ok and "ms round trip" in msg and seen["url"].endswith("/models") and seen["auth"] == "Bearer sk-SECRET"
        ok, msg = dc.check_grok(rig, get=lambda url, headers, timeout: Resp(401))
        assert not ok and "refused the key" in msg and "SECRET" not in msg

        def offline(url, headers, timeout):
            raise OSError("no route")
        ok, msg = dc.check_grok(rig, get=offline)
        assert not ok and "hotspot" in msg
        monkeypatch.delenv("XAI_API_KEY")
        monkeypatch.chdir(dc.tempfile.mkdtemp())            # no .env either
        ok, msg = dc.check_grok(rig, get=ok_get)
        assert not ok and "no XAI_API_KEY" in msg
    finally:
        rig.close()


def test_grok_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OTHER=1\nexport XAI_API_KEY='sk-abc'\n")
    assert dc._xai_key() == "sk-abc"


DEVICES = [{"name": "HDMI 0", "max_input_channels": 0, "max_output_channels": 2},
           {"name": "Logitech BRIO: USB Audio", "max_input_channels": 2, "max_output_channels": 0},
           {"name": "Jabra SPEAK 410 USB", "max_input_channels": 1, "max_output_channels": 2}]


@pytest.mark.parametrize("mic, spk, ok, want", [
    ("jabra", "Jabra", True, "mic #2 Jabra"),
    (1, "jabra", True, "mic #1 Logitech"),
    ("jabra", None, False, "HDMI, silent"),
    ("anker", "jabra", False, "no input device named like 'anker'"),
    ("jabra", "brio", False, "no output device"),                 # the Brio has no speaker
])
def test_mic_and_speaker_are_found_by_name(mic, spk, ok, want):
    rig = dc.Rig(dict(CFG, stt={"input_device": mic}, tts={"output_device": spk}), manual=False)
    try:
        got, msg = dc.check_devices(rig, query=lambda: DEVICES)
        assert got is ok and want in msg, msg
    finally:
        rig.close()


def test_memory_and_log_tail(tmp_path):
    mi = tmp_path / "meminfo"
    log = tmp_path / "app.log"
    rig = dc.Rig(dict(CFG, room_check={"app_log": str(log), "log_lines": 50, "min_ram_mb": 1024}), manual=False)
    try:
        mi.write_text("MemTotal: 7800000 kB\nMemAvailable: 2097152 kB\n")
        ok, msg = dc.check_memory(rig, meminfo=str(mi))
        assert not ok and "no app log" in msg
        old = ["NvMapMemAlloc error 12 at startup"] + ["fine"] * 80          # scrolled out of the tail
        log.write_text("\n".join(old) + "\n")
        ok, msg = dc.check_memory(rig, meminfo=str(mi))
        assert ok and "2.0 GB available" in msg and "0 NvMapMemAlloc" in msg, msg
        with open(log, "a") as f:
            f.write("NvMapMemAlloc error 12\nTraceback (most recent call last):\n")
        ok, msg = dc.check_memory(rig, meminfo=str(mi))
        assert not ok and "1 NvMapMemAlloc, 1 tracebacks" in msg
        log.write_text("fine\n")
        mi.write_text("MemAvailable: 600000 kB\n")
        ok, msg = dc.check_memory(rig, meminfo=str(mi))
        assert not ok and "under 1.0 GB" in msg
    finally:
        rig.close()


def test_namer_queue_is_bounded(fake_rig):
    ok, msg = dc.check_namer(fake_rig)
    assert ok and "-> 8 kept" in msg, msg


def test_live_runs_only_the_checks_that_leave_the_camera(monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(dc, "CHECKS", [(n, (lambda n: lambda rig: (ran.append(n), (True, "ok"))[1])(n))
                                       for n, _ in dc.CHECKS])
    assert dc.main(["--live", "--skip-manual"]) == 0
    assert set(ran) == dc.LIVE_CHECKS and "camera" not in ran and "audio" not in ran


def test_app_url_follows_config_then_the_saved_port(tmp_path):
    args = tmp_path / "app.args"
    assert dc.app_url(CFG, str(args)) == "http://127.0.0.1:8000"
    args.write_text("--port\n8080\n--no-voice\n")
    assert dc.app_url(CFG, str(args)) == "http://127.0.0.1:8080"          # what room_app.sh started
    args.write_text("--port=9001\n")
    assert dc.app_url(CFG, str(args)) == "http://127.0.0.1:9001"
    cfg = dict(CFG, room_check={"app_url": "http://10.0.0.5:8080/"})
    assert dc.app_url(cfg, str(args)) == "http://10.0.0.5:8080"


def asound_tree(root, busy_mic=False):
    """A /proc/asound like the rig's: HDMI (playback), the Brio (capture only), a Jabra (both)."""
    (root / "cards").write_text(
        " 0 [HDA            ]: tegra-hda - NVIDIA Jetson Orin Nano HDA\n"
        "                      NVIDIA Jetson Orin Nano HDA at 0x3518000 irq 110\n"
        " 1 [BRIO           ]: USB-Audio - Logitech BRIO\n"
        "                      Logitech BRIO at usb-3610000.usb-2.3, super speed\n"
        " 2 [USB            ]: USB-Audio - Jabra SPEAK 410 USB\n"
        "                      Jabra SPEAK 410 USB at usb-3610000.usb-2.4, full speed\n")
    for card, pcms in ((0, ["pcm3p"]), (1, ["pcm0c"]), (2, ["pcm0c", "pcm0p"])):
        for pcm in pcms:
            sub = root / f"card{card}" / pcm / "sub0"
            sub.mkdir(parents=True)
            running = busy_mic and card == 2 and pcm.endswith("c")
            (sub / "status").write_text("state: RUNNING\nowner_pid   : 99\n" if running else "closed\n")


@pytest.mark.parametrize("mic, spk, busy, ok, want", [
    ("jabra", "jabra", True, True, "mic card 2 Jabra SPEAK 410 USB (in use by the app)"),
    ("BRIO", "Jabra", False, True, "mic card 1 Logitech BRIO"),
    ("jabra", "brio", False, False, "no output card named like 'brio'"),      # the Brio has no speaker
    ("anker", "jabra", False, False, "no input card named like 'anker'"),
    (2, "jabra", False, False, "indexes shift"),
    ("jabra", None, False, False, "HDMI, silent"),
])
def test_live_devices_come_from_alsa_and_a_busy_mic_passes(tmp_path, mic, spk, busy, ok, want):
    asound_tree(tmp_path, busy_mic=busy)
    rig = dc.Rig(dict(CFG, stt={"input_device": mic}, tts={"output_device": spk}), manual=False)
    rig.live = True
    try:
        got, msg = dc.check_devices(rig, asound=str(tmp_path))
        assert got is ok and want in msg, msg
    finally:
        rig.close()


def test_a_hung_mic_alone_times_out_check_5(monkeypatch, capsys):
    """Rig.record is voice.stt.record_seconds (room/voice): the per-check deadline still bounds a mic that
    never returns, with a speaker that works."""
    import threading
    import time
    import voice.stt
    import voice.tts
    gate = threading.Event()
    monkeypatch.setattr(voice.tts, "resolve_output_device", lambda spec: None)
    monkeypatch.setattr(voice.tts, "play_pcm", lambda pcm, rate, device=None: None)
    monkeypatch.setattr(voice.stt, "record_seconds", lambda seconds, spec=None: gate.wait())
    monkeypatch.setattr(dc, "DEFAULT_DEADLINE_S", 1.0)
    t0 = time.monotonic()
    try:
        assert dc.main(["--skip-manual", "--only", "5"]) == 1
    finally:
        gate.set()
    assert time.monotonic() - t0 < 5.0
    out = capsys.readouterr().out
    assert "[FAIL] 5 audio" in out and "timed out after 1 s" in out, out


# ---------------------------------------------------------------- PulseAudio (askroom:audio, Bluetooth speaker)

class FakePactl:
    """pactl list short sinks|sources and pactl info, as the host's PulseAudio would answer."""

    def __init__(self, sinks, sources=(), default_sink=None, default_source=None, fail=False):
        self.sinks, self.sources, self.fail = sinks, sources, fail
        self.info = f"Server Name: pulseaudio\nDefault Sink: {default_sink}\nDefault Source: {default_source}\n"
        self.calls = []

    def __call__(self, args):
        import subprocess
        self.calls.append(args)
        if self.fail:
            return subprocess.CompletedProcess(args, 1, "", "Connection failure: Connection refused")
        if args == ["info"]:
            return subprocess.CompletedProcess(args, 0, self.info, "")
        rows = self.sinks if args[-1] == "sinks" else self.sources
        out = "".join(f"{i}\t{n}\tmodule-x.c\ts16le 2ch 44100Hz\t{st}\n" for i, (n, st) in enumerate(rows))
        return subprocess.CompletedProcess(args, 0, out, "")


BT = "bluez_sink.00_42_79_AA_BB_CC.a2dp_sink"
HDMI = "alsa_output.platform-3510000.hda.hdmi-stereo"
USB_MIC = "alsa_input.usb-Jabra_SPEAK_410-00.mono-fallback"


def pulse_rig(tmp_path, mic="jabra", spk=None, **room_check):
    asound_tree(tmp_path)
    rig = dc.Rig(dict(CFG, stt={"input_device": mic}, tts={"output_device": spk},
                      room_check=dict(CFG.get("room_check") or {}, **room_check)), manual=False)
    rig.live = True
    return rig


@pytest.mark.parametrize("spk", [None, "pulse", "default"])
def test_bluetooth_speaker_behind_pulseaudio_passes(tmp_path, monkeypatch, spk):
    """Not an ALSA card: check 14 used to fail it. With PULSE_SERVER (scripts/dock.sh) the default sink is
    checked with pactl; the mic, a USB card, is still found in ALSA."""
    monkeypatch.setenv("PULSE_SERVER", "unix:/run/user/1000/pulse/native")
    rig = pulse_rig(tmp_path, spk=spk, pulse_sink="bluez")
    try:
        pactl = FakePactl([(HDMI, "SUSPENDED"), (BT, "RUNNING")], default_sink=BT)
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=pactl)
        assert ok is True and f"speaker via PulseAudio: {BT} (RUNNING)" in msg and "mic card 2 Jabra" in msg, msg
    finally:
        rig.close()


def test_pulse_default_sink_that_is_not_the_bluetooth_speaker_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_SERVER", "unix:/run/user/1000/pulse/native")
    rig = pulse_rig(tmp_path, spk="pulse", pulse_sink="bluez")
    try:
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=FakePactl([(HDMI, "IDLE")], default_sink=HDMI))
        assert ok is False and "not like 'bluez'" in msg and "bluetoothctl connect" in msg, msg
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=FakePactl([]))
        assert ok is False and "no PulseAudio sink" in msg and "Bluetooth speaker connected" in msg
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=FakePactl([], fail=True))
        assert ok is False and "pactl failed" in msg and "PULSE_SERVER right (unix:" in msg
    finally:
        rig.close()


def test_pulse_without_pactl_is_skipped_to_check_by_ear(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_SERVER", "unix:/run/user/1000/pulse/native")
    rig = pulse_rig(tmp_path, spk="pulse")
    try:
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=None)
        assert ok is True and "not checked (no pactl here), check it by ear" in msg and "mic card 2" in msg
        rig.cfg["stt"] = {"input_device": "pulse"}
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=None)
        assert ok is None and msg.count("check it by ear") == 2                 # nothing checked: SKIP
    finally:
        rig.close()


def test_pulse_mic_and_a_named_bluetooth_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_SERVER", "unix:/run/user/1000/pulse/native")
    rig = pulse_rig(tmp_path, mic="pulse", spk="bluez")
    try:
        pactl = FakePactl([(BT, "IDLE")], sources=[(BT.replace("sink", "sink") + ".monitor", "IDLE"), (USB_MIC, "RUNNING")],
                          default_sink=BT, default_source=USB_MIC)
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=pactl)
        assert ok is True and f"mic via PulseAudio: {USB_MIC} (RUNNING)" in msg and f"speaker via PulseAudio: {BT}" in msg
    finally:
        rig.close()


def test_without_pulse_server_a_null_speaker_still_fails(tmp_path, monkeypatch):
    """askroom:latest (no PulseAudio): null is the silent HDMI default, as before."""
    monkeypatch.delenv("PULSE_SERVER", raising=False)
    rig = pulse_rig(tmp_path, spk=None)
    try:
        pactl = FakePactl([(BT, "RUNNING")], default_sink=BT)
        ok, msg = dc.check_devices(rig, asound=str(tmp_path), pactl=pactl)
        assert ok is False and "HDMI, silent" in msg and not pactl.calls
    finally:
        rig.close()
