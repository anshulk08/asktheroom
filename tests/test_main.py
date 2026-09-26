import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from act.calibrate import calibrate
from act.sim import SimRig
from core.config import load_config
from core.events import EventLog
from core.fakeworld import demo_world
from core.types import Answer
from core.world import World
from server.app import create_app
from server.sim import SimTable
from voice.pipeline import make_ask

CFG = load_config()


@pytest.fixture(scope="module")
def cal_path():
    """One simulated-rig calibration shared by the tests (same seed -> same geometry)."""
    rig = SimRig(CFG)
    p = Path(tempfile.mkdtemp(prefix="askroom_test_")) / "laser_cal.json"
    calibrate(rig.make_laser(str(p)))
    return str(p)


class SpeakLog:
    def __init__(self, delay=0.0):
        self.said, self.stopped, self.delay, self.speaking = [], 0, delay, False

    def speak(self, text):
        self.speaking = True
        time.sleep(self.delay)
        self.said.append(text)
        self.speaking = False

    def stop(self):
        self.stopped += 1


def no_model(text, world, events, cfg, online=True):
    return Answer("local")


def make_room(tmp_path, cal=None, world=None, **kw):
    events = EventLog(":memory:", str(tmp_path))
    world = world or demo_world(events)
    rig = SimRig(CFG)
    laser = rig.make_laser(cal or str(tmp_path / "missing.json"))
    ask = make_ask(CFG, world, events, net=None, other=no_model)
    room = main.Room(CFG, world, events, SimTable(CFG), None, laser, ask, tts=SpeakLog(), **kw)
    room.laser_timeout_s = 60
    room.listen_mode = "click"                  # the always-on tests switch it back
    return room, rig


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_where_question_speaks_and_aims(tmp_path, cal_path):
    room, rig = make_room(tmp_path, cal_path)
    ans = room.ask_and_act("where is my wallet?", "dashboard")
    assert ans.point_at == "wallet"
    assert wait_for(lambda: room.tts.said and room.world.laser.get("target") == "wallet")
    assert room.tts.said == [ans.text]
    assert rig.act.writes, "fake actuator never moved"
    pos, _ = room.world.resolve("wallet")
    assert np.linalg.norm(np.subtract(rig.true_dot_cm(), pos)) < 3.0
    assert room.world.laser["on"] is True


def test_nested_object_aims_at_its_container(tmp_path, cal_path):
    room, rig = make_room(tmp_path, cal_path)
    ans = room.ask("where are my pills", "voice")
    assert ans.point_at == "pill_bottle"
    room.aim(ans)
    assert room.world.laser["target"] == "pill_bottle"
    pos, chain = room.world.resolve("pill_bottle")
    assert chain[-1] == "notebook"
    assert np.linalg.norm(np.subtract(rig.true_dot_cm(), pos)) < 3.0


def test_uncalibrated_laser_still_speaks(tmp_path):
    room, rig = make_room(tmp_path)          # no laser_cal.json
    room.ask_and_act("where is my wallet?", "dashboard")
    assert wait_for(lambda: room.tts.said)
    time.sleep(0.1)
    assert room.world.laser["on"] is False and not rig.act.writes


def test_sweep_and_circle_actions(tmp_path, cal_path):
    room, rig = make_room(tmp_path, cal_path)
    room.aim(Answer("carried off the left", point_at="phone", action="sweep:left"))
    assert room.world.laser["target"] == "edge:left"
    n = len(rig.act.writes)
    room.aim(Answer("lost track", point_at="glasses", action="circle"))
    assert room.world.laser["target"] == "glasses" and len(rig.act.writes) > n
    assert room.aim(Answer("bad edge", action="sweep:up")) is None      # ValueError is logged, not raised


def test_answers_without_target_do_not_move_laser(tmp_path, cal_path):
    room, rig = make_room(tmp_path, cal_path)
    assert room.aim(Answer("Nothing has changed.")) is None
    assert not rig.act.writes


def test_sms_answers_without_speaking_or_aiming(tmp_path, cal_path):
    room, rig = make_room(tmp_path, cal_path)
    ans = room.ask_and_act("where is my wallet?", "sms")
    time.sleep(0.2)
    assert ans.point_at == "wallet" and room.tts.said == [] and not rig.act.writes


def test_reset_question_resets_world(tmp_path, cal_path):
    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    world.entities["keys"].pos_cm = (10.0, 10.0)
    room, _ = make_room(tmp_path, cal_path, world=world)
    room.ask("reset the table", "voice")
    assert world.entities["keys"].pos_cm is None


def test_laser_shown_off_after_timeout(tmp_path, cal_path):
    room, _ = make_room(tmp_path, cal_path)
    room.laser_timeout_s = 0.2
    room.aim(Answer("x", point_at="wallet", action="point"))
    assert room.world.laser["on"] is True
    assert wait_for(lambda: room.world.laser["on"] is False, 2.0)


class FakeClicker:
    kind = "keyboard"

    def __init__(self):
        self.presses = threading.Semaphore(0)

    def wait_press(self, timeout=None):
        return self.presses.acquire(timeout=timeout)

    def pressed(self):
        return self.presses.acquire(blocking=False)

    def press(self):
        self.presses.release()

    def clear(self):
        pass


class FakeSTT:
    def __init__(self, text, overheard=()):
        self.text, self.last_ms, self.log_text = text, {}, True
        self.overheard, self.last_stop = list(overheard), "silence"

    def listen(self):
        return self.text

    def hear(self, idle_s):
        """Always-on mic: one scripted utterance per call, then nobody talks."""
        if self.overheard:
            return self.overheard.pop(0)
        time.sleep(0.02)
        return ""


def always_on(room):
    room.listen_mode = "always"
    t = threading.Thread(target=room.voice_loop, daemon=True)
    t.start()
    return t


def stop_voice(room, t):
    room.stop_ev.set()
    t.join(2)


def test_voice_loop_click_to_answer_and_laser(tmp_path, cal_path):
    clicker = FakeClicker()
    room, rig = make_room(tmp_path, cal_path, stt=FakeSTT("where's my wallet"), clicker=clicker)
    t = threading.Thread(target=room.voice_loop, daemon=True)
    t.start()
    clicker.presses.release()
    assert wait_for(lambda: room.last_timing and room.tts.said)
    assert room.tts.stopped == 1 and "wallet" in room.tts.said[0].lower()
    assert room.world.laser["target"] == "wallet"
    assert room.last_timing["click_to_laser_s"] < 3.0
    room.stop_ev.set()
    t.join(2)


def test_voice_loop_nothing_heard(tmp_path, cal_path):
    clicker = FakeClicker()
    room, rig = make_room(tmp_path, cal_path, stt=FakeSTT(""), clicker=clicker)
    t = threading.Thread(target=room.voice_loop, daemon=True)
    t.start()
    clicker.presses.release()
    assert wait_for(lambda: room.tts.said)
    assert "didn't catch" in room.tts.said[0] and not rig.act.writes
    room.stop_ev.set()
    t.join(2)


def test_voice_loop_reports_each_question_to_n8n(tmp_path, cal_path, monkeypatch):
    posted = []
    monkeypatch.setattr(main.requests, "post", lambda url, json, timeout, headers=None: posted.append((url, json)))
    clicker = FakeClicker()
    room, rig = make_room(tmp_path, cal_path, stt=FakeSTT("where's my wallet"), clicker=clicker)
    room.webhook_url = "http://laptop:5678/webhook/ask-the-room"
    t = threading.Thread(target=room.voice_loop, daemon=True)
    t.start()
    clicker.presses.release()
    assert wait_for(lambda: posted)
    url, q = posted[0]
    assert url == room.webhook_url and q["heard"] == "where's my wallet"
    assert (q["intent"], q["object"], q["understood_by"], q["point_at"]) == ("WHERE", "wallet", "rules", "wallet")
    assert "wallet" in q["answer"].lower() and q["click_to_laser_s"] < 3.0
    room.stop_ev.set()
    t.join(2)


def test_always_listening_answers_questions_and_drops_chatter(tmp_path, cal_path, monkeypatch):
    posted = []
    monkeypatch.setattr(main.requests, "post",
                        lambda url, json, timeout, headers=None: posted.append((json, headers)))
    stt = FakeSTT("", overheard=["I'll grab my keys on the way out", "we built this last night",
                                 "okay so where's my wallet"])
    room, rig = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    room.webhook_url, room.webhook_token = "http://n8n/webhook/ask-the-room", "s3cret"
    t = always_on(room)
    assert wait_for(lambda: posted)
    q, headers = posted[0]
    assert (q["heard"], q["mode"], q["ignored_since_last"]) == ("okay so where's my wallet", "overheard", 2)
    assert q["point_at"] == "wallet" and q["speech_end_to_laser_s"] < 3.0 and "click_to_laser_s" not in q
    assert headers == {"x-askroom-token": "s3cret"} and stt.log_text is False
    assert room.tts.said == [q["answer"]] and room.world.laser["target"] == "wallet"
    assert room.events._rows("SELECT text FROM questions") == [("okay so where's my wallet",)]
    stop_voice(room, t)


def test_always_listening_waits_while_the_rig_speaks(tmp_path, cal_path):
    stt = FakeSTT("", overheard=["where is my wallet", "where are my keys"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    room.tts = SpeakLog(delay=0.3)
    heard_while_speaking = []
    hear = stt.hear
    stt.hear = lambda idle_s: (heard_while_speaking.append(room.tts.speaking), hear(idle_s))[1]
    t = always_on(room)
    assert wait_for(lambda: len(room.tts.said) == 2)
    assert not any(heard_while_speaking)
    assert "wallet" in room.tts.said[0].lower() and "keys" in room.tts.said[1].lower()
    stop_voice(room, t)


def test_clicker_is_listen_now_in_always_mode(tmp_path, cal_path):
    clicker = FakeClicker()
    room, _ = make_room(tmp_path, cal_path, stt=FakeSTT("show me my keys"), clicker=clicker)
    clicker.press()
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    assert room.tts.stopped == 1 and room.tts.said == ["local"]    # asked: no overheard filter
    stop_voice(room, t)


def test_no_webhook_no_post(tmp_path, cal_path, monkeypatch):
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: pytest.fail("posted"))
    room, _ = make_room(tmp_path, cal_path)
    assert room.webhook_url == ""
    room.report({"heard": "x"})


def test_dashboard_ask_route_moves_fake_laser(tmp_path, cal_path):
    room, rig = make_room(tmp_path, cal_path)
    app = create_app(CFG, room.world, room.events, frames=None, ask_fn=room.ask_and_act, table=room.table)
    with TestClient(app) as c:
        r = c.post("/ask", json={"text": "where is my wallet?"})
        assert r.status_code == 200 and r.json()["point_at"] == "wallet"
        assert wait_for(lambda: room.world.laser.get("target") == "wallet")


def test_perception_loop_feeds_world(tmp_path):
    from core.types import Detection, Detections, Frame

    class Frames:
        def __init__(self):
            self.i = 0

        def wait_new(self, after, timeout=1.0):
            self.i += 1
            return Frame(t=time.monotonic(), wall=time.time(), img=None, idx=self.i)

    class Det:
        def detect(self, f):
            d = Detection("wallet", 0.9, (100, 100, 150, 150), (60.0, 15.0), (58, 13, 62, 17))
            return Detections(t=f.t, frame_idx=f.idx, items=[d], hands=[])

    class Hands:
        def update(self, hands, t):
            return hands

    class Table:
        ok = True

    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    room = main.Room(CFG, world, events, Table(), Frames(), None, None, detector=Det(), hands=Hands())
    room.max_fps = 200
    t = threading.Thread(target=room.perception_loop, daemon=True)
    t.start()
    assert wait_for(lambda: str(world.get("wallet").status) == "VISIBLE")
    room.stop_ev.set()
    t.join(2)


def test_build_fake_runs_without_hardware(monkeypatch, tmp_path):
    import net
    import voice.understand
    monkeypatch.setattr(net.NetMonitor, "probe", lambda self: False)
    monkeypatch.setattr(voice.understand.Qwen, "health", lambda self, timeout=1.0: False)   # no llama-server
    room, perception = main.build(CFG, fake=True, with_voice=False)
    try:
        assert perception is False and room.laser.fit is not None
        assert wait_for(lambda: room.frames.latest() is not None)
        room.tts = SpeakLog()
        room.ask_and_act("where is my wallet?", "dashboard")
        assert wait_for(lambda: room.tts.said)
    finally:
        room.shutdown()


def test_reset_words_in_a_reminder_do_not_act_on_the_room(tmp_path, cal_path):
    """RESET / RECAL act only when the router answered that intent: a reminder the care layer handled
    ('remind me to reset the router') leaves the world alone, and costs no second interpret call."""
    from voice.care import attach_care
    from voice.understand import Understander
    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    world.entities["keys"].pos_cm = (10.0, 10.0)
    rules = Understander(dict(CFG, understand={"enabled": False}))
    calls = []
    room, _ = make_room(tmp_path, cal_path, world=world,
                        interpret=lambda text, overheard=False: calls.append(text) or rules(text, overheard))
    attach_care(room, CFG)
    recal = []
    room.recalibrate = lambda: recal.append(1)
    for text in ("Room, remind me to reset the router at 5", "remind me to recalibrate the laser at 5"):
        assert "remind" in room.ask(text, "voice").text.lower()
    time.sleep(0.1)
    assert world.entities["keys"].pos_cm == (10.0, 10.0) and recal == [] and calls == []
    room.ask("reset the table", "voice")                  # through the care layer to the router: acts
    assert world.entities["keys"].pos_cm is None
    room.ask("recalibrate the laser", "voice")
    assert wait_for(lambda: recal == [1])


def test_reset_clears_the_detectors_proposals_and_crops_on_the_perception_thread(tmp_path):
    """RESET recaptures the proposer's empty-table reference and forgets crops (Detector.reset_proposals),
    on the perception thread before its next detect (the proposer isn't thread-safe)."""
    from core.types import Detections, Frame
    calls = []

    class Frames:
        def __init__(self):
            self.i = 0

        def wait_new(self, after, timeout=1.0):
            time.sleep(0.01)
            self.i += 1
            return Frame(t=time.monotonic(), wall=time.time(), img=None, idx=self.i)

    class Det:
        def detect(self, f):
            calls.append((threading.current_thread().name, "detect"))
            return Detections(t=f.t, frame_idx=f.idx, items=[], hands=[])

        def reset_proposals(self):
            calls.append((threading.current_thread().name, "reset"))

    class Hands:
        def update(self, hands, t):
            return hands

        def reset(self):
            pass

    class Table:
        ok = True

    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    ask = make_ask(CFG, world, events, net=None, other=no_model)
    room = main.Room(CFG, world, events, Table(), Frames(), None, ask, detector=Det(), hands=Hands())
    t = threading.Thread(target=room.perception_loop, name="perception", daemon=True)
    t.start()
    assert wait_for(lambda: calls)
    room.ask("where are my keys", "voice")
    time.sleep(0.05)
    assert all(c[1] == "detect" for c in calls)
    room.ask("reset the table", "voice")
    assert wait_for(lambda: ("perception", "reset") in calls)
    room.stop_ev.set()
    t.join(2)
    i = calls.index(("perception", "reset"))
    assert calls[i + 1:i + 2] in ([], [("perception", "detect")]) and calls.count(("perception", "reset")) == 1


def test_voice_answers_are_logged_for_the_phone(tmp_path):
    room, _ = make_room(tmp_path)
    logged = []
    room.record_answer = lambda q, ans, src: logged.append((q, ans.text, src))
    room._answer("where is my wallet?", time.monotonic(), time.monotonic(), {"mode": "asked"})
    assert logged and logged[0][0] == "where is my wallet?" and logged[0][2] == "voice"


def test_a_phone_question_heard_by_the_mic_is_answered_once(tmp_path):
    """P1: the judge dictates into the phone next to the rig; the always-on mic hears it too."""
    room, _ = make_room(tmp_path)
    room.ask_and_act("Where is my wallet?", "phone")
    assert wait_for(lambda: len(room.tts.said) == 1)
    room._answer("where is my wallet", time.monotonic(), time.monotonic(), {"mode": "overheard"})
    room._answer("where's my wallet?", time.monotonic(), time.monotonic(), {"mode": "overheard"})
    time.sleep(0.2)
    assert len(room.tts.said) == 1
    room._answer("where are my keys?", time.monotonic(), time.monotonic(), {"mode": "overheard"})
    assert wait_for(lambda: len(room.tts.said) == 2)                      # a different question still counts


def test_the_same_question_by_voice_after_the_window_is_answered(tmp_path, monkeypatch):
    room, _ = make_room(tmp_path)
    now = [100.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    room.ask_and_act("where is my wallet?", "phone")
    now[0] += 3.5
    room._answer("where is my wallet?", now[0], now[0], {"mode": "overheard"})
    assert wait_for(lambda: len(room.tts.said) == 2)


class FlipNet:
    def __init__(self, online=False):
        self.online, self.cbs = online, []

    def on_change(self, cb):
        self.cbs.append(cb)

    def flip(self, online):
        self.online = online
        for cb in self.cbs:
            cb(online)


def test_grok_is_warmed_when_the_network_comes_up_and_again_after_a_drop():
    warmed = []
    net = FlipNet(online=False)
    main.warm_on_connect(net, lambda: warmed.append(1))
    time.sleep(0.05)
    assert warmed == []                                 # offline at start: nothing to warm
    net.flip(True)
    assert wait_for(lambda: len(warmed) == 1)
    net.flip(False)
    time.sleep(0.05)
    assert len(warmed) == 1
    net.flip(True)
    assert wait_for(lambda: len(warmed) == 2)


def test_grok_is_warmed_at_start_when_already_online_and_a_slow_warm_never_blocks_the_monitor():
    warmed = []
    net = FlipNet(online=True)
    t0 = time.monotonic()
    main.warm_on_connect(net, lambda: (time.sleep(0.5), warmed.append(1)))
    net.flip(True)
    assert time.monotonic() - t0 < 0.2                  # the monitor thread is never held up
    assert wait_for(lambda: len(warmed) >= 1)
