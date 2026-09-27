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


def test_slow_answer_gets_a_thinking_cue_first(tmp_path, cal_path, monkeypatch):
    posted = []
    monkeypatch.setattr(main.requests, "post", lambda url, json, timeout, headers=None: posted.append(json))
    room, rig = make_room(tmp_path, cal_path)
    room.webhook_url, room.cue_after_s = "http://laptop/webhook", 0.05
    fast = room.base_ask

    def slow(text, source):
        time.sleep(0.3)
        return fast(text, source)

    room.base_ask = slow
    room._answer("where's my wallet", time.monotonic(), time.monotonic(), {"mode": "asked"})
    assert wait_for(lambda: len(room.tts.said) == 2)
    assert room.tts.said[0] == "Let me look." and "wallet" in room.tts.said[1].lower()
    assert wait_for(lambda: posted) and posted[0]["thinking_cue"] is True


def test_quick_answer_gets_no_cue(tmp_path, cal_path):
    room, rig = make_room(tmp_path, cal_path)
    room._answer("where's my wallet", time.monotonic(), time.monotonic(), {"mode": "asked"})
    assert wait_for(lambda: room.tts.said) and len(room.tts.said) == 1 and "wallet" in room.tts.said[0].lower()
    room.cue_after_s = 0                        # off
    assert room._ask_with_cue("where's my wallet")[1] is False


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
    room, _ = make_room(tmp_path, cal_path, stt=FakeSTT("tell me a joke"), clicker=clicker)
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


def test_perceive_is_one_perception_step_after_the_table_is_calibrated(tmp_path):
    """Room.perceive(frame) is the perception loop's body (eval.score_clip replays clips through it):
    no step until the table calibrates from a frame, then detect -> hand ids -> world, returning both."""
    from core.types import Detection, Detections, Frame

    class Det:
        def detect(self, f):
            d = Detection("wallet", 0.9, (100, 100, 150, 150), (60.0, 15.0), (58, 13, 62, 17))
            h = Detection("hand", 0.8, (400, 400, 500, 500), (40.0, 40.0), (35, 35, 45, 45))
            return Detections(t=f.t, frame_idx=f.idx, items=[d], hands=[h])

    class Hands:
        def update(self, hands, t):
            return [Detection("hand:7", h.conf, h.box_px, h.center_cm, h.box_cm) for h in hands]

    class Table:
        ok = False

        def calibrate(self, img):
            self.ok = img is not None
            return self.ok

    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    room = main.Room(CFG, world, events, Table(), None, None, None, detector=Det(), hands=Hands())
    assert room.perceive(Frame(t=0.0, wall=1000.0, img=None, idx=1)) is None      # calibrating
    out = None
    for i in range(2, 12):
        out = room.perceive(Frame(t=i / 10, wall=1000.0 + i / 10, img=np.zeros((4, 4, 3), np.uint8), idx=i))
    dets, evs = out
    assert [h.cls for h in dets.hands] == ["hand:7"] and evs == []
    assert str(world.get("wallet").status) == "VISIBLE"


def test_the_detector_weights_are_on_state(tmp_path):
    class Backend:
        info = {"path": "models/brio.engine", "size": 1, "mtime": "2026-09-26T17:53:00", "names": ["keys"],
                "missing": ["wallet"]}

    class Det:
        backend = Backend()

    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    main.Room(CFG, world, events, None, None, None, None, detector=Det())
    assert world.state_json()["perception"]["model"]["path"] == "models/brio.engine"


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
        assert "close-up" in room.world.state_json()["auto_name"]["disclosure"]     # auto-naming wired
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
            calls.append((threading.current_thread().name, "hands"))

    class Table:
        ok = True

    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    ask = make_ask(CFG, world, events, net=None, other=no_model)
    class RoomMem:
        def reset(self):
            calls.append((threading.current_thread().name, "room"))

    room = main.Room(CFG, world, events, Table(), Frames(), None, ask, detector=Det(), hands=Hands())
    room.room_memory = RoomMem()            # Frames has no full_at: no room step, only its reset
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
    assert calls[i + 1:i + 3] == [("perception", "room"), ("perception", "hands")]   # room memory, hand ids: same thread
    assert calls[i + 3:i + 4] in ([], [("perception", "detect")]) and calls.count(("perception", "reset")) == 1


def test_a_frame_read_that_raises_never_ends_the_perception_loop(tmp_path):
    """TableView.wait_new cuts the frame; a bad table_view_rect raises ValueError there, outside perceive's try."""
    from core.types import Detections, Frame
    seen = []

    class Frames:
        def __init__(self):
            self.i = 0

        def wait_new(self, after, timeout=1.0):
            time.sleep(0.005)
            self.i += 1
            if self.i <= 3:
                raise ValueError("rect (0, 0, 0, 0) is empty inside a 1920x1080 image")
            return Frame(t=time.monotonic(), wall=time.time(), img=None, idx=self.i)

    class Det:
        def detect(self, f):
            seen.append(f.idx)
            return Detections(t=f.t, frame_idx=f.idx, items=[], hands=[])

    class Hands:
        def update(self, hands, t):
            return hands

    class Table:
        ok = True

    events = EventLog(":memory:", str(tmp_path))
    room = main.Room(CFG, World(CFG, events), events, Table(), Frames(), None, None, detector=Det(), hands=Hands())
    t = threading.Thread(target=room.perception_loop, daemon=True)
    t.start()
    assert wait_for(lambda: len(seen) >= 3) and t.is_alive()
    room.stop_ev.set()
    t.join(2)


def test_where_answers_hedge_while_perception_is_stale(tmp_path):
    room, _ = make_room(tmp_path)
    assert not room.ask("where is my wallet?", "dashboard").text.startswith("My camera view")   # no live loop
    room._perceived_t = time.monotonic() - 5                          # the live loop stuck for 5 s
    ans = room.ask("where is my wallet?", "dashboard")
    assert ans.text.startswith("My camera view is not updating right now.") and ans.point_at == "wallet"
    room._perceived_t = time.monotonic()
    assert not room.ask("where is my wallet?", "dashboard").text.startswith("My camera view")


def test_the_perception_loop_puts_staleness_on_state(tmp_path):
    class Frames:
        def wait_new(self, after, timeout=1.0):
            time.sleep(0.01)
            return None                                               # the camera gave up

    events = EventLog(":memory:", str(tmp_path))
    world = World(CFG, events)
    room = main.Room(CFG, world, events, None, Frames(), None, None)
    room.stale_s = 0.05
    t = threading.Thread(target=room.perception_loop, daemon=True)
    t.start()
    assert wait_for(lambda: world.state_json()["perception"].get("stale") is True)
    room.stop_ev.set()
    t.join(2)


def test_a_spoken_recalibrate_can_be_turned_off_for_the_demo(tmp_path, cal_path):
    room, _ = make_room(tmp_path, cal_path)
    room.voice_recal = False
    recal = []
    room.recalibrate = lambda timeout_s=None: recal.append(1)
    assert "turned off" in room.ask("recalibrate", "voice").text
    time.sleep(0.1)
    assert recal == []


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


def test_camera_is_an_index_or_a_stable_device_path():
    assert main.camera_source("2") == 2 and main.camera_source(" 0 ") == 0
    by_id = "/dev/v4l/by-id/usb-046d_0809_A1C0DC94-video-index0"
    assert main.camera_source(by_id) == by_id and main.camera_source("/dev/video2") == "/dev/video2"


class _TagTable:
    """One-tag table stand-in: fits after `need` calibrate() calls, then shifts the frame by `shift` cm."""
    tag_mode, tag_id, tag_frames = True, 0, 5

    def __init__(self, need=5, shift=0.0):
        self.need, self.shift, self.calls, self.ok = need, shift, 0, True

    def calibrate(self, img):
        self.calls += 1
        return self.calls >= self.need

    def px_to_cm(self, pts):
        import numpy as np
        return np.asarray(pts, dtype=float) / 10 + (self.shift if self.calls >= self.need else 0.0)


class _NewFrames:
    def __init__(self):
        import numpy as np
        from core.types import Frame
        self.Frame, self.img, self.i = Frame, np.zeros((4, 4, 3), np.uint8), 0

    def latest(self):
        return self.Frame(0.0, 0.0, self.img, self.i)

    def wait_new(self, after_idx, timeout=1.0):
        time.sleep(0.005)
        self.i = after_idx + 1
        return self.Frame(0.0, 0.0, self.img, self.i)


def test_recalibrate_feeds_fresh_frames_until_the_tag_fits(tmp_path, caplog):
    room, _ = make_room(tmp_path)
    room.frames, room.table = _NewFrames(), _TagTable(need=5, shift=3.0)
    room.laser.fit = object()
    with caplog.at_level("WARNING"):
        assert room.recalibrate(timeout_s=2.0) is True
    assert room.table.calls == 5
    assert "recalibrate the laser" in caplog.text           # the frame moved 3 cm under a fitted laser


def test_recalibrate_gives_up_when_the_tag_stays_hidden(tmp_path, caplog):
    room, _ = make_room(tmp_path)
    room.frames, room.table = _NewFrames(), _TagTable(need=10 ** 6)
    with caplog.at_level("WARNING"):
        assert room.recalibrate(timeout_s=0.1) is False
    assert "tag 0 not held in view" in caplog.text


def test_laser_fitted_before_the_table_calibration_is_flagged(tmp_path):
    import json
    from types import SimpleNamespace
    p = tmp_path / "table_cal.json"
    p.write_text(json.dumps({"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": 2000.0}))
    assert main.laser_older_than_table(SimpleNamespace(timestamp=1000.0), str(p))
    assert not main.laser_older_than_table(SimpleNamespace(timestamp=3000.0), str(p))
    assert not main.laser_older_than_table(SimpleNamespace(timestamp=1000.0), str(tmp_path / "none.json"))


def test_camera_defaults_to_the_configured_stable_path_when_it_exists(tmp_path):
    dev = tmp_path / "usb-046d_Logitech_BRIO-video-index0"
    assert main.default_camera({"demo_check": {"camera": str(dev)}}) == 0          # not plugged in
    dev.write_text("")
    assert main.default_camera({"demo_check": {"camera": str(dev)}}) == str(dev)
    assert main.default_camera({"demo_check": {"camera": "3"}}) == 3
    assert main.default_camera({}) == 0


def test_the_ask_timeout_matches_the_server():
    import main
    from server.app import ASK_TIMEOUT_S
    assert main.ANSWER_LATE_S == ASK_TIMEOUT_S


def test_an_answer_past_the_server_timeout_is_not_spoken_or_aimed(tmp_path, cal_path, monkeypatch):
    import main
    room, rig = make_room(tmp_path, cal_path)
    monkeypatch.setattr(main, "ANSWER_LATE_S", -1.0)      # every answer is "late"
    ans = room.ask_and_act("where is my wallet?", "phone")
    time.sleep(0.2)
    assert ans.point_at == "wallet" and room.tts.said == [] and not rig.act.writes


def test_a_spoken_recalibrate_that_fails_says_so(tmp_path):
    room, _ = make_room(tmp_path)
    room.frames, room.table = _NewFrames(), _TagTable(need=10 ** 6)
    room.table.size_cm = (80.0, 50.0)
    room.recalibrate = lambda timeout_s=None: False
    assert "couldn't recalibrate" in room._recalibrate_and_tell(speak=True)
    assert room.tts.said and "table tag" in room.tts.said[-1]


def test_a_recalibrate_that_changes_the_tracked_area_asks_for_a_restart(tmp_path):
    room, _ = make_room(tmp_path)
    room.frames, room.table = _NewFrames(), _TagTable(need=1)
    room.table.size_cm = (80.0, 50.0)

    def refit(timeout_s=None):
        room.table.size_cm = (95.0, 55.0)
        return True
    room.recalibrate = refit
    assert "Restart me" in room._recalibrate_and_tell(speak=False)
    assert room.tts.said == []                               # a text question is not answered aloud


def test_a_recalibrate_that_keeps_the_size_says_nothing_more(tmp_path):
    room, _ = make_room(tmp_path)
    room.frames, room.table = _NewFrames(), _TagTable(need=1)
    room.table.size_cm = (80.0, 50.0)
    room.recalibrate = lambda timeout_s=None: True
    assert room._recalibrate_and_tell(speak=True) is None and room.tts.said == []


# -- voice path robustness: the mic never dies, and a stall still gets an answer

def test_an_error_answering_says_sorry_and_the_loop_keeps_listening(tmp_path, cal_path):
    """One exception in interpret / ask used to end voice_loop for good: the rig went deaf."""
    stt = FakeSTT("", overheard=["where is my wallet", "where are my keys"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    fast, calls = room.base_ask, []

    def flaky(text, source):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return fast(text, source)

    room.base_ask = flaky
    real_answer = room._answer
    blown = []

    def answer_once_broken(*a, **k):            # the whole turn raising, not just ask
        if not blown:
            blown.append(1)
            raise RuntimeError("turn failed")
        return real_answer(*a, **k)

    room._answer = answer_once_broken
    t = always_on(room)
    assert wait_for(lambda: any("keys" in s.lower() for s in room.tts.said))
    assert room.tts.said[0].startswith("Sorry") and t.is_alive()
    stop_voice(room, t)


def test_a_failed_ask_is_answered_from_the_rules(tmp_path, cal_path):
    room, _ = make_room(tmp_path, cal_path)

    def broken(text, source):
        raise RuntimeError("router failed")

    room.base_ask = broken
    ans, _ = room._ask_with_cue("where is my wallet")
    assert ans.point_at == "wallet" and room._answer_late


def test_click_mode_survives_an_error_too(tmp_path, cal_path):
    clicker = FakeClicker()
    room, _ = make_room(tmp_path, cal_path, stt=FakeSTT("where is my wallet"), clicker=clicker)
    real, n = room._asked, []

    def asked(t):
        n.append(t)
        if len(n) == 1:
            raise RuntimeError("clicker path failed")
        real(t)

    room._asked = asked
    t = threading.Thread(target=room.voice_loop, daemon=True)
    t.start()
    clicker.press()
    assert wait_for(lambda: room.tts.said)
    clicker.press()
    assert wait_for(lambda: any("wallet" in s.lower() for s in room.tts.said)) and t.is_alive()
    room.stop_ev.set()
    t.join(2)


def test_the_watchdog_restarts_a_dead_voice_loop(tmp_path, cal_path):
    room, _ = make_room(tmp_path, cal_path)
    runs = []

    def dies():
        runs.append(1)
        if len(runs) == 1:
            return                              # died early, however (a BaseException, a bug in the guard)
        room.stop_ev.wait()

    room.voice_loop = dies
    w = threading.Thread(target=room.voice_watchdog, kwargs={"check_s": 0.05}, daemon=True)
    w.start()
    assert wait_for(lambda: len(runs) == 2, 4.0) and room.voice_restarts == 1
    room.stop_ev.set()
    w.join(2)
    assert not w.is_alive()


def test_a_stalled_answer_falls_back_to_the_rules_within_the_limit(tmp_path, cal_path):
    """Wi-Fi stalls mid-answer: done.wait() used to wait forever with the mic shut."""
    room, _ = make_room(tmp_path, cal_path)
    room.cue_after_s, room.answer_limit_s = 0.05, 0.4
    room.base_ask = lambda text, source: time.sleep(30)
    t0 = time.monotonic()
    ans, cued = room._ask_with_cue("where is my wallet")
    assert time.monotonic() - t0 < 1.0 and cued and ans.point_at == "wallet"
    assert room._answer_late
    ans, _ = room._ask_with_cue("tell me a joke")          # OTHER: the offline sentence, not silence
    assert ans.text and ans.point_at is None


def test_a_slow_model_reading_overheard_speech_starts_the_cue_first(tmp_path, cal_path):
    """The model's reading of overheard speech used to run before the cue timer: seconds of silence."""
    room, _ = make_room(tmp_path, cal_path)
    room.cue_after_s = 0.05
    rules = room.interpret

    class SlowReader:
        last_by, last_ms = "grok", 0.0

        def screen(self, text):
            return True

        def __call__(self, text, overheard=False):
            if overheard:
                time.sleep(0.3)
            return rules(text)

    room.interpret = SlowReader()
    assert room._answer("where is my wallet", time.monotonic(), time.monotonic(), {"mode": "overheard"})
    assert room.tts.said[0] == "Let me look." and "wallet" in room.tts.said[1].lower()


def test_overheard_speech_the_model_ignores_is_dropped_quietly(tmp_path, cal_path):
    room, _ = make_room(tmp_path, cal_path)
    rules = room.interpret

    class Ignores:
        def screen(self, text):
            return True

        def __call__(self, text, overheard=False):
            return main.Intent(kind="IGNORE", obj=None, raw=text) if overheard else rules(text)

    room.interpret = Ignores()
    assert room._answer("where is my wallet", time.monotonic(), time.monotonic(), {"mode": "overheard"}) is False
    assert room.tts.said == [] and room.world.laser["on"] is False


@pytest.mark.parametrize("text, target", [("show me my keys", "keys"), ("can you point at the pill bottle", "pill_bottle"),
                                          ("light up my wallet", "wallet"), ("wears my wall it", "wallet")])
def test_offline_show_and_point_aim_the_laser(tmp_path, cal_path, text, target):
    """Offline (no model), 'show me my keys' was OTHER: "I'm offline", and the laser never moved."""
    stt = FakeSTT("", overheard=[text])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    assert not room.world.online
    t = always_on(room)
    assert wait_for(lambda: room.world.laser.get("target") == target), room.tts.said
    assert room.tts.said and "offline" not in room.tts.said[0].lower()
    stop_voice(room, t)


def test_overheard_where_with_nothing_to_find_is_ignored(tmp_path, cal_path):
    stt = FakeSTT("", overheard=["where are you guys from", "where are my keys"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    assert len(room.tts.said) == 1 and "keys" in room.tts.said[0].lower()
    stop_voice(room, t)


class SlowGrokOther:
    """A model that takes 0.3 s and reads everything as OTHER (what Grok says to chatter)."""
    url, name, local = "stub", "grok", False

    def __init__(self):
        self.asked = []

    def ask(self, text, timeout):
        import json
        self.asked.append(text)
        time.sleep(0.3)
        return json.dumps({"kind": "OTHER", "object": "none"})

    def health(self, timeout=1.0):
        return True


def test_no_thinking_cue_for_chatter_the_model_then_rejects(tmp_path, cal_path):
    """'Let me look.' used to start before the overheard verdict: the rig said it, then nothing."""
    from voice.understand import Understander
    room, _ = make_room(tmp_path, cal_path)
    room.cue_after_s = 0.05
    room.interpret = Understander(CFG, model=SlowGrokOther())
    assert room._for_rig("where's my stapler")                # passes the cheap checks; the model decides
    assert room._answer("where's my stapler", time.monotonic(), time.monotonic(), {"mode": "overheard"}) is False
    time.sleep(0.1)
    assert room.tts.said == [] and room.interpret.model.asked == ["where's my stapler"]


def test_a_sure_overheard_question_still_gets_the_cue_while_the_answer_is_slow(tmp_path, cal_path):
    from voice.understand import Understander
    room, _ = make_room(tmp_path, cal_path)
    room.cue_after_s = 0.05
    room.interpret = Understander(CFG, model=SlowGrokOther())
    fast = room.base_ask
    room.base_ask = lambda text, source: (time.sleep(0.3), fast(text, source))[1]
    assert room._answer("room, where is my wallet", time.monotonic(), time.monotonic(), {"mode": "overheard"})
    assert room.tts.said[0] == "Let me look." and "wallet" in room.tts.said[1].lower()


def test_a_reset_that_comes_after_the_fallback_is_not_applied(tmp_path, cal_path):
    """After answer_limit_s the rules answered, but the ask thread ran on and reset the world ~1 s later."""
    room, _ = make_room(tmp_path, cal_path)
    room.cue_after_s, room.answer_limit_s = 0, 0.2
    routed = room.base_ask
    room.base_ask = lambda text, source: (time.sleep(0.5), routed(text, source))[1]
    resets = []
    room.world.reset = lambda: resets.append(1)
    ans, _ = room._ask_with_cue("reset everything")
    assert room._answer_late and resets == []
    time.sleep(0.6)
    assert resets == []                                       # the late RESET was dropped
    ans = room.ask("reset everything", "voice")               # on time, it still resets
    assert resets == [1]


def test_overheard_follow_up_reaches_the_conversation_memory(tmp_path, cal_path):
    from voice.care import attach_care
    from voice.understand import Understander
    stt = FakeSTT("", overheard=["where is it now", "where are my keys", "where is it now"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker(),
                        interpret=Understander(dict(CFG, understand={"enabled": False})))
    attach_care(room, dict(CFG, care=dict(CFG.get("care") or {}, profile_llm=False)))
    t = always_on(room)
    assert wait_for(lambda: len(room.tts.said) == 2)
    assert "keys" in room.tts.said[0].lower() and "keys" in room.tts.said[1].lower()   # the first 'it' was dropped


def test_the_wake_word_alone_makes_the_rig_listen_for_the_question(tmp_path, cal_path, monkeypatch):
    """Rig run (Sat 26 Sep): "Room!" ... pause ... "where is my wallet?" arrived as two utterances and
    neither was answered. The bare wake word now acts like a clicker press: the next thing said is for
    the rig."""
    posted = []
    monkeypatch.setattr(main.requests, "post",
                        lambda url, json, timeout, headers=None: posted.append(json))
    stt = FakeSTT("where is my wallet", overheard=["Room!", "hey room"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    room.webhook_url = "http://n8n/webhook/ask-the-room"
    t = always_on(room)
    assert wait_for(lambda: len(posted) >= 2)
    assert [(q["heard"], q["mode"]) for q in posted[:2]] == [("where is my wallet", "asked")] * 2
    assert all("wallet" in s.lower() for s in room.tts.said[:2])
    assert room._bare_wake("Room!") and room._bare_wake("hey room, um") and not room._bare_wake("room where is it")
    stop_voice(room, t)


def test_overheard_transcripts_are_logged_only_with_the_debug_switch(tmp_path, cal_path):
    stt = FakeSTT("", overheard=["we built this last night"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    room.cfg = dict(room.cfg, listen=dict(room.cfg.get("listen") or {}, log_overheard=True))
    t = always_on(room)
    assert wait_for(lambda: not stt.overheard)
    assert stt.log_text is True
    stop_voice(room, t)


# -- the bare wake word (review): it opens the mic like a click, but chatter still doesn't get answered

def test_chatter_after_a_bare_wake_word_is_not_answered(tmp_path, cal_path, monkeypatch):
    """A lone 'Room.' in hall chatter (or a mishearing) opened the mic, and the next sentence skipped every
    chatter check: 'we built this in twenty hours' was answered as a question, and silence got 'Sorry, I
    didn't catch that.'"""
    posted = []
    monkeypatch.setattr(main.requests, "post", lambda url, json, timeout, headers=None: posted.append(json))
    for follow in ["we built this in like twenty hours", "where are you guys from", "let's reset after this", ""]:
        stt = FakeSTT(follow, overheard=["Room."])
        room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
        room.webhook_url = "http://n8n/webhook/ask-the-room"
        resets = []
        room.world.reset = lambda: resets.append(1)
        t = always_on(room)
        assert wait_for(lambda: not stt.overheard)
        time.sleep(0.2)
        stop_voice(room, t)
        assert room.tts.said == [] and posted == [] and resets == [], follow


def test_a_bare_wake_phrase_of_several_words_opens_the_mic(tmp_path, cal_path):
    cfg = dict(CFG, listen=dict(CFG["listen"], wake_words=["ask the room", "askroom", "room"]))
    room, _ = make_room(tmp_path, cal_path)
    room.cfg = cfg
    for text in ["Ask the room!", "hey, ask the room", "askroom", "Room!", "ok room um"]:
        assert room._bare_wake(text), text
    for text in ["the room", "ask the room where my keys are", "ask the", "room b", "living room"]:
        assert not room._bare_wake(text), text


def test_a_question_after_a_bare_wake_word_is_still_answered(tmp_path, cal_path):
    stt = FakeSTT("did anything change while I was gone", overheard=["Ask the room."])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    stop_voice(room, t)
    assert len(room.tts.said) >= 1 and room.tts.said[0] != main.NOT_HEARD


class ListensSTT(FakeSTT):
    """listen() gives each scripted question in turn (the rig heard the wake word twice)."""

    def __init__(self, listens, overheard=()):
        super().__init__("", overheard=overheard)
        self.listens = list(listens)

    def listen(self):
        return self.listens.pop(0) if self.listens else ""


def wake_room(tmp_path, cal_path, stt):
    from voice.understand import Understander
    cfg = dict(CFG, understand={"enabled": False}, listen=dict(CFG["listen"], mode="wake"))
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker(), interpret=Understander(cfg))
    return room


@pytest.mark.parametrize("wake", ["Room!", "Hey, bro!", "Hey Drew!", "Goodroom."])
def test_wake_mode_answers_the_question_after_a_bare_wake_word(tmp_path, cal_path, wake):
    """Rig (Sat 26 Sep, listen.mode wake): "Room!" ... "What do you see?" was dropped because the question had no
    wake word of its own, and "Hey room" came back as "Hey, bro!" / "Hey Drew!" / "Goodroom."."""
    stt = ListensSTT(["where is my wallet"], overheard=[wake])
    room = wake_room(tmp_path, cal_path, stt)
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    stop_voice(room, t)
    assert "wallet" in room.tts.said[0].lower()


def test_the_wake_word_said_again_listens_again(tmp_path, cal_path):
    """Rig: "Hey Drew!" ... "Okay, room." ... "Where's my pill bottle?"."""
    stt = ListensSTT(["Okay, room.", "Where's my pill bottle?"], overheard=["Hey Drew!"])
    room = wake_room(tmp_path, cal_path, stt)
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    stop_voice(room, t)
    assert "pill bottle" in room.tts.said[0].lower() and stt.listens == []


def test_wake_mode_ignores_room_in_chatter(tmp_path, cal_path):
    stt = ListensSTT([], overheard=["I'm in the room.", "Try this room.", "This is the best news.", "That's my name.",
                                    "Okay, bro, where are my keys?"])
    room = wake_room(tmp_path, cal_path, stt)
    t = always_on(room)
    assert wait_for(lambda: not stt.overheard)
    time.sleep(0.2)
    stop_voice(room, t)
    assert room.tts.said == []
