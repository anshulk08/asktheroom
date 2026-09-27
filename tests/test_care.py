"""The care layer wired together (voice/care.py): the real World + EventLog + router, a simulated clock,
notices spoken through main.Room.respond with the laser on the object, /state and /report on the
dashboard, the morning report on the first question, and the scheduler thread."""
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from act.calibrate import calibrate
from act.sim import SimRig
from core.carewords import pill_claim
from core.config import load_config
from core.events import EventLog
from core.types import Answer, Event, Status
from core.world import World
from server.app import create_app
from server.sim import SimTable
from voice.care import Care
from voice.pipeline import make_ask

CFG = load_config()


def at(h, m=0, s=0, day=25):
    return (datetime(2026, 9, day) + timedelta(hours=h, minutes=m, seconds=s)).timestamp()


def no_grok(text, world, events, cfg, online=True):
    return Answer("grok")


class SpeakLog:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)

    def stop(self):
        pass


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def real_world(events):
    """The real World with the pill bottle under the notebook (as if perception had seen it go there)."""
    w = World(CFG, events)
    nb, pb = w.entities["notebook"], w.entities["pill_bottle"]
    nb.status, nb.pos_cm, nb.confidence, nb.last_seen = Status.VISIBLE, (30.0, 30.0), 1.0, at(7)
    pb.status, pb.parent, pb.pos_cm, pb.confidence, pb.last_seen = Status.UNDER, "notebook", (31.0, 29.0), 0.85, at(7)
    return w


@pytest.fixture
def cal_path(tmp_path_factory):
    p = Path(tmp_path_factory.mktemp("cal")) / "laser_cal.json"
    calibrate(SimRig(CFG).make_laser(str(p)))
    return str(p)


def test_end_to_end_condition_reminder_speaks_and_points(tmp_path, cal_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    now = [at(8, 30)]
    care = Care(CFG, world, events, make_ask(CFG, world, events, net=None, other=no_grok), clock=lambda: now[0])
    rig = SimRig(CFG)
    room = main.Room(CFG, world, events, SimTable(CFG), None, rig.make_laser(cal_path), care.ask, tts=SpeakLog())
    room.room_head_px, room._people_now = (640.0, 0.0), (lambda img: [])     # the laser gate: nobody in view
    room.laser_timeout_s = 60
    care.on_notice = room.respond
    care.morning.mark_delivered(now[0])        # this morning's report already played

    a = room.ask("Remind me if I haven't picked up my pill bottle by 9.", "voice")
    assert a.text == "Okay. At 9 AM today, if your pill bottle hasn't been picked up, I'll remind you."
    events.add(Event(t=at(8, 40), wall=at(8, 40), obj="keys", type="PICKED_UP"))    # not the bottle

    now[0] = at(8, 59, 50)
    assert care.tick() == []
    now[0] = at(9, 0, 3)
    (n,) = care.tick()
    text = "It's 9 o'clock and you haven't picked up your pill bottle yet. It's under the notebook."
    assert n.text == text and n.point_at == "pill_bottle" and n.spoken
    assert wait_for(lambda: room.tts.said == [text] and world.laser.get("target") == "pill_bottle")
    pos, chain = world.resolve("pill_bottle")
    assert chain == ["pill_bottle", "notebook"]
    assert np.linalg.norm(np.subtract(rig.true_dot_cm(), pos)) < 3.0
    assert not pill_claim(text)

    now[0] = at(9, 0, 20)
    assert care.tick() == []                                               # deduped
    assert room.ask("okay, thanks", "voice").text == "Okay, got it."
    assert care.notices_json()[0]["acknowledged"] is True
    events.close()


def test_pickup_before_deadline_stays_silent(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    now = [at(8, 30)]
    spoken = []
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: now[0])
    care.on_notice = spoken.append
    care.morning.mark_delivered(now[0])
    care.ask("remind me if I haven't picked up my pills by 9", "voice")
    events.add(Event(t=at(8, 50), wall=at(8, 50), obj="pill_bottle", type="PICKED_UP"))
    now[0] = at(9, 0, 5)
    assert care.tick() == [] and spoken == []
    events.close()


def test_state_notices_ack_and_report_routes(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    now = [at(8, 0)]
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: now[0])
    care.morning.mark_delivered(now[0])
    care.ask("remind me at 9 to call Sarah", "voice")
    events.add(Event(t=at(8, 5), wall=at(8, 5), obj="pill_bottle", type="PICKED_UP"))
    now[0] = at(9, 0, 5)
    (n,) = care.tick()
    app = create_app(CFG, world, events, ask_fn=care.ask, care=care)
    with TestClient(app) as c:
        body = c.get("/state").json()
        assert "state" in body and body["notices"][0]["text"] == "It's 9 o'clock. Time to call Sarah."
        assert body["notices"][0]["acknowledged"] is False
        assert c.post(f"/notices/{n.id}/ack").json() == {"ok": True}
        assert c.get("/state").json()["notices"][0]["acknowledged"] is True
        assert c.post("/notices/999/ack").status_code == 404
        md = c.get("/report", params={"date": "2026-09-25"}).text
        assert md.startswith("# Daily summary: Friday, September 25, 2026") and "8:05 AM" in md
        js = c.get("/report", params={"date": "2026-09-25", "format": "json"}).json()
        assert js["pill_bottle_pickups"][0]["t"] == at(8, 5)
        assert c.get("/report", params={"date": "yesterday-ish"}).status_code == 400
    events.close()


def test_state_without_care_is_unchanged(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    app = create_app(CFG, real_world(events), events)
    with TestClient(app) as c:
        assert "notices" not in c.get("/state").json()
        assert c.get("/report").status_code == 404
    events.close()


def test_morning_report_on_first_question(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    events.add(Event(t=at(18, day=24), wall=at(18, day=24), obj="pill_bottle", type="COVERED", parent="notebook"))
    now = [at(7, 30)]
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: now[0])
    base = make_ask(CFG, world, events, other=no_grok)
    a = care.ask("where are my pills?", "voice")
    report = ("Good morning. Yesterday your pill bottle was last seen under the notebook at 6 PM. "
              "You don't have any reminders today.")
    assert a.text == report + " " + base("where are my pills?", "voice").text
    assert a.point_at == "pill_bottle"
    now[0] += 60
    assert care.ask("where are my pills?", "voice").text == base("where are my pills?", "voice").text
    events.close()


def test_morning_report_not_for_text_sources(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: at(7, 30))
    assert not care.ask("where are my pills?", "sms").text.startswith("Good morning")
    assert care.ask("where are my pills?", "voice").text.startswith("Good morning")
    events.close()


def test_morning_report_on_activity(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    now = [at(7, 10)]
    spoken = []
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: now[0])
    care.on_notice = spoken.append
    assert care.tick() == []
    events.add(Event(t=at(7, 12), wall=at(7, 12), obj="keys", type="PICKED_UP"))
    now[0] = at(7, 12, 5)
    (n,) = care.tick()
    assert n.kind == "morning" and n.text.startswith("Good morning.") and spoken[0].text == n.text
    now[0] = at(7, 20)
    assert care.tick() == []
    assert not care.ask("where are my keys?", "voice").text.startswith("Good morning")
    events.close()


def test_voice_summary(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    events.add(Event(t=at(8, 5), wall=at(8, 5), obj="pill_bottle", type="PICKED_UP"))
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: at(14))
    a = care.ask("Give me today's summary.", "voice")
    assert a.text.startswith("Today the pill bottle was picked up at 8:05 AM.")
    assert not pill_claim(a.text)
    events.close()


def test_scheduler_thread_starts_and_stops(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok))
    care.tick_s = 0.05
    ticks = []
    orig = care.tick
    care.tick = lambda now=None: ticks.append(1) or orig(now)
    care.start()
    assert wait_for(lambda: len(ticks) >= 2, 2.0)
    care.stop()
    n = len(ticks)
    time.sleep(0.2)
    assert len(ticks) == n
    events.close()


def test_care_ask_never_raises(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: at(14))
    care.reminders.handle = lambda *a, **k: 1 / 0                          # a bug in the care layer ...
    assert care.ask("where is my pill bottle?", "voice").point_at == "pill_bottle"   # ... never costs an answer
    events.close()


def test_room_wiring_attach(tmp_path, cal_path):
    """voice.care.attach_care(room) wires the router, notices -> respond, and the scheduler into the room."""
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    rig = SimRig(CFG)
    room = main.Room(CFG, world, events, SimTable(CFG), None, rig.make_laser(cal_path),
                     make_ask(CFG, world, events, other=no_grok), tts=SpeakLog())
    from voice.care import attach_care
    care = attach_care(room, CFG)
    assert room.care is care and room.base_ask == care.ask
    care.on_notice(Answer("It's 9 o'clock.", point_at="pill_bottle", action="point"))
    assert wait_for(lambda: room.tts.said == ["It's 9 o'clock."])
    events.close()


def test_text_sources_cannot_acknowledge_or_state_facts(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    now = [at(14)]
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: now[0])
    care.ask("remind me at 3 pm to call Sarah", "sms")                     # a caregiver may set one by text
    now[0] = at(15, 0, 5)
    (n,) = care.tick()
    now[0] = at(15, 0, 20)
    assert care.ask("ok", "sms").text != "Okay, got it."                    # ... but only the room acknowledges
    assert care.notices_json()[0]["acknowledged"] is False
    care.ask("my daughter is Sarah", "sms")
    assert care.profile.facts() == []
    assert care.ask("okay", "voice").text == "Okay, got it."
    events.close()


def test_morning_report_said_once_under_concurrency(tmp_path):
    import threading
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: at(7, 30))
    got = []
    ts = [threading.Thread(target=lambda: got.append(care.morning.deliver(at(7, 30)))) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(1 for g in got if g) == 1
    events.close()


def test_morning_on_activity_loses_the_race_to_a_question(tmp_path):
    """tick() saw the report due, then a voice question delivered it before tick's deliver(): no NULL
    notice and nothing handed to the speaker."""
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    spoken, asked = [], []
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: at(7, 30))
    care.on_notice = spoken.append
    events.add(Event(t=at(7, 12), wall=at(7, 12), obj="keys", type="PICKED_UP"))

    def question_first(now):                       # runs after tick's due() check, before its deliver()
        asked.append(care.ask("where are my keys?", "voice"))
        return True

    care.morning.activity_since_morning = question_first
    assert care.tick() == []
    assert asked[0].text.startswith("Good morning")
    assert spoken == [] and all(n.text for n in care.reminders.store.notices(0))
    events.close()


def test_greeting_loses_the_race_to_the_scheduler(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    care = Care(CFG, world, events, make_ask(CFG, world, events, other=no_grok), clock=lambda: at(7, 30))
    deliver = care.morning.deliver

    def scheduler_wins(now):                       # the scheduler says it just before the greeting asks for it
        care.morning.mark_delivered(now)
        return deliver(now)

    care.morning.deliver = scheduler_wins
    assert care.ask("good morning", "voice").text == "Good morning."
    events.close()


def test_demo_hold_notices_keeps_the_rig_quiet_unless_asked(tmp_path):
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = real_world(events)
    now = [at(8, 30)]
    spoken = []
    cfg = dict(CFG, demo={"hold_notices": True})
    care = Care(cfg, world, events, make_ask(cfg, world, events, other=no_grok), clock=lambda: now[0])
    care.on_notice = spoken.append
    base = make_ask(cfg, world, events, other=no_grok)
    assert care.ask("where are my pills?", "voice").text == base("where are my pills?", "voice").text  # no morning report
    care.ask("remind me if I haven't picked up my pills by 9", "voice")
    now[0] = at(9, 0, 5)
    notices = care.tick()
    assert notices and spoken == []                                        # recorded, not spoken
    assert care.notices_json() and care.ask("morning report", "voice").text.startswith("Good morning")
    events.close()
