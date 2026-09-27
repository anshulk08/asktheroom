"""Morning report and caregiver summary (core/reports.py) on a synthetic day: what gets mentioned, the
once-per-day rule, persistence, narration summaries when that store exists, and the pill rule."""
import re
from datetime import date, datetime, timedelta

import pytest

from core.carewords import pill_claim
from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.profile import Profile
from core.reminders import Reminders
from core.reports import MorningReport, caregiver_summary, morning_report, spoken_summary, summary_data
from core.types import Entity, Event, Status

CFG = load_config()


def at(h, m=0, s=0, day=25):
    return (datetime(2026, 9, day, 0, 0, 0) + timedelta(hours=h, minutes=m, seconds=s)).timestamp()


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def safe(text):
    for s in sentences(text):
        assert not pill_claim(s), s
    assert not re.search(r"\b(taken|took|swallow\w*|missed)\b", text, re.I), text


def ev(events, obj, typ, wall, **kw):
    events.add(Event(t=wall, wall=wall, obj=obj, type=typ, **kw))


def scene(events):
    """Yesterday (the 24th): keys went into the box at 3:10 PM, glasses under the notebook at 6 PM, the pill
    bottle was picked up at 8:02 AM and 8:15 PM, the wallet moved around (still visible), and the phone
    left the table two days ago. The remote was lost from view at 7 PM."""
    w = FakeWorld([
        Entity("keys", "target", Status.INSIDE, parent="box", pos_cm=(41.0, 29.0), last_seen=at(15, 10, day=24)),
        Entity("glasses", "target", Status.UNDER, parent="notebook", pos_cm=(20.0, 40.0), last_seen=at(18, day=24)),
        Entity("pill_bottle", "target", Status.VISIBLE, pos_cm=(60.0, 20.0), last_seen=at(20, 16, day=24)),
        Entity("wallet", "target", Status.VISIBLE, pos_cm=(70.0, 10.0), last_seen=at(12, day=24)),
        Entity("phone", "target", Status.GONE, pos_cm=(2.0, 30.0), edge="left", last_seen=at(9, day=23)),
        Entity("remote", "target", Status.UNKNOWN, parent="unknown", pos_cm=(85.0, 55.0),
               last_seen=at(19, day=24), confidence=0.3),
        Entity("box", "container", Status.VISIBLE, pos_cm=(70.0, 38.0), last_seen=at(12, day=24)),
        Entity("notebook", "cover", Status.VISIBLE, pos_cm=(20.5, 39.0), last_seen=at(12, day=24)),
    ], events)
    ev(events, "phone", "EXITED_VIEW", at(9, day=23), edge="left")
    ev(events, "pill_bottle", "PICKED_UP", at(8, 2, day=24), from_cm=(60.0, 20.0))
    ev(events, "pill_bottle", "PUT_BACK", at(8, 3, day=24), to_cm=(60.0, 20.0))
    ev(events, "wallet", "MOVED", at(12, day=24), to_cm=(70.0, 10.0))
    ev(events, "keys", "PICKED_UP", at(15, 9, day=24), from_cm=(10.0, 10.0))
    ev(events, "keys", "PUT_INSIDE", at(15, 10, day=24), parent="box")
    ev(events, "glasses", "COVERED", at(18, 0, day=24), parent="notebook")
    ev(events, "remote", "LOST_TRACK", at(19, 0, day=24))
    ev(events, "pill_bottle", "PICKED_UP", at(20, 15, day=24), from_cm=(60.0, 20.0))
    ev(events, "pill_bottle", "PUT_BACK", at(20, 16, day=24), to_cm=(60.0, 20.0))
    return w


@pytest.fixture
def events(tmp_path):
    e = EventLog(":memory:", str(tmp_path / "snaps"))
    yield e
    e.close()


@pytest.fixture
def w(events):
    return scene(events)


# ---------------------------------------------------------------- morning report

def test_morning_report_content(w, events):
    rem = Reminders(CFG, w, events)
    rem.handle("remind me every day if I haven't picked up my pill bottle by 8", at(6, day=24))
    prof = Profile(events)
    prof.remember("user", "preferred_name", "Grandpa Joe", "call me Grandpa Joe", now=at(9, day=23))
    text = morning_report(w, events, CFG, at(7, 30), reminders=rem, profile=prof)
    assert text.startswith("Good morning, Grandpa Joe.")
    assert "your keys ended up in the box at 3:10 PM" in text
    assert "your glasses were last seen under the notebook at 6 PM" in text
    assert "picked up at 8:02 AM and 8:15 PM" in text
    assert "You have one reminder today: your pill bottle at 8 AM." in text
    assert "wallet" not in text and "phone" not in text               # visible / not yesterday
    assert len(sentences(text)) <= 3
    safe(text)


def test_morning_report_mentions_lost_track(events):
    w = FakeWorld([Entity("remote", "target", Status.UNKNOWN, parent="unknown", pos_cm=(85.0, 55.0),
                          last_seen=at(19, day=24), confidence=0.3)], events)
    ev(events, "remote", "LOST_TRACK", at(19, day=24))
    text = morning_report(w, events, CFG, at(8))
    assert text == ("Good morning. Yesterday I lost track of your remote at 7 PM, on your right, near you. "
                    "You don't have any reminders today.")


def test_morning_report_quiet_day(events):
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(10.0, 10.0))], events)
    ev(events, "keys", "MOVED", at(10, day=24), to_cm=(10.0, 10.0))
    text = morning_report(w, events, CFG, at(8))
    assert text == "Good morning. Nothing went missing yesterday. You don't have any reminders today."


def test_morning_report_uses_narration_when_quiet(events):
    ns = pytest.importorskip("core.narration_store")
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(10.0, 10.0))], events)
    st = ns.NarrationStore(events)
    i = st.add_pending(at(15, day=24), at(15, 1, day=24), None, {})
    st.mark_done(i, "You were reading the newspaper at the table. Then you tidied up.",
                 {"summary": "x", "confidence": 0.9}, "fake", "f", 5)
    text = morning_report(w, events, CFG, at(8))
    assert "reading the newspaper" in text and len(sentences(text)) <= 3
    safe(text)


def test_morning_report_without_narration_table(events, w):
    # store_for(create=False) finds no table: the report is built from events alone
    assert morning_report(w, events, CFG, at(8)).startswith("Good morning.")


def test_morning_report_once_per_day(w, events):
    mr = MorningReport(CFG, w, events)
    assert mr.deliver(at(5, 59)) is None                                  # before morning_after
    first = mr.deliver(at(7, 30))
    assert first and first.startswith("Good morning.")
    assert mr.deliver(at(7, 45)) is None                                  # once a day
    assert mr.deliver(at(12, 30, day=26)) is None                         # after noon: too late
    assert mr.deliver(at(6, 30, day=27)) is not None                      # next morning


def test_morning_report_once_per_day_survives_restart(tmp_path):
    db = str(tmp_path / "e.db")
    e1 = EventLog(db, str(tmp_path / "s"))
    assert MorningReport(CFG, scene(e1), e1).deliver(at(7, 30)) is not None
    e1.close()
    e2 = EventLog(db, str(tmp_path / "s"))
    assert MorningReport(CFG, scene(e2), e2).deliver(at(8)) is None
    e2.close()


def test_morning_report_activity_trigger(w, events):
    mr = MorningReport(CFG, w, events)
    assert mr.activity_since_morning(at(7)) is False
    ev(events, "keys", "PICKED_UP", at(6, 40))
    assert mr.activity_since_morning(at(7)) is True


# ---------------------------------------------------------------- caregiver summary

def test_caregiver_summary_markdown(w, events):
    rem = Reminders(CFG, w, events)
    rem.handle("remind me if I haven't picked up my pill bottle by 7 am", at(6, day=24))
    rem.handle("remind me at 5 pm to call Sarah", at(6, day=24))
    rem.tick(at(7, 0, 5, day=24))
    rem.acknowledge("okay", at(7, 0, 30, day=24))
    rem.tick(at(17, 0, 5, day=24))
    md = caregiver_summary(date(2026, 9, 24), w, events, CFG, reminders=rem, now=at(7, 30))
    assert md.startswith("# Daily summary: Thursday, September 24, 2026")
    pill = md.split("## Pill bottle")[1].split("##")[0]
    assert "8:02 AM" in pill and "8:15 PM" in pill
    missing = md.split("## Items that went missing")[1].split("##")[0]
    assert "Keys" in missing and "inside the box at 3:10 PM" in missing
    assert "Glasses" in missing and "under the notebook at 6 PM" in missing
    assert "Remote" in missing and "Wallet" not in missing
    reminders = md.split("## Reminders")[1].split("##")[0]
    assert "7 AM" in reminders and "acknowledged at 7 AM" in reminders
    assert "5 PM" in reminders and "not acknowledged" in reminders
    lost = md.split("## Lost track")[1].split("##")[0]
    assert "Remote" in lost and "7 PM" in lost
    for line in md.splitlines():
        safe(line)


def test_caregiver_summary_no_pickups_and_text_format(events):
    w = FakeWorld([Entity("pill_bottle", "target", Status.VISIBLE, pos_cm=(10.0, 10.0)),
                   Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 10.0))], events)
    ev(events, "keys", "MOVED", at(10, day=24), to_cm=(20.0, 10.0))
    txt = caregiver_summary("2026-09-24", w, events, CFG, fmt="text", now=at(7, 30))
    assert "#" not in txt and "No pickups of the pill bottle were seen" in txt
    assert "Nothing went missing." in txt
    safe(txt)


def test_summary_data_is_json_friendly(w, events):
    import json
    d = summary_data(date(2026, 9, 24), w, events, CFG, now=at(7, 30))
    json.dumps(d)
    assert [p["t"] for p in d["pill_bottle_pickups"]] == [at(8, 2, day=24), at(20, 15, day=24)]
    assert {m["obj"] for m in d["missing"]} == {"keys", "glasses", "remote"}


def test_spoken_summary(w, events):
    text = spoken_summary(date(2026, 9, 24), w, events, CFG, now=at(7, 30))
    assert len(sentences(text)) <= 3 and "8:02 AM" in text
    safe(text)


# ---------------------------------------------------------------- the user's seat (core/viewframe.py)

SEAT_OFF = {"bottom": "the table on your left", "top": "the table on your right",
            "right": "the far side of the table", "left": "the side of the table nearest you"}


def seat(front):
    return {**CFG, "table": {**CFG["table"], "size_cm": [100, 60]}, "table_area": {"polygon_cm": []},
            "viewer": {"front": front}}


@pytest.mark.parametrize("front", SEAT_OFF)
def test_care_words_say_sides_from_the_seat_and_sweep_the_camera_edge(w, events, front):
    from core.carewords import event_place, where_sentence
    from core.reports import _object_clause
    cfg = seat(front)
    s, action = where_sentence(w, cfg, "phone", named=True)
    assert s == f"Your phone was carried off {SEAT_OFF[front]}." and action == "sweep:left"
    gone = events.last_of_type("phone", ["EXITED_VIEW"])
    assert event_place(gone, w, cfg) == f"off {SEAT_OFF[front]}"
    assert _object_clause(w, cfg, "phone", gone, Status.GONE) == f"your phone went off {SEAT_OFF[front]} at 9 AM"
    moved = Event(t=0.0, wall=0.0, obj="wallet", type="MOVED", to_cm=(90.0, 5.0))
    assert event_place(moved, w, cfg) == "on the table, " + {
        "bottom": "at the far right", "top": "on your left, near you",
        "right": "on your right, near you", "left": "at the far left"}[front]
