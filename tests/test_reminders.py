"""Reminders (core/reminders.py) and the care wording helpers (core/carewords.py): spoken time parsing,
request parsing, condition reminders against synthetic events, dedupe / expiry / acknowledgement, quiet
hours and caps, persistence across restarts, and the pill rule on every sentence produced."""
import re
from datetime import datetime, timedelta

import pytest

from core.carewords import pill_claim, pill_guard, say_time
from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.reminders import ReminderStore, Reminders, parse_request, parse_when
from core.types import Entity, Event, Status

CFG = load_config()
# The tests pin their own quiet hours so they don't depend on the repo's setting (off for the hackathon).
CFG = {**CFG, "care": {**(CFG.get("care") or {}), "quiet_hours": ["22:00", "07:00"]}}
SAID: list[str] = []            # every sentence the tests saw generated; checked against the pill rule


def at(h, m=0, s=0, day=25):
    return (datetime(2026, 9, day, 0, 0, 0) + timedelta(hours=h, minutes=m, seconds=s)).timestamp()


def hm(t):
    d = datetime.fromtimestamp(t)
    return d.day, d.hour, d.minute


def world(events):
    return FakeWorld([
        Entity("pill_bottle", "target", Status.UNDER, parent="notebook", pos_cm=(20.0, 40.0), last_seen=at(7)),
        Entity("notebook", "cover", Status.VISIBLE, pos_cm=(20.5, 39.0), last_seen=at(7)),
        Entity("keys", "target", Status.INSIDE, parent="box", pos_cm=(41.0, 29.0), last_seen=at(7)),
        Entity("box", "container", Status.VISIBLE, pos_cm=(70.0, 38.0), last_seen=at(7)),
        Entity("glasses", "target", Status.VISIBLE, pos_cm=(5.0, 5.0), last_seen=at(7)),
    ], events)


class TaughtWorld(FakeWorld):
    """FakeWorld plus the open-world lookups: 'charger' was taught to thing:1."""

    def find(self, name):
        n = name.lower().strip()
        n = {"charger": "thing:1", "my charger": "thing:1", "chargers": "thing:1"}.get(n, n)
        n = (CFG.get("synonyms") or {}).get(n, n).replace(" ", "_")
        return n if n in self.entities else None

    def alias_phrases(self):
        return ["charger"]

    def thing_labels(self):
        return {"thing:1": "charger"}


def ev(events, obj, typ, wall, **kw):
    events.add(Event(t=wall, wall=wall, obj=obj, type=typ, **kw))


@pytest.fixture
def events(tmp_path):
    e = EventLog(":memory:", str(tmp_path / "snaps"))
    yield e
    e.close()


@pytest.fixture
def rem(events):
    return Reminders(CFG, world(events), events)


def said(text):
    SAID.append(text)
    return text


# ---------------------------------------------------------------- spoken times

@pytest.mark.parametrize("text,now,day,h,m,daily", [
    ("at 8", at(7), 25, 8, 0, False),                    # before 8 am: this morning
    ("at 8", at(9), 25, 20, 0, False),                   # after 8 am: tonight
    ("at 8", at(21), 26, 8, 0, False),                   # after 8 pm: tomorrow morning
    ("at 8", at(8), 25, 20, 0, False),                   # exactly 8:00: the next one
    ("at 8 pm", at(9), 25, 20, 0, False),
    ("at 8 p.m.", at(21), 26, 20, 0, False),
    ("at 8 am", at(9), 26, 8, 0, False),
    ("at 8 a.m.", at(7), 25, 8, 0, False),
    ("at 8:30 PM", at(9), 25, 20, 30, False),
    ("at 8.30", at(7), 25, 8, 30, False),
    ("at 20:15", at(7), 25, 20, 15, False),
    ("at 8 o'clock", at(7), 25, 8, 0, False),
    ("at eight", at(9), 25, 20, 0, False),
    ("at eight thirty", at(7), 25, 8, 30, False),
    ("at 12", at(11, 59), 25, 12, 0, False),             # noon is next
    ("at 12", at(12, 30), 26, 0, 0, False),              # midnight is next
    ("at 12", at(23, 30), 26, 0, 0, False),
    ("at 12", at(0, 10), 25, 12, 0, False),
    ("at 12 am", at(9), 26, 0, 0, False),
    ("at 12 pm", at(13), 26, 12, 0, False),
    ("at noon", at(13), 26, 12, 0, False),
    ("at midnight", at(23), 26, 0, 0, False),
    ("at 9 in the morning", at(10), 26, 9, 0, False),
    ("at 7 in the evening", at(6), 25, 19, 0, False),
    ("at 8 tonight", at(7), 25, 20, 0, False),
    ("tomorrow at 8", at(7), 26, 8, 0, False),
    ("tomorrow at 8 pm", at(7), 26, 20, 0, False),
    ("in 10 minutes", at(23, 55), 26, 0, 5, False),      # across midnight
    ("in an hour", at(9, 30), 25, 10, 30, False),
    ("in half an hour", at(9, 45), 25, 10, 15, False),
    ("in 2 hours", at(23), 26, 1, 0, False),
    ("every day at 8", at(7), 25, 8, 0, True),
    ("every day at 8", at(9), 25, 20, 0, True),
    ("daily at 8 am", at(9), 26, 8, 0, True),
    ("every morning at 8", at(21), 26, 8, 0, True),
    ("at 9 every day", at(10), 25, 21, 0, True),
    ("every night at 10", at(7), 25, 22, 0, True),
])
def test_parse_when(text, now, day, h, m, daily):
    w = parse_when(text, now)
    assert w is not None, text
    assert hm(w.due) == (day, h, m), (text, datetime.fromtimestamp(w.due))
    assert w.daily is daily
    if not w.relative:
        assert w.hm == (h, m)


@pytest.mark.parametrize("text", ["remind me later", "at the table", "in the box", "every day", "at 25"])
def test_parse_when_none(text):
    assert parse_when(text, at(9)) is None


@pytest.mark.parametrize("h,m,out", [(8, 0, "8 AM"), (20, 30, "8:30 PM"), (12, 0, "noon"), (0, 0, "midnight"),
                                     (12, 15, "12:15 PM"), (0, 5, "12:05 AM"), (21, 0, "9 PM")])
def test_say_time(h, m, out):
    assert say_time(h, m) == out


# ---------------------------------------------------------------- the pill rule

@pytest.mark.parametrize("text,bad", [
    ("You haven't taken your pills.", True),
    ("You took your medication at 8.", True),
    ("Did you miss your meds?", True),
    ("It was swallowed.", True),
    ("You skipped a dose.", True),
    ("Remember to take your pills.", True),
    ("Time for your pills.", False),
    ("It's 9 o'clock and you haven't picked up your pill bottle yet.", False),
    ("Your pill bottle was picked up at 8:02 AM.", False),
    ("Only pill bottle pickups are shown here; this is not a medication record.", False),
])
def test_pill_claim(text, bad):
    assert pill_claim(text) is bad


def test_pill_guard_drops_claims_only():
    assert pill_guard("Your pill bottle is under the notebook. You took your pills.") == \
        "Your pill bottle is under the notebook."
    assert pill_guard("You took your pills.")          # never empty
    assert not pill_claim(pill_guard("You took your pills."))


# ---------------------------------------------------------------- request parsing

def test_parse_condition_request(events):
    w = world(events)
    r = parse_request("Remind me if I haven't picked up my pill bottle by 9", CFG, w, at(8, 30))
    assert (r.action, r.kind, r.obj, r.daily, r.since) == ("create", "condition", "pill_bottle", False, None)
    assert hm(r.due) == (25, 9, 0)
    r = parse_request("remind me if I haven't picked up my pills by 9 since 8", CFG, w, at(8, 30))
    assert (r.obj, r.since) == ("pill_bottle", (8, 0)) and hm(r.due) == (25, 9, 0)
    r = parse_request("Remind me every day if I haven't touched my medicine by 8 pm.", CFG, w, at(9))
    assert (r.kind, r.obj, r.daily) == ("condition", "pill_bottle", True) and hm(r.due) == (25, 20, 0)
    r = parse_request("remind me if the pill bottle hasn't been picked up by 9", CFG, w, at(10))
    assert (r.kind, r.obj) == ("condition", "pill_bottle") and hm(r.due) == (25, 21, 0)


def test_parse_time_request(events):
    w = world(events)
    r = parse_request("Remind me to take my pills at 8.", CFG, w, at(7))
    assert (r.action, r.kind, r.obj, r.task) == ("create", "time", "pill_bottle", "take my pills")
    r = parse_request("remind me at 3 to call Sarah", CFG, w, at(9))
    assert (r.kind, r.obj, r.task) == ("time", None, "call Sarah") and hm(r.due) == (25, 15, 0)
    r = parse_request("Remind me in 10 minutes to check the oven", CFG, w, at(9))
    assert (r.task, hm(r.due)) == ("check the oven", (25, 9, 10))
    r = parse_request("remind me every day at 8 to take my pills", CFG, w, at(7))
    assert r.daily and r.obj == "pill_bottle"


def test_parse_request_taught_alias(events):
    w = TaughtWorld([Entity("thing:1", "target", Status.VISIBLE, pos_cm=(10.0, 10.0))], events)
    r = parse_request("remind me if I haven't picked up my charger by 5 pm", CFG, w, at(9))
    assert (r.kind, r.obj) == ("condition", "thing:1")


@pytest.mark.parametrize("text,action", [
    ("remind me to take my keys when I leave the table", "unsupported"),
    ("remind me to call Sarah", "unclear"),
    ("What are my reminders?", "list"),
    ("do I have any reminders", "list"),
    ("cancel my 8 o'clock reminder", "cancel"),
    ("delete all my reminders", "cancel"),
    ("where are my keys", None),
    ("what did I do this morning", None),
])
def test_parse_request_actions(events, text, action):
    r = parse_request(text, CFG, world(events), at(9))
    assert (r.action if r else None) == action


# ---------------------------------------------------------------- condition reminders

def test_condition_fires_without_pickup(rem, events):
    a = rem.handle("Remind me if I haven't picked up my pill bottle by 9", at(8, 30))
    said(a.text)
    assert "9 AM" in a.text and "pill bottle" in a.text
    assert rem.tick(at(8, 59, 59)) == []
    out = rem.tick(at(9, 0, 5))
    assert len(out) == 1
    n, ans = out[0]
    assert ans.text == "It's 9 o'clock and you haven't picked up your pill bottle yet. It's under the notebook."
    assert (ans.point_at, ans.action) == ("pill_bottle", "point")
    assert (n.kind, n.point_at, n.acknowledged) == ("condition", "pill_bottle", False)
    said(ans.text)
    assert rem.tick(at(9, 0, 15)) == [] and rem.tick(at(9, 30)) == []          # one firing only
    assert rem.store.active() == []                                             # one-shot: done


@pytest.mark.parametrize("typ", ["PICKED_UP", "MOVED", "PUT_BACK"])
def test_condition_never_fires_after_pickup(rem, events, typ):
    rem.handle("remind me if I haven't picked up my pill bottle by 9", at(8, 30))
    ev(events, "pill_bottle", typ, at(8, 45))
    assert rem.tick(at(9, 0, 5)) == []
    assert rem.tick(at(9, 10)) == []
    assert rem.store.active() == []


def test_condition_window(rem, events):
    rem.handle("remind me if I haven't picked up my pill bottle by 9", at(8, 30))
    ev(events, "pill_bottle", "PICKED_UP", at(20, day=24))       # yesterday: outside today's window
    ev(events, "keys", "PICKED_UP", at(8, 40))                    # another object
    ev(events, "pill_bottle", "COVERED", at(8, 41), parent="notebook")   # not a pickup
    assert len(rem.tick(at(9, 0, 5))) == 1


def test_condition_since(rem, events):
    rem.handle("remind me if I haven't picked up my pill bottle by 9 since 8", at(7))
    ev(events, "pill_bottle", "PICKED_UP", at(7, 30))             # before 'since 8'
    assert len(rem.tick(at(9, 0, 5))) == 1
    rem.handle("remind me if I haven't picked up my pill bottle by 9 pm since 8 pm", at(10))
    ev(events, "pill_bottle", "PICKED_UP", at(20, 10))
    assert rem.tick(at(21, 0, 5)) == []


def test_daily_condition_once_per_day(rem, events):
    rem.handle("remind me every day if I haven't picked up my pill bottle by 9", at(8))
    assert len(rem.tick(at(9, 0, 5))) == 1
    assert rem.tick(at(9, 0, 15)) == [] and rem.tick(at(12)) == []
    assert len(rem.store.active()) == 1                           # daily stays active
    ev(events, "pill_bottle", "PICKED_UP", at(8, 30, day=26))
    assert rem.tick(at(9, 0, 5, day=26)) == []                     # picked up on day 2
    out = rem.tick(at(9, 0, 5, day=27))                            # not on day 3
    assert len(out) == 1 and out[0][0].day == "2026-09-27"


def test_late_reminder_is_skipped_not_fired(rem):
    rem.handle("remind me if I haven't picked up my pill bottle by 9", at(8))
    assert rem.tick(at(10)) == []                                  # scheduler was down past the grace
    assert rem.store.active() == []


def test_things_resolve_and_fire(events):
    w = TaughtWorld([Entity("thing:1", "target", Status.INSIDE, parent="box", pos_cm=(10.0, 10.0)),
                     Entity("box", "container", Status.VISIBLE, pos_cm=(30.0, 30.0))], events)
    rem = Reminders(CFG, w, events)
    rem.handle("remind me if I haven't picked up my charger by 5 pm", at(9))
    (n, ans), = rem.tick(at(17, 0, 5))
    assert ans.text == "It's 5 o'clock and you haven't picked up your charger yet. It's inside the box."
    assert ans.point_at == "thing:1"


# ---------------------------------------------------------------- time reminders

def test_time_reminder_pills_is_neutral(rem):
    a = rem.handle("Remind me to take my pills at 8", at(7))
    said(a.text)
    assert "8 AM" in a.text and not pill_claim(a.text)
    (n, ans), = rem.tick(at(8, 0, 5))
    assert ans.text == "It's 8 o'clock. Time for your pills. Your pill bottle is under the notebook."
    assert ans.point_at == "pill_bottle" and n.kind == "reminder"
    said(ans.text)


def test_time_reminder_task(rem):
    said(rem.handle("remind me at 3:30 to call my daughter", at(9)).text)
    (n, ans), = rem.tick(at(15, 30, 2))
    assert ans.text == "It's 3:30. Time to call your daughter." and ans.point_at is None
    said(ans.text)


def test_time_reminder_with_object(rem):
    rem.handle("remind me to take my keys at 5 pm", at(9))
    (n, ans), = rem.tick(at(17, 0, 2))
    assert ans.text == "It's 5 o'clock. Time to take your keys. They're inside the box."
    assert ans.point_at == "keys"


# ---------------------------------------------------------------- notices: ack, expiry, quiet, cap

def test_ack_within_window(rem):
    rem.handle("remind me if I haven't picked up my pill bottle by 9", at(8))
    (n, _), = rem.tick(at(9, 0, 5))
    assert rem.acknowledge("where are my keys", at(9, 0, 20)) is None       # not an ack
    a = rem.acknowledge("Okay, thanks!", at(9, 0, 30))
    assert a is not None and said(a.text)
    assert rem.store.notice(n.id).acknowledged
    assert rem.acknowledge("okay", at(9, 0, 40)) is None                    # nothing left to ack


@pytest.mark.parametrize("word", ["okay", "thanks", "done", "I did it", "got it", "ok thank you"])
def test_ack_words(rem, word):
    rem.handle("remind me at 9 to call Sarah", at(8))
    rem.tick(at(9, 0, 5))
    assert rem.acknowledge(word, at(9, 0, 50)) is not None


def test_ack_too_late(rem):
    rem.handle("remind me at 9 to call Sarah", at(8))
    rem.tick(at(9, 0, 5))
    assert rem.acknowledge("okay", at(9, 2)) is None


def test_notice_expiry_in_json(rem):
    rem.handle("remind me at 9 to call Sarah", at(8))
    (n, _), = rem.tick(at(9, 0, 5))
    js = rem.notices_json(at(9, 10))
    assert [d["id"] for d in js] == [n.id]
    assert set(js[0]) >= {"id", "t", "kind", "text", "point_at", "acknowledged"}
    assert js[0]["acknowledged"] is False
    assert rem.notices_json(at(9, 31)) == []                                # 30 min expiry
    assert rem.acknowledge("okay", at(9, 0, 30)) is not None
    assert rem.notices_json(at(9, 10))[0]["acknowledged"] is True


@pytest.mark.parametrize("h,m,quiet", [(22, 0, True), (23, 30, True), (0, 0, True), (6, 59, True),
                                       (7, 0, False), (12, 0, False), (21, 59, False)])
def test_quiet_hours(rem, h, m, quiet):
    assert rem.quiet(at(h, m)) is quiet


def test_quiet_notice_recorded_but_not_spoken(rem):
    rem.handle("remind me at 11 pm to lock the door", at(9))
    (n, ans), = rem.tick(at(23, 0, 5))
    assert n.spoken is False and ans is None
    assert rem.notices_json(at(23, 1))[0]["id"] == n.id


def test_cap_per_hour(events):
    cfg = {**CFG, "care": {**(CFG.get("care") or {}), "max_notices_per_hour": 2}}
    rem = Reminders(cfg, world(events), events)
    for mm in (0, 1, 2):
        rem.handle(f"remind me at 9:0{mm} to call Sarah", at(8))
    spoken = []
    for mm in (0, 1, 2):
        spoken += [n.spoken for n, _ in rem.tick(at(9, mm, 5))]
    assert spoken == [True, True, False]
    rem.handle("remind me at 10:30 to call Sarah", at(8))
    (n, _), = rem.tick(at(10, 30, 5))
    assert n.spoken is True                                               # an hour later: allowed again


# ---------------------------------------------------------------- list and cancel by voice

def test_list_and_cancel(rem):
    assert said(rem.handle("what are my reminders?", at(7)).text) == "You don't have any reminders."
    rem.handle("remind me every day if I haven't picked up my pill bottle by 8", at(7))
    rem.handle("remind me at 3 to call Sarah", at(7))
    a = rem.handle("What are my reminders?", at(7, 5))
    said(a.text)
    assert a.text.startswith("You have 2 reminders:")
    assert "every day at 8 AM" in a.text and "3 PM" in a.text and "call Sarah" in a.text
    a = rem.handle("cancel my 3 o'clock reminder", at(7, 6))
    said(a.text)
    assert "3 PM" in a.text and len(rem.store.active()) == 1
    rem.handle("cancel the pill bottle reminder", at(7, 7))
    assert rem.store.active() == []


def test_cancel_ambiguous_then_specific(rem):
    rem.handle("remind me every day at 8 am to take my pills", at(7))
    rem.handle("remind me every day at 8 pm to take my pills", at(7))
    a = rem.handle("cancel my 8 o'clock reminder", at(7, 5))
    said(a.text)
    assert "8 AM" in a.text and "8 PM" in a.text and len(rem.store.active()) == 2
    rem.handle("cancel my 8 pm reminder", at(7, 6))
    assert [r.at_h for r in rem.store.active()] == [8]
    said(rem.handle("cancel all my reminders", at(7, 7)).text)
    assert rem.store.active() == []


def test_unsupported_and_unclear(rem):
    a = rem.handle("remind me to take my keys when I leave the table", at(9))
    said(a.text)
    assert "time" in a.text.lower() and rem.store.active() == []
    a = rem.handle("remind me to call Sarah", at(9))
    said(a.text)
    assert "when" in a.text.lower() and rem.store.active() == []


# ---------------------------------------------------------------- persistence

def test_persistence_across_restart(tmp_path):
    db, snaps = str(tmp_path / "care.db"), str(tmp_path / "snaps")
    e1 = EventLog(db, snaps)
    r1 = Reminders(CFG, world(e1), e1)
    r1.handle("remind me every day if I haven't picked up my pill bottle by 9", at(8))
    r1.handle("remind me at 3 pm to call Sarah", at(8))
    (n, _), = r1.tick(at(9, 0, 5))
    e1.close()

    e2 = EventLog(db, snaps)
    r2 = Reminders(CFG, world(e2), e2)
    assert sorted(r.at_h for r in r2.store.active()) == [9, 15]
    assert r2.tick(at(9, 0, 30)) == []                                      # no duplicate after restart
    assert r2.acknowledge("okay", at(9, 0, 40)) is not None
    assert r2.store.notice(n.id).acknowledged
    e2.close()

    e3 = EventLog(db, snaps)
    r3 = ReminderStore(e3)
    assert r3.notice(n.id).acknowledged and len(r3.active()) == 2
    e3.close()


# ---------------------------------------------------------------- the pill rule, everywhere

def test_every_generated_sentence_keeps_the_pill_rule():
    if not SAID:
        pytest.skip("collects the sentences of the tests above; run the whole file")
    for text in SAID:
        for s in re.split(r"(?<=[.!?])\s+", text):
            assert not pill_claim(s), s
            assert not re.search(r"\b(taken|took|swallow\w*|missed)\b", s, re.I), s
