"""Narration memory in questions and answers: time windows, the WHAT_DOING intent, 'while I was away',
object questions enriched with the latest narration, fallback to the event log, hedging, and the
medication rule on the way out."""
import re
import time
from datetime import datetime, timedelta

import pytest

from core.config import load_config
from core.fakeworld import FakeWorld, demo_world
from core.narration_store import NarrationStore, has_time_phrase, parse_window
from core.types import Entity, Event, Status
from voice.answers import answer
from voice.intents import parse

CFG = load_config()
NOW_DT = datetime(2026, 9, 25, 15, 30, 0)          # local 3:30 PM
NOW = NOW_DT.timestamp()


def at(h, m=0, s=0, days=0):
    return (datetime(2026, 9, 25, h, m, s) + timedelta(days=days)).timestamp()


def sentences(text):
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


def spoken_ok(a):
    assert a.text and a.text[0].isupper() and a.text.endswith((".", "?")), a.text
    assert not re.search(r"[*#`_\[\]]|thing:|None", a.text), a.text
    assert len(sentences(a.text)) <= 3, a.text


def narrate(events, t0, summary, conf=0.9, secs=40, objects=(), tags=()):
    st = NarrationStore(events)
    i = st.add_pending(t0, t0 + secs, None, {})
    st.mark_done(i, summary, {"summary": summary, "confidence": conf, "objects_involved": list(objects),
                              "activity_tags": list(tags), "actions": []}, "fake", "f", 5)
    return i


def ask(world, text, now=NOW, **kw):
    return answer(parse(text, CFG), world, world.events, CFG, now=now, **kw)


@pytest.fixture
def fw():
    return FakeWorld([Entity("keys", "target", Status.INSIDE, parent="box", pos_cm=(10.0, 10.0)),
                      Entity("box", "container", Status.VISIBLE, pos_cm=(30.0, 30.0)),
                      Entity("pill_bottle", "target", Status.VISIBLE, pos_cm=(50.0, 30.0))])


# ---------------------------------------------------------------- intents

@pytest.mark.parametrize("text,kind,obj,name", [
    ("What was I doing before lunch?", "WHAT_DOING", None, None),
    ("what did I do this morning", "WHAT_DOING", None, None),
    ("What was I doing at 3?", "WHAT_DOING", None, None),
    ("What have I been up to today?", "WHAT_DOING", None, None),
    ("what was I doing?", "WHAT_DOING", None, None),
    ("what happened this morning", "WHAT_DOING", None, None),
    ("What happened at 3 p.m.?", "WHAT_DOING", None, None),
    ("Summarize my afternoon.", "WHAT_DOING", None, None),
    ("What did I do with my keys?", "HISTORY", "keys", None),
    ("what was I doing with my keys", "HISTORY", "keys", None),
    ("What did I do with my charger?", "HISTORY", None, "charger"),
    ("Did I use the stove?", "HANDLED", None, "stove"),
    ("Did I use my keys?", "HANDLED", "keys", None),
    ("What happened while I was away?", "CHANGES", None, None),
    ("What did I miss?", "CHANGES", None, None),
    ("Did I take my medication this morning?", "HANDLED", "pill_bottle", None),
])
def test_intents(text, kind, obj, name):
    it = parse(text, CFG)
    assert (it.kind, it.obj, it.name) == (kind, obj, name), text


# ---------------------------------------------------------------- time windows

@pytest.mark.parametrize("text,t0,t1,label", [
    ("what did I do this morning", at(5), at(12), "this morning"),
    ("what was I doing before lunch", at(0), at(12), "before lunch"),
    ("what did I do after lunch", at(13), NOW, "after lunch"),
    ("what was I doing at 3", at(14, 30), at(15, 30), "around 3:00 PM"),
    ("what was I doing at 3:30", at(15), NOW, "around 3:30 PM"),
    ("what was I doing at 9", at(8, 30), at(9, 30), "around 9:00 AM"),
    ("what was I doing at 4", at(3, 30), at(4, 30), "around 4:00 AM"),
    ("what was I doing at 9 p.m.", at(20, 30, days=-1), at(21, 30, days=-1), "around 9:00 PM yesterday"),
    ("what happened yesterday", at(0, days=-1), at(0), "yesterday"),
    ("what did I do today", at(0), NOW, "today"),
    ("what was I doing an hour ago", at(14, 15), at(14, 45), "an hour ago"),
    ("what was I doing 10 minutes ago", at(15, 15), at(15, 25), "10 minutes ago"),
    ("what happened in the last 20 minutes", at(15, 10), NOW, "in the last 20 minutes"),
    ("what happened in the past hour", at(14, 30), NOW, "in the last hour"),
    ("what was I doing this afternoon", at(12), NOW, "this afternoon"),
])
def test_parse_window(text, t0, t1, label):
    w = parse_window(text, NOW)
    assert w is not None, text
    assert (abs(w.t0 - t0), abs(w.t1 - t1), w.label) == (0, 0, label), (text, w)
    assert has_time_phrase(text)


@pytest.mark.parametrize("text", ["what was I doing", "where are my keys", "what's in the box",
                                  "what happened while I was away"])
def test_no_window(text):
    assert parse_window(text, NOW) is None and not has_time_phrase(text)


# ---------------------------------------------------------------- WHAT_DOING answers

def test_what_doing_in_a_window_cites_times_and_skips_others(fw):
    narrate(fw.events, at(10, 5), "You sorted some papers, then put your keys in the box.")
    narrate(fw.events, at(11, 40), "You moved the box to the right.")
    narrate(fw.events, at(14, 0), "You read the newspaper.")
    a = ask(fw, "What was I doing before lunch?")
    spoken_ok(a)
    assert "10:05 AM" in a.text and "11:40 AM" in a.text and "newspaper" not in a.text
    assert "you sorted some papers" in a.text and a.text.index("10:05") < a.text.index("11:40")


def test_what_doing_without_a_time_is_the_latest(fw):
    narrate(fw.events, at(14, 0), "You read the newspaper.")
    narrate(fw.events, at(15, 10), "You put your keys in the box. Then you moved the box.")
    a = ask(fw, "what was I doing?")
    spoken_ok(a)
    assert a.text == "At 3:10 PM, you put your keys in the box. Then you moved the box."


def test_many_narrations_mention_the_rest(fw):
    for m in (0, 10, 20, 30):
        narrate(fw.events, at(9, m), f"You moved the box {m} times.")
    a = ask(fw, "what did I do this morning")
    spoken_ok(a)
    assert "9:20 AM" in a.text and "9:30 AM" in a.text and "9:00 AM" not in a.text
    assert "2 more" in a.text


def test_low_confidence_is_hedged(fw):
    narrate(fw.events, at(15, 0), "You wrote in the notebook.", conf=0.3)
    a = ask(fw, "what was I doing?")
    assert a.text == "At 3:00 PM, it looked like you wrote in the notebook."


def test_unclear_narration_is_not_used(fw):
    narrate(fw.events, at(15, 0), "Unclear.", conf=0.1)
    a = ask(fw, "what was I doing?")
    assert "unclear" not in a.text.lower()


def test_falls_back_to_the_event_log(fw):
    fw.events.add(Event(t=1.0, wall=at(14, 2, 11), obj="keys", type="PICKED_UP", parent="hand:1"))
    fw.events.add(Event(t=5.0, wall=at(14, 2, 15), obj="keys", type="PUT_INSIDE", parent="box"))
    a = ask(fw, "what was I doing at 2?")
    spoken_ok(a)
    assert a.text == ("I don't have a description of that, but the keys were picked up at 2:02 PM "
                      "and put inside the box.")


def test_falls_back_with_two_objects(fw):
    fw.events.add(Event(t=1.0, wall=at(9, 2), obj="keys", type="PICKED_UP", parent="hand:1"))
    fw.events.add(Event(t=5.0, wall=at(9, 3), obj="keys", type="PUT_INSIDE", parent="box"))
    fw.events.add(Event(t=9.0, wall=at(9, 10), obj="box", type="MOVED"))
    a = ask(fw, "what did I do this morning")
    spoken_ok(a)
    assert a.text.startswith("I don't have a description of that, but ")
    assert "keys were picked up at 9:02 AM" in a.text and "box was moved at 9:10 AM" in a.text


def test_nothing_seen(fw):
    a = ask(fw, "what did I do this morning")
    assert a.text == "I didn't see anything happen on the table this morning."
    a = ask(fw, "what was I doing?")
    assert a.text == "I haven't seen anything happen on the table lately."


def test_what_doing_never_says_pills_were_taken(fw):
    st = NarrationStore(fw.events)
    i = st.add_pending(at(15, 0), at(15, 1), None, {})
    with fw.events._locked():                        # an unredacted row, as if written by an old build
        c = fw.events._conn()
        c.execute("UPDATE narrations SET status='done', summary=?, json='{}' WHERE id=?",
                  ("You opened the pill bottle. You took your pills.", i))
        c.commit()
    a = ask(fw, "what was I doing?")
    assert "took" not in a.text and "taken" not in a.text and "pill bottle" in a.text


# ---------------------------------------------------------------- while I was away

def test_away_uses_narrations_since_the_last_question_in_third_person(fw):
    fw.events.log_question("where are my keys", "WHERE", "keys", "…", True, 10, t=NOW - 3600)
    narrate(fw.events, NOW - 7200, "You read the newspaper.")
    narrate(fw.events, NOW - 1800, "You put your keys in the box, then you moved the box.")
    a = ask(fw, "What happened while I was away?")
    spoken_ok(a)
    assert "newspaper" not in a.text
    assert "someone put your keys in the box, then they moved the box" in a.text, a.text
    assert "3:00 PM" in a.text


def test_away_without_narrations_uses_events_since_the_last_question(fw):
    fw.events.log_question("where are my keys", "WHERE", "keys", "…", True, 10, t=NOW - 3600)
    fw.events.add(Event(t=1.0, wall=NOW - 4000, obj="box", type="MOVED"))
    fw.events.add(Event(t=2.0, wall=NOW - 600, obj="keys", type="PICKED_UP", parent="hand:1"))
    a = ask(fw, "What happened while I was away?")
    assert a.text == "The keys were picked up 10 minutes ago."


def test_away_ignores_a_question_asked_seconds_ago(fw):
    fw.events.log_question("where are my keys", "WHERE", "keys", "…", True, 10, t=NOW - 20)
    fw.events.add(Event(t=2.0, wall=NOW - 300, obj="keys", type="PICKED_UP", parent="hand:1"))
    assert ask(fw, "what did I miss?").text == "The keys were picked up 5 minutes ago."


def test_plain_changes_is_unchanged_by_narrations(fw):
    narrate(fw.events, NOW - 100, "You read the newspaper.")
    fw.events.add(Event(t=2.0, wall=NOW - 300, obj="keys", type="PICKED_UP", parent="hand:1"))
    assert ask(fw, "What changed?").text == "The keys were picked up 5 minutes ago."


# ---------------------------------------------------------------- objects

def test_history_is_enriched_with_the_latest_relevant_narration():
    w = demo_world()
    base = answer(parse("what happened to my keys", CFG), w, w.events, CFG).text
    now = time.time()
    narrate(w.events, now - 380, "You sorted some papers, then put your keys in the box.", objects=["keys"])
    narrate(w.events, now - 60, "You read the newspaper.")
    a = answer(parse("what happened to my keys", CFG), w, w.events, CFG, now=now)
    spoken_ok(a)
    assert a.text.startswith(base)
    assert a.text.endswith("you sorted some papers, then put your keys in the box.")
    assert "newspaper" not in a.text


def test_handled_known_object_gets_the_narration_too(fw):
    fw.events.add(Event(t=1.0, wall=NOW - 300, obj="keys", type="PICKED_UP", parent="hand:1"))
    narrate(fw.events, NOW - 310, "You picked up your keys and dropped them in the box.")
    a = ask(fw, "Did I use my keys?")
    spoken_ok(a)
    assert a.text.startswith("The keys were picked up at")
    assert "dropped them in the box" in a.text


def test_unknown_name_is_answered_from_narrations(fw):
    narrate(fw.events, at(12, 15), "You turned on the stove and put a pan on it.", conf=0.8)
    a = ask(fw, "Did I use the stove?")
    spoken_ok(a)
    assert a.text == "At 12:15 PM, you turned on the stove and put a pan on it."


def test_unknown_name_without_narration_keeps_the_teach_prompt(fw):
    a = ask(fw, "Did I use the stove?")
    assert a.text.startswith("I don't know what your stove is yet.")


class AliasWorld(FakeWorld):
    """FakeWorld plus the open-world read API (find / alias_phrases / thing_labels)."""

    def find(self, name):
        n = name.lower()
        return "thing:1" if n in ("charger", "phone charger") else (n if n in self.entities else None)

    def alias_phrases(self):
        return ["phone charger"]

    def thing_labels(self):
        return {"thing:1": "phone charger"}


def test_taught_alias_matches_narrations():
    w = AliasWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(10.0, 10.0)),
                    Entity("thing:1", "target", Status.VISIBLE, pos_cm=(40.0, 30.0),
                           aliases=["phone charger"])])
    w.events.add(Event(t=1.0, wall=NOW - 500, obj="thing:1", type="MOVED"))
    narrate(w.events, NOW - 510, "You plugged the phone charger into your phone.", objects=["phone charger"])
    a = answer(parse("What did I do with my charger?", CFG), w, w.events, CFG, now=NOW)
    spoken_ok(a)
    assert "phone charger was moved" in a.text.lower() and "plugged the phone charger" in a.text


def test_pill_bottle_handled_with_narration_never_says_taken(fw):
    fw.events.add(Event(t=1.0, wall=NOW - 300, obj="pill_bottle", type="PICKED_UP", parent="hand:1"))
    fw.events.add(Event(t=3.0, wall=NOW - 290, obj="pill_bottle", type="PUT_BACK"))
    narrate(fw.events, NOW - 305, "You opened the pill bottle and put it back.", objects=["pill bottle"])
    for q in ["Did I take my medication?", "Did I take my pills this morning?", "What did I do with my pills?"]:
        a = ask(fw, q)
        spoken_ok(a)
        assert not re.search(r"\b(took|taken|take|swallow\w*)\b", a.text.lower()), (q, a.text)
    assert "opened the pill bottle" in ask(fw, "Did I take my medication?").text
