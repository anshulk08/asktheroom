"""Conversation memory (voice/conversation.py): follow-ups resolved from the last turns of the same
source, through the real router (voice.pipeline.make_ask with Grok stubbed), and their expiry."""
import time

import pytest

from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.types import Answer, Entity, Event, Status
from voice.conversation import Conversation
from voice.pipeline import make_ask

CFG = load_config()
NOW = time.time()


def ev(events, obj, typ, wall, **kw):
    events.add(Event(t=wall, wall=wall, obj=obj, type=typ, **kw))


class World(FakeWorld):
    def find(self, name):
        n = (CFG.get("synonyms") or {}).get(name, name).replace(" ", "_")
        return n if n in self.entities else None


def no_grok(text, world, events, cfg, online=True):
    return Answer("grok")


@pytest.fixture
def room(tmp_path):
    events = EventLog(":memory:", str(tmp_path))
    w = World([
        Entity("keys", "target", Status.INSIDE, parent="box", pos_cm=(41.0, 29.0), last_seen=NOW - 95,
               confidence=0.6, candidates=["notebook"]),
        Entity("glasses", "target", Status.VISIBLE, pos_cm=(80.0, 50.0), last_seen=NOW, confidence=1.0),
        Entity("pill_bottle", "target", Status.UNDER, parent="notebook", pos_cm=(20.0, 40.0),
               last_seen=NOW - 300, confidence=0.85),
        Entity("wallet", "target", Status.VISIBLE, pos_cm=(60.0, 15.0), last_seen=NOW),
        Entity("box", "container", Status.VISIBLE, pos_cm=(70.0, 38.0), last_seen=NOW),
        Entity("notebook", "cover", Status.VISIBLE, pos_cm=(20.5, 39.0), last_seen=NOW),
    ], events)
    ev(events, "keys", "APPEARED", NOW - 600, to_cm=(10.0, 10.0))
    ev(events, "keys", "PICKED_UP", NOW - 100, from_cm=(10.0, 10.0), parent="hand:1")
    ev(events, "keys", "PUT_INSIDE", NOW - 95, parent="box")
    ev(events, "pill_bottle", "PICKED_UP", NOW - 400, from_cm=(20.0, 40.0))
    ev(events, "pill_bottle", "PUT_BACK", NOW - 390, to_cm=(20.0, 40.0))
    ev(events, "pill_bottle", "COVERED", NOW - 300, parent="notebook")
    base = make_ask(CFG, w, events, net=None, other=no_grok)
    conv = Conversation(CFG)

    def ask(text, source="voice", now=NOW):
        return conv.ask(text, source, base, w, events, CFG, now=now)

    return ask, base, conv


def test_and_my_glasses_reuses_where(room):
    ask, base, _ = room
    ask("where are my keys?")
    a = ask("and my glasses?", now=NOW + 5)
    assert a.text == base("where are my glasses", "voice").text and a.point_at == "glasses"


def test_what_about_reuses_handled(room):
    ask, base, _ = room
    ask("did anyone touch my pill bottle?")
    a = ask("what about my keys?", now=NOW + 5)
    assert a.text == base("did anyone move my keys", "voice").text
    assert a.text.startswith("The keys were picked up")


def test_when_is_history_of_last_entity(room):
    ask, base, _ = room
    ask("where are my keys?")
    for q in ("when?", "how long ago?", "When was that?"):
        a = ask(q, now=NOW + 5)
        assert a.text == base("what happened to my keys", "voice").text, q
        assert "picked up" in a.text


def test_when_after_handled_stays_handled(room):
    ask, base, _ = room
    ask("did anyone touch my pill bottle?")
    assert ask("when?", now=NOW + 5).text == base("did anyone move my pill bottle", "voice").text


@pytest.mark.parametrize("q", ["point at it again", "show me", "Show me again.", "point to them"])
def test_point_again(room, q):
    ask, _, _ = room
    first = ask("where are my keys?")
    a = ask(q, now=NOW + 5)
    assert (a.point_at, a.action) == (first.point_at, first.action) == ("keys", "point")
    assert a.text == "Here they are."


def test_where_was_it_before_that(room):
    ask, _, _ = room
    ask("where are my keys?")
    a = ask("where were they before that?", now=NOW + 5)
    assert a.text == "Before that, your keys were on the table, near the top left, until 2 minutes ago."
    ask("where is my pill bottle?", now=NOW + 10)
    a = ask("where was it before that?", now=NOW + 15)
    assert a.text.startswith("Before that, your pill bottle was on the table, on the left side")


def test_before_that_without_history(room):
    ask, _, _ = room
    ask("where is my wallet?")
    a = ask("where was it before that?", now=NOW + 5)
    assert a.text == "I don't know where your wallet was before that."


def test_the_other_one(room):
    ask, _, _ = room
    first = ask("where are my keys?")
    assert "notebook" in first.text                              # the answer offered an alternative
    a = ask("what about the other one?", now=NOW + 5)
    assert (a.text, a.point_at) == ("Or they could be under the notebook.", "notebook")
    a = ask("the other one?", now=NOW + 8)
    assert a.text == "That's the only other place I know of." and a.point_at is None


def test_the_other_one_without_alternatives(room):
    ask, _, _ = room
    ask("where is my wallet?")
    assert ask("what about the other one?", now=NOW + 5).text == "I only know of one wallet."


def test_other_one_for_look_alike_things(tmp_path):
    events = EventLog(":memory:", str(tmp_path))

    class Things(World):
        def similar_to(self, name):
            return [("thing:3", 0.8)] if name == "thing:1" else []

        def thing_labels(self):
            return {"thing:1": "mug", "thing:3": None}

        def alias_phrases(self):
            return ["mug"]

        def find(self, name):
            return "thing:1" if name in ("mug", "my mug") else super().find(name)

    w = Things([Entity("thing:1", "target", Status.GONE, pos_cm=(2.0, 30.0), edge="left", last_seen=NOW - 50),
                Entity("thing:3", "target", Status.VISIBLE, pos_cm=(80.0, 10.0), last_seen=NOW)], events)
    conv = Conversation(CFG)
    base = make_ask(CFG, w, events, other=no_grok)
    conv.ask("where is my mug", "voice", base, w, events, CFG, now=NOW)
    a = conv.ask("what about the other one", "voice", base, w, events, CFG, now=NOW + 3)
    assert a.point_at == "thing:3" and "on the table" in a.text


def test_pronoun_follow_up(room):
    ask, base, _ = room
    ask("where are my keys?")
    assert ask("who moved them?", now=NOW + 5).text == base("who moved my keys", "voice").text
    ask("where is my pill bottle?", now=NOW + 6)
    assert ask("did anyone touch it?", now=NOW + 8).text == base("did anyone touch my pill bottle", "voice").text


def test_expiry(room):
    ask, base, conv = room
    ask("where are my keys?")
    a = ask("and my glasses?", now=NOW + 121)
    assert a.text == base("and my glasses?", "voice").text           # not rewritten: context expired
    assert conv.turns("voice", NOW + 121)[-1].question == "and my glasses?"
    assert len(conv.turns("voice", NOW + 400)) == 0


def test_silence_is_measured_from_the_last_turn(room):
    ask, _, _ = room
    ask("where are my keys?")
    ask("and my glasses?", now=NOW + 100)
    assert ask("show me", now=NOW + 200).point_at == "glasses"          # 100 s after the last turn


def test_sources_are_separate(room):
    ask, base, _ = room
    ask("where are my keys?", source="voice")
    assert ask("when?", source="sms", now=NOW + 5).text == base("when?", "sms").text


def test_keeps_last_five_turns(room):
    ask, _, conv = room
    for i in range(7):
        ask("where is my wallet?", now=NOW + i)
    assert len(conv.turns("voice", NOW + 7)) == 5


def test_context_for_the_llm(room):
    ask, _, conv = room
    ask("where are my keys?")
    ask("and my glasses?", now=NOW + 5)
    ctx = conv.context("voice", NOW + 10)
    assert [m["role"] for m in ctx] == ["user", "assistant", "user", "assistant"]
    assert ctx[2]["content"] == "and my glasses?" and "glasses" in ctx[3]["content"]
    assert conv.context("voice", NOW + 500) == []


def test_unrelated_questions_pass_through(room):
    ask, base, _ = room
    ask("where are my keys?")
    for q in ("where is my wallet?", "what changed?", "what's the weather?"):
        assert ask(q, now=NOW + 5).text == base(q, "voice").text


@pytest.mark.parametrize("q", ["what time is it?", "What time is it now", "and then what?", "what about you?"])
def test_not_follow_ups(room, q):
    ask, base, _ = room
    ask("where are my keys?")
    assert ask(q, now=NOW + 5).text == base(q, "voice").text


def test_what_about_after_a_non_object_turn_is_not_an_object_question(room):
    ask, base, conv = room
    ask("what's the weather?")
    q = "what about my appointment tomorrow?"
    assert conv.resolve(q, "voice", None, None, CFG, NOW + 5) is None
    assert ask(q, now=NOW + 5).text == base(q, "voice").text


def test_what_about_a_known_object_after_a_non_object_turn(room):
    ask, base, _ = room
    ask("what's the weather?")
    assert ask("what about my keys?", now=NOW + 5).text == base("where is my keys", "voice").text


def test_and_an_unknown_name_after_an_object_question_is_still_a_follow_up(room):
    _, _, conv = room
    conv.ask("where are my keys?", "voice", lambda t, s: Answer("x"), None, None, CFG, now=NOW)
    f = conv.resolve("and my charger?", "voice", None, None, CFG, NOW + 5)
    assert f is not None and f.text == "where is my charger"
