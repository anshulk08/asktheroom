import json

import pytest
import requests

from core.config import load_config
from core.events import EventLog
from core.fakeworld import demo_world
from voice.pipeline import make_ask
from voice.understand import IGNORE, Understander, gate, schema, sounds_like, system_prompt, to_intent

CFG = load_config()


class StubQwen:
    """Stands in for llama-server: replies with a fixed kind/object, or raises."""

    url = "http://stub/v1"

    def __init__(self, kind="WHERE", obj="keys", raw=None, error=None):
        self.raw = raw if raw is not None else json.dumps({"kind": kind, "object": obj})
        self.error, self.asked = error, []

    def ask(self, text, timeout):
        self.asked.append(text)
        if self.error:
            raise self.error
        return self.raw

    def health(self, timeout=1.0):
        return self.error is None


def understander(**kw):
    return Understander(CFG, qwen=StubQwen(**kw))


def test_rules_answer_what_they_are_sure_of_without_qwen():
    u = understander(kind="OTHER", obj="none")
    for text, kind, obj in [("where are my keys", "WHERE", "keys"), ("what did I miss", "CHANGES", None),
                            ("reset everything", "RESET", None), ("recalibrate", "RECAL", None)]:
        i = u(text)
        assert (i.kind, i.obj) == (kind, obj) and u.last_by == "rules"
    assert u.qwen.asked == []


def test_qwen_reads_what_the_rules_cannot():
    u = understander(kind="HANDLED", obj="pill_bottle")
    i = u("has anybody messed with my meds")
    assert (i.kind, i.obj, i.raw) == ("HANDLED", "pill_bottle", "has anybody messed with my meds")
    assert u.last_by == "qwen"


def test_rules_object_beats_qwens():
    u = understander(kind="WHERE", obj="phone")
    assert u("I've lost the clicker again").obj == "remote"      # clicker is a synonym for the remote


@pytest.mark.parametrize("text, obj, keep", [
    ("wears my wall it", "wallet", True),                # misheard, still sounds like it
    ("where did my coffee mug go", "glasses", False),    # invented
    ("where is the charger", "phone", False),
])
def test_qwen_object_must_sound_like_what_was_said(text, obj, keep):
    i = understander(kind="WHERE", obj=obj)(text)
    assert (i.kind, i.obj) == (("WHERE", obj) if keep else ("OTHER", None))


@pytest.mark.parametrize("stub", [
    dict(error=requests.ConnectionError("refused")),
    dict(error=requests.Timeout("slow")),
    dict(raw="not json"),
    dict(raw=json.dumps({"kind": "RESET", "object": "none"})),      # Qwen may never act on the room
    dict(raw=json.dumps({"kind": "DANCE", "object": "keys"})),
])
def test_rules_answer_when_qwen_cannot(stub):
    u = understander(**stub)
    i = u("can you point at the pill bottle")
    assert (i.kind, i.obj) == ("OTHER", "pill_bottle") and u.last_by == "rules"


def test_last_transcript_is_cached():
    u = understander(kind="WHERE", obj="keys")
    assert u("show me my keys") is u("show me my keys")
    assert u.qwen.asked == ["show me my keys"]


def test_disabled_uses_rules_only():
    cfg = dict(CFG, understand=dict(CFG["understand"], enabled=False))
    u = Understander(cfg)
    assert u.qwen is None and u("show me my keys").kind == "OTHER" and not u.warm()


def test_to_intent_normalizes():
    assert to_intent(json.dumps({"kind": "CHANGES", "object": "keys"}), "anything new with keys", CFG).obj is None
    i = to_intent(json.dumps({"kind": "HANDLED", "object": "none"}), "did anyone touch stuff", CFG)
    assert (i.kind, i.obj) == ("CHANGES", None)
    assert to_intent("[1]", "x", CFG) is None


def test_schema_and_prompt_cover_the_objects():
    s = schema(CFG)["properties"]
    assert set(s["object"]["enum"]) == set(CFG["objects"]) | {"none"}
    assert "RESET" not in s["kind"]["enum"] and "OTHER" in s["kind"]["enum"]
    assert all(o in system_prompt(CFG) for o in CFG["objects"])
    assert sounds_like("notebook", "wheres the note book", CFG)


def test_pipeline_answers_with_qwens_reading(tmp_path):
    events = EventLog(":memory:", str(tmp_path))
    world = demo_world(events)
    ask = make_ask(CFG, world, events, net=None, interpret=understander(kind="WHERE", obj="keys"))
    ans = ask("show me my keys", "voice")                   # the rules alone say OTHER
    assert ans.point_at == "keys" and "box" in ans.text


@pytest.mark.parametrize("text", ["we built this in twenty hours", "that's so cool", ""])
def test_overheard_chatter_without_keywords_never_reaches_qwen(text):
    u = understander(kind="WHERE", obj="keys")
    assert not gate(text, CFG)
    assert u(text, overheard=True).kind == IGNORE and u.qwen.asked == []


@pytest.mark.parametrize("text", ["I'll grab my keys on the way out", "put the wallet in the box",
                                  "yeah the keys are under there now", "hold on let me check my phone"])
def test_overheard_statements_are_not_for_the_rig(text):
    u = understander(kind="WHERE", obj="keys")
    assert u(text, overheard=True).kind == IGNORE and u.qwen.asked == []
    assert u(text).kind != IGNORE                     # asked (clicker) speech never is


@pytest.mark.parametrize("text,kind,obj", [("okay so where's my wallet", "WHERE", "wallet"),
                                           ("hey room, what did I miss", "CHANGES", None),
                                           ("can you show me the remote", "WHERE", "remote")])
def test_overheard_questions_are_answered(text, kind, obj):
    u = understander(kind="WHERE", obj="remote")
    i = u(text, overheard=True)
    assert (i.kind, i.obj) == (kind, obj)


def test_overheard_reset_needs_the_wake_word():
    u = understander(kind="OTHER", obj="none")
    assert u("let's reset after this", overheard=True).kind == IGNORE
    assert u("reset the table", overheard=True).kind == IGNORE
    assert u("room, reset the table", overheard=True).kind == "RESET"


def test_overheard_open_question_needs_an_object_or_the_wake_word():
    u = understander(kind="OTHER", obj="none")
    assert u("where are you guys from", overheard=True).kind == IGNORE
    assert u("room, where are you guys from", overheard=True).kind == "OTHER"
    assert u("what is in the box", overheard=True).kind == "OTHER"
