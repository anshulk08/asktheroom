"""Profile facts (core/profile.py): deterministic extraction of what the user states, dedupe with history,
forgetting, 'what do you know about me', persistence, the optional Grok extraction (with a fake client),
and the routine -> suggested reminder flow through the care layer."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from core.carewords import pill_claim
from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.profile import Profile, extract, make_grok_extractor, validate
from core.types import Answer, Entity, Status
from voice.care import Care

CFG = load_config()


def at(h, m=0, s=0, day=25):
    return (datetime(2026, 9, day) + timedelta(hours=h, minutes=m, seconds=s)).timestamp()


@pytest.fixture
def events(tmp_path):
    e = EventLog(":memory:", str(tmp_path / "s"))
    yield e
    e.close()


@pytest.fixture
def prof(events):
    return Profile(events)


def facts(text):
    return [(s.subject, s.relation, s.value) for s in extract(text)]


# ---------------------------------------------------------------- deterministic patterns

@pytest.mark.parametrize("text,want", [
    ("Call me Grandpa Joe.", [("user", "preferred_name", "Grandpa Joe")]),
    ("you can call me grandpa joe", [("user", "preferred_name", "Grandpa Joe")]),
    ("My name is Joe", [("user", "preferred_name", "Joe")]),
    ("Speak slower, please.", [("user", "speech_rate", "slower")]),
    ("can you talk more slowly", [("user", "speech_rate", "slower")]),
    ("speak louder", [("user", "speech_volume", "louder")]),
    ("My daughter is Sarah.", [("person", "daughter", "Sarah")]),
    ("Sarah is my daughter", [("person", "daughter", "Sarah")]),
    ("my son's name is Tom", [("person", "son", "Tom")]),
    ("My nurse is called Maria Lopez", [("person", "nurse", "Maria Lopez")]),
    ("I take my pills at 8 and 8", [("routine", "take your pills", "at 8 AM and 8 PM")]),
    ("I take my medicine at 9 am and 9 pm", [("routine", "take your medicine", "at 9 AM and 9 PM")]),
    ("I usually take my pills at 8", [("routine", "take your pills", "at 8 AM")]),
    ("I walk the dog every morning", [("routine", "walk the dog", "every morning")]),
    ("I go to church every Sunday at 10", [("routine", "go to church", "every Sunday at 10 AM")]),
])
def test_extract(text, want):
    assert facts(text) == want


@pytest.mark.parametrize("text", [
    "where is my daughter", "I left my keys at 3", "this is my charger", "call this my wallet",
    "where are my pills", "I took my pills at 8", "remind me to take my pills at 8", "is Sarah my daughter",
    "what did I do this morning",
])
def test_extract_ignores(text):
    assert extract(text) == []


def test_routine_times():
    (s,) = extract("I take my pills at 8 and 8")
    assert s.times == [(8, 0), (20, 0)]
    (s,) = extract("I take my pills at 8 and 6")
    assert s.times == [(8, 0), (18, 0)]
    (s,) = extract("I walk the dog every morning")
    assert s.times == []


# ---------------------------------------------------------------- storage

def test_dedupe_keeps_history(prof):
    prof.remember("person", "daughter", "Sarah", "my daughter is Sarah", now=at(9))
    prof.remember("person", "daughter", "Anna", "my daughter is Anna", now=at(10))
    assert prof.get("person", "daughter") == "Anna"
    assert [f.value for f in prof.history("person", "daughter")] == ["Sarah", "Anna"]
    assert len(prof.facts()) == 1


def test_forget_that_and_everything(prof):
    prof.remember("user", "preferred_name", "Joe", "call me Joe", now=at(9))
    prof.remember("person", "daughter", "Sarah", "my daughter is Sarah", now=at(10))
    a = prof.handle("forget that", at(10, 1)).answer
    assert a.text == "Okay, I've forgotten that your daughter is Sarah."
    assert prof.get("person", "daughter") is None and prof.preferred_name() == "Joe"
    prof.handle("Forget everything you know about me.", at(10, 2))
    assert prof.facts() == []
    assert prof.handle("forget that", at(10, 3)).answer.text == "There's nothing to forget."


def test_what_do_you_know(prof):
    assert prof.handle("what do you know about me?", at(9)).answer.text == \
        "I don't know anything about you yet. You can tell me things like 'call me Joe' or 'my daughter is Sarah'."
    for t in ("call me Grandpa Joe", "my daughter is Sarah", "I take my pills at 8 and 8",
              "I walk the dog every morning", "speak slower"):
        prof.handle(t, at(9))
    a = prof.handle("What do you know about me?", at(9, 5)).answer
    assert a.text == ("You asked me to call you Grandpa Joe. Your daughter is Sarah. "
                      "Your routines: your pills at 8 AM and 8 PM; you walk the dog every morning. "
                      "You asked me to speak slower.")
    assert not pill_claim(a.text)


def test_statement_confirmations(prof):
    assert prof.handle("call me Grandpa Joe", at(9)).answer.text == "Okay, I'll call you Grandpa Joe."
    assert prof.handle("my daughter is Sarah", at(9)).answer.text == "Got it, your daughter is Sarah."
    r = prof.handle("I take my pills at 8 and 8", at(9))
    assert r.answer.text == "Got it: your pills at 8 AM and 8 PM."
    assert r.suggest is not None and r.suggest.obj_phrase == "take your pills"
    assert r.suggest.times == [(8, 0), (20, 0)]
    assert prof.handle("speak slower", at(9)).answer.text == \
        "Okay, I've noted that you'd like me to speak slower."
    assert prof.handle("I walk the dog every morning", at(9)).suggest is None
    assert prof.speech() == {"speech_rate": "slower"}


def test_persistence(tmp_path):
    db = str(tmp_path / "p.db")
    e1 = EventLog(db, str(tmp_path / "s"))
    Profile(e1).handle("call me Grandpa Joe", at(9))
    e1.close()
    e2 = EventLog(db, str(tmp_path / "s"))
    assert Profile(e2).preferred_name() == "Grandpa Joe"
    e2.close()


# ---------------------------------------------------------------- Grok extraction (fake client)

def test_validate_rejects_inference_and_unknown_relations():
    text = "I love gardening and my granddaughter Lily visits on Sundays"
    raw = {"facts": [
        {"subject": "user", "relation": "likes", "value": "gardening"},
        {"subject": "person", "relation": "granddaughter", "value": "Lily"},
        {"subject": "user", "relation": "diagnosis", "value": "dementia"},          # not a relation we keep
        {"subject": "user", "relation": "likes", "value": "tomatoes"},             # never said
        {"subject": "person", "relation": "", "value": "Lily"},
        "junk",
    ]}
    got = [(s.subject, s.relation, s.value) for s in validate(raw, text)]
    assert got == [("user", "likes", "gardening"), ("person", "granddaughter", "Lily")]
    assert validate("not json", text) == [] and validate({"facts": "x"}, text) == []


class FakeCompletions:
    def __init__(self, content):
        self.content, self.calls = content, []

    def create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_grok_extractor_with_fake_client(monkeypatch):
    fake = FakeCompletions(json.dumps({"facts": [{"subject": "user", "relation": "likes", "value": "gardening"}]}))
    import voice.llm
    monkeypatch.setattr(voice.llm, "_make_client",
                        lambda base_url, api_key, timeout: SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    ex = make_grok_extractor(CFG)
    assert ex("I love gardening") == {"facts": [{"subject": "user", "relation": "likes", "value": "gardening"}]}
    kw = fake.calls[0]
    assert kw["model"] == CFG["llm"]["model"] and kw["response_format"] == {"type": "json_object"}
    assert kw.get("reasoning_effort") == CFG["llm"].get("reasoning_effort")


def test_grok_extractor_without_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert make_grok_extractor(CFG)("I love gardening") is None


def test_learn_uses_llm_only_online_and_only_for_statements(events):
    seen = []

    def fake(text):
        seen.append(text)
        return {"facts": [{"subject": "user", "relation": "likes", "value": "gardening"}]}

    p = Profile(events, extractor=fake)
    assert p.learn("I love gardening", at(9), online=False) == []
    assert p.learn("where are my keys?", at(9), online=True) == []
    assert seen == []
    assert [f.value for f in p.learn("I love gardening", at(9), online=True)] == ["gardening"]
    assert p.get("user", "likes") == "gardening"


def test_learn_survives_extractor_failure(events):
    def boom(text):
        raise RuntimeError("down")
    assert Profile(events, extractor=boom).learn("I love gardening", at(9), online=True) == []


# ---------------------------------------------------------------- suggestion flow and greetings (care layer)

def fw(events):
    return FakeWorld([Entity("pill_bottle", "target", Status.UNDER, parent="notebook", pos_cm=(20.0, 40.0)),
                      Entity("notebook", "cover", Status.VISIBLE, pos_cm=(20.5, 39.0))], events)


def base_ask(text, source):
    return Answer(f"router: {text}")


def test_routine_suggests_reminder_and_yes_creates_it(events):
    now = [at(13)]
    care = Care(CFG, fw(events), events, base_ask, clock=lambda: now[0])
    a = care.ask("I take my pills at 8 and 8", "voice")
    assert a.text == ("Got it: your pills at 8 AM and 8 PM. Should I remind you at 8 AM and 8 PM every day "
                      "if you haven't picked up the pill bottle?")
    assert not pill_claim(a.text)
    now[0] += 10
    a = care.ask("Yes please", "voice")
    assert a.text == "Okay. I'll check your pill bottle every day at 8 AM and 8 PM."
    rs = sorted(care.reminders.store.active(), key=lambda r: r.at_h)
    assert [(r.kind, r.obj, r.daily, r.at_h, r.since) for r in rs] == [
        ("condition", "pill_bottle", True, 8, None), ("condition", "pill_bottle", True, 20, (14, 0))]
    assert care.ask("yes", "voice").text == "router: yes"                   # nothing pending any more


def test_routine_suggestion_declined_or_expired(events):
    now = [at(13)]
    care = Care(CFG, fw(events), events, base_ask, clock=lambda: now[0])
    care.ask("I take my pills at 8 and 8", "voice")
    assert care.ask("no thanks", "voice").text == "Okay, I won't."
    assert care.reminders.store.active() == []
    care.ask("I take my pills at 9", "voice")
    now[0] += 300
    assert care.ask("yes", "voice").text == "router: yes"
    assert care.reminders.store.active() == []


def test_greeting_uses_preferred_name(events):
    care = Care(CFG, fw(events), events, base_ask, clock=lambda: at(14))
    assert care.ask("hello", "voice").text == "Hello."
    care.ask("call me Grandpa Joe", "voice")
    assert care.ask("Good afternoon!", "voice").text == "Good afternoon, Grandpa Joe."


def test_profile_read_back_keeps_the_pill_rule(events):
    p = Profile(events)
    p.remember("routine", "take your pills", "right after I take them with water", "x", now=at(9))
    for text in (p.describe(), p.handle("forget that", at(9, 1)).answer.text):
        assert not pill_claim(text), text
