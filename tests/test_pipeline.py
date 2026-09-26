from types import SimpleNamespace

from core.config import load_config
from core.fakeworld import demo_world
from core.types import Answer
from voice.pipeline import make_ask


def _setup(online):
    cfg = load_config()
    w = demo_world()
    calls = []

    def fake_grok(text, world, events, cfg, online=True):
        calls.append(text)
        return Answer("from grok", "box", "point")

    ask = make_ask(cfg, w, w.events, net=SimpleNamespace(online=online), grok=fake_grok)
    return ask, calls, w


def test_known_intents_never_reach_grok():
    ask, calls, _ = _setup(online=True)
    ans = ask("where are my keys", "voice")
    assert calls == [] and "box" in ans.text and ans.point_at == "keys"


def test_other_goes_to_grok_only_when_online():
    ask, calls, _ = _setup(online=True)
    assert ask("which things are hidden right now?", "voice").text == "from grok"
    ask_off, calls_off, _ = _setup(online=False)
    ans = ask_off("which things are hidden right now?", "voice")
    assert calls_off == [] and ans.text != "from grok"


def test_questions_are_logged_with_latency():
    ask, _, w = _setup(online=False)
    ask("where are my keys", "voice")
    [(intent, obj, latency_ms)] = w.events._rows("SELECT intent, obj, latency_ms FROM questions")
    assert intent == "WHERE" and obj == "keys" and latency_ms >= 0
