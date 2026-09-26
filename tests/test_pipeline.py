from types import SimpleNamespace

from core.config import load_config
from core.fakeworld import demo_world
from core.types import Answer
from voice.pipeline import make_ask


def _setup(online):
    cfg = load_config()
    w = demo_world()
    calls = []

    def fake_local(text, world, events, cfg, online=True):
        calls.append(text)
        return Answer("from qwen", "box", "point")

    ask = make_ask(cfg, w, w.events, net=SimpleNamespace(online=online), other=fake_local)
    return ask, calls, w


def test_known_intents_never_reach_the_model():
    ask, calls, _ = _setup(online=True)
    ans = ask("where are my keys", "voice")
    assert calls == [] and "box" in ans.text and ans.point_at == "keys"


def test_other_is_answered_locally_online_or_not():
    for online in (True, False):
        ask, calls, _ = _setup(online=online)
        assert ask("which things are hidden right now?", "voice").text == "from qwen"
        assert calls == ["which things are hidden right now?"]


def test_questions_are_logged_with_latency():
    ask, _, w = _setup(online=False)
    ask("where are my keys", "voice")
    [(intent, obj, latency_ms)] = w.events._rows("SELECT intent, obj, latency_ms FROM questions")
    assert intent == "WHERE" and obj == "keys" and latency_ms >= 0
