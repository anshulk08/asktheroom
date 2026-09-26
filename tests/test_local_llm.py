import json

import pytest
import requests

from core.config import load_config
from core.fakeworld import demo_world
from voice import local_llm
from voice.llm import FALLBACK_TEXT, PILLS_SAFE
from voice.local_llm import ask_local, schema, templated, to_local_answer

CFG = load_config()
NAMES = list(CFG["objects"])


@pytest.fixture
def world():
    return demo_world()          # keys INSIDE box, pill_bottle UNDER notebook, wallet VISIBLE, ...


@pytest.mark.parametrize("q,text,target", [
    ("what's in the box", "The keys are inside the box.", "box"),
    ("is anything under the notebook", "The pill bottle is under the notebook.", "notebook"),
])
def test_contents_are_templated_and_circle_the_place(world, q, text, target):
    a = templated(q, world, CFG)
    assert (a.text, a.point_at, a.action) == (text, target, "circle")


def test_hidden_on_table_privacy_help_meds(world):
    assert "keys inside the box" in templated("which things are hidden right now", world, CFG).text
    assert templated("how many things are on the table", world, CFG).text == \
        "I can see the wallet, the box and the notebook on the table."
    privacy = templated("are you recording me", world, CFG).text
    assert "never save a recording" in privacy and "Grok" in privacy and "picture of the table" in privacy
    assert "your pill bottle" in templated("what can you do", world, CFG).text
    a = templated("did I take my meds", world, CFG)
    assert (a.text, a.point_at) == (PILLS_SAFE, "pill_bottle")
    assert templated("is my wallet safe", world, CFG) is None


def test_model_reply_is_cleaned_and_guarded():
    raw = json.dumps({"action": "circle", "point_at": "wallet", "text": "Your **wallet** is on the table. Yes. Really."})
    a = to_local_answer(raw, NAMES, CFG)
    assert (a.text, a.point_at, a.action) == ("Your wallet is on the table. Yes.", "wallet", "circle")
    raw = json.dumps({"action": "point", "point_at": "keys", "text": "The notebook sits on the left."})
    assert to_local_answer(raw, NAMES, CFG).point_at is None       # sentence doesn't mention the keys
    raw = json.dumps({"action": "none", "point_at": "none", "text": "You took your pills an hour ago."})
    assert to_local_answer(raw, NAMES, CFG).text == PILLS_SAFE


def test_schema_puts_the_action_first():
    s = schema(CFG)
    assert list(s["properties"]) == ["action", "point_at", "text"]
    assert set(s["properties"]["point_at"]["enum"]) == set(NAMES) | {"none"}


class Resp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def test_ask_local_calls_qwen_for_the_long_tail(world, monkeypatch):
    sent = []

    def post(url, timeout, json):
        sent.append(json)
        return Resp('{"action":"point","point_at":"wallet","text":"Your wallet looks safe on the table."}')

    monkeypatch.setattr(local_llm.requests, "post", post)
    a = ask_local("is my wallet safe", world, world.events, CFG)
    assert a.point_at == "wallet" and "safe" in a.text
    body = sent[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0
    assert "Question: is my wallet safe" in body["messages"][1]["content"]
    assert '"name":"keys","status":"INSIDE"' in body["messages"][1]["content"]
    assert "State" not in body["messages"][0]["content"]           # system prompt stays cacheable


def test_ask_local_never_raises(world, monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(local_llm.requests, "post", down)
    assert ask_local("is my wallet safe", world, world.events, CFG).text == FALLBACK_TEXT
    monkeypatch.setattr(local_llm.requests, "post", lambda *a, **k: Resp("not json"))
    assert ask_local("is my wallet safe", world, world.events, CFG).text == FALLBACK_TEXT
    off = dict(CFG, understand=dict(CFG["understand"], enabled=False))
    assert ask_local("is my wallet safe", world, world.events, off).text == FALLBACK_TEXT
