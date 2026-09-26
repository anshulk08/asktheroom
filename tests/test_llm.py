import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from core.config import load_config
from core.fakeworld import demo_world
from core.types import Status
from voice import llm
from voice.llm import FALLBACK_TEXT, ask_grok

CFG = load_config()


def tool_call(name, args, id_="c1"):
    return NS(id=id_, type="function", function=NS(name=name, arguments=json.dumps(args)))


def reply(*calls, content=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=list(calls) or None))])


class FakeClient:
    """Scripted chat.completions.create; records every request."""

    def __init__(self, script, delay=0.0):
        self.script = list(script)
        self.delay = delay
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw["messages"])))
        self.requests[-1] = {"messages": self.requests[-1],
                             **{k: v for k, v in kw.items() if k != "messages"}}
        if self.delay:
            time.sleep(self.delay)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
def world():
    return demo_world()


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")


def install(monkeypatch, client):
    made = []

    def make(base_url, api_key, timeout):
        made.append((base_url, api_key, timeout))
        return client

    monkeypatch.setattr(llm, "_make_client", make)
    return made


def tool_results(req):
    return {m["tool_call_id"]: json.loads(m["content"]) for m in req["messages"] if m["role"] == "tool"}


# ------------------------------------------------------------------ tool dispatch

def test_tools_return_facts_from_demo_world(world):
    t = llm._Tools(world, world.events, CFG)
    loc = t.call("locate", {"object": "keys"})
    assert loc["status"] == "INSIDE" and loc["parent"] == "box" and loc["chain"] == ["keys", "box"]
    assert loc["area"] == "right"                         # box at x=70.4 of 90
    assert t.call("locate", {"object": "pills"})["parent"] == "notebook"   # synonym
    assert t.call("locate", {"object": "phone"})["edge"] == "left"
    assert t.call("locate", {"object": "glasses"})["status"] == "UNKNOWN"

    h = t.call("history", {"object": "keys", "limit": 5})
    assert [e["type"] for e in h["events"]] == ["PUT_INSIDE", "PICKED_UP"]
    assert 390 <= h["events"][0]["s_ago"] <= 400

    since = (datetime.now().astimezone() - timedelta(seconds=150)).isoformat()
    ch = t.call("changes_since", {"iso_time": since})
    assert [(e["object"], e["type"]) for e in ch["events"]] == [
        ("phone", "PICKED_UP"), ("phone", "EXITED_VIEW"), ("box", "MOVED"), ("remote", "PICKED_UP")]
    utc = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None) - timedelta(seconds=150)
    assert tool_count(t.call("changes_since", {"iso_time": utc.isoformat() + "Z"})) == 4

    assert "error" in t.call("locate", {"object": "banana"})
    assert "error" in t.call("changes_since", {"iso_time": "yesterday-ish"})
    assert "error" in t.call("nope", {})


def tool_count(r):
    return r["count"]


def test_compact_state_drops_noise(world):
    s = llm.compact_state(world, CFG)
    keys = next(e for e in s if e["name"] == "keys")
    assert keys["status"] == "INSIDE" and keys["parent"] == "box" and keys["confidence"] == 0.85
    assert "pos_cm" not in keys and "resolved_cm" not in keys and "kind" not in keys
    wallet = next(e for e in s if e["name"] == "wallet")
    assert "parent" not in wallet and "candidates" not in wallet


# ------------------------------------------------------------------ full loop

def test_tool_round_then_respond(monkeypatch, world, key):
    client = FakeClient([
        reply(tool_call("locate", {"object": "keys"}, "a"), tool_call("history", {"object": "keys"}, "b")),
        reply(tool_call("respond", {"text": "Your keys are in the box.", "point_at": "keys"})),
    ])
    made = install(monkeypatch, client)
    a = ask_grok("could my keys have gone anywhere else?", world, world.events, CFG)
    assert a.text == "Your keys are in the box."
    assert a.point_at == "keys" and a.action == "point"
    assert made[0][:2] == ("https://api.x.ai/v1", "test-key")

    first, second = client.requests
    assert first["model"] == "grok-4.3" and first["reasoning_effort"] == "none"
    assert first["tool_choice"] == "required"
    assert 0 < first["timeout"] <= 4
    sys = first["messages"][0]["content"]
    assert '"name":"keys","status":"INSIDE","parent":"box"' in sys
    assert "pill" in sys.lower() and "probably" in sys and "I lost track" in sys
    res = tool_results(second)
    assert res["a"]["parent"] == "box"
    assert res["b"]["events"][0]["type"] == "PUT_INSIDE"
    assert second["messages"][2]["role"] == "assistant"
    assert {"locate", "history", "changes_since", "respond"} == {
        t["function"]["name"] for t in first["tools"]}


def test_last_round_forces_respond(monkeypatch, world, key):
    loc = tool_call("locate", {"object": "wallet"})
    client = FakeClient([reply(loc), reply(loc),
                         reply(tool_call("respond", {"text": "It is on the table."}))])
    install(monkeypatch, client)
    a = ask_grok("where is it", world, world.events, CFG)
    assert a.text == "It is on the table." and a.point_at is None and a.action is None
    assert client.requests[-1]["tool_choice"] == {"type": "function", "function": {"name": "respond"}}


@pytest.mark.parametrize("point_at", ["banana", "hand:2", "", None, "table"])
def test_invalid_point_at_dropped(monkeypatch, world, key, point_at):
    args = {"text": "The remote is in someone's hand."}
    if point_at is not None:
        args["point_at"] = point_at
    install(monkeypatch, FakeClient([reply(tool_call("respond", args))]))
    a = ask_grok("who has the remote", world, world.events, CFG)
    assert a.text == "The remote is in someone's hand."
    assert a.point_at is None and a.action is None


def test_point_at_synonym_and_markdown_cleanup(monkeypatch, world, key):
    install(monkeypatch, FakeClient([reply(tool_call("respond", {
        "text": "**Probably** under the notebook.\nIt was covered 5 minutes ago. Extra sentence.",
        "point_at": "pill bottle"}))]))
    a = ask_grok("pills?", world, world.events, CFG)
    assert a.text == "Probably under the notebook. It was covered 5 minutes ago."
    assert a.point_at == "pill_bottle" and a.action == "point"


def test_never_claims_pills_taken(monkeypatch, world, key):
    install(monkeypatch, FakeClient([reply(tool_call("respond", {
        "text": "Yes, you took your pills five minutes ago.", "point_at": "pill_bottle"}))]))
    a = ask_grok("did I take my meds", world, world.events, CFG)
    assert a.text == llm.PILLS_SAFE and a.point_at == "pill_bottle"
    ok = llm.to_answer("You took the pill bottle five minutes ago.", None, ["pill_bottle"], CFG)
    assert ok.text == "You took the pill bottle five minutes ago."


def test_plain_content_without_tool_call_is_accepted(monkeypatch, world, key):
    install(monkeypatch, FakeClient([reply(content="The wallet is on the table.")]))
    a = ask_grok("wallet", world, world.events, CFG)
    assert a.text == "The wallet is on the table." and a.point_at is None


# ------------------------------------------------------------------ fallbacks

def assert_fallback(a):
    assert a.text == FALLBACK_TEXT and a.point_at is None and a.action is None


def test_timeout_returns_fallback_within_budget(monkeypatch, world, key):
    cfg = {**CFG, "llm": {**CFG["llm"], "timeout_s": 0.3}}
    install(monkeypatch, FakeClient([reply(tool_call("respond", {"text": "late"}))], delay=1.0))
    t0 = time.perf_counter()
    a = ask_grok("anything", world, world.events, cfg)
    assert time.perf_counter() - t0 < 0.5
    assert_fallback(a)


def test_http_timeout_returns_fallback(monkeypatch, world, key):
    import requests
    install(monkeypatch, FakeClient([requests.Timeout("read timed out")]))
    assert_fallback(ask_grok("anything", world, world.events, CFG))


def test_api_error_returns_fallback(monkeypatch, world, key):
    install(monkeypatch, FakeClient([RuntimeError("500")]))
    assert_fallback(ask_grok("anything", world, world.events, CFG))


def test_garbage_arguments_return_fallback(monkeypatch, world, key):
    bad = NS(id="x", type="function", function=NS(name="respond", arguments="{not json"))
    install(monkeypatch, FakeClient([reply(bad)]))
    assert_fallback(ask_grok("anything", world, world.events, CFG))


def test_missing_key_returns_fallback(monkeypatch, world):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    client = FakeClient([])
    install(monkeypatch, client)
    assert_fallback(ask_grok("anything", world, world.events, CFG))
    assert client.requests == []


def test_offline_returns_fallback_without_any_call(monkeypatch, world, key):
    def boom(*a, **k):
        raise AssertionError("client must not be created offline")

    monkeypatch.setattr(llm, "_make_client", boom)
    assert_fallback(ask_grok("anything", world, world.events, CFG, online=False))


def test_world_error_never_raises(monkeypatch, key):
    class Broken:
        def state_json(self):
            raise RuntimeError("world down")

    install(monkeypatch, FakeClient([]))
    assert_fallback(ask_grok("anything", Broken(), None, CFG))


def test_status_values_match_prompt():
    for s in Status:
        assert s.value in llm.SYSTEM_TEMPLATE
