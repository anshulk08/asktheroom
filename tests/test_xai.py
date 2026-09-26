"""core/xai.py: the one small client every Grok call goes through (plain requests, no openai package)."""
import subprocess
import sys

import pytest
import requests

from core import xai


class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.posts, self.gets = resp or Resp(), exc, [], []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.resp

    def get(self, url, headers=None, timeout=None):
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.resp


REPLY = {"choices": [{"message": {"content": None, "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "respond", "arguments": "{\"text\": \"hi\"}"}}]}}],
    "usage": {"prompt_tokens": 900, "completion_tokens": 12}}


def test_create_posts_openai_format_to_xai_and_returns_attribute_objects():
    s = FakeSession(Resp(body=REPLY))
    c = xai.Client("https://api.x.ai/v1/", "k1", timeout=7.0, session=s)
    r = c.chat.completions.create(model="grok-4.3", messages=[{"role": "user", "content": "q"}], timeout=2.5)
    [p] = s.posts
    assert p["url"] == "https://api.x.ai/v1/chat/completions" and p["timeout"] == 2.5
    assert p["headers"] == {"Authorization": "Bearer k1"}
    assert p["json"] == {"model": "grok-4.3", "messages": [{"role": "user", "content": "q"}]}
    tc = r.choices[0].message.tool_calls[0]
    assert (tc.id, tc.function.name, tc.function.arguments) == ("c1", "respond", "{\"text\": \"hi\"}")
    assert r.choices[0].message.content is None and r.usage.prompt_tokens == 900
    c.chat.completions.create(model="m", messages=[])
    assert s.posts[1]["timeout"] == 7.0                       # the client's default


def test_missing_fields_read_as_none():
    r = xai.Client("u", "k", session=FakeSession(Resp(body={"choices": [{"message": {"content": "x"}}]}))) \
        .chat.completions.create(model="m", messages=[])
    assert r.choices[0].message.tool_calls is None and r.usage is None


def test_http_errors_carry_the_status_and_the_apis_message():
    s = FakeSession(Resp(400, text='{"error": "Argument not supported: reasoning_effort"}'))
    with pytest.raises(xai.XAIError) as ei:
        xai.Client("u", "k", session=s).chat.completions.create(model="m", messages=[])
    assert ei.value.status_code == 400 and "reasoning_effort" in str(ei.value)


def test_timeouts_propagate_as_requests_timeouts():
    s = FakeSession(exc=requests.Timeout("slow"))
    with pytest.raises(requests.Timeout):
        xai.Client("u", "k", session=s).chat.completions.create(model="m", messages=[])


def test_one_shared_session_per_process():
    assert xai.session() is xai.session() and isinstance(xai.session(), requests.Session)


def test_warm_opens_the_connection_with_a_models_call(monkeypatch):
    s = FakeSession(Resp(200, body={"data": []}))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert xai.warm(session=s) is False and s.gets == []                 # no key: nothing sent
    monkeypatch.setenv("XAI_API_KEY", "k2")
    assert xai.warm("https://api.x.ai/v1", session=s) is True
    assert s.gets == [{"url": "https://api.x.ai/v1/models", "headers": {"Authorization": "Bearer k2"}, "timeout": 5.0}]
    assert xai.warm(session=FakeSession(exc=requests.ConnectionError("down"))) is False
    assert xai.warm(session=FakeSession(Resp(500))) is False


def test_nothing_on_the_rig_imports_the_openai_package():
    code = ("import sys, main, voice.llm, voice.understand, voice.visual, voice.pipeline, core.narration, core.xai\n"
            "print('openai' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(
        __import__("pathlib").Path(__file__).resolve().parent.parent), timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False"


def test_no_source_file_on_the_rig_uses_the_openai_package():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    files = [root / "main.py", *[p for d in ("core", "voice", "server", "act", "mobile") for p in (root / d).rglob("*.py")]]
    hits = [str(p.relative_to(root)) for p in files if re.search(r"^\s*(?:from|import)\s+openai\b", p.read_text(), re.M)]
    assert hits == []
