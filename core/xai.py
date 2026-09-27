"""The one client every Grok call on the rig goes through: xAI's chat API (https://api.x.ai/v1, the
OpenAI request format) over plain requests. voice/llm.py (open questions, with tools), voice/understand.py
(reading what the rules can't) and core/narration.py (narration and visual questions, with images) all
use it, so there is no openai package on the rig, and one process-wide requests.Session keeps a single
warm TLS connection to xAI. warm() opens it before the first question (main.py: at start and whenever
the network comes back), so a visitor's first question doesn't pay for the handshake.

    Client(base_url, api_key).chat.completions.create(model=..., messages=..., timeout=...)

has the OpenAI SDK's call shape and returns attribute objects (r.choices[0].message.tool_calls[0]
.function.name, r.usage.prompt_tokens; absent fields read as None). HTTP errors raise XAIError with
status_code (core/narration.py's retry rules read it); timeouts and connection errors propagate as
requests exceptions. The key is read from the environment by callers and never logged.
"""
from __future__ import annotations

import logging
import os
import threading
from types import SimpleNamespace
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.x.ai/v1"
_session: Optional[requests.Session] = None
_lock = threading.Lock()


class XAIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def session() -> requests.Session:
    """The process-wide session (connection pool) for api.x.ai."""
    global _session
    with _lock:
        if _session is None:
            _session = requests.Session()
        return _session


def api_key(env: str = "XAI_API_KEY") -> str:
    return os.environ.get(env, "").strip()


class _Obj(SimpleNamespace):
    def __getattr__(self, name: str) -> Any:          # absent fields read as None, like the SDK's
        return None


def _obj(x: Any) -> Any:
    if isinstance(x, dict):
        return _Obj(**{k: _obj(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_obj(v) for v in x]
    return x


class Client:
    def __init__(self, base_url: str, key: str, timeout: float = 30.0, session: Optional[requests.Session] = None):
        self.base_url, self.key, self.timeout = (base_url or BASE_URL).rstrip("/"), key, timeout
        self._session = session
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, timeout: Optional[float] = None, **body: Any) -> Any:
        s = self._session if self._session is not None else session()
        r = s.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.key}"},
                   json=body, timeout=self.timeout if timeout is None else timeout)
        if r.status_code >= 400:
            raise XAIError(f"HTTP {r.status_code}: {(r.text or '')[:300]}", r.status_code)
        return _obj(r.json())


def warm(base_url: str = BASE_URL, timeout: float = 5.0, session: Optional[requests.Session] = None,
         env: str = "XAI_API_KEY") -> bool:
    """Open the connection to xAI (one GET /models, no tokens used). False without a key or on any error."""
    key = api_key(env)
    if not key:
        return False
    s = session if session is not None else globals()["session"]()
    try:
        r = s.get(f"{(base_url or BASE_URL).rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"},
                  timeout=timeout)
    except requests.RequestException as ex:
        log.info("xai warm-up failed: %s", type(ex).__name__)
        return False
    if r.status_code >= 400:
        log.warning("xai warm-up: HTTP %s", r.status_code)
        return False
    return True
