"""Question → Answer router shared by the voice loop, POST /ask and /sms.

Everything is answered on the device, online or not. The known intents use the templates in
voice.answers; OTHER goes to voice.local_llm (more templates, then the local Qwen). Grok is
not on this path (team decision: Grok only helps the detector).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from core.types import Answer, Intent

AskFn = Callable[[str, str], Answer]


def make_ask(cfg: dict, world, events, net=None, other: Optional[Callable] = None,
             interpret: Optional[Callable[[str], Intent]] = None) -> AskFn:
    """Returns ask(text, source) -> Answer. `net` has `.online` (logged with the question); `other`
    answers OTHER and defaults to voice.local_llm.ask_local; `interpret` (text -> Intent) defaults to
    the rule parser (main.py passes voice.understand's Qwen)."""
    from voice.answers import answer

    if interpret is None:
        from voice.intents import parse
        interpret = lambda text: parse(text, cfg)     # noqa: E731

    if other is None:
        from voice.local_llm import ask_local as other

    def ask(text: str, source: str = "voice") -> Answer:
        t0 = time.perf_counter()
        online = bool(net and net.online)
        intent = interpret(text)
        if intent.kind == "OTHER":
            ans = other(text, world, events, cfg, online=online)
        else:
            ans = answer(intent, world, events, cfg)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        try:
            events.log_question(text, intent.kind, intent.obj, ans.text, online, latency_ms)
        except Exception:
            pass  # logging must never cost an answer
        return ans

    return ask
