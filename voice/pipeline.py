"""Question → Answer router shared by the voice loop, POST /ask and /sms.

Only the OTHER intent goes to Grok, and only when online. Everything else uses the offline
templates, so core answers never depend on the network.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from core.types import Answer

AskFn = Callable[[str, str], Answer]


def make_ask(cfg: dict, world, events, net=None, grok: Optional[Callable] = None) -> AskFn:
    """Returns ask(text, source) -> Answer. `net` has `.online`; `grok` defaults to voice.llm.ask_grok."""
    from voice.answers import answer
    from voice.intents import parse

    if grok is None:
        from voice.llm import ask_grok as grok

    def ask(text: str, source: str = "voice") -> Answer:
        t0 = time.perf_counter()
        online = bool(net and net.online)
        intent = parse(text, cfg)
        if intent.kind == "OTHER" and online:
            ans = grok(text, world, events, cfg, online=True)
        else:
            ans = answer(intent, world, events, cfg)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        try:
            events.log_question(text, intent.kind, intent.obj, ans.text, online, latency_ms)
        except Exception:
            pass  # logging must never cost an answer
        return ans

    return ask
