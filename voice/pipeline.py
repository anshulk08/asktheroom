"""Question → Answer router shared by the voice loop, POST /ask and /sms.

The known intents are answered on the device from the world model (voice.answers templates), online
or not. OTHER goes to voice.llm.ask_other: more templates, then Grok when online, else a fallback
sentence. Questions about what the camera sees go to voice.visual first (Grok with the frame).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from core.types import Answer, Intent

AskFn = Callable[[str, str], Answer]
log = logging.getLogger(__name__)


def make_ask(cfg: dict, world, events, net=None, other: Optional[Callable] = None,
             interpret: Optional[Callable[[str], Intent]] = None, visual=None,
             clock: Optional[Callable[[], float]] = None) -> AskFn:
    """Returns ask(text, source) -> Answer. `net` has `.online` (logged with the question); `other`
    answers OTHER and defaults to voice.llm.ask_other; `interpret` (text -> Intent) defaults to
    the rule parser (main.py passes voice.understand's model interpreter). `visual`
    (voice.visual.VisualQA, when visual memory is on) gets first say: it answers questions about what
    the camera sees or saw and returns None for everything the world model and templates handle.
    `clock` (wall time) is what the templates measure 'ago' from; live it is None (time.time()), and
    eval.score_clip passes the clip's time so replayed answers say '20 seconds ago', not '3 days ago'."""
    from voice.answers import answer

    if interpret is None:
        from voice.intents import parse
        interpret = lambda text: parse(text, cfg)     # noqa: E731

    if other is None:
        from voice.llm import ask_other as other

    def ask(text: str, source: str = "voice") -> Answer:
        t0 = time.perf_counter()
        online = bool(net and net.online)
        intent = interpret(text)
        ans = None
        if visual is not None:
            try:
                ans = visual.route(intent, text, online)
            except Exception:
                log.exception("visual route failed")
        if ans is None and intent.kind == "OTHER":
            ans = other(text, world, events, cfg, online=online)
        elif ans is None:
            ans = answer(intent, world, events, cfg, now=clock() if clock is not None else None)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        try:
            events.log_question(text, intent.kind, intent.obj, ans.text, online, latency_ms)
        except Exception:
            pass  # logging must never cost an answer
        return ans

    return ask
