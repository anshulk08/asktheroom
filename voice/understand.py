"""Qwen interprets the spoken question, on the Jetson: transcript -> Intent (same as voice.intents.parse).

The rule parser knows the phrasings it was written for. People at a demo talk however they like
("has anybody messed with my meds", "can you point at the pill bottle") and Whisper mishears, and
every "I can tell you where things are..." breaks the illusion. Qwen2.5 1.5B Instruct, served by
llama.cpp's llama-server on the Jetson (scripts/qwen_server.sh), reads what the rules can't and
returns {kind, object}. A JSON schema limits it to a valid kind and one of the known objects.
Nothing leaves the device.

Rules first, then Qwen (on the laptop, 24 loosely worded questions not in the prompt: rules alone
13/24, rules then Qwen 18/24, no invented objects; tests/stt_questions.json stays 20/20. Qwen takes
~135 ms on an M-series GPU; time it on the Jetson):
  - The rules answer when they found a question word and the object it needs. They are exact on
    the phrasings they know and on synonyms ("clicker" is the remote), where the 1.5B model slips.
    RESET and RECAL act on the room, so only the explicit words count: a misheard "reset"
    mid-demo would wipe the world model.
  - Otherwise (rules say OTHER, or found no object) Qwen decides the kind. An object the rules
    recognised still wins over Qwen's.
  - Qwen's object must sound like something that was said (sounds_like), so "my coffee mug"
    doesn't become the glasses.
  - Qwen down, slower than understand.timeout_s, or bad output: the rules' answer stands.

    python -m voice.understand "ugh where did I put my specs"      # try a transcript
"""
from __future__ import annotations

import difflib
import json
import logging
import sys
import threading
import time
from typing import Callable, Optional

import requests

from core.config import load_config
from core.types import INTENT_KINDS, Intent
from voice.intents import normalize, parse

log = logging.getLogger(__name__)

ACTS = ("RESET", "RECAL")               # act on the room: the rules alone decide these
NO_OBJECT = ("RESET", "RECAL", "CHANGES")
SOUNDS_LIKE = 0.6                       # difflib ratio: "wall it" ~ wallet, "note book" ~ notebook, "mug" !~ glasses
QWEN_KINDS = [k for k in INTENT_KINDS if k not in ACTS]

SYSTEM = """You sort questions asked out loud to "Ask the Room", a device watching a tabletop with a camera. The question comes from speech recognition, so words may be misheard (wear = where). Work out what the person meant.

Kinds:
- WHERE: where an object is now, or they want it found, shown or pointed at. "where did I put my specs", "I can't find my phone", "have you seen my wallet", "is the pill bottle still in the box", "show me my keys", "light up my remote"
- HISTORY: what happened to one object, who did something or when. "what happened to my keys", "who moved the box", "when did I last have the remote"
- HANDLED: yes or no, was an object touched, moved or taken. "did anyone touch my pills", "has my wallet been moved", "did the dog get into my wallet"
- CHANGES: what changed on the table in general, no one object. "what did I miss", "did anything change while I was gone"
- OTHER: anything else, including what is inside or under something. "what's in the box", "what's under the notebook", "how many things are on the table", "what can you do"

Objects: {objects}.
object is the object the question is about, or "none". Map other words for an object to its name (specs -> glasses, meds -> pill_bottle). If an object is named but it is not one of these, use "none".
Reply with JSON only."""


def _objects(cfg: dict) -> list[str]:
    return list(cfg.get("objects") or {})


def schema(cfg: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["kind", "object"],
            "properties": {"kind": {"type": "string", "enum": QWEN_KINDS},
                           "object": {"type": "string", "enum": _objects(cfg) + ["none"]}}}


def system_prompt(cfg: dict) -> str:
    return SYSTEM.format(objects=", ".join(_objects(cfg)))


def sounds_like(obj: str, text: str, cfg: dict) -> bool:
    """Was something like obj said? Keeps Qwen from inventing an object ("my coffee mug" -> glasses)
    while letting it fix mishearings the rules can't ("wears my wall it" -> wallet)."""
    names = [obj.replace("_", " ")] + [str(k) for k, v in (cfg.get("synonyms") or {}).items() if v == obj]
    words = normalize(text).split()
    spans = [" ".join(words[i:i + n]) for n in (1, 2) for i in range(len(words) - n + 1)]
    return any(difflib.SequenceMatcher(None, s, n).ratio() >= SOUNDS_LIKE for s in spans for n in names)


def rules_sure(i: Intent) -> bool:
    return i.kind in NO_OBJECT or (i.kind != "OTHER" and i.obj is not None)


def to_intent(raw: str, text: str, cfg: dict, obj_hint: Optional[str] = None) -> Optional[Intent]:
    """Qwen's JSON -> Intent, or None if it isn't a valid answer. obj_hint (the rules' object)
    replaces Qwen's object."""
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or d.get("kind") not in QWEN_KINDS:
        return None
    kind, obj = d["kind"], d.get("object")
    obj = obj_hint or (obj if obj in _objects(cfg) and sounds_like(obj, text, cfg) else None)
    if kind in ("WHERE", "HISTORY", "HANDLED") and obj is None:
        # nothing to point at or look up: CHANGES covers "did anyone touch anything",
        # OTHER lets Grok (online) or the fallback handle "where is the charger"
        kind = "CHANGES" if kind != "WHERE" else "OTHER"
    if kind in NO_OBJECT:
        obj = None
    return Intent(kind=kind, obj=obj, raw=text)


class Qwen:
    """Talks to llama-server's OpenAI-compatible chat endpoint. ask() returns the raw JSON text."""

    def __init__(self, cfg: dict):
        u = cfg.get("understand") or {}
        self.url = str(u.get("url", "http://127.0.0.1:8081/v1")).rstrip("/")
        self.model = str(u.get("model", "qwen"))
        self.system = system_prompt(cfg)
        self.schema = schema(cfg)
        self.session = requests.Session()

    def ask(self, text: str, timeout: float) -> str:
        r = self.session.post(f"{self.url}/chat/completions", timeout=timeout, json={
            "model": self.model, "temperature": 0, "max_tokens": 40, "cache_prompt": True,
            "messages": [{"role": "system", "content": self.system},
                         {"role": "user", "content": text}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "intent", "schema": self.schema}}})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def health(self, timeout: float = 1.0) -> bool:
        try:
            base = self.url[:-3] if self.url.endswith("/v1") else self.url
            return self.session.get(f"{base}/health", timeout=timeout).ok
        except requests.RequestException:
            return False


class Understander:
    """interpret(text) -> Intent. Qwen when it's up and quick, the rule parser otherwise.

    Remembers the last transcript, so main.Room (RESET/RECAL) and the ask pipeline share one call."""

    def __init__(self, cfg: dict, qwen: Optional[Qwen] = None):
        u = cfg.get("understand") or {}
        self.cfg = cfg
        self.enabled = bool(u.get("enabled", True))
        self.timeout_s = float(u.get("timeout_s", 1.5))
        self.qwen = qwen if qwen is not None else (Qwen(cfg) if self.enabled else None)
        self._lock = threading.Lock()
        self._last: tuple[str, Intent] | None = None
        self.last_ms: float = 0.0
        self.last_by = "rules"             # who decided the last intent: qwen | rules

    def warm(self) -> bool:
        """Check llama-server and run one question, so the prompt is cached before the first visitor."""
        if self.qwen is None:
            return False
        if not self.qwen.health():
            log.warning("Qwen not reachable at %s; questions use the rule parser (scripts/qwen_server.sh)",
                        self.qwen.url)
            return False
        try:
            self.qwen.ask("where are my keys", timeout=30)
        except Exception as ex:
            log.warning("Qwen warm-up failed: %s", ex)
            return False
        log.info("Qwen interpreting questions at %s", self.qwen.url)
        return True

    def __call__(self, text: str) -> Intent:
        with self._lock:
            if self._last is not None and self._last[0] == text:
                return self._last[1]
            intent = self._interpret(text)
            self._last = (text, intent)
            return intent

    def _interpret(self, text: str) -> Intent:
        rules = parse(text, self.cfg)
        if rules_sure(rules) or self.qwen is None or not text.strip():
            self.last_by, self.last_ms = "rules", 0.0
            return rules
        t0 = time.monotonic()
        try:
            got = to_intent(self.qwen.ask(text, self.timeout_s), text, self.cfg, rules.obj)
        except Exception as ex:
            log.warning("Qwen failed (%s: %s); using the rule parser", type(ex).__name__, ex)
            got = None
        self.last_ms = 1000 * (time.monotonic() - t0)
        if got is None:
            self.last_by = "rules"
            return rules
        self.last_by = "qwen"
        if (got.kind, got.obj) != (rules.kind, rules.obj):
            log.info("Qwen read %r as %s %s (rules: %s %s) in %.0f ms", text, got.kind, got.obj,
                     rules.kind, rules.obj, self.last_ms)
        return got


def make_interpreter(cfg: dict) -> Callable[[str], Intent]:
    return Understander(cfg)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    u = Understander(cfg)
    up = u.warm()
    for text in argv or [line.strip() for line in sys.stdin if line.strip()]:
        i = u(text)
        print(f"{i.kind:8} {str(i.obj):12} {u.last_by:5} {u.last_ms:5.0f} ms  {text}")
    return 0 if up else 1


if __name__ == "__main__":
    raise SystemExit(main())
