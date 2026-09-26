"""Grok (default) or a local Qwen interprets spoken questions: transcript -> Intent (same as
voice.intents.parse).

The rule parser knows the phrasings it was written for. People at a demo talk however they like
("has anybody messed with my meds", "can you point at the pill bottle") and Whisper mishears, and
every "I can tell you where things are..." breaks the illusion. A model reads what the rules can't
and returns {kind, object}; a JSON schema limits it to a valid kind and one of the known objects.
understand.backend: grok (the default) sends only the transcript to Grok; qwen / auto use a local Qwen
served by llama.cpp (scripts/qwen_server.sh, not installed on the rig), where nothing leaves the device.
Scores per model: scripts/eval_understand.py on tests/understand_eval.json.

Two ways speech arrives (listen.mode in config.yaml):
  asked      the visitor pressed the clicker, so the speech is meant for the rig.
  overheard  the mic is always on; most speech near the table is people talking to each other.

Asked: rules first, then Qwen.
  - The rules answer when they found a question word and the object it needs, named as such. They
    are exact on the phrasings they know and on synonyms ("clicker" is the remote), where a small
    model slips. An object the rules only guessed from a mishearing ("where are my kiss" -> keys) or a
    name nobody taught ("where is my kiss") goes to the model, which may know better; if it names no
    object, the rules' guess or name stands (world.find resolves names).
    RESET and RECAL act on the room, so only the explicit words count: a misheard "reset"
    mid-demo would wipe the world model.
  - So do TEACH, WHAT_DOING and questions about a taught name ("where is my charger" once someone
    said "this is my charger": Intent.name, for world.find); the model knows neither.
  - Otherwise (rules say OTHER, or found no object) Qwen decides the kind. An object the rules
    recognised still wins over Qwen's.
  - Qwen's object must sound like something that was said (sounds_like), so "my coffee mug"
    doesn't become the glasses.
  - Qwen down, slower than understand.timeout_s (a wall-clock limit on the whole call, DNS included),
    or bad output: the rules' answer stands.

Which model (understand.backend): grok (the default; offline, the rules answer: use a phone hotspot
if the venue Wi-Fi drops), qwen (the local llama-server, asked online or not), or auto (Grok online,
Qwen offline). Qwen isn't installed on the Jetson; qwen and auto need scripts/qwen_server.sh running.

Overheard: decide "was that for me?" without the model, then read it like an asked question.
  - A whole teaching sentence ("this is my vaseline") is for the rig, whatever else is true below.
  - listen.mode wake: only speech with the wake word counts (a loud hall defeats the rest).
  - Keyword gate: no object, command word or wake word ("we built this in twenty hours") -> IGNORE.
  - Addressed: the wake word ("room, ...") or a question/request opening ("where", "did", "can",
    "show", after fillers like "okay so"). "I'll grab my keys on the way out" and "put the wallet
    in the box" aren't -> IGNORE.
  - Then the asked path (rules, then Qwen). Two more guards, because a wrong answer to people
    talking to each other breaks the illusion more than silence: RESET/RECAL need the wake word
    ("let's reset after this" must not wipe the world), and OTHER needs the wake word or an
    object ("where are you guys from" is for the team, "what's in the box" is for the rig), and so
    does a WHERE with nothing to look for. screen() makes every check that needs no model, so the
    voice loop can drop chatter before starting its thinking cue.
  - Qwen isn't asked to judge IGNORE: on tests/understand_eval.json it got 8/16 overheard lines
    right, the checks above 14/16 with nothing to load.

    python -m voice.understand "ugh where did I put my specs"      # try a transcript
    python -m voice.understand --overheard "I'll grab my keys later"   # always-on mic
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import sys
import threading
import time
from typing import Callable, Optional

import requests

from core.config import load_config
from core.types import INTENT_KINDS, Intent
from net import call_with_deadline
from voice.intents import _vocab, fuzzy_match, matched_exactly, normalize, parse

log = logging.getLogger(__name__)

ACTS = ("RESET", "RECAL")               # act on the room: the rules alone decide these
RULES_ONLY = ACTS + ("TEACH", "WHAT_DOING")   # TEACH binds a name; Qwen's schema has neither
NO_OBJECT = ("RESET", "RECAL", "CHANGES")
SOUNDS_LIKE = 0.6                       # difflib ratio: "wall it" ~ wallet, "note book" ~ notebook, "mug" !~ glasses
QWEN_KINDS = [k for k in INTENT_KINDS if k not in RULES_ONLY]
IGNORE = "IGNORE"                       # overheard speech not meant for the rig; say and do nothing
COMMAND_WORDS = {"where", "whered", "wheres", "find", "found", "seen", "lost", "show", "point", "light",
                 "miss", "missed", "change", "changed", "different", "happened", "happen", "touch",
                 "touched", "moved", "took", "taken", "grabbed", "who", "when", "reset", "recalibrate",
                 "calibrate", "laser"}
QUESTION_START = {"where", "whered", "wheres", "what", "whats", "who", "whos", "when", "did", "has",
                  "have", "is", "are", "was", "were", "can", "could", "would", "will", "show", "find",
                  "point", "light", "tell", "any", "anything", "anyone", "anybody"}
FILLERS = {"um", "uh", "umm", "uhh", "ok", "okay", "so", "hey", "hi", "yeah", "oh", "well", "and",
           "wait", "hmm", "alright", "right", "please", "excuse", "me"}

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


def wake_words(cfg: dict) -> list[str]:
    return [normalize(w) for w in ((cfg.get("listen") or {}).get("wake_words") or ["room"])]


def system_prompt(cfg: dict) -> str:
    return SYSTEM.format(objects=", ".join(_objects(cfg)))


def has_wake_word(text: str, cfg: dict) -> bool:
    t = f" {normalize(text)} "
    return any(f" {w} " in t for w in wake_words(cfg))


def names_object(text: str, cfg: dict) -> bool:
    return bool(_vocab(cfg)[1].search(normalize(text)))


def gate(text: str, cfg: dict) -> bool:
    """Cheap first check for overheard speech: an object (or synonym), a command word or the wake word."""
    return bool(names_object(text, cfg) or COMMAND_WORDS & set(normalize(text).split())
                or has_wake_word(text, cfg))


def addressed(text: str, cfg: dict) -> bool:
    """Said to the rig: the wake word, or it opens like a question or request ("okay so where's...")."""
    words = [w for w in normalize(text).split() if w not in wake_words(cfg)]
    while words and words[0] in FILLERS:
        words.pop(0)
    return has_wake_word(text, cfg) or bool(words and words[0] in QUESTION_START)


def sounds_like(obj: str, text: str, cfg: dict) -> bool:
    """Was something like obj said? Keeps Qwen from inventing an object ("my coffee mug" -> glasses)
    while letting it fix mishearings the rules can't ("wears my wall it" -> wallet, "my kiss" -> keys)."""
    names = [obj.replace("_", " ")] + [str(k) for k, v in (cfg.get("synonyms") or {}).items() if v == obj]
    words = normalize(text).split()
    spans = [" ".join(words[i:i + n]) for n in (1, 2) for i in range(len(words) - n + 1)]
    return any(difflib.SequenceMatcher(None, s, n).ratio() >= SOUNDS_LIKE or fuzzy_match(s, n) > 0
               for s in spans for n in names)


def _taught(name: Optional[str], aliases) -> bool:
    return bool(name) and normalize(name) in {normalize(a) for a in aliases}


def rules_sure(i: Intent, cfg: Optional[dict] = None, aliases=()) -> bool:
    """A kind only the rules produce, or a question with its object named as such (with cfg: not
    guessed from a mishearing) or a taught name ('where is my charger' once 'charger' was taught:
    world.find resolves it, and the model has never heard of it)."""
    if i.kind in NO_OBJECT or i.kind in RULES_ONLY:
        return True
    if i.kind == "OTHER":
        return False
    if i.obj is not None:
        return cfg is None or matched_exactly(i.raw, i.obj, cfg)
    return _taught(i.name, aliases)


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
        # OTHER lets voice/local_llm handle "where did it go"
        kind = "CHANGES" if kind != "WHERE" else "OTHER"
    if kind in NO_OBJECT:
        obj = None
    return Intent(kind=kind, obj=obj, raw=text)


class Qwen:
    """Talks to llama-server's OpenAI-compatible chat endpoint. ask() returns the raw JSON text.
    Offline option (understand.backend: qwen); the default is Grok."""

    name = "qwen"
    local = True                                  # on the Jetson: asked online or offline

    def __init__(self, cfg: dict):
        u = cfg.get("understand") or {}
        self.url = str(u.get("url", "http://127.0.0.1:8081/v1")).rstrip("/")
        self.model = str(u.get("model", "qwen"))
        self.system = system_prompt(cfg)
        self.schema = schema(cfg)
        self.session = requests.Session()

    def ask(self, text: str, timeout: float) -> str:
        # Qwen3 thinks by default, and llama.cpp skips the JSON schema grammar while it does;
        # enable_thinking=False turns that off (other models' templates ignore it).
        r = self.session.post(f"{self.url}/chat/completions", timeout=timeout, json={
            "model": self.model, "temperature": 0, "max_tokens": 40, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
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


class Grok:
    """Same ask()/health() as Qwen, against xAI's OpenAI-compatible API (the llm: section's base_url,
    model and reasoning_effort; key in $XAI_API_KEY). Only the transcript leaves the device."""

    name = "grok"
    local = False

    def __init__(self, cfg: dict, session=None):
        llm = cfg.get("llm") or {}
        self.url = str(llm.get("base_url", "https://api.x.ai/v1")).rstrip("/")
        self.model = str(llm.get("model", "grok-4.3"))
        self.reasoning = llm.get("reasoning_effort")
        self.system = system_prompt(cfg)
        self.schema = schema(cfg)
        if session is None:
            from core.xai import session as shared
            session = shared()                    # the process-wide connection to xAI (core/xai.py)
        self.session = session

    @staticmethod
    def _key() -> str:
        return os.environ.get("XAI_API_KEY", "").strip()

    def ask(self, text: str, timeout: float) -> str:
        key = self._key()
        if not key:
            raise RuntimeError("XAI_API_KEY is not set")
        body = {"model": self.model, "temperature": 0, "max_tokens": 40,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "intent", "schema": self.schema, "strict": True}}}
        if self.reasoning:
            body["reasoning_effort"] = str(self.reasoning)
        r = self.session.post(f"{self.url}/chat/completions", timeout=timeout,
                              headers={"Authorization": f"Bearer {key}"}, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def health(self, timeout: float = 1.0) -> bool:
        return bool(self._key())


class Understander:
    """interpret(text) -> Intent. The model (understand.backend: Grok by default) when
    the rules aren't sure and it is up and quick; the rule parser otherwise.

    Remembers the last transcript, so main.Room (RESET/RECAL) and the ask pipeline share one call."""

    def __init__(self, cfg: dict, model=None, qwen=None, online: Optional[Callable[[], bool]] = None,
                 local=None, aliases: Optional[Callable[[], list]] = None):
        """aliases: names people taught ("this is my charger"; world.alias_phrases), which the rules
        trust without asking the model."""
        u = cfg.get("understand") or {}
        self.cfg = cfg
        self.enabled = bool(u.get("enabled", True))
        self.timeout_s = float(u.get("timeout_s", 1.5))
        self.backend = str(u.get("backend", "grok"))   # grok | qwen | auto
        if model is None:
            model = qwen
        if model is None and self.enabled:
            model = Qwen(cfg) if self.backend == "qwen" else Grok(cfg)
            if local is None and self.backend == "auto":
                local = Qwen(cfg)                  # offline stand-in for Grok
        self.model = model
        self.local = local if self.enabled else None
        self.online = online or (lambda: True)
        self.aliases = aliases or (lambda: [])
        self._lock = threading.Lock()
        self._last: tuple[tuple[str, bool], Intent] | None = None
        self.last_ms: float = 0.0
        self.last_by = "rules"             # who decided the last intent: grok | qwen | rules | gate
        self.wake_only = (cfg.get("listen") or {}).get("mode") == "wake"   # overheard needs "room, ..."

    @property
    def qwen(self):
        """The model, by its old name (tests and scripts written when it was always Qwen)."""
        return self.model

    @staticmethod
    def _name(model) -> str:
        return getattr(model, "name", "qwen")

    def _pick(self):
        """The model for this question: the main one if it is local, or online (and, for Grok, has a
        key); else the local stand-in (backend auto); else None, and the rules answer."""
        m = self.model
        if m is not None and (getattr(m, "local", False) or (
                self.online() and (not isinstance(m, Grok) or m.health()))):
            return m
        return self.local

    def warm(self) -> bool:
        """Check each model and run one question (Qwen: caches the prompt; Grok: opens the connection).
        True if any model is ready."""
        ready = False
        for m in (self.model, self.local):
            if m is None:
                continue
            name, local = self._name(m), getattr(m, "local", False)
            if not m.health():
                log.warning("%s not reachable at %s", name, m.url)
                continue
            if not local and not self.online():
                log.info("offline: %s is used once the connection is back", name)
                continue
            try:
                m.ask("where are my keys", timeout=30 if local else 5)
            except Exception as ex:
                log.warning("%s warm-up failed: %s", name, ex)
                continue
            log.info("%s interpreting questions at %s%s", name, m.url,
                     " (offline stand-in)" if m is self.local else "")
            ready = True
        if not ready and (self.model is not None or self.local is not None):
            log.warning("no model ready; questions use the rule parser")
        return ready

    def __call__(self, text: str, overheard: bool = False) -> Intent:
        """overheard: always-on mic speech, which may be IGNORE. Asked (clicker) speech never is."""
        with self._lock:
            key = (text, overheard)
            if self._last is not None and (self._last[0] == key or (
                    self._last[0] == (text, True) and self._last[1].kind != IGNORE)):
                return self._last[1]            # accepted overheard speech reads the same when asked
            intent = self._overheard(text) if overheard else self._interpret(text)
            self._last = (key, intent)
            return intent

    def screen(self, text: str) -> bool:
        """Overheard speech: could it be for the rig? Every check that needs no model (fast, never
        blocks); False means IGNORE. The voice loop calls it before starting the thinking cue."""
        with self._lock:
            return self._screen(text)

    def _screen(self, text: str) -> bool:
        rules = parse(text, self.cfg)
        if rules.kind == "TEACH":              # "this is my vaseline": only a whole teaching sentence parses so
            return True                        # (idioms like "call it a day" don't), and it has no question
        woke = has_wake_word(text, self.cfg)   # opening or known object for the gate
        if (not text.strip() or not gate(text, self.cfg) or not addressed(text, self.cfg)
                or (self.wake_only and not woke)):
            return False
        if rules.kind in ACTS and not woke:    # "let's reset after this"
            return False
        # "where are you guys from": nothing to look for, and not said to the rig
        return woke or rules.obj is not None or rules.name is not None or rules.kind not in ("OTHER", "WHERE")

    def _overheard(self, text: str) -> Intent:
        ignore = Intent(kind=IGNORE, obj=None, raw=text)
        if not self._screen(text):
            self.last_by, self.last_ms = "gate", 0.0
            return ignore
        i = self._interpret(text)
        if i.kind == "TEACH":
            return i
        woke = has_wake_word(text, self.cfg)
        if i.kind in ACTS and not woke:
            return ignore
        if i.kind in ("OTHER", "WHERE") and not (woke or i.obj or i.name or names_object(text, self.cfg)):
            return ignore
        return i

    def _interpret(self, text: str) -> Intent:
        rules = parse(text, self.cfg)
        try:
            taught = list(self.aliases() or [])
        except Exception:
            log.exception("taught names unavailable")
            taught = []
        model = None if rules_sure(rules, self.cfg, taught) or not text.strip() else self._pick()
        if model is None:
            self.last_by, self.last_ms = "rules", 0.0
            return rules
        name = self._name(model)
        hint = rules.obj if matched_exactly(text, rules.obj, self.cfg) else None   # not a guessed one
        t0 = time.monotonic()
        try:
            # requests' timeout is per phase and skips DNS: bound the whole call (Wi-Fi at a venue)
            raw = call_with_deadline(model.ask, self.timeout_s + 0.2, text, self.timeout_s, name=f"understand-{name}")
            got = to_intent(raw, text, self.cfg, hint)
        except Exception as ex:
            log.warning("%s failed (%s: %s); using the rule parser", name, type(ex).__name__, ex)
            got = None
        self.last_ms = 1000 * (time.monotonic() - t0)
        if got is not None and got.obj is None and rules.kind != "OTHER" and (rules.obj or rules.name):
            got = None                         # the model found nothing better than the rules' guess or name
        if got is None:
            self.last_by = "rules"
            return rules
        self.last_by = name
        if (got.kind, got.obj) != (rules.kind, rules.obj):
            log.info("%s read %r as %s %s (rules: %s %s) in %.0f ms", name, text, got.kind, got.obj,
                     rules.kind, rules.obj, self.last_ms)
        return got


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    overheard = "--overheard" in argv
    argv = [a for a in argv if a != "--overheard"]
    cfg = load_config()
    u = Understander(cfg)
    up = u.warm()
    for text in argv or [line.strip() for line in sys.stdin if line.strip()]:
        i = u(text, overheard)
        print(f"{i.kind:8} {str(i.obj):12} {u.last_by:5} {u.last_ms:5.0f} ms  {text}")
    return 0 if up else 1


if __name__ == "__main__":
    raise SystemExit(main())
