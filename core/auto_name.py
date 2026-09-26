"""Automatic names for new things: a soft guess of what an unnamed thing:N is, so "where's my deodorant?"
can be answered after it was hidden, even though nobody taught its name.

When the world confirms a new thing (an APPEARED event for thing:N), one close-up of it is queued: the
crop store's view of that thing (core.crops.active().for_entity: only views confirmed as it), else the
current frame cut at the thing's box with a margin. A worker thread sends each close-up to Grok once
(the visual_memory: section's provider, model, base_url and timeout, through core/xai.py) and asks for
strict JSON {name: 1-3 word common noun, also: up to 3 alternatives, confidence}. Perception never waits:
the update hook only copies a crop and queues it.

Rules: calls only while online (offline, the job waits); at most max_per_minute calls; one successful
name per thing; a failed call is retried once, retry_after_s later, then given up; a thing that has a
taught alias (or was merged or reset away) is never sent. A reply under min_confidence, or a name that
says nothing ('object'), is kept as no guess and not asked again.

The guess is not an alias. It lives here (thing -> {name, also, confidence}), is merged into
state_json's thing entries as "guess" (dashboard, phone, Grok's world state) and is reached through
world.find_guess(said), which voice/answers.py consults only after configured names and taught
aliases found nothing. A thing with a taught alias is never found by its guess: taught names win.
Answers that rely on a guess hedge ("Your deodorant, I think, is under the notebook.").

Medication: a pill bottle is named plainly ('pill bottle'); the prompt forbids any claim about its
contents or use, and an alternative that makes one (core.narration_store.med_claim) is dropped.

Privacy: one close-up per new thing goes to Grok while online (README's privacy statement; the
disclosure is in state_json's auto_name status, shown on the dashboard).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from typing import Callable, Optional

import numpy as np

from core.things import is_thing, norm_name
from core.types import EventType, Status

log = logging.getLogger(__name__)

GENERIC = {"object", "objects", "thing", "things", "item", "items", "unknown", "something", "stuff",
           "unclear", "none", "nothing", "table", "tabletop", "unidentified object"}
MAX_WORDS = 3
MAX_ALSO = 3

NAME_SYSTEM = """You name one object from an overhead close-up of a tabletop (the camera looks straight down).
Reply with the everyday name a person would use when asking where it is.

Rules:
- name: a common noun of 1 to 3 words, lowercase, no brand, no colour, e.g. "deodorant stick", "coffee mug", "phone charger".
- also: up to 3 other short names people might say for it, e.g. "deodorant"; an empty list if there are none.
- confidence: 0 to 1. If you can't tell what it is, set it below 0.5.
- If it is a medicine or pill bottle, just name it plainly ("pill bottle"). Never say anything about medication being taken, its contents or its use.
- Name the main object in the middle of the image only; ignore hands and the table.
Reply with the JSON object only."""

NAME_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["name", "also", "confidence"],
    "properties": {"name": {"type": "string"}, "also": {"type": "array", "items": {"type": "string"}},
                   "confidence": {"type": "number"}},
}


@dataclass
class AutoNameConfig:
    """The auto_name: section of config.yaml (off when the section is absent). The VLM itself is the
    visual_memory: section's."""
    enabled: bool = False
    max_per_minute: int = 10       # Grok calls per minute; more new things wait their turn
    min_confidence: float = 0.5    # a name below this is not kept
    retry_after_s: float = 30.0    # a failed call is tried once more this much later
    crop_px: int = 384             # the close-up's long side as sent (small crops are upscaled)
    margin: float = 0.15           # frame crop fallback: the box grown by this fraction per side
    max_pending: int = 32          # queued close-ups at most (the oldest go first)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AutoNameConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d or {}).items() if k in known})


@dataclass
class Job:
    name: str                      # thing:N
    img: np.ndarray                # BGR close-up
    attempts: int = 0
    due: float = 0.0               # clock() before which it is not tried


# ---------------------------------------------------------------- names and matching

def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def clean_name(text) -> Optional[str]:
    """A spoken-style name from the model's text: lowercase, no articles or punctuation, at most
    MAX_WORDS words; None for nothing or a word that names no object ('object', 'unknown')."""
    if not isinstance(text, str):
        return None
    key = norm_name(text)
    words = [w for w in key.split() if not w.isdigit()][:MAX_WORDS]
    name = " ".join(words)
    if not name or name in GENERIC or all(w in GENERIC for w in words):
        return None
    return name


MODIFIERS = {"red", "blue", "green", "black", "white", "yellow", "orange", "purple", "pink", "brown", "grey",
             "gray", "silver", "gold", "clear", "big", "small", "little", "large", "tiny", "old", "new"}


def _tokens(phrase: str) -> list[str]:
    return [_singular(w) for w in norm_name(phrase).split()]


def match_score(said: str, guess: dict) -> float:
    """How well a spoken name fits a guess (0: not at all), plurals folded and a colour or size word in
    what was said ignored ('my blue mug'). The same words: 3 for the guessed name, 2.5 for an
    alternative. Sharing the head noun, one's words all inside the other's ('mug' / 'coffee mug'): 2
    (1.5). The spoken words all inside the guess ('deodorant' in 'deodorant stick'): 1.5 (1). Nothing
    else: 'phone case' never fits 'phone charger', nor 'glue stick' 'deodorant stick'."""
    s = _tokens(said)
    s = [w for w in s if w not in MODIFIERS] or s
    if not s or not isinstance(guess, dict):
        return 0.0
    best = 0.0
    for phrase, cut in [(guess.get("name"), 0.0)] + [(a, 0.5) for a in (guess.get("also") or [])]:
        g = _tokens(phrase) if isinstance(phrase, str) else []
        if not g:
            continue
        if s == g:
            sc = 3.0
        elif s[-1] == g[-1] and (set(s) <= set(g) or set(g) <= set(s)):
            sc = 2.0
        elif set(s) <= set(g):
            sc = 1.5
        else:
            continue
        best = max(best, sc - cut)
    return best


# ---------------------------------------------------------------- the namer

def _number(name: str) -> int:
    try:
        return int(str(name).split(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def _jpeg(img: np.ndarray, long_side: int) -> bytes:
    """JPEG bytes of img scaled so its long side is long_side (small crops are enlarged: tiny images
    hurt the VLM)."""
    import cv2
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s != 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _crop_store():
    try:
        from core.crops import active
        return active()
    except Exception:
        return None


class AutoNamer:
    def __init__(self, cfg: dict, world, provider=None, online: Optional[Callable[[], bool]] = None,
                 clock: Callable[[], float] = time.monotonic, c: Optional[AutoNameConfig] = None,
                 start: bool = True):
        self.cfg, self.world, self.clock = cfg, world, clock
        self.c = c or AutoNameConfig.from_dict((cfg or {}).get("auto_name"))
        if provider is None:
            from core.narration import NarrationConfig, make_provider
            from core.visual_memory import VisualConfig
            v = VisualConfig.from_dict((cfg or {}).get("visual_memory"))
            provider = make_provider(NarrationConfig.from_dict({
                "provider": v.provider, "model": v.model, "base_url": v.base_url, "api_key_env": v.api_key_env,
                "reasoning_effort": v.reasoning_effort, "timeout_s": v.timeout_s, "max_tokens": 200}))
        self.provider = provider
        self.online = online or (lambda: bool(getattr(world, "online", False)))
        self._lock = threading.Lock()
        self._jobs: deque = deque()
        self._guesses: dict[str, dict] = {}
        self._done: set[str] = set()           # named, or given up on: never asked again
        self._failed = 0
        self._calls: deque = deque()           # clock() of recent calls (the per-minute cap)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if start:
            self._thread = threading.Thread(target=self._run, name="auto-name", daemon=True)
            self._thread.start()

    # -- reading (any thread)

    def guess(self, name: str) -> Optional[dict]:
        with self._lock:
            g = self._guesses.get(name)
            return {"name": g["name"], "also": list(g["also"]), "confidence": g["confidence"]} if g else None

    def guesses(self) -> dict[str, dict]:
        with self._lock:
            return {n: {"name": g["name"], "also": list(g["also"]), "confidence": g["confidence"]}
                    for n, g in self._guesses.items()}

    def find_guess(self, said: str) -> list[tuple[str, float]]:
        """Things whose guess fits a spoken name, as (thing, score), best first: by score, then a
        visible thing, then the most recently seen, then the newest. Things with a taught alias are
        left out."""
        hits: dict[str, tuple] = {}
        for n, g in self.guesses().items():
            sc = match_score(said, g)
            if sc <= 0:
                continue
            try:
                ent = self.world.get(n)
                if getattr(ent, "merged_into", None):
                    n = self.world.find(n) if hasattr(self.world, "find") else None
                    ent = self.world.get(n) if n else None
            except Exception:
                continue
            if ent is None or ent.aliases or getattr(ent, "merged_into", None):
                continue
            key = (sc, ent.status == Status.VISIBLE, ent.last_seen or 0.0, _number(n))
            if n not in hits or key > hits[n]:
                hits[n] = key
        return [(n, k[0]) for n, k in sorted(hits.items(), key=lambda kv: kv[1], reverse=True)]

    def status(self) -> dict:
        with self._lock:
            pending, named = len(self._jobs), len(self._guesses)
        return {"enabled": True, "provider": getattr(self.provider, "name", "grok"),
                "model": getattr(self.provider, "model", None), "named": named, "pending": pending,
                "failed": self._failed,
                "disclosure": (f"New objects are named automatically: while online, one close-up of each new "
                               f"object on the table is sent once to {getattr(self.provider, 'name', 'grok')} "
                               f"({getattr(self.provider, 'model', '')}) for a guessed name.")}

    # -- queueing (the perception thread, inside world.update's hook)

    def observe(self, events, dets, frame) -> None:
        """Queue a close-up of every thing this batch confirmed as new. Cheap: a crop copy at most."""
        for ev in events or []:
            if str(ev.type) != EventType.APPEARED.value or not is_thing(ev.obj):
                continue
            with self._lock:
                if ev.obj in self._done or any(j.name == ev.obj for j in self._jobs):
                    continue
            try:
                ent = self.world.get(ev.obj)
            except Exception:
                continue
            if ent.aliases:
                continue
            img = self._close_up(ent, dets, frame)
            if img is None:
                log.debug("no close-up for %s; not named", ev.obj)
                continue
            with self._lock:
                self._jobs.append(Job(ev.obj, img, due=self.clock()))
                while len(self._jobs) > self.c.max_pending:
                    self._jobs.popleft()
            self._wake.set()

    def _close_up(self, ent, dets, frame) -> Optional[np.ndarray]:
        """The crop store's confirmed view of ent, else the frame cut at ent's box (+ margin)."""
        store = _crop_store()
        if store is not None:
            try:
                tr = store.for_entity(ent)
                crop = (tr.best or tr.recent) if tr is not None else None
                if crop is not None and crop.img is not None:
                    return crop.img
            except Exception:
                log.debug("crop store failed", exc_info=True)
        img = getattr(frame, "img", None)
        if img is None or dets is None:
            return None
        box = next((d.box_px for d in dets.items if d.box_cm is ent.box_cm), None)
        if box is None and ent.pos_cm is not None:
            near = [d for d in dets.items if d.cls == "thing"
                    and abs(d.center_cm[0] - ent.pos_cm[0]) + abs(d.center_cm[1] - ent.pos_cm[1]) < 2.0]
            box = near[0].box_px if near else None
        if box is None:
            return None
        from core.crops import _clip
        c = _clip(img, box, self.c.margin)
        return None if c is None else img[c[1]:c[3], c[0]:c[2]].copy()

    # -- the worker

    def _capped(self, now: float) -> bool:
        while self._calls and self._calls[0] <= now - 60.0:
            self._calls.popleft()
        return len(self._calls) >= self.c.max_per_minute

    def _stale(self, name: str) -> bool:
        """Not worth a call: taught meanwhile, merged into another thing, or gone with a reset."""
        try:
            ent = self.world.get(name)
        except Exception:
            return True
        return bool(ent.aliases) or bool(getattr(ent, "merged_into", None))

    def step(self, now: Optional[float] = None) -> bool:
        """Try the next due close-up. True when a call was made."""
        now = self.clock() if now is None else now
        with self._lock:
            names = [j.name for j in self._jobs]
        stale = {n for n in names if self._stale(n)}      # world.get outside our lock: no lock order to keep
        with self._lock:
            for j in [j for j in self._jobs if j.name in stale]:
                self._jobs.remove(j)
            job = next((j for j in self._jobs if j.due <= now), None)
            if job is None or not self.online() or self._capped(now):
                return False
            self._jobs.remove(job)
            self._calls.append(now)
        job.attempts += 1
        try:
            g = self._ask(job.img)
        except Exception as ex:
            log.info("naming %s failed (attempt %d): %s", job.name, job.attempts, ex)
            with self._lock:
                if job.attempts < 2:
                    job.due = now + self.c.retry_after_s
                    self._jobs.append(job)
                else:
                    self._done.add(job.name)
                    self._failed += 1
            return True
        keep = g is not None and not self._stale(job.name)
        with self._lock:
            self._done.add(job.name)
            if keep:
                self._guesses[job.name] = g
                log.info("%s looks like a %s (%.2f)", job.name, g["name"], g["confidence"])
        return True

    def _ask(self, img: np.ndarray) -> Optional[dict]:
        from core.narration import _parse_json
        from core.narration_store import med_claim
        jpg = _jpeg(img, self.c.crop_px)
        reply = self.provider.narrate(NAME_SYSTEM, [("text", "Close-up of one object on the table:"),
                                                    ("image", jpg), ("text", "What is it called?")], NAME_SCHEMA)
        d = _parse_json(reply.text)
        try:
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(1.0, max(0.0, conf)) if conf == conf else 0.0
        name = clean_name(d.get("name"))
        if name is None or med_claim(name) or conf < self.c.min_confidence:
            return None
        also = []
        for a in d.get("also") or []:
            a = clean_name(a)
            if a and a != name and a not in also and not med_claim(a):
                also.append(a)
        return {"name": name, "also": also[:MAX_ALSO], "confidence": round(conf, 3)}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                made = self.step()
            except Exception:
                log.exception("auto-name step failed")
                made = False
            if not made:
                self._wake.wait(0.5)
                self._wake.clear()

    # -- wiring

    def attach(self, world) -> "AutoNamer":
        """After every world.update, queue close-ups of new things; state_json's things carry their
        guess and the status carries the disclosure; world.find_guess(said) looks guesses up."""
        update = getattr(world, "update", None)
        if callable(update):
            def update_and_name(dets, frame):
                out = update(dets, frame)
                try:
                    self.observe(out, dets, frame)
                except Exception:
                    log.exception("auto-name observe failed")
                return out

            world.update = update_and_name
        state = world.state_json

        def state_json(*a, **kw):
            st = state(*a, **kw)
            try:
                g = self.guesses()
                for e in st.get("entities") or []:
                    if isinstance(e, dict) and e.get("name") in g:
                        e["guess"] = g[e["name"]]
                st["auto_name"] = self.status()
            except Exception:
                log.exception("auto-name status failed")
            return st

        world.state_json = state_json
        world.find_guess = self.find_guess
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def from_config(cfg: dict, world, online=None, start: bool = True) -> Optional[AutoNamer]:
    """The app's namer, attached to the world, or None when auto_name.enabled is false (the default)."""
    c = AutoNameConfig.from_dict((cfg or {}).get("auto_name"))
    if not c.enabled:
        return None
    namer = AutoNamer(cfg, world, online=online, c=c, start=start).attach(world)
    log.info("auto-naming on: %s", namer.status()["disclosure"])
    return namer
