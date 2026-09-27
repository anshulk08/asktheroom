"""Automatic names for new things: a soft guess of what an unnamed thing:N is, so "where's my deodorant?"
can be answered after it was hidden, even though nobody taught its name.

When the world confirms a new thing (an APPEARED event for thing:N), one close-up of it is queued. With
room memory on, the camera runs at 2560x1440 and the table pipeline sees a 1280x720 resize of a ~817x460
region (core/room_view.TableView); the close-up is then cut from the full frame of the same moment
(frames.full_at(t)) at native resolution, with a wider marked view of the spot (the thing in a red box)
as context. Without a full frame: the crop store's view of that thing (core.crops.active().for_entity:
only views confirmed as it), else the current frame cut at the thing's box with a margin, and the frame's
marked spot as context. A worker thread sends each close-up to Grok once (the visual_memory: section's
provider, model, base_url and timeout, through core/xai.py) and asks for strict JSON {object: is it a
separate object, name: 1-3 word common noun, also: up to 3 alternatives, confidence}. Perception never
waits: the update hook only copies crops and queues them.

Naming accuracy (eval/naming.py, 120 hand-labelled rig crops, Sun 27 Sep, mean of 3 runs): the Sat 26 Sep
namer (a 128 px table-view crop, an overhead-camera prompt) got 44/120 right: it named 48 of 61 hands,
feet, worn watches and jeans as objects. The corner-camera prompt, the context view, the object flag,
NOT_OBJECTS and min_confidence 0.65 get 83/120: 50 of 61 rejected, wrong names for real things 26 -> 9 of 59.

Rules: calls only while online (offline, the job waits); at most max_per_minute calls; one successful
name per thing; a failed call is retried once, retry_after_s later, then given up; a thing that has a
taught alias (or was merged or reset away) is never sent. A reply under min_confidence, one that says
it is no object (a person, something worn, furniture), a name that says nothing ('object', a colour
alone, 'white object') or one naming a body part, a person or worn clothing (NOT_OBJECTS) is kept as no
guess and not asked again.

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
           "unclear", "none", "nothing", "table", "tabletop", "unidentified object", "unidentifiable",
           "unidentified", "shape", "blob", "piece"}
# A name whose last word is one of these is not an object someone would ask about: a body part, a person,
# worn clothing, the furniture or the room, a trick of the light. Rig logs, Sat 26 Sep: room tracks named
# hand x97, arm, finger, ear, person, persons leg, shirt, shorts, fabric, wooden table. Shoes, watches and
# glasses are not here: lying on a table they are objects (the prompt's object flag says when they are worn).
NOT_OBJECTS = {"hand", "hands", "finger", "fingers", "thumb", "palm", "fist", "arm", "arms", "forearm",
               "elbow", "wrist", "shoulder", "leg", "legs", "knee", "thigh", "foot", "feet", "toe", "ankle",
               "face", "head", "hair", "nose", "ear", "eye", "mouth", "lip", "neck", "chin", "skin", "body",
               "person", "people", "man", "woman", "boy", "girl", "child", "human", "lap",
               "sleeve", "shirt", "tshirt", "t shirt", "sweater", "hoodie", "sweatshirt", "jeans", "pants",
               "trousers", "shorts", "sock", "socks", "fabric", "clothing",
               "button", "zipper", "pocket", "collar", "logo", "clothing tag",
               "floor", "wall", "carpet", "rug", "couch", "sofa", "tabletop", "surface", "wooden table",
               "shadow", "reflection", "glare"}
MAX_WORDS = 3
TRAILING = {"of", "with", "and", "for", "on", "in", "or", "a", "the"}      # never the last word of a name
MAX_ALSO = 3

NAME_SYSTEM = """You name one object for a person who will later ask where it is. The camera is high in a corner of a room and looks down at an angle, so objects are seen from above and from the side, often small, and the light is warm and dim, so colours can look off.

You get a close-up of the object and, when there is one, a wider view of the same spot with the object in a red box. Name only the object inside the red box (the middle of the close-up), never one next to it.

Rules:
- object: false if what is in the box is a person or part of one (hand, arm, finger, wrist, leg, knee, foot, face, head, hair), something a person is wearing or a detail of it (a watch or band on a wrist, a sock or shoe on a foot, a shirt, jeans, a button, zipper or logo on clothes, glasses on a face, a necklace), part of the furniture or the room (table, table leg, couch, floor, wall), a shadow or reflection, or nothing you can make out. Otherwise true.
- name: the everyday name a person would use when asking where it is: a common noun of 1 to 3 words, lowercase, no brand, e.g. "coffee mug", "deodorant stick", "house keys". Put a colour first only if you are sure of it ("white mug"); never a colour alone and never "white object". If you can't tell what the object is, describe it plainly ("plastic part", "small box") with a low confidence. Empty if object is false.
- also: up to 3 other short names people might say for it; an empty list if there are none.
- confidence: 0 to 1, how sure you are of the name. A guess from a blurry or partly hidden shape is below 0.5.
- If it is a medicine or pill bottle, just name it plainly ("pill bottle"). Never say anything about medication being taken, its contents or its use.
- A hand holding or touching the object is fine: name the object, not the hand.
Reply with the JSON object only."""

NAME_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["object", "name", "also", "confidence"],
    "properties": {"object": {"type": "boolean"}, "name": {"type": "string"},
                   "also": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number"}},
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
    jpeg_quality: int = 90         # of the close-up and the context view as sent
    margin: float = 0.15           # a frame close-up: the box grown by this fraction per side
    context: bool = True           # also send a wider view of the spot with the thing in a red box
    context_px: int = 384          # the context view's long side as sent
    context_min_px: int = 240      # the context patch is at least this many source px (and 3x the box)
    max_pending: int = 32          # queued close-ups at most (the oldest go first)
    rename_after_s: float = 20.0   # a thing whose reply was no usable name gets a fresh close-up this much later ...
    rename_max: int = 1            # ... at most this many times (0: never), while it is visible on the table

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AutoNameConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d or {}).items() if k in known})


@dataclass
class Job:
    name: str                      # thing:N
    img: np.ndarray                # BGR close-up
    ctx: Optional[np.ndarray] = None   # BGR wider view with the thing in a red box, or None
    attempts: int = 0
    due: float = 0.0               # clock() before which it is not tried


# ---------------------------------------------------------------- names and matching

def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


MODIFIERS = {"red", "blue", "green", "black", "white", "yellow", "orange", "purple", "pink", "brown", "grey",
             "gray", "silver", "gold", "clear", "big", "small", "little", "large", "tiny", "old", "new"}


def clean_name(text) -> Optional[str]:
    """A spoken-style name from the model's text: lowercase, no articles or punctuation, at most
    MAX_WORDS words; None for nothing or a name that names no object: only words like 'object' or
    'unknown', or only colour and size words around them ('white', 'white object', 'small black thing').
    'orange' alone stays: it is a fruit too."""
    if not isinstance(text, str):
        return None
    key = norm_name(text)
    words = [w for w in key.split() if not w.isdigit()]
    if len(words) > MAX_WORDS and words[MAX_WORDS - 1] in TRAILING:     # 'white bar of soap': not 'white bar of'
        words = [w for w in words if w not in MODIFIERS] or words
    words = words[:MAX_WORDS]
    while len(words) > 1 and words[-1] in TRAILING:
        words.pop()
    name = " ".join(words)
    if not name or name in GENERIC or (all(w in GENERIC or w in MODIFIERS for w in words) and name != "orange"):
        return None
    return name


def not_object(name: str) -> bool:
    """True when a cleaned name is a body part, a person, worn clothing or part of the room (NOT_OBJECTS,
    by its last word or the whole name, plurals folded): 'hand', 'persons leg', 'table leg', 'grey shirt'."""
    words = norm_name(name).split()
    return bool(words) and (" ".join(words) in NOT_OBJECTS or words[-1] in NOT_OBJECTS
                            or _singular(words[-1]) in NOT_OBJECTS)


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


def judge(d: dict, min_confidence: float) -> Optional[dict]:
    """The guess kept from Grok's parsed reply, or None: it says no object, the name names nothing or no
    object (clean_name, not_object), makes a medical claim, or is under min_confidence. Alternatives
    that fail the same checks are dropped."""
    from core.narration_store import med_claim
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(1.0, max(0.0, conf)) if conf == conf else 0.0
    name = clean_name(d.get("name"))
    if d.get("object") is False or name is None or not_object(name) or med_claim(name) or conf < min_confidence:
        return None
    also = []
    for a in d.get("also") or []:
        a = clean_name(a)
        if a and a != name and a not in also and not med_claim(a) and not not_object(a):
            also.append(a)
    return {"name": name, "also": also[:MAX_ALSO], "confidence": round(conf, 3)}


# ---------------------------------------------------------------- the namer

def _number(name: str) -> int:
    try:
        return int(str(name).split(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def _jpeg(img: np.ndarray, long_side: int, quality: int = 85) -> bytes:
    """JPEG bytes of img scaled so its long side is long_side (small crops are enlarged: tiny images
    hurt the VLM)."""
    import cv2
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s != 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _to_full(box, rect, out_size) -> tuple[float, float, float, float]:
    """A table-view box (px of the out_size image) in full-frame px, for the view cut at rect."""
    sx = (rect[2] - rect[0]) / float(out_size[0])
    sy = (rect[3] - rect[1]) / float(out_size[1])
    return (rect[0] + box[0] * sx, rect[1] + box[1] * sy, rect[0] + box[2] * sx, rect[1] + box[3] * sy)


def _crop_store():
    try:
        from core.crops import active
        return active()
    except Exception:
        return None


class AutoNamer:
    def __init__(self, cfg: dict, world, provider=None, online: Optional[Callable[[], bool]] = None,
                 clock: Callable[[], float] = time.monotonic, c: Optional[AutoNameConfig] = None,
                 start: bool = True, frames=None):
        self.cfg, self.world, self.clock = cfg, world, clock
        # core.room_view.TableView (full_at, rect, out_size) when room memory runs the camera at full size:
        # close-ups are then cut from the full frame at native resolution
        self.frames = frames
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
        self._unnamed: dict[str, list] = {}    # replied with no usable name -> [clock() of that, re-asks so far]
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
            views = self._close_up(ent, dets, frame)
            if views is None:
                log.debug("no close-up for %s; not named", ev.obj)
                continue
            with self._lock:
                self._jobs.append(Job(ev.obj, views[0], views[1], due=self.clock()))
                while len(self._jobs) > self.c.max_pending:
                    self._jobs.popleft()
            self._wake.set()
        self._rename(dets, frame)

    def _rename(self, dets, frame) -> None:
        """One more close-up for a thing whose reply was no usable name, rename_after_s later, while it
        is visible on the table. The first close-up is taken the moment it appears (often a hand is still
        on it, or it is small and dark from a corner camera: spec 0009 rig run); a settled view often gets
        a name."""
        if not self._unnamed:
            return
        now = self.clock()
        with self._lock:
            due = [n for n, (t, k) in self._unnamed.items() if k < self.c.rename_max and now - t >= self.c.rename_after_s
                   and not any(j.name == n for j in self._jobs)]
        for name in due:
            try:
                ent = self.world.get(name)
            except Exception:
                continue
            if ent.aliases or ent.status != Status.VISIBLE or getattr(ent, "zone", "table") != "table":
                continue
            views = self._close_up(ent, dets, frame)
            with self._lock:
                t, k = self._unnamed.get(name, [now, 0])
                self._unnamed[name] = [now, k if views is None else k + 1]
                if views is None:
                    continue
                self._done.discard(name)
                self._jobs.append(Job(name, views[0], views[1], due=now))
            self._wake.set()

    def _close_up(self, ent, dets, frame) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        """(close-up, context view or None) of ent. Its box in this frame's detections, cut from the full
        frame of the same moment when there is one (native resolution: the table view is a ~1.6x
        enlargement of an 817 px wide region at 1440p); else the crop store's confirmed view of ent, else
        this frame cut at its box. The context view is the same frame's patch around the box with ent in a
        red box (off with context: false, and never for a crop-store view, whose frame is gone)."""
        from core.crops import close_up, marked_view, shrink
        views = self._views(ent, dets, frame, close_up, marked_view)
        return None if views is None else (shrink(views[0], self.c.crop_px), shrink(views[1], self.c.context_px))

    def _views(self, ent, dets, frame, close_up, marked_view):
        img = getattr(frame, "img", None)
        box = self._box(ent, dets) if img is not None else None
        full = self._full(frame) if box is not None else None
        if full is not None:
            fbox = _to_full(box, self.frames.rect, self.frames.out_size)
            crop = close_up(full, fbox, self.c.margin)
            if crop is not None:
                return crop, (marked_view(full, fbox, self.c.context_min_px) if self.c.context else None)
        store = _crop_store()
        if store is not None:
            try:
                tr = store.for_entity(ent)
                crop = (tr.best or tr.recent) if tr is not None else None
                if crop is not None and crop.img is not None:
                    return crop.img, None
            except Exception:
                log.debug("crop store failed", exc_info=True)
        if box is None:
            return None
        crop = close_up(img, box, self.c.margin)
        if crop is None:
            return None
        return crop, (marked_view(img, box, self.c.context_min_px) if self.c.context else None)

    @staticmethod
    def _box(ent, dets):
        """ent's box (px) among this frame's detections: the one the world took as its latest observation,
        else a proposal within 2 cm of its position; None when it was not seen in this frame."""
        if dets is None:
            return None
        box = next((d.box_px for d in dets.items if d.box_cm is ent.box_cm), None)
        if box is None and ent.pos_cm is not None:
            near = [d for d in dets.items if d.cls == "thing"
                    and abs(d.center_cm[0] - ent.pos_cm[0]) + abs(d.center_cm[1] - ent.pos_cm[1]) < 2.0]
            box = near[0].box_px if near else None
        return box

    def _full(self, frame) -> Optional[np.ndarray]:
        """The full camera frame `frame` (a table-view Frame) was cut from, or None: no full-frame source,
        or the ring no longer holds that exact frame (a box from another moment would miss the thing)."""
        full_at = getattr(self.frames, "full_at", None)
        if full_at is None or getattr(frame, "t", None) is None:
            return None
        try:
            f = full_at(frame.t)
        except Exception:
            log.debug("full frame lookup failed", exc_info=True)
            return None
        if f is None or f.img is None or f.idx != frame.idx:
            return None
        return f.img

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
            g = self._ask(job.img, job.ctx)
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
                self._unnamed.pop(job.name, None)
                log.info("%s looks like a %s (%.2f)", job.name, g["name"], g["confidence"])
            elif g is None:
                n = self._unnamed.get(job.name, [0.0, 0])[1]
                self._unnamed[job.name] = [now, n]
                log.info("naming %s: no usable name (low confidence or 'object')%s", job.name,
                         "; a fresh close-up later" if n < self.c.rename_max else "")
        return True

    def _ask(self, img: np.ndarray, ctx: Optional[np.ndarray] = None) -> Optional[dict]:
        """Grok's guess for a close-up (and the context view, when there is one), or None: no object, a
        name that names nothing or no object (clean_name, NOT_OBJECTS), a medical claim, or under
        min_confidence."""
        from core.narration import _parse_json
        q = self.c.jpeg_quality
        parts = [("text", "Close-up of the object:"), ("image", _jpeg(img, self.c.crop_px, q))]
        if ctx is not None:
            parts += [("text", "The same spot, wider; the object is in the red box:"),
                      ("image", _jpeg(ctx, self.c.context_px, q))]
        reply = self.provider.narrate(NAME_SYSTEM, parts + [("text", "What is it called?")], NAME_SCHEMA)
        return judge(_parse_json(reply.text), self.c.min_confidence)

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
        guess and the status carries the disclosure; world.find_guess(said) looks guesses up and
        world.thing_guess(thing) reads one (the room handoff, core/room_world.py)."""
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
        world.thing_guess = self.guess     # room memory: a departed thing's name, matched against room tracks
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def from_config(cfg: dict, world, online=None, start: bool = True, frames=None) -> Optional[AutoNamer]:
    """The app's namer, attached to the world, or None when auto_name.enabled is false (the default).
    frames: the app's frame source; a TableView (room memory on) gives native-resolution close-ups."""
    c = AutoNameConfig.from_dict((cfg or {}).get("auto_name"))
    if not c.enabled:
        return None
    namer = AutoNamer(cfg, world, online=online, c=c, start=start, frames=frames).attach(world)
    log.info("auto-naming on: %s", namer.status()["disclosure"])
    return namer
