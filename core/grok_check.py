"""Grok settle check: each time the table settles, Grok looks at the tracker's boxes and says which are
real, what they are, and what the tracker missed. The verdicts go to SQLite for answers and review.

Why not Grok as the detector (docs/specs/0007-grok-settle-check.md): the world rules need hand and
object boxes at 10+ fps (contact, container dwell, debounce, edge exits), a VLM at ~1 Hz misses the
moments that hide things, its boxes are loose (grok-4.3 boxes scored 0-2/5 on our desk photo) and it
sees nothing offline. So YOLO keeps proposing and tracking, and Grok verifies by picking marks, the
form that measured 5/5 (voice/visual.py).

When. The narration Segmenter (core/narration.py) cuts the stream into activity episodes: hands or
world events open one, quiet_s with neither closes it. The close is the settled moment and its frame
is the one checked (so is the first quiet_s after start-up). feed() runs on the perception thread and
only swaps a reference into a one-slot box; the 'grok-check' worker sends the newest settled frame, at
most one call per min_gap_s and max_per_hour, and only while online. If an episode has opened again by
the time the worker gets to it, the frame is dropped (the world's boxes no longer match it).

What. Every VISIBLE tracked entity with a box is drawn as a numbered mark (voice.visual.draw_marks);
Grok replies per mark {real, label, confidence} and lists objects no mark covers with a centre point.
Each becomes a grok_checks row (verdict agree | relabel | phantom | named | unsure | unmarked, table cm,
time, latency, model) in the EventLog's SQLite file. Effects, none of them inside core/world.py:
  - an unnamed thing:N that Grok labels at >= bind_conf takes the label (world.bind_alias), only when no
    other entity answers to it; a taught name is never replaced;
  - state_json()['grok_check'] carries the last check and its disagreements for the dashboard;
  - lookup(name) gives the newest sighting of a name. voice.visual uses it for WHERE when the world has
    never had a position for the object; it reads stored rows, so it also works offline.
A phantom verdict is only recorded: vetoing world entities is a proposal for core/world.py's owner.

Privacy: off by default (grok_check.enabled). When on, one still frame of the table goes to the
provider each time the table settles, and status()['disclosure'] says so. No frame is kept on disk.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from typing import Callable, Optional

from core.narration import (FakeProvider, NarrationConfig, NarrationError, ProviderError, Segmenter,
                            _parse_json, make_provider)
from core.narration_store import med_claim, stem
from core.types import Event

log = logging.getLogger(__name__)

MAX_UNMARKED = 8
MOVED_CM = 5.0                                            # a belief starts over past this
SIGHTED = ("agree", "relabel", "named", "unmarked")      # verdicts whose label and position lookup() trusts


@dataclass
class GrokCheckConfig:
    """The config.yaml grok_check: section. Every key is optional."""
    enabled: bool = False
    provider: str = "grok"                  # grok | claude | fake (canned agreement, for --fake runs)
    model: Optional[str] = None             # default grok-4.3
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None       # default XAI_API_KEY
    reasoning_effort: Optional[str] = "none"
    quiet_s: float = 1.5                    # no hands and no world events this long = settled
    min_hand_frames: int = 2
    min_gap_s: float = 5.0                  # between calls; a newer settle waits and replaces an older one
    max_per_hour: int = 120
    look_px: int = 1280
    timeout_s: float = 8.0
    max_tokens: int = 1024
    min_conf: float = 0.6                   # below this a verdict is stored as 'unsure' and has no effect
    bind_names: bool = True
    bind_conf: float = 0.7                  # an unnamed thing takes Grok's label at or above this
    merge_same: bool = False                # one per kind: a thing named what a lost thing answers to is it
    # Belief: an unnamed thing's top-3 guesses, summed over checks (older ones fade), name it or retire it.
    belief_enabled: bool = False
    min_obs: int = 3                        # checks before the belief acts
    name_p: float = 0.6                     # top label's share to name the thing
    name_margin: float = 0.2                # ... and its lead over the next label
    retire_p: float = 0.7                   # 'not an object' share to retire it as clutter
    half_life_s: float = 120.0
    veto_s: float = 120.0                   # no new thing where a retired one lay, this long
    sighting_max_age_s: float = 900.0       # lookup() ignores older sightings
    keep_h: float = 24.0                    # rows older than this are deleted on start

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GrokCheckConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def narration_config(self) -> NarrationConfig:
        """The provider settings and the segmenter timing, in the form core.narration takes."""
        return NarrationConfig.from_dict({
            "provider": self.provider, "model": self.model, "base_url": self.base_url,
            "api_key_env": self.api_key_env, "reasoning_effort": self.reasoning_effort,
            "timeout_s": self.timeout_s, "max_tokens": self.max_tokens, "quiet_s": self.quiet_s,
            "min_hand_frames": self.min_hand_frames, "max_episode_s": 1e9, "keyframe_every_s": 1e9})


SYSTEM = """You check what an object tracker believes is on a tabletop. The image is the camera's view of the table: the camera is above it, looking straight down or, mounted high in a corner of the room, down at an angle, so the edges of the image may show the floor, furniture or people around the table. Numbered yellow boxes (marks) are drawn around the objects the tracker follows, and the text says what the tracker calls each one.

For every mark:
- real: true if the box shows a physical object on the table; false if it shows only empty table, a shadow, a reflection, a hand or arm, or a printed black-and-white marker.
- label: what the object in the box is, the way a person would say it: 1 to 4 plain words, colour first if it helps ("red mug", "phone charger"). Null if real is false. For any medicine container say only "pill bottle".
- confidence: 0 to 1; below 0.5 if you aren't sure.
- guesses: up to 3 labels the object could be, most likely first, each with p = your probability that it is that (0 to 1, together at most 1). Empty if real is false.
- not_object: your probability (0 to 1) that the box shows no object at all (table, shadow, cable, tape, part of the desk).
- held: true if a hand is touching or holding the object.

unmarked: objects on the table that no box covers (at most 8), each with a label as above, point = its centre as {"x": fraction of the image width, "y": fraction of the height}, each 0 to 1, and a confidence. Ignore the table itself, anything off the table, cables, hands and the printed square markers.

Reply with the JSON object only."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["marks", "unmarked"],
    "properties": {
        "marks": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["mark", "real", "label", "confidence", "guesses", "not_object", "held"],
            "properties": {"mark": {"type": "integer"}, "real": {"type": "boolean"},
                           "label": {"anyOf": [{"type": "null"}, {"type": "string"}]},
                           "confidence": {"type": "number"},
                           "guesses": {"type": "array", "maxItems": 3, "items": {
                               "type": "object", "additionalProperties": False, "required": ["label", "p"],
                               "properties": {"label": {"type": "string"}, "p": {"type": "number"}}}},
                           "not_object": {"type": "number"}, "held": {"type": "boolean"}}}},
        "unmarked": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["label", "point", "confidence"],
            "properties": {"label": {"type": "string"},
                           "point": {"type": "object", "additionalProperties": False, "required": ["x", "y"],
                                     "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
                           "confidence": {"type": "number"}}}},
    },
}


def _fake_reply(job) -> str:
    """The fake provider's reply: every mark is real and is what the tracker calls it."""
    text = " ".join(v for k, v in job.parts if k == "text")
    marks = [{"mark": int(i), "real": True, "label": None if n.strip() == "unnamed object" else n.strip(),
              "confidence": 0.9, "guesses": [], "not_object": 0.05, "held": False}
             for i, n in re.findall(r"(\d+) = ([^,.\n]+)", text)]
    return json.dumps({"marks": marks, "unmarked": []})


NOT_OBJECT = "not an object"                              # the belief's clutter label


def _p(v) -> Optional[float]:
    """A probability from the reply, clipped to 0-1, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        return None
    return round(min(1.0, max(0.0, float(v))), 3)


def guesses(v) -> list[list]:
    """The reply's guesses as [[label, p], ...]: safe labels only, at most 3, p rescaled if they sum over 1."""
    out = []
    for g in (v if isinstance(v, list) else [])[:3]:
        if not isinstance(g, dict):
            continue
        label, p = safe_label(g.get("label")), _p(g.get("p"))
        if label and p and all(not same_thing(label, o) for o, _ in out):
            out.append([label, p])
    total = sum(p for _, p in out)
    return [[lab, round(p / total, 3)] for lab, p in out] if total > 1 else out


def safe_label(v) -> Optional[str]:
    """Grok's name for a mark or find, as something safe to store and say: 1-4 plain lower-case words,
    never a medication claim, or None. Own copy (voice/visual.py's pick has one only on some branches)."""
    t = re.sub(r"[^a-z' -]", "", str(v or "").lower()).strip()
    t = re.sub(r"^(?:a|an|the|my|your)\s+", "", t)
    if not t or len(t.split()) > 4 or len(t) > 30 or med_claim(t) or t in ("object", "thing", "item", "unknown"):
        return None
    return t


def same_thing(a: Optional[str], b: Optional[str]) -> bool:
    """Loose name match on the head noun (the last word): 'car keys' ~ 'keys', 'smartphone' ~ 'phone',
    'eyeglasses' ~ 'glasses'."""
    sa = [stem(w) for w in re.findall(r"[a-z]+", (a or "").lower())]
    sb = [stem(w) for w in re.findall(r"[a-z]+", (b or "").lower())]
    if not sa or not sb:
        return False
    ha, hb = sa[-1], sb[-1]
    return ha == hb or (min(len(ha), len(hb)) >= 3 and (ha.endswith(hb) or hb.endswith(ha)))


# ---------------------------------------------------------------- storage

TABLE = """
CREATE TABLE IF NOT EXISTS grok_checks (id INTEGER PRIMARY KEY, t REAL, wall REAL, episode TEXT, mark INTEGER,
  entity TEXT, world_label TEXT, grok_label TEXT, verdict TEXT, x_cm REAL, y_cm REAL, confidence REAL,
  latency_ms INTEGER, model TEXT, guesses TEXT, not_object REAL, held INTEGER);
CREATE INDEX IF NOT EXISTS grok_checks_wall ON grok_checks(wall);
"""
ADDED = (("guesses", "TEXT"), ("not_object", "REAL"), ("held", "INTEGER"))    # to tables made before them
COLS = ("t", "wall", "episode", "mark", "entity", "world_label", "grok_label", "verdict", "x_cm", "y_cm",
        "confidence", "latency_ms", "model", "guesses", "not_object", "held")


class CheckStore:
    """The grok_checks table in the EventLog's SQLite file, through the EventLog's connection layer
    (core/narration_store.py explains why)."""

    def __init__(self, events):
        self.events = events
        with events._locked():
            c = events._conn()
            c.executescript(TABLE)
            have = {r[1] for r in c.execute("PRAGMA table_info(grok_checks)")}
            for col, kind in ADDED:
                if col not in have:
                    c.execute(f"ALTER TABLE grok_checks ADD COLUMN {col} {kind}")
            c.commit()

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.events._locked():
            c = self.events._conn()
            c.executemany(f"INSERT INTO grok_checks ({', '.join(COLS)}) VALUES ({', '.join('?' * len(COLS))})",
                          [tuple(r.get(k) for k in COLS) for r in rows])
            c.commit()

    def rows(self, since: float = 0.0, verdicts: Optional[tuple] = None) -> list[dict]:
        """Rows with wall >= since, newest first."""
        sql = f"SELECT {', '.join(COLS)} FROM grok_checks WHERE wall >= ?"
        args: list = [since]
        if verdicts:
            sql += f" AND verdict IN ({', '.join('?' * len(verdicts))})"
            args += list(verdicts)
        with self.events._locked():
            out = self.events._conn().execute(sql + " ORDER BY wall DESC, id DESC", args).fetchall()
        return [dict(zip(COLS, r)) for r in out]

    def prune(self, keep_h: float, now: Optional[float] = None) -> None:
        cutoff = (time.time() if now is None else now) - keep_h * 3600
        with self.events._locked():
            c = self.events._conn()
            c.execute("DELETE FROM grok_checks WHERE wall < ?", (cutoff,))
            c.commit()


# ---------------------------------------------------------------- the checker

class GrokCheck:
    """See the module docstring. feed(frame, dets, new_events) once per perception frame, or
    attach(world) to have World.update do it. start=False runs no thread: tests call run_once()."""

    def __init__(self, cfg: dict, events, world=None, table=None, provider=None,
                 online: Optional[Callable[[], bool]] = None, clock: Callable[[], float] = time.time,
                 start: bool = True):
        self.cfg = cfg
        self.c = GrokCheckConfig.from_dict(cfg.get("grok_check"))
        nc = self.c.narration_config()
        if provider is None:
            provider = FakeProvider(_fake_reply) if nc.provider == "fake" else make_provider(nc)
        self.provider = provider
        self.seg = Segmenter(nc)
        self.store = CheckStore(events)
        self.world, self.table = world, table
        self.online = online or (lambda: True)
        self.clock = clock
        self.last: Optional[dict] = None
        self._belief: dict[str, dict] = {}       # unnamed thing -> {'a': {label: weight}, 'wall', 'n'}
        self._slot: Optional[tuple] = None       # (frame, episode id): the newest settled frame
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._calls: deque = deque()
        self._t0: Optional[float] = None
        self._settled_once = False
        self._feed_errors = 0
        self.dropped = 0                         # settled frames dropped because activity began again
        self._thread: Optional[threading.Thread] = None
        try:
            self.store.prune(self.c.keep_h, now=clock())
        except Exception:
            log.exception("grok check prune failed")
        if start:
            self.start()

    # -- perception thread

    def feed(self, frame, dets, new_events=()) -> None:
        """One call per processed frame. O(1); never blocks on disk or network; never raises."""
        try:
            hands = bool(getattr(dets, "hands", None))
            evs = [e for e in (new_events or ()) if isinstance(e, Event)]
            step = self.seg.update(frame.t, frame.wall, hands, evs)
            if self._t0 is None:
                self._t0 = frame.t
            settled = None
            if step.ended is not None and self.seg.ep is None:
                settled = step.ended.id
            elif not self._settled_once and self.seg.ep is None and frame.t - self._t0 >= self.c.quiet_s - 1e-9:
                settled = "start"
            if settled is not None and getattr(frame, "img", None) is not None:
                self._settled_once = True
                # The capture thread allocates a new array per frame, so a reference is safe to hand off.
                with self._lock:
                    self._slot = (frame, settled)
                self._wake.set()
        except Exception:
            self._feed_errors += 1
            if self._feed_errors in (1, 10, 100) or self._feed_errors % 1000 == 0:
                log.exception("grok check feed failed (%d so far)", self._feed_errors)

    def attach(self, world) -> "GrokCheck":
        """Hook into a World (as core.narration.Narrator.attach does): each update() feeds this checker
        and state_json() gains a 'grok_check' entry. Instance attributes, so the class is untouched."""
        update, state = world.update, world.state_json
        self.world = world

        def fed_update(dets, frame):
            out = update(dets, frame)
            if frame is not None:
                self.feed(frame, dets, out)
            return out

        def state_json(*a, **kw):
            st = state(*a, **kw)
            try:
                for e in st.get("entities") or []:
                    top = self.belief(e.get("name") or "")[:3]
                    if top and not e.get("label"):
                        e["belief"] = top
                st["grok_check"] = self.status()
            except Exception:
                log.exception("grok check status failed")
            return st

        world.update, world.state_json = fed_update, state_json
        return self

    # -- the worker

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="grok-check", daemon=True)
        self._thread.start()

    def _prune_calls(self, now: float) -> None:
        while self._calls and self._calls[0] <= now - 3600:
            self._calls.popleft()

    def _wait_s(self, now: float) -> float:
        """0 when a call may go now; otherwise how long to wait (offline, the hourly cap, min_gap_s)."""
        if not self.online():
            return 2.0
        self._prune_calls(now)
        if len(self._calls) >= self.c.max_per_hour:
            return max(1.0, self._calls[0] + 3600 - now)
        if self._calls:
            return max(0.0, self._calls[-1] + self.c.min_gap_s - now)
        return 0.0

    def run_once(self) -> Optional[dict]:
        """Check the newest settled frame if a call may go now. Returns the check's summary, or None
        (nothing waiting, not allowed yet, or activity began again since)."""
        if self._slot is None or self._wait_s(self.clock()) > 0:
            return None
        with self._lock:
            item, self._slot = self._slot, None
        if item is None:
            return None
        if self.seg.ep is not None:
            self.dropped += 1
            return None
        frame, episode = item
        return self.check(frame.img, frame.wall, episode=episode, t=frame.t)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._slot is None:
                self._wake.wait(2.0)
                self._wake.clear()
                continue
            wait = self._wait_s(self.clock())
            if wait > 0:
                self._stop.wait(min(wait, 2.0))
                continue
            try:
                self.run_once()
            except Exception:
                log.exception("grok check failed")

    # -- one check

    def _names(self) -> dict:
        from voice.visual import spoken_names
        return spoken_names(self.world, self.cfg) if self.world is not None else {}

    def _marks(self) -> list[tuple[str, tuple]]:
        from voice.visual import tracked_marks
        return tracked_marks(self.world, self.table) if self.world is not None else []

    def _cm(self, px: tuple) -> tuple:
        from voice.visual import px_to_cm
        try:
            c = px_to_cm(self.table, [px])
        except Exception:
            c = None
        return (None, None) if c is None else (round(float(c[0][0]), 1), round(float(c[0][1]), 1))

    def check(self, img, wall: float, episode: str = "", t: Optional[float] = None,
              marks: Optional[list] = None, names: Optional[dict] = None) -> dict:
        """One Grok call on img with the tracker's marks (default: the attached world's VISIBLE entities)
        -> rows stored, unnamed things named, and the summary (also self.last). Never raises on a
        provider or reply failure: the summary carries the error."""
        from voice.visual import _conf, _jpeg, draw_marks, point_to_px
        marks = self._marks() if marks is None else marks
        names = self._names() if names is None else names
        h, w = img.shape[:2]
        jpeg, _ = _jpeg(draw_marks(img, [b for _, b in marks]) if marks else img, self.c.look_px)
        listed = ", ".join(f"{i} = {names.get(n) or 'unnamed object'}" for i, (n, _) in enumerate(marks, 1))
        parts = [("image", jpeg), ("text", f"Marks: {listed or 'none'}.\n"
                                           "Check every mark and list anything on the table that no mark covers.")]
        self._calls.append(self.clock())
        try:
            reply = self.provider.narrate(SYSTEM, parts, SCHEMA)
            d = _parse_json(reply.text)
        except (ProviderError, NarrationError) as ex:
            log.warning("grok check failed: %s", ex)
            self.last = {"wall": wall, "error": str(ex)[:120]}
            return self.last
        model = getattr(self.provider, "model", "")
        base = {"t": t, "wall": wall, "episode": episode, "latency_ms": reply.latency_ms, "model": model}
        rows, seen = [], set()
        for m in d.get("marks") or []:
            i = m.get("mark") if isinstance(m, dict) else None
            if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= len(marks) or i in seen:
                continue
            seen.add(i)
            ent, box = marks[i - 1]
            conf, label, real, said = _conf(m), safe_label(m.get("label")), m.get("real"), names.get(ent)
            if conf < self.c.min_conf or not isinstance(real, bool) or (real and label is None):
                verdict = "unsure"
            elif not real:
                verdict = "phantom"
            elif said is None:
                verdict = "named"
            elif same_thing(said, label) or self._finds(label) == ent:
                verdict = "agree"
            else:
                verdict = "relabel"
            x, y = self._cm(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            held = m.get("held")
            rows.append({**base, "mark": i, "entity": ent, "world_label": said, "grok_label": label,
                         "verdict": verdict, "x_cm": x, "y_cm": y, "confidence": round(conf, 3),
                         "guesses": json.dumps(guesses(m.get("guesses"))), "not_object": _p(m.get("not_object")),
                         "held": int(held) if isinstance(held, bool) else None})
        for u in (d.get("unmarked") or [])[:MAX_UNMARKED]:
            if not isinstance(u, dict):
                continue
            label, px = safe_label(u.get("label")), point_to_px(u.get("point"), (w, h))
            if label is None or px is None:
                continue
            if any(b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3] for _, b in marks):
                continue                             # it named a marked object again
            x, y = self._cm(px)
            if x is not None and not self._on_table(x, y):
                continue
            conf = _conf(u)
            rows.append({**base, "mark": None, "entity": None, "world_label": None, "grok_label": label,
                         "verdict": "unmarked" if conf >= self.c.min_conf else "unsure",
                         "x_cm": x, "y_cm": y, "confidence": round(conf, 3)})
        try:
            self.store.add(rows)
        except Exception:
            log.exception("grok check rows not stored")
        bound = self._bind(rows)
        named, retired = self._believe(rows, wall, t)
        self.last = self._summary(rows, wall, reply.latency_ms, bound + named, len(marks))
        self.last["retired"] = retired
        self.last["usage"] = reply.usage
        log.info("grok check (%d ms): %s", reply.latency_ms, self.last["text"])
        return self.last

    def _finds(self, label: Optional[str]) -> Optional[str]:
        """The entity the world knows by this name (configured synonyms, taught aliases), or None."""
        if not label or self.world is None or not hasattr(self.world, "find"):
            return None
        try:
            return self.world.find(label)
        except Exception:
            return None

    def _on_table(self, x: float, y: float) -> bool:
        tw, th = ((self.cfg.get("table") or {}).get("size_cm") or [90, 60])
        return -5 <= x <= tw + 5 and -5 <= y <= th + 5

    def _bind(self, rows: list[dict]) -> list[str]:
        """Unnamed things Grok named with confidence take the label, if nothing else answers to it."""
        if not self.c.bind_names or self.world is None or not hasattr(self.world, "bind_alias"):
            return []
        out = []
        marked = {r["entity"] for r in rows if r["entity"]}
        for r in rows:
            ent = r["entity"]
            if r["verdict"] != "named" or not (ent or "").startswith("thing:") or r["confidence"] < self.c.bind_conf:
                continue
            owner = self._finds(r["grok_label"])
            if owner is not None:
                if self._one_of_kind(r, rows) and owner not in marked:
                    kept = self._merge(owner, ent)
                    if kept:
                        out.append(f"{ent} = {r['grok_label']} ({kept})")
                continue
            if self._name(ent, r["grok_label"]):
                log.info("grok check: %s is now '%s' (%.2f)", ent, r["grok_label"], r["confidence"])
                out.append(f"{ent} = {r['grok_label']}")
        return out

    def _name(self, ent: str, label: str) -> bool:
        """bind_alias, marked as Grok's (worlds without the by= argument take it plain)."""
        try:
            try:
                return bool(self.world.bind_alias(ent, label, by="grok"))
            except TypeError:
                return bool(self.world.bind_alias(ent, label))
        except Exception:
            log.exception("bind_alias failed")
            return False

    # -- belief: what an unnamed thing is, over several checks

    def _believe(self, rows: list[dict], wall: float, t: Optional[float]) -> tuple[list[str], list[str]]:
        """Add this check's guesses for each unnamed thing to its belief (weights fade with half_life_s),
        then, after min_obs checks, name it (top label >= name_p, name_margin ahead of the next) or retire
        it as clutter ('not an object' >= retire_p). Returns (named, retired)."""
        if not self.c.belief_enabled:
            return [], []
        named, retired = [], []
        for r in rows:
            ent = r["entity"]
            if not (ent or "").startswith("thing:") or r["world_label"] is not None:
                continue
            b = self._belief.setdefault(ent, {"a": {}, "wall": wall, "n": 0, "xy": None})
            xy = (r["x_cm"], r["y_cm"]) if r["x_cm"] is not None else None
            if xy and b["xy"] and math.dist(xy, b["xy"]) > MOVED_CM:
                b.update(a={}, n=0)             # carried, or the tracker moved its name to another object
            fade = 0.5 ** (max(0.0, wall - b["wall"]) / self.c.half_life_s)
            a = {k: v * fade for k, v in b["a"].items()}
            got = json.loads(r["guesses"] or "[]")
            if not got and r["grok_label"]:
                got = [[r["grok_label"], r["confidence"]]]
            for label, p in got:
                key = next((k for k in a if k != NOT_OBJECT and same_thing(k, label)), label)
                a[key] = a.get(key, 0.0) + p
            no = r["not_object"] or 0.0
            if r["verdict"] == "phantom":
                no = max(no, r["confidence"])
            if no > 0:
                a[NOT_OBJECT] = a.get(NOT_OBJECT, 0.0) + no
            b.update(a=a, wall=wall, n=b["n"] + 1, xy=xy or b["xy"])
            top = self.belief(ent)
            if b["n"] < self.c.min_obs or not top:
                continue
            (label, p), p2 = top[0], (top[1][1] if len(top) > 1 else 0.0)
            if label == NOT_OBJECT:
                if p >= self.c.retire_p and t is not None and self._retire(ent, t):
                    retired.append(ent)
                    self._belief.pop(ent, None)
            elif p >= self.c.name_p and p - p2 >= self.c.name_margin and self._finds(label) is None \
                    and self._name(ent, label):
                log.info("grok check: %s is now '%s' (belief %.2f over %d checks)", ent, label, p, b["n"])
                named.append(f"{ent} = {label}")
                self._belief.pop(ent, None)
        return named, retired

    def belief(self, ent: str) -> list[list]:
        """[[label, share], ...] for an unnamed thing, largest first ('not an object' is NOT_OBJECT)."""
        a = (self._belief.get(ent) or {}).get("a") or {}
        total = sum(a.values())
        if total <= 0:
            return []
        return sorted(([k, round(v / total, 3)] for k, v in a.items()), key=lambda kv: -kv[1])

    def _retire(self, ent: str, t: float) -> bool:
        if self.world is None or not hasattr(self.world, "retire_thing"):
            return False
        try:
            ok = bool(self.world.retire_thing(ent, t, veto_s=self.c.veto_s))
        except Exception:
            log.exception("retire_thing failed")
            return False
        if ok:
            log.info("grok check: %s retired (not an object)", ent)
        return ok

    def _one_of_kind(self, r: dict, rows: list[dict]) -> bool:
        """Grok saw only one of this kind in the frame: no other real mark with a label like it."""
        return self.c.merge_same and not any(
            o is not r and o["entity"] and o["verdict"] in ("agree", "relabel", "named")
            and same_thing(o["grok_label"] or "", r["grok_label"]) for o in rows)

    def _merge(self, owner: str, ent: str) -> Optional[str]:
        if not hasattr(self.world, "merge_same_kind"):
            return None
        try:
            kept = self.world.merge_same_kind(owner, ent)
        except Exception:
            log.exception("merge_same_kind failed")
            return None
        if kept:
            log.info("grok check: %s is %s again (one '%s')", ent, kept, owner)
        return kept

    @staticmethod
    def _summary(rows: list[dict], wall: float, latency_ms: int, bound: list[str], n_marks: int) -> dict:
        by = {v: [r for r in rows if r["verdict"] == v] for v in ("agree", "relabel", "phantom", "named",
                                                                   "unsure", "unmarked")}
        s = {"wall": wall, "latency_ms": latency_ms, "marks": n_marks, "agree": len(by["agree"]),
             "unsure": len(by["unsure"]),
             "phantom": [r["world_label"] or r["entity"] for r in by["phantom"]],
             "relabel": [f"{r['world_label']} -> {r['grok_label']}" for r in by["relabel"]],
             "named": [f"{r['entity']} = {r['grok_label']}" for r in by["named"]],
             "unmarked": [r["grok_label"] for r in by["unmarked"]], "bound": bound}
        bits = [f"{s['agree']}/{n_marks} agree"]
        bits += [f"{k}: {', '.join(s[k])}" for k in ("phantom", "relabel", "named", "unmarked") if s[k]]
        s["text"] = "; ".join(bits)
        return s

    # -- answers

    def lookup(self, name: str, now: Optional[float] = None) -> Optional[dict]:
        """The newest confident sighting of `name` with a table position, no older than
        sighting_max_age_s: {grok_label, x_cm, y_cm, wall, verdict, entity, ...}, or None."""
        now = self.clock() if now is None else now
        try:
            rows = self.store.rows(now - self.c.sighting_max_age_s, SIGHTED)
        except Exception:
            return None
        for r in rows:
            if r["x_cm"] is not None and same_thing(name, r["grok_label"]):
                return r
        return None

    # -- lifecycle and status

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def status(self) -> dict:
        self._prune_calls(self.clock())
        return {"enabled": True, "provider": self.provider.name, "model": self.provider.model,
                "calls_last_hour": len(self._calls), "last": self.last,
                "disclosure": (f"Grok check is on: each time the table settles, one still frame of the table "
                               f"(objects, sometimes hands) is sent to {self.provider.name} "
                               f"({self.provider.model}) to check what the tracker sees. No frame is kept.")}


def from_config(cfg: dict, events, world=None, table=None, online: Optional[Callable[[], bool]] = None,
                start: bool = True) -> Optional[GrokCheck]:
    """The app's GrokCheck, attached to the world, or None when cfg grok_check.enabled is false (the
    default)."""
    if not GrokCheckConfig.from_dict(cfg.get("grok_check")).enabled:
        return None
    g = GrokCheck(cfg, events, table=table, online=online, start=start)
    if world is not None:
        g.attach(world)
    log.info("grok check on: %s", g.status()["disclosure"])
    return g


# ---------------------------------------------------------------- eval (real provider calls)

def _eval(argv=None) -> int:
    """python -m core.grok_check --eval DIR [--n 20]

    Runs the check on saved rig frames (DIR/*.jpg, *.png) with the configured provider (default Grok;
    needs $XAI_API_KEY; ~$0.003 per frame). A sidecar DIR/<frame>.json, [{"name": "keys", "box_px":
    [x1, y1, x2, y2]}, ...], gives the tracker's boxes (e.g. what YOLO found); without one the frame is
    checked with no marks, so everything Grok lists is 'unmarked'. Prints each frame's verdicts, then
    latency p50/p90, prompt tokens per frame and the verdict counts."""
    import argparse
    import glob
    import os
    import tempfile

    import cv2
    import numpy as np

    from core.config import load_config
    from core.events import EventLog
    ap = argparse.ArgumentParser(description=_eval.__doc__)
    ap.add_argument("--eval", required=True, metavar="DIR")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config()
    c = GrokCheckConfig.from_dict(dict(cfg.get("grok_check") or {}, enabled=True))
    key_env = c.api_key_env or "XAI_API_KEY"
    if c.provider != "fake" and not os.environ.get(key_env):
        print(f"${key_env} is not set: the real {c.provider} call is pending a key.")
        return 2
    paths = sorted(p for p in glob.glob(os.path.join(a.eval, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        print(f"no frames in {a.eval}")
        return 2
    events = EventLog(":memory:", tempfile.mkdtemp(prefix="grok_check_eval_"))
    tw, th = ((cfg.get("table") or {}).get("size_cm") or [90, 60])
    counts: dict = {}
    lat, toks = [], []
    for p in paths[: a.n]:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: unreadable")
            continue
        h, w = img.shape[:2]

        class Table:                                  # frame == table: positions are only indicative
            ok = True

            def px_to_cm(self, pts):
                return np.asarray(pts, float).reshape(-1, 2) * [tw / w, th / h]

        side = os.path.splitext(p)[0] + ".json"
        boxes = json.load(open(side)) if os.path.exists(side) else []
        marks = [(f"m{i}:{b['name']}", tuple(b["box_px"])) for i, b in enumerate(boxes, 1)]
        names = {n: n.split(":", 1)[1] for n, _ in marks}
        g = GrokCheck({**cfg, "grok_check": dict(cfg.get("grok_check") or {}, enabled=True)}, events,
                      table=Table(), start=False)
        s = g.check(img, time.time(), episode=os.path.basename(p), marks=marks, names=names)
        if "error" in s:
            print(f"{os.path.basename(p)}: FAILED {s['error']}")
            continue
        lat.append(s["latency_ms"])
        u = s.get("usage") or {}
        if u.get("input_tokens"):
            toks.append(u["input_tokens"])
        for r in g.store.rows(0.0):
            if r["episode"] == os.path.basename(p):
                counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(f"{os.path.basename(p)}: {s['latency_ms']} ms; {s['text']}")
    if lat:
        q = np.percentile(lat, [50, 90])
        print(f"\n{len(lat)} frames: latency p50 {q[0]:.0f} ms, p90 {q[1]:.0f} ms"
              + (f"; prompt tokens/frame {np.mean(toks):.0f}" if toks else "") + f"; verdicts {counts}")
    events.close()
    return 0 if lat else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_eval(sys.argv[1:]))
