"""Narration memory storage and retrieval (the text side of core/narration.py).

A `narrations` table in the EventLog's own SQLite database, so one file holds what happened (events)
and what it looked like the person was doing (narrations). It goes through the EventLog's connection
layer (one connection per thread under WAL, or the shared ':memory:' connection), which is why this
module uses EventLog's private _conn/_locked: a second connection pool would break ':memory:' logs.

A row is a queue entry before it is a memory: an episode is written 'pending' (with its keyframe list
and events as JSON) the moment it ends, so a restart or a network outage never loses it; the narrator
turns it into 'done' (summary + validated JSON), 'failed' (the provider kept returning unusable output)
or 'dropped' (queue overflow, or its frames aged out before it could be sent).

Also here, because answers need them without importing the image code: the medication rule for any
narration text (redact_meds) and spoken time windows ('before lunch', 'at 3') for retrieval.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS narrations (id INTEGER PRIMARY KEY, t_start REAL, t_end REAL, summary TEXT,
  json TEXT, frames_dir TEXT, provider TEXT, model TEXT, latency_ms INTEGER, status TEXT,
  attempts INTEGER DEFAULT 0, next_try REAL DEFAULT 0, called REAL, error TEXT, episode TEXT, created REAL);
CREATE INDEX IF NOT EXISTS narrations_t ON narrations(t_start, t_end);
CREATE INDEX IF NOT EXISTS narrations_status ON narrations(status, next_try);
"""
_COLS = ("id, t_start, t_end, summary, json, frames_dir, provider, model, latency_ms, status, attempts, "
         "next_try, called, error, episode")


@dataclass
class Row:
    id: int
    t_start: float
    t_end: float
    summary: Optional[str]
    data: dict
    frames_dir: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    latency_ms: Optional[int]
    status: str
    attempts: int
    next_try: float
    called: Optional[float]
    error: Optional[str]
    episode: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        try:
            return float(self.data.get("confidence", 0.5))
        except (TypeError, ValueError):
            return 0.5


def _loads(s) -> dict:
    try:
        v = json.loads(s) if s else {}
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def _row(r) -> Row:
    (i, t0, t1, summary, js, fdir, prov, model, lat, status, att, nxt, called, err, ep) = r
    return Row(i, t0, t1, summary, _loads(js), fdir, prov, model, lat, status, att or 0, nxt or 0.0,
               called, err, _loads(ep))


# ---------------------------------------------------------------- the medication rule

# Mirrors voice/llm.py's _PILLS_TAKEN, widened for narration: a VLM describing hands near a pill bottle
# is exactly where 'you took your pills' would slip in. Any sentence naming medication together with a
# take / miss / skip verb goes, and ingestion words go wherever they appear: an overhead tabletop camera
# never sees anyone swallow anything.
MEDS = r"\b(?:pills?|medications?|medicines?|meds|doses?|dosage|tablets?|capsules?|vitamins?|prescriptions?)\b"
CLAIM = (r"\b(?:took|taken|take|takes|taking|had|miss(?:ed|es|ing)?|skip(?:ped|s|ping)?|forg[eo]t\w*"
         r"|popp(?:ed|ing)|consum\w*|ingest\w*|ate|eat\w*|drank|drink\w*|dosed|dosing)\b")
ALWAYS = r"\b(?:swallow\w*|ingest\w*|dose|doses|dosage|dosed|mouth)\b"
_MEDS, _CLAIM, _ALWAYS = (re.compile(p, re.I) for p in (MEDS, CLAIM, ALWAYS))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]


def med_claim(text: str) -> bool:
    """True if this text states or implies medication was taken, swallowed or missed."""
    t = text or ""
    return bool(_ALWAYS.search(t) or (_MEDS.search(t) and _CLAIM.search(t)))


def redact_meds(text: str) -> tuple[str, int]:
    """(text without the sentences that break the medication rule, how many were dropped)."""
    keep, n = [], 0
    for s in _sentences(text):
        if med_claim(s):
            n += 1
        else:
            keep.append(s)
    return " ".join(keep), n


# ---------------------------------------------------------------- spoken time windows

@dataclass
class Window:
    t0: float
    t1: float
    label: str          # how to say it: 'this morning', 'around 3:00 PM'


_MEALS = {"breakfast": 8, "lunch": 12, "dinner": 18, "noon": 12}
_PARTS = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 24), "night": (19, 24)}
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "few": 3, "couple": 2}
_N = r"(\d+|an?|one|two|three|four|five|ten|fifteen|twenty|thirty|forty|fifty|few|couple(?: of)?)"
_UNIT = r"(minutes?|mins?|hours?|hrs?)"

_RX_PART = re.compile(r"\b(this|yesterday|last|my) (morning|afternoon|evening|night)\b|\btonight\b")
_RX_MEAL = re.compile(r"\b(before|after|during|at|around) (breakfast|lunch|dinner|noon)\b")
_RX_CLOCK = re.compile(r"\b(?:at|around|about|by) (\d{1,2})(?: (\d\d))?(?: ?(a|p) ?m\b)?(?! (?:minutes?|hours?|mins?))")
_RX_AGO = re.compile(rf"\b{_N} {_UNIT} ago\b")
_RX_LAST = re.compile(rf"\b(?:in the |over the |during the )?(?:last|past) (?:{_N} )?{_UNIT}\b")
_RX_DAY = re.compile(r"\b(today|yesterday|earlier|recently|lately)\b")


def _norm(text: str) -> str:
    t = re.sub(r"['’`]", "", (text or "").lower())
    t = re.sub(r"\b([ap])\.? ?m\b\.?", r"\1m", t)          # 'p.m.' / 'p m' -> 'pm'
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", t).split())


def _num(s: Optional[str]) -> int:
    if not s:
        return 1
    s = s.replace(" of", "")
    return int(s) if s.isdigit() else _NUM.get(s, 1)


def _clock(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"{h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def parse_window(text: str, now: Optional[float] = None) -> Optional[Window]:
    """The time span a question points at, or None. Local time; hours without am/pm mean the most
    recent past one ('at 9' asked at 3:30 PM is 9 AM today; 'at 4' is 4 AM today)."""
    now = time.time() if now is None else now
    t = _norm(text)
    nd = datetime.fromtimestamp(now)
    day0 = nd.replace(hour=0, minute=0, second=0, microsecond=0)

    def span(a: datetime, b: datetime, label: str) -> Window:
        return Window(a.timestamp(), min(b.timestamp(), now) if a.timestamp() <= now else b.timestamp(), label)

    m = _RX_MEAL.search(t)
    if m:
        rel, meal = m.groups()
        h = _MEALS[meal]
        if rel == "before":
            return span(day0, day0 + timedelta(hours=h), f"before {meal}")
        if rel == "after":
            return span(day0 + timedelta(hours=h + 1), day0 + timedelta(hours=h + 5), f"after {meal}")
        return span(day0 + timedelta(hours=h - 0.5), day0 + timedelta(hours=h + 1.5), f"at {meal}")
    m = _RX_PART.search(t)
    if m:
        which, part = m.groups() if m.group(1) else ("this", "night")
        a, b = _PARTS[part]
        which = "this" if which == "my" else which
        base = day0 - timedelta(days=1) if which in ("yesterday", "last") else day0
        label = "tonight" if m.group(0) == "tonight" else f"{which} {part}"
        return span(base + timedelta(hours=a), base + timedelta(hours=b), label)
    m = _RX_AGO.search(t)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        secs = n * (3600 if unit.startswith("h") else 60)
        half = max(300.0, secs / 4)
        label = (("an hour" if n == 1 else f"{n} hours") if unit.startswith("h")
                 else ("a minute" if n == 1 else f"{n} minutes")) + " ago"
        return Window(now - secs - half, now - secs + half, label)
    m = _RX_LAST.search(t)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        secs = n * (3600 if unit.startswith("h") else 60)
        label = ("the last hour" if n == 1 else f"the last {n} hours") if unit.startswith("h") \
            else f"the last {n} minutes"
        return Window(now - secs, now, f"in {label}")
    m = _RX_CLOCK.search(t)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if 1 <= h <= 12 and mi < 60 or (h <= 23 and mi < 60 and ap is None):
            hours = [h % 12 + (12 if ap == "p" else 0)] if ap else ([h] if h > 12 else [h % 12, h % 12 + 12])
            cands = [day0 + timedelta(days=d, hours=hh, minutes=mi) for d in (0, -1) for hh in hours]
            past = [c for c in cands if c.timestamp() <= now + 900]
            c = max(past) if past else min(cands)
            label = f"around {_clock(c)}" + (" yesterday" if c < day0 else "")
            return Window(c.timestamp() - 1800, min(c.timestamp() + 1800, now) if c.timestamp() <= now
                          else c.timestamp() + 1800, label)
    m = _RX_DAY.search(t)
    if m:
        w = m.group(1)
        if w == "today":
            return Window(day0.timestamp(), now, "today")
        if w == "yesterday":
            return Window((day0 - timedelta(days=1)).timestamp(), day0.timestamp(), "yesterday")
        return Window(now - 3 * 3600, now, "earlier")
    return None


def has_time_phrase(text: str) -> bool:
    return parse_window(text, time.time()) is not None


# ---------------------------------------------------------------- search terms

def stem(word: str) -> str:
    """Crude suffix stripping, enough for 'chargers' ~ 'charger', 'tidying' ~ 'tidy', 'glasses' ~ 'glass'."""
    w = word.lower().strip()
    if len(w) <= 3:
        return w
    if re.search(r"i(?:es|ed)$", w) and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(?:x|ch|sh|ss|z)es$", w):
        return w[:-2]
    if w.endswith("ing") and len(w) >= 6:
        return w[:-3]
    if w.endswith("ed") and len(w) >= 5:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _pattern(terms: list[str]) -> Optional[re.Pattern]:
    stems = [" ".join(stem(w) for w in str(t).split()) for t in terms if str(t).strip()]
    stems = [s for s in dict.fromkeys(stems) if s]
    if not stems:
        return None
    alts = "|".join(r"\s+".join(re.escape(w) + r"\w*" for w in s.split()) for s in stems)
    return re.compile(rf"\b(?:{alts})", re.I)


# ---------------------------------------------------------------- the store

class NarrationStore:
    """The narrations table. Thread-safe the way EventLog is: each thread uses its own connection."""

    def __init__(self, events):
        self.events = events
        with events._locked():
            events._conn().executescript(SCHEMA)

    def _exec(self, sql: str, args=()) -> int:
        with self.events._locked():
            c = self.events._conn()
            cur = c.execute(sql, args)
            c.commit()
            return cur.lastrowid

    def _rows(self, where: str, args=(), order: str = "t_start", limit: Optional[int] = None) -> list[Row]:
        sql = f"SELECT {_COLS} FROM narrations WHERE {where} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.events._locked():
            return [_row(r) for r in self.events._conn().execute(sql, args).fetchall()]

    # -- the queue

    def add_pending(self, t_start: float, t_end: float, frames_dir: Optional[str], episode: dict,
                    now: Optional[float] = None) -> int:
        return self._exec("INSERT INTO narrations (t_start, t_end, frames_dir, status, attempts, next_try, "
                          "episode, created) VALUES (?,?,?,?,?,?,?,?)",
                          (t_start, t_end, frames_dir, "pending", 0, 0.0, json.dumps(episode),
                           time.time() if now is None else now))

    def next_due(self, now: float) -> Optional[Row]:
        rows = self._rows("status = 'pending' AND next_try <= ?", (now,), order="t_start, id", limit=1)
        return rows[0] if rows else None

    def next_try_after(self) -> Optional[float]:
        with self.events._locked():
            r = self.events._conn().execute(
                "SELECT MIN(next_try) FROM narrations WHERE status = 'pending'").fetchone()
        return r[0] if r else None

    def pending_count(self) -> int:
        return self.counts().get("pending", 0)

    def counts(self) -> dict:
        with self.events._locked():
            rows = self.events._conn().execute(
                "SELECT status, COUNT(*) FROM narrations GROUP BY status").fetchall()
        return {s: n for s, n in rows}

    def calls_since(self, wall: float) -> list[float]:
        with self.events._locked():
            rows = self.events._conn().execute(
                "SELECT called FROM narrations WHERE called >= ? ORDER BY called", (wall,)).fetchall()
        return [r[0] for r in rows]

    def mark_called(self, id_: int, t: float) -> None:
        self._exec("UPDATE narrations SET called = ? WHERE id = ?", (t, id_))

    def mark_done(self, id_: int, summary: str, data: dict, provider: str, model: str,
                  latency_ms: Optional[int]) -> None:
        """Stores the narration. The medication rule is enforced here too: whatever a provider or a
        future caller hands in, the stored text never says medication was taken."""
        summary, _ = redact_meds(summary)
        data = dict(data)
        if "summary" in data:
            data["summary"] = summary
        self._exec("UPDATE narrations SET status = 'done', summary = ?, json = ?, provider = ?, model = ?, "
                   "latency_ms = ?, error = NULL WHERE id = ?",
                   (summary, json.dumps(data), provider, model, latency_ms, id_))

    def mark_retry(self, id_: int, next_try: float, error: str) -> None:
        self._exec("UPDATE narrations SET attempts = attempts + 1, next_try = ?, error = ? WHERE id = ?",
                   (next_try, error[:500], id_))

    def mark_failed(self, id_: int, error: str) -> None:
        self._exec("UPDATE narrations SET status = 'failed', attempts = attempts + 1, error = ? WHERE id = ?",
                   (error[:500], id_))

    def drop_oldest_pending(self, keep: int) -> list[Row]:
        """Bounds the queue: marks all but the newest `keep` pending rows 'dropped' and returns them
        (the caller deletes their frames)."""
        rows = self._rows("status = 'pending'", order="t_start DESC, id DESC")
        old = rows[max(0, keep):]
        for r in old:
            self._exec("UPDATE narrations SET status = 'dropped', frames_dir = NULL, error = 'queue full' "
                       "WHERE id = ?", (r.id,))
        return old

    def get(self, id_: int) -> Optional[Row]:
        rows = self._rows("id = ?", (id_,))
        return rows[0] if rows else None

    # -- retrieval (done rows only)

    def between(self, t0: float, t1: float) -> list[Row]:
        """Narrations overlapping [t0, t1], oldest first."""
        return self._rows("status = 'done' AND t_end >= ? AND t_start <= ?", (t0, t1))

    def latest(self, n: int = 1, before: Optional[float] = None) -> list[Row]:
        if before is None:
            return self._rows("status = 'done'", order="t_end DESC, id DESC", limit=n)
        return self._rows("status = 'done' AND t_start <= ?", (before,), order="t_end DESC, id DESC", limit=n)

    def search(self, terms: list[str], since: Optional[float] = None, until: Optional[float] = None,
               limit: int = 5) -> list[Row]:
        """Done narrations whose summary / actions / objects / tags mention any term (stemmed, whole-word
        prefix), newest first. SQL LIKE narrows the rows; a word-boundary regex confirms them."""
        rx = _pattern(terms)
        if rx is None:
            return []
        stems = [stem(str(t).split()[0]) for t in terms if str(t).strip()]
        likes = " OR ".join("(lower(summary) LIKE ? OR lower(json) LIKE ?)" for _ in stems)
        args: list = []
        for s in stems:
            args += [f"%{s}%", f"%{s}%"]
        where = f"status = 'done' AND ({likes})"
        if since is not None:
            where += " AND t_end >= ?"
            args.append(since)
        if until is not None:
            where += " AND t_start <= ?"
            args.append(until)
        out = []
        for r in self._rows(where, tuple(args), order="t_end DESC, id DESC"):
            d = r.data
            blob = " ".join([r.summary or "", json.dumps(d.get("actions", [])),
                             " ".join(map(str, d.get("objects_involved", []))),
                             " ".join(map(str, d.get("activity_tags", [])))])
            if rx.search(blob):
                out.append(r)
                if len(out) >= limit:
                    break
        return out

    # -- retention

    def prune(self, max_age_h: float = 24, now: Optional[float] = None, root: Optional[str] = None) -> int:
        """Delete keyframe folders older than max_age_h, keeping the text (like EventLog.prune keeps
        event rows); a pending row whose frames went is dropped. Stray folders under root go too."""
        now = time.time() if now is None else now
        cutoff = now - max_age_h * 3600.0
        n = 0
        for r in self._rows("t_end < ? AND (frames_dir IS NOT NULL OR status = 'pending')", (cutoff,)):
            if r.frames_dir:
                shutil.rmtree(r.frames_dir, ignore_errors=True)
                n += 1
            if r.status == "pending":
                self._exec("UPDATE narrations SET status = 'dropped', frames_dir = NULL, "
                           "error = 'expired' WHERE id = ?", (r.id,))
            else:
                self._exec("UPDATE narrations SET frames_dir = NULL WHERE id = ?", (r.id,))
        if root and os.path.isdir(root):
            live = {r.frames_dir for r in self._rows("frames_dir IS NOT NULL")}
            for e in os.scandir(root):
                if e.is_dir() and e.path not in live and e.stat().st_mtime < cutoff:
                    shutil.rmtree(e.path, ignore_errors=True)
                    n += 1
        return n


_STORES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def store_for(events, create: bool = True) -> Optional[NarrationStore]:
    """The NarrationStore on this EventLog, or None (no log, a log without SQLite, or create=False and
    no narrations table yet: answers then behave exactly as before narration existed)."""
    if events is None or not hasattr(events, "_conn") or not hasattr(events, "_locked"):
        return None
    try:
        st = _STORES.get(events)
        if st is not None:
            return st
        if not create:
            with events._locked():
                hit = events._conn().execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'narrations'").fetchone()
            if not hit:
                return None
        st = NarrationStore(events)
        _STORES[events] = st
        return st
    except Exception:
        return None
