"""Reminders that watch the table (care layer, feature 1).

Two kinds, both said by voice and stored in the EventLog's SQLite file so they survive restarts:

  time        'remind me to take my pills at 8', 'remind me at 3 to call Sarah', 'in 10 minutes'
  condition   'remind me if I haven't picked up my pill bottle by 9' (optionally 'since 8'): at 9 it
              fires only if the event log has no PICKED_UP / MOVED / PUT_BACK for that object since the
              window start (default: the start of that day)

Either can repeat ('every day at 8'). A condition is checked against the event log, never a model, so the
answer is deterministic and testable. Firing makes a notice: a row the dashboard and phone see in /state,
plus an Answer (text + point_at) the app speaks and aims like any other answer. At most one firing per
reminder per day (a unique index, so restarts can't repeat it); notices expire unacknowledged after
care.notice_expiry_min; 'okay' / 'thanks' / 'done' within care.ack_window_s acknowledges the latest one.
In quiet hours, or past care.max_notices_per_hour, a notice is still recorded but not spoken.

Spoken times: 'at 8' is the next 8:00 (8 AM before 8 AM, 8 PM before 8 PM, else tomorrow 8 AM); 'am' /
'pm' / 'in the morning' / 'tonight' / 'tomorrow' / 'noon' / 'midnight' pin it down. Local time zone
from the system. 'When I leave the table' reminders are out of scope and say so.

Event-triggered and time-based reminders for people with dementia follow Project Memoria's idea (MIT);
the parsing, storage and firing rules here are our own, built on the event log.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, Optional

from core.carewords import (med_phrase, pill_guard, plural, say_day, say_oclock, say_time, second_person,
                            spoken, where_sentence, your)
from core.types import Answer

log = logging.getLogger(__name__)

PICKUP_TYPES = ("PICKED_UP", "MOVED", "PUT_BACK")
DEFAULTS = {"quiet_hours": ["22:00", "07:00"], "max_notices_per_hour": 4, "notice_expiry_min": 30,
            "ack_window_s": 60, "late_grace_min": 30}

SCHEMA = """
CREATE TABLE IF NOT EXISTS care_reminders (id INTEGER PRIMARY KEY, created REAL, kind TEXT, task TEXT, obj TEXT,
  due REAL, daily INTEGER, at_h INTEGER, at_m INTEGER, since_h INTEGER, since_m INTEGER, active INTEGER,
  last_day TEXT, ended TEXT, said TEXT);
CREATE TABLE IF NOT EXISTS care_notices (id INTEGER PRIMARY KEY, t REAL, kind TEXT, reminder_id INTEGER, day TEXT,
  text TEXT, point_at TEXT, action TEXT, spoken INTEGER, acknowledged INTEGER DEFAULT 0, ack_t REAL);
CREATE UNIQUE INDEX IF NOT EXISTS care_notices_once ON care_notices(reminder_id, day)
  WHERE reminder_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS care_notices_t ON care_notices(t);
CREATE TABLE IF NOT EXISTS care_state (key TEXT PRIMARY KEY, value TEXT);
"""


def care_cfg(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg or {}).get("care") or {})}


class CareDB:
    """Care tables inside the EventLog's database, through its connection layer (one connection per
    thread under WAL, or the shared ':memory:' one), the way core/narration_store.py does it: a second
    pool would see an empty ':memory:' database."""

    def __init__(self, events, schema: str):
        self.events = events
        with events._locked():
            events._conn().executescript(schema)

    def exec(self, sql: str, args=()) -> int:
        """Run a write; returns the number of rows changed."""
        with self.events._locked():
            c = self.events._conn()
            cur = c.execute(sql, args)
            c.commit()
            return cur.rowcount

    def insert(self, sql: str, args=()) -> Optional[int]:
        """Run an INSERT; returns the new row id, or None if it was ignored (INSERT OR IGNORE)."""
        with self.events._locked():
            c = self.events._conn()
            cur = c.execute(sql, args)
            c.commit()
            return cur.lastrowid if cur.rowcount else None

    def rows(self, sql: str, args=()) -> list:
        with self.events._locked():
            return self.events._conn().execute(sql, args).fetchall()

    def get_state(self, key: str) -> Optional[str]:
        r = self.rows("SELECT value FROM care_state WHERE key = ?", (key,))
        return r[0][0] if r else None

    def set_state(self, key: str, value: str) -> None:
        self.exec("INSERT OR REPLACE INTO care_state (key, value) VALUES (?, ?)", (key, value))


# ---------------------------------------------------------------- spoken times

_HOURS = {w: i for i, w in enumerate("one two three four five six seven eight nine ten eleven twelve".split(), 1)}
_MINS = {"oh five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty five": 45, "fortyfive": 45,
         "forty": 40, "fifty": 50, "twenty five": 25}
_PREP = r"(?:at|by|since|after|from|around|until|in|every day at|before)"
_REPEAT = r"(?:every ?day|daily|each day|every (?:morning|afternoon|evening|night))"
_PART = r"(?:in the (?:morning|afternoon|evening|night)|at night|tonight|this (?:morning|afternoon|evening))"
_DAYW = r"(?:today|tomorrow|tonight)"
_CLOCK = (r"(?P<clock>noon|midday|midnight|(?P<h>\d{1,2})(?:(?::|\.|\s)(?P<m>[0-5]\d))?(?:\s*(?P<ap>am|pm))?"
          r"(?:\s*oclock)?)(?![\d:.])")
ABS = re.compile(rf"(?:(?P<pre>{_REPEAT}|{_DAYW})\s+)?\b(?:at|by|around)\s+{_CLOCK}"
                 rf"(?:\s+(?P<post1>{_PART}|{_REPEAT}|{_DAYW}))?(?:\s+(?P<post2>{_PART}|{_REPEAT}|{_DAYW}))?")
REL = re.compile(r"\bin\s+(?:(?P<n>\d+|an?|a couple of|a few)\s+(?P<u>minutes?|mins?|hours?|hrs?)"
                 r"|(?P<half>half an? hour))\b")
SINCE = re.compile(rf"\b(?:since|after|from)\s+{_CLOCK}(?:\s+(?P<post1>{_PART}))?")
REPEAT_RX = re.compile(rf"\b{_REPEAT}\b")


def normalize(text: str) -> str:
    """Lowercase, 'p.m.' -> 'pm', "o'clock" -> 'oclock', spoken numbers after a time word -> digits,
    punctuation -> space (keeping '8:30' and '8.30')."""
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"o'?\s?clock", "oclock", t)
    t = re.sub(r"[`']", "", t)
    words = "|".join(_HOURS)
    t = re.sub(rf"\b({_PREP}\s+)({words})\b", lambda m: m.group(1) + str(_HOURS[m.group(2)]), t)
    t = re.sub(rf"\b({words})(?=\s+(?:oclock|am|pm|a\.\s?m|p\.\s?m|thirty|fifteen|forty|o'clock))",
               lambda m: str(_HOURS[m.group(1)]), t)
    t = re.sub(r"\b(in\s+)(" + "|".join(["ten", "fifteen", "twenty", "thirty", "forty five", "five", "two",
                                          "three", "four"]) + r")\b",
               lambda m: m.group(1) + str({**_HOURS, **_MINS}[m.group(2)]), t)
    mins = "|".join(sorted(_MINS, key=len, reverse=True))
    t = re.sub(rf"(\d)\s+({mins})\b", lambda m: f"{m.group(1)} {_MINS[m.group(2)]:02d}", t)
    t = re.sub(r"(\d)\s*([ap])\.?\s?m\b\.?", r"\1 \2m", t)
    t = re.sub(r"(?<!\d)[.:](?!\d)|[^a-z0-9:.\s]", " ", t)
    t = re.sub(r"(?<=\D)\.|\.(?=\D)|\.$", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class When:
    due: float                  # epoch of the first occurrence
    daily: bool
    hm: tuple[int, int]         # local hour, minute of the occurrence
    relative: bool = False


def _part(*phrases: Optional[str]) -> Optional[str]:
    for p in phrases:
        if not p:
            continue
        if "morning" in p:
            return "am"
        if any(w in p for w in ("afternoon", "evening", "night", "tonight")):
            return "pm"
    return None


def _candidates(m: re.Match, part: Optional[str]) -> Optional[list[tuple[int, int]]]:
    """Every (hour, minute) the spoken clock could mean, or None if it isn't a time."""
    c = m.group("clock")
    if c in ("noon", "midday"):
        return [(12, 0)]
    if c == "midnight":
        return [(0, 0)]
    h, mi, ap = int(m.group("h")), int(m.group("m") or 0), m.group("ap")
    if h > 23 or mi > 59:
        return None
    if ap:
        if h == 0 or h > 12:
            return None
        return [(h % 12 + (12 if ap == "pm" else 0), mi)]
    if h > 12 or h == 0:
        return [(h, mi)]
    if part == "am":
        return [(h % 12, mi)]
    if part == "pm":
        return [(h % 12 + 12, mi)]
    return [(h % 12, mi), (h % 12 + 12, mi)]


def _at(d: date, h: int, m: int) -> float:
    return datetime.combine(d, dtime(h, m)).timestamp()


def _next(cands: list[tuple[int, int]], now: float, only: Optional[date] = None) -> Optional[tuple[float, tuple]]:
    today = datetime.fromtimestamp(now).date()
    days = [only] if only else [today + timedelta(days=k) for k in range(3)]
    best = None
    for d in days:
        for h, m in cands:
            t = _at(d, h, m)
            if t > now and (best is None or t < best[0]):
                best = (t, (h, m))
    return best


def daily_when(h: int, m: int, now: float) -> When:
    """Every day at h:m, starting with the next occurrence."""
    t, hm = _next([(h, m)], now)
    return When(t, True, hm)


def find_when(t: str, now: float, pos: int = 0) -> Optional[tuple[When, tuple[int, int]]]:
    """(When, (start, end) of the time words in t) for normalized text t, looking from pos on; 'every day'
    anywhere in t makes it daily."""
    daily = bool(REPEAT_RX.search(t))
    m = REL.search(t, pos)
    if m:
        n = m.group("n")
        if m.group("half"):
            secs = 1800
        else:
            k = {"a": 1, "an": 1, "a couple of": 2, "a few": 3}.get(n) or int(n)
            secs = k * (3600 if m.group("u").startswith(("h",)) else 60)
        due = now + secs
        lt = datetime.fromtimestamp(due)
        return When(due, False, (lt.hour, lt.minute), relative=True), m.span()
    m = ABS.search(t, pos)
    if not m:
        return None
    words = [m.group("pre"), m.group("post1"), m.group("post2")]
    rep = next((w for w in REPEAT_RX.findall(t)), None)
    cands = _candidates(m, _part(*words, rep))
    if not cands:
        return None
    today = datetime.fromtimestamp(now).date()
    only = today + timedelta(days=1) if "tomorrow" in words else None
    if "today" in words or "tonight" in words:
        only = today
    best = _next(cands, now, only) or _next(cands, now)
    if best is None:
        return None
    return When(best[0], daily, best[1]), m.span()


def parse_when(text: str, now: float) -> Optional[When]:
    got = find_when(normalize(text), now)
    return got[0] if got else None


def _since(t: str, due: float) -> Optional[tuple[int, int]]:
    """'since 8' as a time of day on the due day: the latest reading of it that is not after the due time."""
    m = SINCE.search(t)
    if not m:
        return None
    cands = _candidates(m, _part(m.group("post1")))
    if not cands:
        return None
    due_hm = datetime.fromtimestamp(due)
    ok = [c for c in cands if c <= (due_hm.hour, due_hm.minute)]
    return max(ok) if ok else None


# ---------------------------------------------------------------- records

@dataclass
class Reminder:
    id: int
    created: float
    kind: str                   # 'time' | 'condition'
    task: Optional[str]         # as said: 'take my pills', 'call Sarah'
    obj: Optional[str]          # entity to check / point at
    due: float                  # next occurrence (epoch)
    daily: bool
    at_h: int
    at_m: int
    since: Optional[tuple[int, int]]
    active: bool
    last_day: Optional[str]
    ended: Optional[str]
    said: str


_RCOLS = "id, created, kind, task, obj, due, daily, at_h, at_m, since_h, since_m, active, last_day, ended, said"


def _reminder(r) -> Reminder:
    (i, created, kind, task, obj, due, daily, h, m, sh, sm, active, last_day, ended, said) = r
    return Reminder(i, created, kind, task, obj, due, bool(daily), h, m, None if sh is None else (sh, sm),
                    bool(active), last_day, ended, said or "")


@dataclass
class Notice:
    id: int
    t: float
    kind: str                   # 'reminder' | 'condition' | 'morning'
    reminder_id: Optional[int]
    day: str
    text: str
    point_at: Optional[str]
    action: Optional[str]
    spoken: bool
    acknowledged: bool
    ack_t: Optional[float]

    def to_json(self, expiry_s: float) -> dict:
        return {"id": self.id, "t": self.t, "kind": self.kind, "text": self.text, "point_at": self.point_at,
                "action": self.action, "acknowledged": self.acknowledged, "ack_t": self.ack_t,
                "spoken": self.spoken, "reminder_id": self.reminder_id, "expires": self.t + expiry_s}


_NCOLS = "id, t, kind, reminder_id, day, text, point_at, action, spoken, acknowledged, ack_t"


def _notice(r) -> Notice:
    (i, t, kind, rid, day, text, point_at, action, spoken_, ack, ack_t) = r
    return Notice(i, t, kind, rid, day, text, point_at, action, bool(spoken_), bool(ack), ack_t)


def day_of(t: float) -> str:
    return datetime.fromtimestamp(t).date().isoformat()


class ReminderStore(CareDB):
    """care_reminders, care_notices and care_state."""

    def __init__(self, events):
        super().__init__(events, SCHEMA)

    def add(self, kind: str, when: When, obj: Optional[str] = None, task: Optional[str] = None,
            since: Optional[tuple[int, int]] = None, said: str = "", now: Optional[float] = None) -> Reminder:
        i = self.insert(f"INSERT INTO care_reminders ({_RCOLS}) VALUES (NULL,?,?,?,?,?,?,?,?,?,?,1,NULL,NULL,?)",
                      (now if now is not None else when.due, kind, task, obj, when.due, int(when.daily),
                       when.hm[0], when.hm[1], since[0] if since else None, since[1] if since else None, said))
        return self.get(i)

    def get(self, i: int) -> Optional[Reminder]:
        r = self.rows(f"SELECT {_RCOLS} FROM care_reminders WHERE id = ?", (i,))
        return _reminder(r[0]) if r else None

    def active(self) -> list[Reminder]:
        return [_reminder(r) for r in self.rows(f"SELECT {_RCOLS} FROM care_reminders WHERE active = 1 "
                                                "ORDER BY due, id")]

    def all(self) -> list[Reminder]:
        return [_reminder(r) for r in self.rows(f"SELECT {_RCOLS} FROM care_reminders ORDER BY id")]

    def end(self, i: int, why: str) -> None:
        self.exec("UPDATE care_reminders SET active = 0, ended = ? WHERE id = ?", (why, i))

    def advance(self, r: Reminder, day: str, why: str, now: float, grace_s: float) -> None:
        """Done with r for day: a daily reminder moves to its next occurrence, a one-shot ends."""
        if not r.daily:
            self.exec("UPDATE care_reminders SET active = 0, ended = ?, last_day = ? WHERE id = ?", (why, day, r.id))
            return
        d = datetime.fromtimestamp(r.due).date() + timedelta(days=1)
        while _at(d, r.at_h, r.at_m) < now - grace_s:
            d += timedelta(days=1)
        self.exec("UPDATE care_reminders SET due = ?, last_day = ? WHERE id = ?", (_at(d, r.at_h, r.at_m), day, r.id))

    # -- notices

    def add_notice(self, t: float, kind: str, text: str, point_at: Optional[str], action: Optional[str],
                   spoken_: bool, reminder_id: Optional[int] = None, day: Optional[str] = None) -> Optional[Notice]:
        """None if this reminder already fired that day (the unique index), so no path can repeat it."""
        i = self.insert(f"INSERT OR IGNORE INTO care_notices ({_NCOLS}) VALUES (NULL,?,?,?,?,?,?,?,?,0,NULL)",
                      (t, kind, reminder_id, day or day_of(t), text, point_at, action, int(spoken_)))
        return self.notice(i) if i else None

    def notice(self, i: int) -> Optional[Notice]:
        r = self.rows(f"SELECT {_NCOLS} FROM care_notices WHERE id = ?", (i,))
        return _notice(r[0]) if r else None

    def notices(self, t0: float, t1: Optional[float] = None) -> list[Notice]:
        rows = self.rows(f"SELECT {_NCOLS} FROM care_notices WHERE t >= ? AND t <= ? ORDER BY t, id",
                         (t0, t1 if t1 is not None else 1e18))
        return [_notice(r) for r in rows]

    def fired(self, reminder_id: int, day: str) -> bool:
        return bool(self.rows("SELECT 1 FROM care_notices WHERE reminder_id = ? AND day = ?", (reminder_id, day)))

    def ack(self, i: int, now: float) -> bool:
        """True if notice i exists (acknowledging twice keeps the first time)."""
        self.exec("UPDATE care_notices SET acknowledged = 1, ack_t = ? WHERE id = ? AND acknowledged = 0", (now, i))
        return self.notice(i) is not None

    def spoken_since(self, t: float) -> int:
        return self.rows("SELECT COUNT(*) FROM care_notices WHERE t >= ? AND spoken = 1", (t,))[0][0]


# ---------------------------------------------------------------- voice requests

LIST = re.compile(r"\b(?:what|which) (?:are |is )?(?:my |the )?reminders?\b|\b(?:do|have) i (?:have |got )?(?:any )?"
                  r"reminders\b|\blist (?:my |the |all )?reminders\b|\bany reminders\b|\bmy reminders\b")
CANCEL = re.compile(r"\b(?:cancel|delete|remove|clear|turn off|stop|forget|drop)\b.*\breminders?\b")
REMIND = re.compile(r"\bremind me\b|\bset (?:a |an )?reminder\b")
EVENT_TRIGGER = re.compile(r"\b(?:when|whenever|once|as soon as|before|after) (?:i|we|you|someone|somebody|they)"
                           r" (?:leave|get|come|go|walk|arrive|stand|sit|wake|finish|am|are|move)\b")
_VERBS = r"(?:picked|pick|touched|touch|moved|move|used|use|taken|take|grabbed|grab|opened|open|lifted|lift)"
COND1 = re.compile(rf"\bif (?:i|we|you) (?:have not|havent|did not|didnt|do not|dont|has not|hasnt)(?: yet)?"
                   rf" {_VERBS}(?: up)? (?P<obj>.+?)(?: up)?(?: yet)? by\b")
COND2 = re.compile(r"\bif (?P<obj>.+?) (?:has not|hasnt|have not|havent|was not|wasnt|were not|werent|is not|isnt)"
                   r"(?: yet)? (?:been )?(?:picked up|touched|moved|used|taken|grabbed|opened|lifted)(?: yet)? by\b")
ACK = re.compile(r"^(?:(?:ok|okay|k|kay|thanks|thank you|thank|thx|done|i did it|did it|i did|got it|will do"
                 r"|all right|alright|yes|yeah|yep|sure|i know|on it|i will|ill do it|noted|cheers|great"
                 r"|good|fine|i see)\s*)+$")


@dataclass
class Request:
    action: str                 # create | list | cancel | unsupported | unclear | unclear_obj
    kind: Optional[str] = None
    when: Optional[When] = None
    obj: Optional[str] = None
    task: Optional[str] = None
    since: Optional[tuple[int, int]] = None
    name: Optional[str] = None

    @property
    def due(self) -> Optional[float]:
        return self.when.due if self.when else None

    @property
    def daily(self) -> bool:
        return bool(self.when and self.when.daily)


def resolve_obj(phrase: str, cfg: dict, world) -> Optional[str]:
    """An entity for the words: configured names and synonyms (voice.intents), taught aliases, then
    world.find on the bare noun phrase."""
    from voice.intents import parse
    try:
        aliases = world.alias_phrases() if hasattr(world, "alias_phrases") else []
    except Exception:
        aliases = []
    it = parse(phrase, cfg, aliases=aliases)
    names = [it.obj, it.name]
    bare = re.sub(r"^(?:(?:up|my|the|your|our|a)\s+)+", "", phrase.strip())
    names += [bare, re.sub(r"\b(?:my|the|your|our)\b", " ", phrase).strip()]
    for n in [x for x in names if x]:
        if n in (cfg.get("objects") or {}) and not hasattr(world, "find"):
            try:
                world.get(n)
                return n
            except Exception:
                continue
        if hasattr(world, "find"):
            try:
                hit = world.find(n)
            except Exception:
                hit = None
            if hit:
                return hit
    return None


def raw_words(raw: str, norm: str) -> str:
    """The words of norm as they were spoken in raw (keeps 'Sarah' capitalized)."""
    ws = norm.split()
    if not ws:
        return norm
    m = re.search(r"\W+".join(re.escape(w).replace("'", "'?") for w in ws), raw.replace("'", ""), re.I)
    return m.group(0) if m else norm


def parse_request(text: str, cfg: dict, world, now: float) -> Optional[Request]:
    """A reminder request in text, or None if it isn't about reminders."""
    t = normalize(text)
    if CANCEL.search(t):
        return Request("cancel")
    if LIST.search(t) and not REMIND.search(t):
        return Request("list")
    if not REMIND.search(t):
        return None
    if EVENT_TRIGGER.search(t):
        return Request("unsupported")
    cond = COND1.search(t) or COND2.search(t)
    if cond:
        got = find_when(t, now, cond.end() - 2)
        if got is None:
            return Request("unclear")
        when = got[0]
        phrase = cond.group("obj")
        obj = resolve_obj(phrase, cfg, world)
        if obj is None:
            return Request("unclear_obj", name=re.sub(r"^(?:my|the|your)\s+", "", phrase))
        return Request("create", "condition", when, obj, since=_since(t, when.due))
    got = find_when(t, now)
    if got is None:
        return Request("unclear")
    when, (a, b) = got
    rest = f"{t[:a]} {t[b:]}"
    rest = REPEAT_RX.sub(" ", rest)
    rest = re.sub(r"\b(?:remind me|set (?:a |an )?reminder|please)\b", " ", rest)
    rest = re.sub(r"\s+", " ", rest).strip()
    rest = re.sub(r"^(?:to|that i|that|about)\s+", "", rest)
    rest = re.sub(r"\s+(?:to|at)$", "", rest).strip()
    if not rest:
        return Request("unclear")
    task = raw_words(text, rest)
    return Request("create", "time", when, resolve_obj(task, cfg, world), task=task)


# ---------------------------------------------------------------- the logic

class Reminders:
    """Voice handling, firing and notices for reminders. Every method takes `now` (epoch seconds), so a
    simulated clock drives it in tests and the care scheduler drives it with time.time()."""

    def __init__(self, cfg: dict, world, events, store: Optional[ReminderStore] = None):
        self.cfg, self.world, self.events = cfg, world, events
        self.c = care_cfg(cfg)
        self.store = store or ReminderStore(events)
        self.on_change: Optional[Callable[[], None]] = None

    # -- settings

    @property
    def expiry_s(self) -> float:
        return float(self.c["notice_expiry_min"]) * 60

    def quiet(self, now: float) -> bool:
        q = self.c.get("quiet_hours")
        if not q or len(q) != 2:
            return False
        a, b = (int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in map(str, q))
        lt = datetime.fromtimestamp(now)
        x = lt.hour * 60 + lt.minute
        return a <= x < b if a <= b else (x >= a or x < b)

    def may_speak(self, now: float) -> bool:
        cap = int(self.c.get("max_notices_per_hour") or 0)
        return not self.quiet(now) and (cap <= 0 or self.store.spoken_since(now - 3600) < cap)

    # -- voice

    def handle(self, text: str, now: float) -> Optional[Answer]:
        """Answer a reminder request (create / list / cancel), or None if text isn't one."""
        req = parse_request(text, self.cfg, self.world, now)
        if req is None:
            return None
        if req.action == "list":
            return Answer(self.describe_all(now))
        if req.action == "cancel":
            return self.cancel(text, now)
        if req.action == "unsupported":
            return Answer("I can't do reminders for when you leave or arrive yet. I can remind you at a time, "
                          "or if something hasn't been picked up by a time.")
        if req.action == "unclear":
            return Answer("When should I remind you? You can say something like 'remind me at 8 to call Sarah'.")
        if req.action == "unclear_obj":
            return Answer(f"I don't know what your {req.name} is yet, so I can't watch it. "
                          f"Put it in the teach square and say 'this is my {req.name}'.")
        r = self.store.add(req.kind, req.when, obj=req.obj, task=req.task, since=req.since, said=text, now=now)
        self._changed()
        return Answer(pill_guard(self.confirm(r, now)))

    def confirm(self, r: Reminder, now: float) -> str:
        T = say_time(r.at_h, r.at_m)
        if r.kind == "condition":
            n, has = self._name(r.obj), "haven't" if plural(spoken(self.world, self.cfg, r.obj)) else "hasn't"
            when = f"Every day at {T}" if r.daily else f"At {T} {say_day(r.due, now)}"
            since = f" since {say_time(*r.since)}" if r.since else ""
            return f"Okay. {when}, if {n} {has} been picked up{since}, I'll remind you."
        if r.daily:
            return f"Okay, I'll remind you every day at {T}."
        return f"Okay, I'll remind you at {T} {say_day(r.due, now)}."

    def _name(self, obj: Optional[str]) -> str:
        return f"{your(self.world, self.cfg, obj)} {spoken(self.world, self.cfg, obj)}" if obj else "it"

    def label(self, r: Reminder) -> str:
        """What a reminder is about, short: 'your pill bottle', 'your pills', 'call Sarah'."""
        if r.kind == "condition":
            return self._name(r.obj)
        return med_phrase(r.task or "") or second_person(r.task or "")

    def describe(self, r: Reminder, now: float) -> str:
        T = say_time(r.at_h, r.at_m)
        when = f"every day at {T}" if r.daily else f"at {T} {say_day(r.due, now)}"
        if r.kind == "condition":
            has = "haven't" if plural(spoken(self.world, self.cfg, r.obj)) else "hasn't"
            return f"{when}, if {self._name(r.obj)} {has} been picked up"
        meds = med_phrase(r.task or "")
        return f"{when}, {'for ' + meds if meds else 'to ' + second_person(r.task or '')}"

    def describe_all(self, now: float) -> str:
        rs = self.store.active()
        if not rs:
            return "You don't have any reminders."
        parts = [self.describe(r, now) for r in rs]
        if len(parts) == 1:
            return pill_guard(f"You have one reminder: {parts[0]}.")
        body = "; ".join(parts[:-1]) + f"; and {parts[-1]}"
        return pill_guard(f"You have {len(parts)} reminders: {body}.")

    def short(self, r: Reminder) -> str:
        T = say_time(r.at_h, r.at_m)
        if r.kind == "condition":
            return f"{T} {spoken(self.world, self.cfg, r.obj)} reminder"
        meds = med_phrase(r.task or "")
        return f"{T} reminder {'for ' + meds if meds else 'to ' + second_person(r.task or '')}"

    def cancel(self, text: str, now: float) -> Answer:
        rs = self.store.active()
        if not rs:
            return Answer("You don't have any reminders.")
        t = normalize(text)
        if re.search(r"\b(?:all|every|everything|both)\b", t):
            for r in rs:
                self.store.end(r.id, "cancelled")
            self._changed()
            return Answer("Okay, I cancelled all of your reminders." if len(rs) > 1
                          else f"Okay, I cancelled your {self.short(rs[0])}.")
        match = rs
        m = re.search(r"\b(?P<h>\d{1,2})(?:[:.](?P<m>\d\d))?(?:\s*(?P<ap>am|pm))?\b", t)
        if m:
            h, mi, ap = int(m.group("h")), m.group("m"), m.group("ap")
            match = [r for r in match if (r.at_h == h if h > 12 else r.at_h % 12 == h % 12)
                     and (mi is None or r.at_m == int(mi)) and (ap is None or (r.at_h >= 12) == (ap == "pm"))]
        else:
            words = re.sub(r"\b(?:cancel|delete|remove|clear|turn off|stop|forget|drop|reminders?|my|the|that|it"
                           r"|this|please|last)\b", " ", t).strip()
            obj = resolve_obj(words, self.cfg, self.world) if words else None
            if obj is not None:
                match = [r for r in match if r.obj == obj]
            elif re.search(r"\b(?:that|it|this|last)\b", t):
                match = [max(rs, key=lambda r: r.id)]
            elif words:
                ws = set(words.split())
                match = [r for r in match if ws & set(normalize(r.task or "").split())]
        if not match:
            return Answer(pill_guard("I couldn't find that reminder. " + self.describe_all(now)))
        times = sorted({(r.at_h, r.at_m) for r in match})
        if len(times) > 1:
            said = " and ".join(say_time(*x) for x in times)
            return Answer(f"You have reminders at {said}. Which one should I cancel?")
        for r in match:
            self.store.end(r.id, "cancelled")
        self._changed()
        return Answer(pill_guard(f"Okay, I cancelled your {self.short(match[0])}."))

    def _changed(self) -> None:
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                log.exception("reminder change hook failed")

    # -- firing

    def today(self, now: float) -> list[tuple[float, Reminder]]:
        """(time, reminder) for occurrences still ahead today, soonest first."""
        d = datetime.fromtimestamp(now).date()
        out = []
        for r in self.store.active():
            t = _at(d, r.at_h, r.at_m) if r.daily else r.due
            if t >= now and datetime.fromtimestamp(t).date() == d and r.last_day != d.isoformat():
                out.append((t, r))
        return sorted(out, key=lambda x: x[0])

    def picked_up(self, r: Reminder, now: float) -> bool:
        """Any PICKED_UP / MOVED / PUT_BACK of r's object (or a thing merged into it) in r's window."""
        d = datetime.fromtimestamp(r.due).date()
        start = _at(d, *r.since) if r.since else _at(d, 0, 0)
        names = {r.obj}
        try:
            names |= {n for n, e in getattr(self.world, "entities", {}).items()
                      if getattr(e, "merged_into", None) == r.obj}
        except Exception:
            pass
        return any(ev.obj in names and str(ev.type) in PICKUP_TYPES and ev.wall <= now
                   for ev in self.events.since(start))

    def tick(self, now: float) -> list[tuple[Notice, Optional[Answer]]]:
        """Fire what is due. Returns (notice, answer to speak or None when quiet / capped)."""
        out = []
        grace = float(self.c["late_grace_min"]) * 60
        for r in self.store.active():
            if r.due > now:
                continue
            day = day_of(r.due)
            if r.last_day == day or self.store.fired(r.id, day):
                self.store.advance(r, day, "done", now, grace)
                continue
            if now - r.due > grace:
                log.info("reminder %d skipped: %.0f s late (scheduler not running?)", r.id, now - r.due)
                self.store.advance(r, day, "skipped", now, grace)
                continue
            if r.kind == "condition" and self.picked_up(r, now):
                self.store.advance(r, day, "satisfied", now, grace)
                continue
            ans = self.notice_answer(r)
            speak = self.may_speak(now)
            n = self.store.add_notice(now, "condition" if r.kind == "condition" else "reminder", ans.text,
                                      ans.point_at, ans.action, speak, reminder_id=r.id, day=day)
            self.store.advance(r, day, "done", now, grace)
            if n is not None:
                log.info("reminder %d fired%s: %s", r.id, "" if speak else " (silent)", ans.text)
                out.append((n, ans if speak else None))
        return out

    def notice_answer(self, r: Reminder) -> Answer:
        """The spoken notice. Condition: 'It's 9 o'clock and you haven't picked up your pill bottle yet.
        It's under the notebook.' Medication tasks are said without a verb ('Time for your pills.')."""
        now_s = f"It's {say_oclock(r.at_h, r.at_m)}"
        point, action, tail = None, None, ""
        if r.kind == "condition":
            head = f"{now_s} and you haven't picked up {self._name(r.obj)} yet."
            tail, action = where_sentence(self.world, self.cfg, r.obj)
        else:
            meds = med_phrase(r.task or "")
            head = f"{now_s}. " + (f"Time for {meds}." if meds else f"Time to {second_person(r.task or '')}.")
            if r.obj:
                tail, action = where_sentence(self.world, self.cfg, r.obj, named=bool(meds))
        if r.obj and action:
            point = r.obj
        text = pill_guard(f"{head} {tail}".strip())
        return Answer(text, point_at=point, action=action if point else None)

    # -- acknowledgement and the dashboard

    def acknowledge(self, text: str, now: float) -> Optional[Answer]:
        """'okay' / 'thanks' / 'done' within ack_window_s of the latest unacknowledged notice."""
        if not ACK.match(normalize(text)):
            return None
        win = float(self.c["ack_window_s"])
        open_ = [n for n in self.store.notices(now - win, now) if not n.acknowledged]
        if not open_:
            return None
        self.store.ack(open_[-1].id, now)
        return Answer("Okay, got it.")

    def notices_json(self, now: float) -> list[dict]:
        """Notices of the last notice_expiry_min, newest first; older unacknowledged ones have expired."""
        return [n.to_json(self.expiry_s) for n in reversed(self.store.notices(now - self.expiry_s, now))]
