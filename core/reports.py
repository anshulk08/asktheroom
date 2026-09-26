"""Morning report and caregiver summary (care layer, feature 2), built deterministically from the event
log, the world state, today's reminders and (when that store exists) narration summaries.

Morning report: spoken once a day, on the first question or the first activity on the table after
care.morning_after (default 06:00) and before care.morning_until (noon). At most three sentences and only
what matters in the morning: things that ended yesterday hidden, gone or lost (with the time), the pill
bottle's pickup times, and today's reminders. 'Good morning, Grandpa Joe. Yesterday your keys ended up in
the box at 3:10 PM, and your glasses were last seen under the notebook at 6 PM. You have one reminder
today: your pill bottle at 8 AM.'

Caregiver summary: the longer digest of one day, as Markdown, plain text (SMS / n8n) or JSON
(summary_data): every pill-bottle pickup, what went missing and where it ended up, reminders fired and
acknowledged, lost-track events, narration summaries, and how many questions were asked. The dashboard
serves it at GET /report?date=YYYY-MM-DD; voice asks for the spoken version ('give me today's summary').

Pill rule throughout: pickups of the bottle, never whether medication was taken.

Daily briefings and caregiver digests for dementia care are ideas from Project Memoria (MIT); what is
selected and how it is said here is our own, built on our event log and world model.
"""
from __future__ import annotations

import re
import threading
from datetime import date, datetime, time as dtime, timedelta
from typing import Optional, Union

from core.carewords import (event_place, pill_claim, pill_guard, plural, say_clock, spoken, where_sentence,
                            your)
from core.reminders import ReminderStore, Reminders, care_cfg
from core.types import Status

PILL = "pill_bottle"
NOTABLE = (Status.INSIDE, Status.UNDER, Status.GONE, Status.UNKNOWN)
HIDE = ("PUT_INSIDE", "COVERED", "EXITED_VIEW", "LOST_TRACK")
PUT_DOWN = ("PUT_BACK", "MOVED", "PUT_INSIDE", "COVERED", "EXITED_VIEW")
MAX_OBJECTS = 2                 # things named in the morning report
MAX_PICKUPS = 3                 # pickup times listed before 'N times, last at ...'
PILL_NOTE = "Only pill bottle pickups are shown here; this is not a medication record."


# ---------------------------------------------------------------- helpers

def _midnight(d: date) -> float:
    return datetime.combine(d, dtime(0, 0)).timestamp()


def _hm(s: str) -> tuple[int, int]:
    h, m = str(s).split(":")
    return int(h), int(m)


def as_date(day: Union[date, str, None], now: float) -> date:
    """date, 'YYYY-MM-DD', 'today', 'yesterday' or None (today)."""
    today = datetime.fromtimestamp(now).date()
    if day is None or day == "today":
        return today
    if day == "yesterday":
        return today - timedelta(days=1)
    if isinstance(day, date):
        return day
    return date.fromisoformat(str(day))


def _entities(world) -> list[dict]:
    try:
        return [e for e in world.state_json().get("entities", []) if not e.get("merged_into")]
    except Exception:
        return []


def _targets(world) -> list[str]:
    return [e["name"] for e in _entities(world) if e.get("kind") == "target"]


def _day_events(events, t0: float, t1: float) -> list:
    return [e for e in events.since(t0) if e.wall < t1] if events is not None else []


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 2 else f"{parts[0]} and {parts[1]}"


def _times(walls: list[float]) -> str:
    if len(walls) > MAX_PICKUPS:
        return f"{len(walls)} times, last at {say_clock(walls[-1])}"
    return "at " + _join([say_clock(w) for w in walls])


def _count(n: int) -> str:
    return ("one two three four five six seven eight nine ten".split()[n - 1]) if 1 <= n <= 10 else str(n)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _narrations(events, t0: float, t1: float) -> list:
    """Done narration rows overlapping [t0, t1) if core.narration_store and its table exist, else []."""
    try:
        from core.narration_store import store_for
    except Exception:
        return []
    try:
        st = store_for(events, create=False)
        return [r for r in st.between(t0, t1) if r.summary] if st is not None else []
    except Exception:
        return []


# ---------------------------------------------------------------- morning report

def _object_clause(world, cfg: dict, obj: str, ev, st: Status) -> str:
    n = f"{your(world, cfg, obj)} {spoken(world, cfg, obj)}"
    pl = plural(spoken(world, cfg, obj))
    T = say_clock(ev.wall)
    e = world.get(obj)
    if st == Status.INSIDE:
        p = e.parent if e.parent and e.parent != "unknown" else None
        return f"{n} ended up in the {spoken(world, cfg, p)} at {T}" if p else f"{n} ended up inside something at {T}"
    if st == Status.UNDER:
        p = e.parent if e.parent and e.parent != "unknown" else None
        where = f"under the {spoken(world, cfg, p)}" if p else "under something"
        return f"{n} {'were' if pl else 'was'} last seen {where} at {T}"
    if st == Status.GONE:
        side = f"the {e.edge} side of the table" if e.edge else "the table"
        return f"{n} went off {side} at {T}"
    from voice.answers import area
    return f"I lost track of {n} at {T}, {area(e.pos_cm, cfg)}"


def _reminder_sentence(rem: Reminders, now: float) -> str:
    today = rem.today(now)
    if not today:
        return "You don't have any reminders today."
    items = [f"{rem.label(r)} at {say_clock(t)}" for t, r in today[:3]]
    if len(today) == 1:
        return f"You have one reminder today: {items[0]}."
    return f"You have {len(today)} reminders today: {_join(items)}."


def morning_report(world, events, cfg: dict, now: float, reminders: Optional[Reminders] = None,
                   profile=None) -> str:
    """The morning report text (always returns one; MorningReport decides when to say it)."""
    today = datetime.fromtimestamp(now).date()
    y0, t0 = _midnight(today - timedelta(days=1)), _midnight(today)
    name = None
    try:
        name = profile.preferred_name() if profile is not None else None
    except Exception:
        pass
    parts = [f"Good morning, {name}." if name else "Good morning."]

    clauses = []
    for obj in _targets(world):
        try:
            st = world.get(obj).status
        except Exception:
            continue
        if st not in NOTABLE:
            continue
        last = events.last(obj, 1) if events is not None else []
        if last and y0 <= last[0].wall < t0:
            clauses.append((last[0].wall, _object_clause(world, cfg, obj, last[0], st)))
    clauses = [c for _, c in sorted(clauses)[:MAX_OBJECTS]]
    day_evs = _day_events(events, y0, t0)
    picks = [e.wall for e in day_evs if e.obj == PILL and str(e.type) == "PICKED_UP"]
    pill = f"your pill bottle was picked up {_times(picks)}" if picks else None
    if clauses or pill:
        body = ", and ".join(clauses)
        body = f"{body}; {pill}" if clauses and pill else (body or pill)
        parts.append(f"Yesterday {body}.")
    else:
        line = _narration_line(events, y0, t0)
        if line:
            parts.append(line)
        elif day_evs:
            parts.append("Nothing went missing yesterday.")

    parts.append(_reminder_sentence(reminders or Reminders(cfg, world, events), now))
    return pill_guard(" ".join(parts))


def _narration_line(events, t0: float, t1: float) -> Optional[str]:
    rows = [r for r in _narrations(events, t0, t1) if r.confidence >= 0.5]
    if not rows:
        return None
    first = re.split(r"(?<=[.!?])\s+", rows[-1].summary.strip())[0].rstrip(".!?")
    if not first or pill_claim(first):
        return None
    if re.match(r"(?:You|Someone|Somebody|A person)\b", first):
        return f"Yesterday {first[0].lower()}{first[1:]}."
    return f"Yesterday: {first}."


class MorningReport:
    """Once-per-day delivery. The day it was said is kept in care_state, so a restart doesn't repeat it."""

    KEY = "morning_day"

    def __init__(self, cfg: dict, world, events, reminders: Optional[Reminders] = None, profile=None):
        self.cfg, self.world, self.events, self.profile = cfg, world, events, profile
        self.reminders = reminders or Reminders(cfg, world, events)
        self.db: ReminderStore = self.reminders.store
        c = care_cfg(cfg)
        self.after, self.until = _hm(c.get("morning_after", "06:00")), _hm(c.get("morning_until", "12:00"))
        self.enabled = bool(c.get("morning_report", True))
        self._lock = threading.Lock()          # the scheduler and a question may both try at once

    def _bounds(self, now: float) -> tuple[float, float]:
        d = datetime.fromtimestamp(now).date()
        return (datetime.combine(d, dtime(*self.after)).timestamp(),
                datetime.combine(d, dtime(*self.until)).timestamp())

    def due(self, now: float) -> bool:
        a, b = self._bounds(now)
        return self.enabled and a <= now < b and \
            self.db.get_state(self.KEY) != datetime.fromtimestamp(now).date().isoformat()

    def mark_delivered(self, now: float) -> None:
        self.db.set_state(self.KEY, datetime.fromtimestamp(now).date().isoformat())

    def deliver(self, now: float) -> Optional[str]:
        """The report if it is due now (and marks it said), else None."""
        with self._lock:
            if not self.due(now):
                return None
            text = morning_report(self.world, self.events, self.cfg, now, self.reminders, self.profile)
            self.mark_delivered(now)
            return text

    def activity_since_morning(self, now: float) -> bool:
        a, _ = self._bounds(now)
        return any(e.wall <= now for e in self.events.since(a)) if now >= a else False


# ---------------------------------------------------------------- caregiver summary

def _place_now(world, cfg: dict, obj: str) -> str:
    try:
        e = world.get(obj)
    except Exception:
        return "unknown"
    if e.status in (Status.INSIDE, Status.UNDER, Status.GONE):
        s, _ = where_sentence(world, cfg, obj)
        s = re.sub(r"^(?:It's|They're|It was|They were)\s+", "", s).rstrip(".")
        return s.replace("carried off", "off")
    if e.status == Status.VISIBLE:
        return "back on the table"
    if e.status == Status.HELD:
        return "being held"
    return "not found yet"


def summary_data(day, world, events, cfg: dict, reminders: Optional[Reminders] = None,
                 now: Optional[float] = None) -> dict:
    """One day as JSON-friendly data: what caregivers, n8n, SMS and the phone app are sent."""
    import time as _time
    now = _time.time() if now is None else now
    d = as_date(day, now)
    t0, t1 = _midnight(d), _midnight(d + timedelta(days=1))
    is_today = d == datetime.fromtimestamp(now).date()
    evs = _day_events(events, t0, t1)
    targets = set(_targets(world))

    pickups = []
    for i, e in enumerate(evs):
        if e.obj == PILL and str(e.type) == "PICKED_UP":
            down = next((x for x in evs[i + 1:] if x.obj == PILL and str(x.type) in PUT_DOWN), None)
            pickups.append({"t": e.wall, "clock": say_clock(e.wall),
                            "put_down_t": down.wall if down else None,
                            "put_down_clock": say_clock(down.wall) if down else None})

    missing = []
    for obj in sorted({e.obj for e in evs if str(e.type) in HIDE and e.obj in targets},
                      key=lambda o: next(e.wall for e in evs if e.obj == o and str(e.type) in HIDE)):
        hides = [e for e in evs if e.obj == obj and str(e.type) in HIDE]
        mine = [e for e in evs if e.obj == obj]
        if is_today:
            end = _place_now(world, cfg, obj)
        else:
            last = mine[-1]
            end = "lost from view" if str(last.type) == "LOST_TRACK" else \
                ("being held" if str(last.type) == "PICKED_UP" else (event_place(last, world, cfg) or "unknown"))
        missing.append({"obj": obj, "name": spoken(world, cfg, obj),
                        "events": [{"t": e.wall, "clock": say_clock(e.wall), "type": str(e.type),
                                    "place": event_place(e, world, cfg) or "lost from view"} for e in hides],
                        "end_place": end})

    rem = reminders or Reminders(cfg, world, events)
    notices = [{"id": n.id, "t": n.t, "clock": say_clock(n.t), "kind": n.kind, "text": n.text,
                "acknowledged": n.acknowledged, "ack_t": n.ack_t,
                "ack_clock": say_clock(n.ack_t) if n.ack_t else None, "spoken": n.spoken}
               for n in rem.store.notices(t0, t1 - 1e-6)]
    lost = [{"obj": e.obj, "name": spoken(world, cfg, e.obj), "t": e.wall, "clock": say_clock(e.wall)}
            for e in evs if str(e.type) == "LOST_TRACK" and e.obj in targets]
    narr = [{"t_start": r.t_start, "t_end": r.t_end, "summary": pill_guard(r.summary)}
            for r in _narrations(events, t0, t1)]
    questions = None
    try:
        questions = events._rows("SELECT COUNT(*) FROM questions WHERE t >= ? AND t < ?", (t0, t1))[0][0]
    except Exception:
        pass
    return {"date": d.isoformat(), "is_today": is_today, "pill_bottle_pickups": pickups, "missing": missing,
            "reminders": notices, "lost_track": lost, "narrations": narr, "questions": questions,
            "watching": bool(evs)}


def caregiver_summary(day, world, events, cfg: dict, reminders: Optional[Reminders] = None,
                      fmt: str = "markdown", now: Optional[float] = None) -> str:
    """The day's digest as Markdown (fmt='markdown') or plain text (fmt='text')."""
    s = summary_data(day, world, events, cfg, reminders, now)
    d = date.fromisoformat(s["date"])
    md = fmt == "markdown"
    lines = [f"{'# ' if md else ''}Daily summary: {d.strftime('%A, %B')} {d.day}, {d.year}", "", PILL_NOTE]

    def section(title: str, items: list[str]) -> None:
        lines.extend(["", f"## {title}" if md else title.upper()])
        lines.extend(f"- {x}" for x in items)

    section("Pill bottle", [f"Picked up at {p['clock']}" + (f" (put down at {p['put_down_clock']})"
                                                            if p["put_down_clock"] else "")
                            for p in s["pill_bottle_pickups"]] or ["No pickups of the pill bottle were seen."])
    end = "now" if s["is_today"] else "ended the day"
    section("Items that went missing",
            [f"{_cap(m['name'])}: " + ", ".join(f"{e['place']} at {e['clock']}" for e in m["events"])
             + f"; {end} {m['end_place']}." for m in s["missing"]] or ["Nothing went missing."])
    section("Reminders", [f"{r['clock']}: \"{r['text']}\" "
                          + (f"(acknowledged at {r['ack_clock']})" if r["acknowledged"] else "(not acknowledged)")
                          + ("" if r["spoken"] else " [not spoken: quiet hours or limit]")
                          for r in s["reminders"]] or ["No reminders went off."])
    section("Lost track", [f"{_cap(x['name'])} at {x['clock']}" for x in s["lost_track"]] or ["None."])
    if s["narrations"]:
        section("Activity", [f"{say_clock(n['t_start'])} to {say_clock(n['t_end'])}: {n['summary']}"
                             for n in s["narrations"]])
    if s["questions"] is not None:
        section("Questions", [f"{s['questions']} question{'' if s['questions'] == 1 else 's'} asked."])
    if not s["watching"]:
        lines.extend(["", "No table events were recorded this day; the system may not have been running."])
    return "\n".join(pill_guard(x) if x.strip() and pill_claim(x) else x for x in lines) + "\n"


def spoken_summary(day, world, events, cfg: dict, reminders: Optional[Reminders] = None,
                   now: Optional[float] = None) -> str:
    """At most three spoken sentences for 'give me today's summary'."""
    import time as _time
    now = _time.time() if now is None else now
    s = summary_data(day, world, events, cfg, reminders, now)
    d, today = date.fromisoformat(s["date"]), datetime.fromtimestamp(now).date()
    when = "Today" if d == today else ("Yesterday" if d == today - timedelta(days=1) else f"On {d.strftime('%A')}")
    out = []
    picks = [p["t"] for p in s["pill_bottle_pickups"]]
    if picks:
        out.append(f"{when} the pill bottle was picked up {_times(picks)}.")
    elif s["watching"]:
        out.append(f"{when} I didn't see the pill bottle picked up.")
    else:
        out.append(f"I don't have anything recorded for {when.lower() if when in ('Today', 'Yesterday') else d.strftime('%A')}.")
        return out[0]
    gone = []
    for m in s["missing"][:3]:
        n = f"{your(world, cfg, m['obj'])} {m['name']}"
        last = m["events"][-1]
        gone.append(f"I lost track of {n}" if last["type"] == "LOST_TRACK" else f"{n} went {last['place']}")
    out.append(_cap(_join(gone)) + "." if gone else "Nothing went missing.")
    rs = s["reminders"]
    if rs:
        acked = sum(1 for r in rs if r["acknowledged"])
        out.append(f"{_count(len(rs)).capitalize()} reminder{'s' if len(rs) > 1 else ''} went off, and "
                   f"{'none' if acked == 0 else _count(acked)} {'was' if acked == 1 else 'were'} acknowledged.")
    return pill_guard(" ".join(out))
