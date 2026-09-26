"""Offline answer templates (spec V4): Intent + world + event log -> Answer (speech + laser).

Answers are spoken: 1-2 short sentences, no markdown. Pill bottle wording is deliberately neutral:
no template says a pill or medication was taken ('taken out of' is phrased 'lifted out of').
"""
from __future__ import annotations

import time
from typing import Optional

from core.config import display_name, load_config
from core.types import Answer, Entity, Event, Intent, Point, Status

__all__ = ["Answer", "answer", "ago", "clock", "area"]

PLURAL = {"keys", "glasses", "pills"}
NEAR_CM = 25.0          # 'near the X' only when another visible object is this close
CHANGES_WINDOW_S = 600  # default look-back for 'what changed'
PUT_DOWN = ("PUT_BACK", "MOVED", "PUT_INSIDE", "COVERED", "EXITED_VIEW")


# ---------- wording helpers ----------

def _plural(obj: str) -> bool:
    return obj in PLURAL or (obj.endswith("s") and not obj.endswith("ss"))


def _be(obj: str, past: bool = False) -> str:
    return ("were" if past else "are") if _plural(obj) else ("was" if past else "is")


def _your(cfg: dict, obj: str) -> str:
    """'Your keys' for the user's things, 'The box' for props."""
    return "Your" if (cfg.get("objects") or {}).get(obj, "target") == "target" else "The"


def _it(obj: str) -> str:
    return "them" if _plural(obj) else "it"


def _It(obj: str) -> str:
    return "They" if _plural(obj) else "It"


def ago(wall: Optional[float], now: Optional[float] = None) -> str:
    """Human 'ago': 'just now', '40 seconds ago', 'a minute ago', '6 minutes ago', 'an hour ago'."""
    if wall is None:
        return "a while ago"
    d = max(0.0, (now if now is not None else time.time()) - wall)
    if d < 10:
        return "just now"
    if d < 55:
        return f"{int(round(d, -1))} seconds ago"
    if d < 90:
        return "a minute ago"
    if d < 3300:
        return f"{int(round(d / 60))} minutes ago"
    if d < 5400:
        return "an hour ago"
    if d < 86400:
        return f"{int(round(d / 3600))} hours ago"
    days = int(round(d / 86400))
    return "yesterday" if days == 1 else f"{days} days ago"


def clock(wall: float) -> str:
    """Local clock time as spoken: '3:42 PM'."""
    lt = time.localtime(wall)
    h = lt.tm_hour % 12 or 12
    return f"{h}:{lt.tm_min:02d} {'AM' if lt.tm_hour < 12 else 'PM'}"


def area(pos: Optional[Point], cfg: dict) -> str:
    """Coarse table region for a table-cm point: 'near the top left', 'in the middle', ..."""
    if pos is None:
        return "somewhere on the table"
    w, h = (cfg.get("table") or {}).get("size_cm", [90, 60])
    col = "left" if pos[0] < w / 3 else "right" if pos[0] > 2 * w / 3 else ""
    row = "top" if pos[1] < h / 3 else "bottom" if pos[1] > 2 * h / 3 else ""
    if row and col:
        return f"near the {row} {col}"
    if row:
        return f"near the {row} edge"
    if col:
        return f"on the {col} side"
    return "in the middle"


def _kind(world, cfg: dict, name: str) -> Optional[str]:
    try:
        return world.get(name).kind
    except Exception:
        return (cfg.get("objects") or {}).get(name)


def _rel(world, cfg: dict, parent: Optional[str], status: Optional[Status] = None) -> str:
    """Where-phrase for being attached to parent: 'inside the box', 'under the notebook'."""
    if parent is None or parent == "unknown":
        return "under something" if status == Status.UNDER else "inside something"
    if parent.startswith("hand"):
        return "in someone's hand"
    k = _kind(world, cfg, parent)
    under = status == Status.UNDER if status in (Status.UNDER, Status.INSIDE) else k == "cover"
    return f"{'under' if under else 'inside'} the {display_name(cfg, parent)}"


def _clause(e: Entity, world, cfg: dict) -> Optional[str]:
    """Where-phrase for an entity in a parent chain ('under the notebook'); None if on the table."""
    if e.status in (Status.INSIDE, Status.UNDER):
        return _rel(world, cfg, e.parent, e.status)
    if e.status == Status.HELD:
        return "being held right now"
    if e.status == Status.GONE:
        return f"off the {e.edge} side of the table" if e.edge else "off the table"
    return None


def _event_phrase(ev: Event, cfg: dict) -> str:
    """Passive participle phrase for an event: 'picked up', 'put inside the box'."""
    p = display_name(cfg, ev.parent) if ev.parent and not ev.parent.startswith("hand") \
        and ev.parent != "unknown" else None
    return {
        "PICKED_UP": "picked up",
        "PUT_BACK": "put back down",
        "MOVED": "moved",
        "COVERED": f"covered by the {p}" if p else "covered up",
        "UNCOVERED": "uncovered",
        "PUT_INSIDE": f"put inside the {p}" if p else "put inside something",
        "TAKEN_OUT": f"lifted out of the {p}" if p else "lifted out of something",
        "EXITED_VIEW": f"carried off the {ev.edge} side of the table" if ev.edge else "carried off the table",
        "LOST_TRACK": "lost from view",
        "CORRECTED": "given a corrected location",
        "FOUND": "found again",
    }.get(ev.type, "seen changing")


def _chain(ev_list: list[Event], cfg: dict, now: float) -> str:
    """Oldest-first events -> 'picked up 6 minutes ago, then put inside the box'."""
    parts, prev = [], None
    for ev in ev_list:
        ph = _event_phrase(ev, cfg)
        if prev is None or ev.wall - prev >= 30:
            ph = f"{ph} {ago(ev.wall, now)}"
        parts.append(ph if prev is None else f"then {ph}")
        prev = ev.wall
    return ", ".join(parts)


def _located(world, obj: str) -> bool:
    try:
        return world.resolve(obj)[0] is not None
    except Exception:
        return False


def _which(cfg: dict) -> Answer:
    names = [display_name(cfg, o) for o, k in (cfg.get("objects") or {}).items() if k == "target"]
    lst = ", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 1 else "".join(names)
    return Answer(f"Which object do you mean? I can track your {lst}.")


# ---------- per-intent answers ----------

def _where(obj: str, world, events, cfg: dict, now: float) -> Answer:
    e = world.get(obj)
    n, be, It, Y = display_name(cfg, obj), _be(obj), _It(obj), _your(cfg, obj)
    hedge, plain = float(cfg.get("answer_hedge", 0.5)), float(cfg.get("answer_plain", 0.7))

    if e.status == Status.UNKNOWN or e.confidence < hedge:
        return Answer(f"I lost track of {Y.lower()} {n}. I last saw {_it(obj)} {area(e.pos_cm, cfg)} "
                      f"{ago(e.last_seen, now)}.", point_at=obj, action="circle")

    prob = " probably" if e.confidence < plain or e.candidates else ""

    if e.status == Status.VISIBLE:
        near = _nearest(obj, e, world, cfg)
        tail = f", near the {near}" if near else ""
        return Answer(f"{Y} {n} {be}{prob} on the table{tail}.", point_at=obj, action="point")

    if e.status == Status.HELD:
        pos = e.pos_cm if e.pos_cm is not None else (world.resolve(obj)[0] if _located(world, obj) else None)
        return Answer(f"Someone is{prob} holding {Y.lower()} {n} right now.",
                      point_at=obj if pos is not None else None, action="point" if pos is not None else None)

    if e.status == Status.GONE:
        ev = events.last_of_type(obj, ["EXITED_VIEW"]) if events is not None else None
        side = f"the {e.edge} side of the table" if e.edge else "the table"
        when = ago(ev.wall if ev else e.last_seen, now)
        return Answer(f"{Y} {n} {_be(obj, True)}{prob} carried off {side} {when}.",
                      point_at=obj, action=f"sweep:{e.edge}" if e.edge else "circle")

    # INSIDE / UNDER
    if e.status == Status.UNDER and (e.parent in (None, "unknown")):
        where = "under something near where it was" if not _plural(obj) else "under something near where they were"
    else:
        where = _rel(world, cfg, e.parent, e.status)
        try:
            _, chain = world.resolve(obj)
        except Exception:
            chain = [obj]
        for name in chain[1:3]:  # the parent's own location, then the grandparent's
            try:
                c = _clause(world.get(name), world, cfg)
            except Exception:
                c = None
            if not c:
                break
            where += f", which is {c}"
    alts = [c for c in e.candidates if c != e.parent]
    if alts:
        where += "".join(f", or {_rel(world, cfg, c)}" for c in alts[:2])
    text = f"{Y} {n} {be}{prob} {where}."
    if e.status == Status.INSIDE and not alts:
        ev = events.last_of_type(obj, ["PUT_INSIDE"]) if events is not None else None
        if ev or e.last_seen:
            text += f" {It} {_be(obj, True)} put there {ago(ev.wall if ev else e.last_seen, now)}."
    return Answer(text, point_at=obj, action="point")


def _nearest(obj: str, e: Entity, world, cfg: dict) -> Optional[str]:
    if e.pos_cm is None:
        return None
    try:
        ents = world.state_json().get("entities", [])
    except Exception:
        return None
    best, bd = None, NEAR_CM
    for d in ents:
        if d.get("name") == obj or d.get("status") != Status.VISIBLE.value or not d.get("pos_cm"):
            continue
        dist = ((d["pos_cm"][0] - e.pos_cm[0]) ** 2 + (d["pos_cm"][1] - e.pos_cm[1]) ** 2) ** 0.5
        if dist < bd:
            best, bd = d["name"], dist
    return display_name(cfg, best) if best else None


def _history(obj: str, world, cfg: dict, now: float) -> Answer:
    n = display_name(cfg, obj)
    evs = list(reversed(world.history(obj, 3)))
    point = ("point" if _located(world, obj) else None)
    if not evs:
        return Answer(f"I haven't seen anything happen to {_your(cfg, obj).lower()} {n} since I started watching.",
                      point_at=obj if point else None, action=point)
    text = f"{_your(cfg, obj)} {n} {_be(obj, True)} {_chain(evs, cfg, now)}."
    try:  # a nice touch: the container/cover it is in moved after that
        parent = world.get(obj).parent
        if parent and parent in (cfg.get("objects") or {}):
            pe = world.history(parent, 1)
            if pe and pe[0].wall > evs[-1].wall and pe[0].type in ("MOVED", "PICKED_UP", "EXITED_VIEW"):
                pn = display_name(cfg, parent)
                text += f" The {pn} {_be(parent, True)} {_event_phrase(pe[0], cfg)} {ago(pe[0].wall, now)}."
    except Exception:
        pass
    return Answer(text, point_at=obj if point else None, action=point)


def _handled(obj: str, world, events, cfg: dict, now: float) -> Answer:
    n, was = display_name(cfg, obj), _be(obj, True)
    evs = list(reversed(events.last(obj, 50))) if events is not None else []
    point = "point" if _located(world, obj) else None
    picks = [i for i, ev in enumerate(evs) if ev.type == "PICKED_UP"]
    if picks:
        i = picks[-1]
        ev = evs[i]
        text = f"The {n} {was} picked up at {clock(ev.wall)}, {ago(ev.wall, now)}"
        down = next((d for d in evs[i + 1:] if d.type in PUT_DOWN), None)
        if down is not None:
            ph = "put down" if down.type in ("PUT_BACK", "MOVED") else _event_phrase(down, cfg)
            text += f", and {ph} at {clock(down.wall)}."
        else:
            text += "."
        if len(picks) > 1:
            text += f" That's {len(picks)} times since I started watching."
        return Answer(text, point_at=obj if point else None, action=point)
    moved = next((ev for ev in reversed(evs) if ev.type == "MOVED"), None)
    if moved is not None:
        return Answer(f"The {n} {was} moved at {clock(moved.wall)}, {ago(moved.wall, now)}.",
                      point_at=obj if point else None, action=point)
    return Answer(f"I haven't seen anyone touch the {n} since I started watching.",
                  point_at=obj if point else None, action=point)


def _changes(events, cfg: dict, now: float, since: Optional[float]) -> Answer:
    t0 = since if since is not None else now - CHANGES_WINDOW_S
    evs = events.since(t0) if events is not None else []
    if not evs:
        when = f"since {clock(t0)}" if since is not None else "in the last 10 minutes"
        return Answer(f"Nothing has changed {when}.")
    groups: dict[str, list[Event]] = {}
    for ev in evs:
        groups.setdefault(ev.obj, []).append(ev)
    order = sorted(groups, key=lambda o: groups[o][-1].wall, reverse=True)
    sents = []
    for o in order[:3] if len(order) <= 3 else order[:2]:
        g = groups[o]
        pair = g if len(g) == 1 else [g[0], g[-1]]
        ph = _chain(pair, cfg, now) if len(pair) == 1 else \
            f"{_event_phrase(pair[0], cfg)}, then {_event_phrase(pair[1], cfg)} {ago(pair[1].wall, now)}"
        sents.append(f"The {display_name(cfg, o)} {_be(o, True)} {ph}.")
    if len(order) > 3:
        rest = [display_name(cfg, o) for o in order[2:]]
        lst = ", ".join(rest[:-1]) + f" and {rest[-1]}"
        sents.append(f"The {lst} also changed.")
    return Answer(" ".join(sents))


def answer(intent: Intent, world, events, cfg: Optional[dict] = None,
           since: Optional[float] = None, now: Optional[float] = None) -> Answer:
    """Render the offline answer for an intent. `since` (wall time) scopes CHANGES."""
    cfg = cfg if cfg is not None else load_config()
    now = now if now is not None else time.time()
    k, obj = intent.kind, intent.obj
    if k == "RESET":
        return Answer("Okay, resetting the table.")
    if k == "RECAL":
        return Answer("Recalibrating now.")
    if k == "CHANGES":
        return _changes(events, cfg, now, since)
    if k in ("WHERE", "HISTORY", "HANDLED"):
        if obj is None:
            return _which(cfg)
        try:
            world.get(obj)
        except Exception:
            return Answer(f"I'm not tracking a {display_name(cfg, obj)} right now.")
        if k == "WHERE":
            return _where(obj, world, events, cfg, now)
        if k == "HISTORY":
            return _history(obj, world, cfg, now)
        return _handled(obj, world, events, cfg, now)
    return Answer("I can tell you where things are, what happened to them, or what changed.")
