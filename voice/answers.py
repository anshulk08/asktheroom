"""Offline answer templates (spec V4): Intent + world + event log -> Answer (speech + laser).

Answers are spoken: 1-2 short sentences, no markdown. Pill bottle wording is deliberately neutral:
no template says a pill or medication was taken ('taken out of' is phrased 'lifted out of').

Open world: a question may name a thing taught with 'this is my charger' (or a name not taught yet).
Names resolve through world.find; a thing is spoken by its alias (or as the thing I haven't been
told about), and a thing that left while something similar came back is hedged, never asserted.
TEACH itself is answered by voice/teach.py.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional

from core.config import display_name, load_config
from core.types import Answer, Entity, Event, Intent, Point, Status
from core.viewframe import View
from voice.intents import AWAY, normalize, parse

__all__ = ["Answer", "answer", "ago", "clock", "area"]

PLURAL = {"keys", "glasses", "pills"}
NEAR_CM = 25.0          # 'near the X' only when another visible object is this close
CHANGES_WINDOW_S = 600  # default look-back for 'what changed'
ALSO_NAMED_MAX = 3      # 'what changed': other changed things named, the rest counted
PUT_DOWN = ("PUT_BACK", "MOVED", "PUT_INSIDE", "COVERED", "EXITED_VIEW")
UNNAMED = "thing I haven't been told about"


# ---------- wording helpers ----------

def _plural(obj: str) -> bool:
    return obj in PLURAL or (obj.endswith("s") and not obj.endswith("ss"))


def _be(obj: str, past: bool = False) -> str:
    return ("were" if past else "are") if _plural(obj) else ("was" if past else "is")


def _your(cfg: dict, obj: str) -> str:
    """'Your keys' for the user's things, 'The box' for props."""
    return "Your" if (cfg.get("objects") or {}).get(obj, "target") == "target" else "The"


def _dn(cfg: dict, obj: str) -> str:
    """display_name, except a thing id is never spoken: an unlabelled thing is the UNNAMED phrase."""
    if obj.startswith("thing:") and obj not in (cfg.get("display_names") or {}):
        return UNNAMED
    return display_name(cfg, obj)


def _pk(cfg: dict, obj: str) -> str:
    """The word whose number decides 'is' / 'are': a thing's spoken name ('headphones'), else obj."""
    return _dn(cfg, obj) if obj.startswith("thing:") else obj


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
    """Coarse table region for a table-cm point, from the user's seat: 'at the far left', 'in the middle', ..."""
    return View.from_cfg(cfg).area(pos)


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
    return f"{'under' if under else 'inside'} {_pn(cfg, parent)}"


def _pn(cfg: dict, parent: str) -> str:
    """A parent as spoken after 'inside' / 'put inside': 'the box'; a thing that holds others by its
    taught name ('the toy bin') or automatic guess ('the plastic tub'), else 'a container'."""
    if not parent.startswith("thing:"):
        return f"the {_dn(cfg, parent)}"
    name = (cfg.get("display_names") or {}).get(parent)
    if not name or name == UNNAMED:
        name = (cfg.get("thing_guesses") or {}).get(parent)
    return f"the {name}" if name else "a container"


def _clause(e: Entity, world, cfg: dict) -> Optional[str]:
    """Where-phrase for an entity in a parent chain ('under the notebook'); None if on the table."""
    if e.status in (Status.INSIDE, Status.UNDER):
        return _rel(world, cfg, e.parent, e.status)
    if e.status == Status.HELD:
        return "being held right now"
    if e.status == Status.GONE:
        return f"off {View.from_cfg(cfg).off_table(e.edge)}"
    return None


def _event_phrase(ev: Event, cfg: dict) -> str:
    """Passive participle phrase for an event: 'picked up', 'put inside the box'."""
    p = _dn(cfg, ev.parent) if ev.parent and not ev.parent.startswith("hand") \
        and ev.parent != "unknown" else None
    return {
        "PICKED_UP": "picked up",
        "PUT_BACK": "put back down",
        "MOVED": "moved",
        "COVERED": f"covered by the {p}" if p else "covered up",
        "UNCOVERED": "uncovered",
        "PUT_INSIDE": f"put inside {_pn(cfg, ev.parent)}" if p else "put inside something",
        "TAKEN_OUT": f"lifted out of {_pn(cfg, ev.parent)}" if p else "lifted out of something",
        "EXITED_VIEW": f"carried off {View.from_cfg(cfg).off_table(ev.edge)}",
        "LOST_TRACK": "lost from view",
        "CORRECTED": "given a corrected location",
        "FOUND": "found again",
        "APPEARED": "first seen",
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
    names = [_dn(cfg, o) for o, k in (cfg.get("objects") or {}).items() if k == "target"]
    lst = ", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 1 else "".join(names)
    return Answer(f"Which object do you mean? I can track your {lst}.")


# ---------- per-intent answers ----------

def _where(obj: str, world, events, cfg: dict, now: float) -> Answer:
    """world.place() first (room memory, spec 0009): a room place gets the room templates, anything else
    today's table templates; then one sentence for a conflicting sighting of the object, if any."""
    try:
        place = world.place(obj, now) if hasattr(world, "place") else None
    except Exception:
        place = None
    if place is not None and place.kind == "room":
        ans = _tentative(_where_room(obj, place, cfg, now, world), place, cfg, obj)
    else:
        ans = _where_table(obj, world, events, cfg, now)
    if place is not None and place.conflicts:
        ans = Answer(f"{ans.text} {_conflict_tail(place.conflicts[0], cfg)}", ans.point_at, ans.action)
    return ans


def _where_room(obj: str, place, cfg: dict, now: float, world=None) -> Answer:
    """Spoken room place (spec 0009 section 4). Room answers never point the laser and never say who put
    the thing there: the camera saw it arrive, not whose hand it was."""
    pk = _pk(cfg, obj)
    n, be, It, it, Y = _dn(cfg, obj), _be(pk), _It(pk), _it(pk), _your(cfg, obj)
    say, arrived, last = place.say, place.arrived_wall, place.last_seen_wall
    at = f" at {clock(last)}" if last is not None else ""
    if place.via and place.via != obj:               # seen only through its outermost container
        pn = _pn(cfg, place.via)
        try:
            prep = "under" if world is not None and world.get(obj).status == Status.UNDER else "in"
        except Exception:
            prep = "in"
        if place.fresh:
            return Answer(f"{Y} {n} {be} {prep} {pn}. {pn[:1].upper()}{pn[1:]} is on {say}.")
        return Answer(f"{Y} {n} {be} {prep} {pn}, which I last saw on {say}{at}.")
    if place.absent:
        return Answer(f"I last saw {Y.lower()} {n} on {say}{at}. I can't see {it} there now.")
    if not place.fresh:
        return Answer(f"I last saw {Y.lower()} {n} on {say}{at}.")
    text = f"{Y} {n} {be} on {say}."
    if place.arrival_observed:
        text += f" {It} appeared there {ago(arrived, now)}."
    elif arrived is not None or last is not None:
        text += f" I've seen {it} there since {clock(arrived if arrived is not None else last)}."
    return Answer(text)


def _tentative(ans: Answer, place, cfg: dict, obj: str) -> Answer:
    """A room place reached by a Grok name match, not a known class (spec 0009): never asserted."""
    if not getattr(place, "tentative", False) or "I think" in ans.text:
        return ans
    return Answer(_hedge(ans.text, _dn(cfg, obj)), ans.point_at, ans.action)


def _conflict_tail(c, cfg: dict) -> str:
    """'I also see keys on the bookshelf.': a sighting of the object's class that isn't the object."""
    n = _dn(cfg, c.entity)
    if not _plural(_pk(cfg, c.entity)):
        n = f"{'an' if n[:1] and n[:1] in 'aeiou' else 'a'} {n}"
    return f"I also see {n} on {c.say}."


def _where_table(obj: str, world, events, cfg: dict, now: float) -> Answer:
    e = world.get(obj)
    pk = _pk(cfg, obj)
    n, be, It, Y = _dn(cfg, obj), _be(pk), _It(pk), _your(cfg, obj)
    hedge, plain = float(cfg.get("answer_hedge", 0.5)), float(cfg.get("answer_plain", 0.7))

    if e.status == Status.UNKNOWN and e.pos_cm is None and e.last_seen is None:
        return Answer(f"I haven't seen {Y.lower()} {n} yet. Put {_it(pk)} on the table and I'll keep track.")

    if e.status == Status.UNKNOWN or e.confidence < hedge:
        at = area(e.pos_cm, cfg)
        return Answer(f"I lost track of {Y.lower()} {n}. I last saw {_it(pk)} {at}{',' if ',' in at else ''} "
                      f"{ago(e.last_seen, now)}.", point_at=obj, action="circle")

    prob = " probably" if e.confidence < plain or e.candidates else ""

    if e.status == Status.VISIBLE:
        try:
            holder = world.open_container_of(obj) if hasattr(world, "open_container_of") else None
        except Exception:
            holder = None
        if holder:                          # seen lying in an open box: 'in the box', not 'on the table'
            return Answer(f"{Y} {n} {be}{prob} in {_pn(cfg, holder)}.", point_at=obj, action="point")
        near = _nearest(obj, e, world, cfg)
        tail = f", near the {near}" if near else ""
        return Answer(f"{Y} {n} {be}{prob} on the table{tail}.", point_at=obj, action="point")

    if e.status == Status.HELD:
        pos = e.pos_cm if e.pos_cm is not None else (world.resolve(obj)[0] if _located(world, obj) else None)
        return Answer(f"Someone is{prob} holding {Y.lower()} {n} right now.",
                      point_at=obj if pos is not None else None, action="point" if pos is not None else None)

    if e.status == Status.GONE:
        ev = events.last_of_type(obj, ["EXITED_VIEW"]) if events is not None else None
        side = View.from_cfg(cfg).off_table(e.edge)
        when = ago(ev.wall if ev else e.last_seen, now)
        return Answer(f"{Y} {n} {_be(pk, True)}{prob} carried off {side} {when}.",
                      point_at=obj, action=f"sweep:{e.edge}" if e.edge else "circle")

    # INSIDE / UNDER
    if e.status == Status.UNDER and (e.parent in (None, "unknown")):
        where = "under something near where it was" if not _plural(pk) else "under something near where they were"
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
            text += f" {It} {_be(pk, True)} put there {ago(ev.wall if ev else e.last_seen, now)}."
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
        if str(d.get("name")).startswith("thing:") and not d.get("label"):
            continue                    # an unnamed thing is no landmark
        dist = ((d["pos_cm"][0] - e.pos_cm[0]) ** 2 + (d["pos_cm"][1] - e.pos_cm[1]) ** 2) ** 0.5
        if dist < bd:
            best, bd = d["name"], dist
    return _dn(cfg, best) if best else None


def _history(obj: str, world, cfg: dict, now: float) -> Answer:
    n = _dn(cfg, obj)
    evs = list(reversed(world.history(obj, 3)))
    point = ("point" if _located(world, obj) else None)
    if not evs:
        return Answer(f"I haven't seen anything happen to {_your(cfg, obj).lower()} {n} since I started watching.",
                      point_at=obj if point else None, action=point)
    text = f"{_your(cfg, obj)} {n} {_be(_pk(cfg, obj), True)} {_chain(evs, cfg, now)}."
    try:  # a nice touch: the container/cover it is in moved after that
        parent = world.get(obj).parent
        if parent and parent in (cfg.get("objects") or {}):
            pe = world.history(parent, 1)
            if pe and pe[0].wall > evs[-1].wall and pe[0].type in ("MOVED", "PICKED_UP", "EXITED_VIEW"):
                pn = _dn(cfg, parent)
                text += f" The {pn} {_be(parent, True)} {_event_phrase(pe[0], cfg)} {ago(pe[0].wall, now)}."
    except Exception:
        pass
    return Answer(text, point_at=obj if point else None, action=point)


def _handled(obj: str, world, events, cfg: dict, now: float) -> Answer:
    n, was = _dn(cfg, obj), _be(_pk(cfg, obj), True)
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
        s = f"The {_dn(cfg, o)} {_be(_pk(cfg, o), True)} {ph}."
        if s not in sents:              # two unnamed things first seen together read the same
            sents.append(s)
    if len(order) > 3:
        # A busy room changes hundreds of unnamed things: name a few, count the rest (a spoken list of
        # every one ran to 21,000 characters on the rig and took the app down with it)
        names: list[str] = []
        for o in order[2:]:
            n = _dn(cfg, o)
            if n != UNNAMED and n not in names:
                names.append(n)
        names = names[:ALSO_NAMED_MAX]
        others = len(order) - 2 - len(names)
        if others:
            names.append(f"{others} other thing" + ("s" if others > 1 else ""))
        lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        sents.append(f"{'The ' if names[0][0].isalpha() else ''}{lst} also changed.")
    return Answer(" ".join(sents))


def answer(intent: Intent, world, events, cfg: Optional[dict] = None,
           since: Optional[float] = None, now: Optional[float] = None) -> Answer:
    """Render the offline answer for an intent. `since` (wall time) scopes CHANGES."""
    cfg = cfg if cfg is not None else load_config()
    now = now if now is not None else time.time()
    k = intent.kind
    if k == "RESET":
        return Answer("Okay, resetting the table.")
    if k == "RECAL":
        return Answer("Recalibrating now.")
    if k == "TEACH":
        from voice.teach import teach_answer
        return teach_answer(intent.name or intent.obj, world, cfg)
    if k == "CHANGES":
        if since is None and AWAY.search(normalize(intent.raw or "")):
            return _away(events, _with_things(cfg, world), now)
        return _changes(events, _with_things(cfg, world), now, since)
    if k == "WHAT_DOING":
        return _what_doing(intent, events, _with_things(cfg, world), now)
    if k in ("WHERE", "HISTORY", "HANDLED"):
        obj, guessed = _resolve(intent, world, cfg)
        if obj is None:
            said = intent.name or intent.obj
            narrated = _narrated_about([said], events, now) if said and k != "WHERE" else None
            if narrated is not None:
                return narrated
            if said:
                return Answer(f"I don't know what your {said} is yet. "
                              f"Put it in the teach square and say 'this is my {said}'.")
            return _which(cfg)
        try:
            world.get(obj)
        except Exception:
            return Answer(f"I'm not tracking a {_dn(cfg, obj)} right now.")
        cfg = _with_things(cfg, world)
        spoken = _named(intent, world, cfg)
        if spoken and spoken != obj and spoken in (getattr(world.get(obj), "aliases", None) or []):
            cfg = {**cfg, "display_names": {**(cfg.get("display_names") or {}), obj: spoken}}   # 'brown wallet'
        if guessed:
            return _guessed_answer(k, intent, guessed, world, events, cfg, now)
        if k == "WHERE":
            return _maybe_back(obj, _where(obj, world, events, cfg, now), world, cfg)
        if k == "HISTORY":
            return _plus_narration(_history(obj, world, cfg, now), obj, world, events, cfg, now)
        return _plus_narration(_handled(obj, world, events, cfg, now), obj, world, events, cfg, now)
    return Answer("I can tell you where things are, what happened to them, or what changed.")


# ---------- open world ----------

def _target(intent: Intent, world, cfg: dict) -> Optional[str]:
    """Entity the question is about. A taught alias in the words wins ('where is my phone charger'
    is the charger, not the phone); names outside the config resolve through world.find, and failing
    that through an automatic guess of what an unnamed thing is (core/auto_name.py)."""
    return _resolve(intent, world, cfg)[0]


def _resolve(intent: Intent, world, cfg: dict) -> tuple[Optional[str], list[str]]:
    """(entity, guessed): guessed is empty when a configured name or taught alias decided, else the
    things whose automatic guess fits the spoken name (see _guesses), entity being the first."""
    obj = _named(intent, world, cfg)
    if obj is None or not (intent.name or intent.obj):
        return obj, []
    if obj in (cfg.get("objects") or {}):
        # A configured prop the detector doesn't label from this camera (or labels only now and then)
        # is usually tracked as an unnamed thing that Grok named: while the prop itself is not in view
        # and not placed in a room zone, a fresh Grok-named match answers, hedged. (Rig, Sat 26 Sep: the
        # detector caught the wallet once at 0.9, and "where is my wallet" then said "I lost track" from
        # the prop while the Grok-named wallet sat on the side table.)
        if not _prop_in_view(world, obj):
            guessed = _prop_guesses(obj, intent, world, cfg)
            if guessed and _guess_beats_prop(world, obj, guessed[0]):
                return guessed[0], guessed
        return obj, []
    try:
        hit = world.find(obj) if hasattr(world, "find") else None
    except Exception:
        hit = None
    if hit is not None:
        return hit, []
    guessed = _guesses(obj, world)
    return (guessed[0] if guessed else None), guessed


def _prop_in_view(world, obj: str) -> bool:
    """The prop entity itself is VISIBLE (on the table or in a room zone) or believed hidden on the table
    (UNDER / INSIDE / HELD): then it, not a look-alike thing, is the answer."""
    try:
        e = world.get(obj)
        if e.status in (Status.VISIBLE, Status.UNDER, Status.INSIDE, Status.HELD):
            return True
        place = world.place(obj) if hasattr(world, "place") else None
        return place is not None and place.kind == "room" and not place.absent
    except Exception:
        return True


def _guess_beats_prop(world, obj: str, thing: str) -> bool:
    """A Grok-named thing answers for a prop that is UNKNOWN / GONE only if the thing was seen at least as
    recently as the prop (never seen: always)."""
    try:
        e, t = world.get(obj), world.get(thing)
        return e.last_seen is None or (t.last_seen or 0.0) >= (e.last_seen or 0.0)
    except Exception:
        return False


def _prop_guesses(obj: str, intent: Intent, world, cfg: dict) -> list[str]:
    """Things whose automatic guess fits a configured prop: the words said, then its display name and
    its detector prompts ('remote control', 'tv remote')."""
    said = [intent.name or intent.obj, _dn(cfg, obj)] + list((cfg.get("prompts") or {}).get(obj) or [])
    hits: set[str] = set()
    for words in dict.fromkeys(w for w in said if w):
        try:
            hits |= {n for n, sc in (world.find_guess(words) if hasattr(world, "find_guess") else []) if sc > 0}
        except Exception:
            continue
    if not hits:
        return []

    def fresh(n: str):
        """Seen now first, then most recently: several things can carry the prop's guess (each trip back
        to the table starts a new thing, and a stale room copy lingers until its absence is confirmed);
        the one just seen is the answer, whichever Grok phrase ('tv remote', 'remote control') fits best."""
        try:
            e = world.get(n)
            p = world.place(n) if hasattr(world, "place") else None
            live = e.status == Status.VISIBLE and (p is None or p.kind != "room" or p.fresh)
            return (live, e.last_seen or 0.0)
        except Exception:
            return (False, 0.0)

    return [max(hits, key=fresh)]


def _guesses(said: str, world) -> list[str]:
    """Things whose automatic name guess fits said (world.find_guess, best first). One, or of several
    equally good fits the one in view; else the two best (visible first, then most recently seen):
    the answer names both places."""
    try:
        hits = list(world.find_guess(said)) if hasattr(world, "find_guess") else []
        top = [n for n, s in hits if s == hits[0][1]] if hits else []
        if len(top) > 1:
            vis = [n for n in top if world.get(n).status == Status.VISIBLE]
            if len(vis) == 1:
                return vis
        return [str(n) for n in top[:2]]
    except Exception:
        return []


def _named(intent: Intent, world, cfg: dict) -> Optional[str]:
    obj = intent.obj
    try:
        phrases = world.alias_phrases() if hasattr(world, "alias_phrases") else []
    except Exception:
        phrases = []
    if phrases and intent.raw:
        again = parse(intent.raw, cfg, aliases=phrases)
        if again.obj in phrases:
            obj = again.obj
    return obj or intent.name


def _with_things(cfg: dict, world) -> dict:
    """cfg as the templates see it, with each thing as a target spoken by its alias."""
    try:
        labels = world.thing_labels() if hasattr(world, "thing_labels") else {}
    except Exception:
        labels = {}
    if not labels:
        return cfg
    names = dict(cfg.get("display_names") or {})
    objects = dict(cfg.get("objects") or {})
    for n, label in labels.items():
        names[n], objects[n] = label or UNNAMED, "target"
    return {**cfg, "display_names": names, "objects": objects, "thing_guesses": _thing_guesses(world)}


def _thing_guesses(world) -> dict:
    """Thing -> its automatic name guess (core/auto_name.py adds 'guess' to state_json), for naming a
    thing that holds others: 'inside the plastic tub'."""
    try:
        out = {}
        for e in world.state_json().get("entities") or []:
            g = e.get("guess") if isinstance(e, dict) else None
            if isinstance(g, dict) and g.get("name") and str(e.get("name", "")).startswith("thing:"):
                out[e["name"]] = str(g["name"])
        return out
    except Exception:
        return {}


def _hedge(text: str, n: str) -> str:
    """An answer about a thing found only by its guessed name, hedged: 'Your deodorant, I think, is
    under the notebook.' (a 'probably' in the same place goes: one hedge is enough)."""
    for lead in (f"Your {n} ", f"The {n} "):
        if text.startswith(lead):
            rest = re.sub(r"^(is|are|was|were) probably\b", r"\1", text[len(lead):])
            return f"{lead.rstrip()}, I think, {rest}"
    for phrase in (f"your {n}", f"the {n}"):
        if phrase in text:
            return text.replace(phrase, f"what I think is {phrase}", 1)
    return f"I think {text[:1].lower()}{text[1:]}"


def _place(e: Entity, world, cfg: dict) -> str:
    """Predicate for where an entity is, for listing two places: 'is under the notebook'."""
    if e.status == Status.VISIBLE:
        return f"is {area(e.pos_cm, cfg)}"
    if e.status in (Status.INSIDE, Status.UNDER):
        return f"is {_rel(world, cfg, e.parent, e.status)}"
    if e.status == Status.HELD:
        return "is in someone's hand"
    if e.status == Status.GONE:
        return f"went off {View.from_cfg(cfg).off_table(e.edge)}"
    return f"was last seen {area(e.pos_cm, cfg)}"


def _guessed_answer(k: str, intent: Intent, guessed: list[str], world, events, cfg: dict, now: float) -> Answer:
    """A question answered through automatic name guesses: the thing is spoken as the person named
    it, and the answer hedges. Two equally good fits (neither, or both, in view): both places."""
    from core.things import norm_name
    said = norm_name(intent.name or intent.obj) or "thing"
    cfg = {**cfg, "display_names": {**(cfg.get("display_names") or {}), **{n: said for n in guessed}}}
    obj = guessed[0]
    if len(guessed) > 1 and k == "WHERE":
        try:
            p1, p2 = (_place(world.get(n), world, cfg) for n in guessed[:2])
            return Answer(f"Two things might be your {said}: one {p1}, the other {p2}.",
                          point_at=obj, action="point")
        except Exception:
            pass
    if k == "WHERE":
        ans = _where(obj, world, events, cfg, now)
    elif k == "HISTORY":
        ans = _history(obj, world, cfg, now)
    else:
        ans = _handled(obj, world, events, cfg, now)
    text = ans.text if "I think" in ans.text else _hedge(ans.text, said)   # a tentative room place hedged already
    return Answer(text, ans.point_at, ans.action)


def _maybe_back(obj: str, ans: Answer, world, cfg: dict) -> Answer:
    """A thing that left or was lost, while something that may be it came back: say so, hedged, and
    point at the look-alike. Identity is never asserted (see core/things.py)."""
    try:
        e0 = world.get(obj)
        if e0.status not in (Status.GONE, Status.UNKNOWN) or not hasattr(world, "similar_to"):
            return ans
        if getattr(e0, "zone", "table") != "table":    # a room object (spec 0009): its answer is the room one
            return ans
        for other, _ in world.similar_to(obj):
            e = world.get(other)
            if e.status in (Status.GONE, Status.UNKNOWN):
                continue
            where = f"on the table, {area(e.pos_cm, cfg)}" if e.status == Status.VISIBLE \
                else (_clause(e, world, cfg) or "on the table")
            return Answer(f"{ans.text} Something similar came back and is {where}; it might be yours.",
                          point_at=other, action="point")
    except Exception:
        pass
    return ans


# ---------- narration memory (core/narration.py) ----------
#
# Narrations say what the person was doing; the world model stays the authority on where things are.
# Every narration read here passes the medication rule again (redact_meds), whatever wrote it.

NARR_HEDGE = 0.5        # below this confidence: 'it looked like you ...'
NARR_MIN = 0.2          # below this a narration is not used at all
RECENT_S = 3 * 3600     # 'what was I doing?' with no time: the latest narration within this
AWAY_MIN_S = 120        # a question asked less than this ago doesn't mark when you left
AWAY_GAP_S = 900        # else: the last quiet stretch this long does


def _store(events):
    from core.narration_store import store_for
    return store_for(events, create=False)


def _usable(r) -> bool:
    s = (r.summary or "").strip().lower().rstrip(".")
    return bool(s) and s != "unclear" and r.confidence >= NARR_MIN


def _third_person(text: str) -> str:
    """'You put your keys in the box' -> 'Someone put your keys in the box': while you were away,
    the hands on the table weren't yours (the camera can't tell whose they were)."""
    for a, b in ((r"\bYou were\b", "Someone was"), (r"\bYou have\b", "Someone has"), (r"\bYou\b", "Someone"),
                 (r"\byou were\b", "they were"), (r"\byourself\b", "themselves"), (r"\byou\b", "they")):
        text = re.sub(a, b, text)
    return text


def _narr_text(r, now: float, third: bool = False, first_only: bool = False, terms=()) -> str:
    """'At 2:05 PM, you sorted some papers.' (hedged when unsure; the sentence naming a term first)."""
    from core.narration_store import _pattern, redact_meds
    text, _ = redact_meds(r.summary or "")
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if not sents:
        return ""
    if first_only:
        rx = _pattern(list(terms)) if terms else None
        sents = [next((s for s in sents if rx and rx.search(s)), sents[0])]
    text = " ".join(sents)
    text = _third_person(text) if third else text
    text = text[0].lower() + text[1:]
    if r.confidence < NARR_HEDGE:
        text = "it looked like " + text
    when = clock(r.t_start)
    d, today = datetime.fromtimestamp(r.t_start).date(), datetime.fromtimestamp(now).date()
    if d != today:
        when += " yesterday" if (today - d).days == 1 else f" on {datetime.fromtimestamp(r.t_start):%A}"
    text = f"At {when}, {text}"
    return text if text.endswith((".", "!", "?")) else text + "."


def _narrations_text(rows, now: float, third: bool = False) -> str:
    """At most two narrations (the latest), one sentence each when there are two, plus a count of the rest."""
    shown = rows[-2:]
    text = " ".join(_narr_text(r, now, third, first_only=len(shown) > 1) for r in shown)
    if len(rows) > 2:
        n = len(rows) - 2
        text += f" I saw {n} more {'bit' if n == 1 else 'bits'} of activity then."
    return text


def _events_text(evs: list[Event], cfg: dict) -> str:
    """'the keys were picked up at 2:02 PM and put inside the box' for up to two objects."""
    groups: dict[str, list[Event]] = {}
    for ev in sorted(evs, key=lambda e: e.wall):
        groups.setdefault(ev.obj, []).append(ev)
    parts = []
    for o in sorted(groups, key=lambda o: groups[o][-1].wall)[-2:]:
        g = groups[o]
        s = f"the {_dn(cfg, o)} {_be(_pk(cfg, o), True)} {_event_phrase(g[0], cfg)} at {clock(g[0].wall)}"
        if len(g) > 1:
            s += f" and {_event_phrase(g[-1], cfg)}"
            if clock(g[-1].wall) != clock(g[0].wall):
                s += f" at {clock(g[-1].wall)}"
        parts.append(s)
    return ", and ".join(parts)


def _what_doing(intent: Intent, events, cfg: dict, now: float) -> Answer:
    """Narrations in the question's time window (or the latest one), else the event log for it."""
    from core.narration_store import parse_window
    w = parse_window(intent.raw or "", now)
    st = _store(events)
    if w is None:
        rows = [r for r in (st.latest(5, before=now) if st else []) if _usable(r) and r.t_end >= now - RECENT_S]
        if rows:
            return Answer(_narr_text(rows[0], now))
        t0, t1, none = now - CHANGES_WINDOW_S, now, "I haven't seen anything happen on the table lately."
    else:
        rows = [r for r in (st.between(w.t0, w.t1) if st else []) if _usable(r)]
        if rows:
            return Answer(_narrations_text(rows, now))
        t0, t1, none = w.t0, w.t1, f"I didn't see anything happen on the table {w.label}."
    evs = [e for e in (events.since(t0) if events is not None else []) if e.wall <= t1]
    if evs:
        return Answer(f"I don't have a description of that, but {_events_text(evs, cfg)}.")
    return Answer(none)


def _away_since(events, now: float) -> Optional[float]:
    """When 'while I was away' began: the last question asked (at least AWAY_MIN_S ago), else the start
    of the activity after the last quiet stretch of AWAY_GAP_S, or of that stretch if it runs to now."""
    try:
        q = events._rows("SELECT t FROM questions WHERE t <= ? ORDER BY t DESC LIMIT 1", (now - AWAY_MIN_S,))
        if q:
            return float(q[0][0])
        times = sorted(e.wall for e in events.since(now - 86400) if e.wall <= now)
    except Exception:
        return None
    st = _store(events)
    if st:
        times = sorted(times + [x for r in st.between(now - 86400, now) for x in (r.t_start, r.t_end)])
    if not times:
        return None
    if now - times[-1] >= AWAY_GAP_S:
        return times[-1] + 1
    for a, b in reversed(list(zip(times, times[1:]))):
        if b - a >= AWAY_GAP_S:
            return b - 1
    return None


def _away(events, cfg: dict, now: float) -> Answer:
    """'What happened while I was away?': narrations since then in third person, else the event changes."""
    since = _away_since(events, now)
    st = _store(events)
    if since is not None and st:
        rows = [r for r in st.between(since, now) if _usable(r) and r.t_end >= since]
        if rows:
            return Answer(_narrations_text(rows, now, third=True))
    return _changes(events, cfg, now, since)


def _terms(obj: str, world, cfg: dict) -> list[str]:
    """Every name an entity goes by: spoken name, id, synonyms, taught aliases."""
    out = [_dn(cfg, obj)]
    if not obj.startswith("thing:"):
        out.append(obj.replace("_", " "))
    out += [k for k, v in (cfg.get("synonyms") or {}).items() if v == obj and k not in ("the box", "container")]
    try:
        out += list(world.get(obj).aliases or [])
    except Exception:
        pass
    return [t for t in dict.fromkeys(out) if t and t != UNNAMED]


def _plus_narration(ans: Answer, obj: str, world, events, cfg: dict, now: float) -> Answer:
    """An object answer plus the latest narration of the last day that mentions the object (by any of
    its names), as one more sentence, when the answer has room (3 sentences at most)."""
    st = _store(events)
    if st is None or len(re.split(r"(?<=[.!?])\s+", ans.text.strip())) >= 3:
        return ans
    terms = _terms(obj, world, cfg)
    rows = [r for r in st.search(terms, since=now - 86400, until=now, limit=3) if _usable(r)]
    s = _narr_text(rows[0], now, first_only=True, terms=terms) if rows else ""
    return Answer(f"{ans.text} {s}", ans.point_at, ans.action) if s else ans


def _narrated_about(words: list[str], events, now: float) -> Optional[Answer]:
    """A thing the world model doesn't know ('did I use the stove?'), found in narrations."""
    st = _store(events)
    rows = [r for r in (st.search(words, since=now - 86400, until=now, limit=3) if st else []) if _usable(r)]
    return Answer(_narr_text(rows[0], now, first_only=True, terms=words)) if rows else None
