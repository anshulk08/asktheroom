"""WHERE from the room tracker's own tracks (room.aim_tracks, off by default). Room memory confirms and
Grok-names tracks on the couch, side table or counter ('room track r:387 in couch looks like a laptop'), but a
track becomes a world thing only through a table handoff, so "where's my laptop?" got a spoken room look and
no aim. When the world has no place for the asked name, a confirmed, fresh, named track that fits it answers
"Your laptop is on the couch." and points there through the room aim path (Answer.action 'room:u,v,x1,y1,
x2,y2' -> Room.aim -> _aim_room_px: zone, people gate, closed loop, dwell, laser_locked).

Never: a track named like a body part or a person (core/proposals PEOPLE: a head, an arm, a leg), a track that
is a tentatively handed-over thing's (the world answers for that thing), a track not matched in the last
aim_track_fresh_s or missed on its zone's latest visit. Worn things (a shoe, a sneaker) are valid targets:
whether someone is wearing it is the aim gate's call (main.Room._unsafe / _blockers: footwear touching a
person box blocks, and a person near the target blocks by the gate's margin). Glasses answer to eyeglasses, sunglasses, spectacles and the like."""
from __future__ import annotations

import logging
import re
import time
from typing import Callable, Iterable, Optional

from core.auto_name import match_score
from core.proposals import PEOPLE
from core.types import Answer, Status

log = logging.getLogger(__name__)

MATCH_MIN = 2.0           # core.auto_name.match_score: the head noun shared (room_memory name_match_min)
NOT_OBJECTS = set(PEOPLE) | {'feet', 'legs', 'face', 'torso'}
SYNONYMS = {'glasses': ('eyeglasses', 'eye glasses', 'sunglasses', 'reading glasses', 'spectacles',
                        'pair of glasses', 'specs'),
            'shoe': ('sneaker', 'trainer', 'footwear', 'boot', 'loafer', 'sandal', 'slipper'),
            'sneaker': ('shoe', 'trainer', 'footwear')}


def _words(s: str) -> set[str]:
    return set(re.findall(r'[a-z]+', (s or '').lower()))


def with_synonyms(guess: dict) -> dict:
    """guess, with the plain word for a synonym it uses ('eyeglasses' also answers to 'glasses')."""
    phrases = [str(p).lower() for p in [guess.get('name') or ''] + list(guess.get('also') or [])]
    extra = [k for k, alts in SYNONYMS.items() if any(a in p for p in phrases for a in alts) and k not in phrases]
    return {**guess, 'also': list(guess.get('also') or []) + extra} if extra else guess


def worn_or_person(guess: dict) -> bool:
    """A guess naming a person or a body part, by any of its phrases ('left leg', 'hand')."""
    phrases = [guess.get('name') or ''] + list(guess.get('also') or [])
    for p in phrases:
        p = str(p).lower()
        if p in NOT_OBJECTS or any(w in NOT_OBJECTS or w.rstrip('s') in NOT_OBJECTS for w in _words(p)):
            return True
    return False


def world_has_place(world, target: Optional[str]) -> bool:
    """The world can answer for target itself: it is seen, held, hidden in or under something, or already
    placed in a room zone. A configured prop the corner camera never labels (UNKNOWN on the table) is not."""
    if target is None:
        return False
    try:
        e = world.get(target)
    except Exception:
        return False
    if e is None:
        return False
    return e.status not in (Status.UNKNOWN, Status.GONE) or getattr(e, 'zone', 'table') != 'table'


def pick_track(said: str, tracks: Iterable, now_wall: float, fresh_s: float,
               tentative: Callable[[str], bool] = lambda n: False):
    """The best confirmed track named like `said`, fresh and not worn: (track, score) or None."""
    best = None
    for tr in tracks:
        g = getattr(tr, 'guess', None)
        if not tr.confirmed or not isinstance(g, dict) or not g.get('name') or tr.misses > 0:
            continue
        if now_wall - tr.last_wall > fresh_s or worn_or_person(g):
            continue
        if tr.entity is not None and tentative(tr.entity):
            continue
        sc = match_score(said, with_synonyms(g))
        if sc >= MATCH_MIN and (best is None or (sc, tr.last_wall) > (best[1], best[0].last_wall)):
            best = (tr, sc)
    return best


def answer_from_tracks(said: str, tracks: Iterable, zone_say: dict, now_wall: Optional[float] = None,
                       fresh_s: float = 5.0, tentative: Callable[[str], bool] = lambda n: False
                       ) -> Optional[Answer]:
    """'Your laptop, I think, is on the couch.' with the room aim at the track's full-frame box, or None."""
    now_wall = time.time() if now_wall is None else now_wall
    hit = pick_track(said, tracks, now_wall, fresh_s, tentative)
    if hit is None:
        return None
    tr = hit[0]
    x1, y1, x2, y2 = (float(v) for v in tr.box_px)
    where = zone_say.get(tr.zone) or f'the {tr.zone.replace("_", " ")}'
    verb = 'are' if said.endswith('s') and not said.endswith('ss') else 'is'
    log.info("room track %s in %s (%s) answers for %r", tr.tid, tr.zone, tr.guess.get('name'), said)
    from voice.answers import _hedge      # a Grok-name match, not an identity (spec 0010): hedged
    return Answer(_hedge(f"Your {said} {verb} on {where}.", said),
                  action=f"room:{(x1 + x2) / 2:.0f},{(y1 + y2) / 2:.0f},{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")
