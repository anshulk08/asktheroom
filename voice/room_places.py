"""'Point to the couch': the laser shows a drawn room zone itself (room memory's zones: couch, side table, kitchen
counter). Only a question that asks to point (voice.intents.asks_to_point) and whose whole target is the zone's
name or spoken name ('the couch', 'the side table', 'the kitchen', 'the kitchen counter') is answered here,
"That's the couch.", aimed at the zone polygon's centre through the room aim path (Answer.action 'room:u,v,x1,y1,
x2,y2' -> Room.aim -> _aim_room_px: zone, people gate, closed loop, dwell, laser_locked). 'Point to the remote on
the couch' names a thing, not a zone, and goes on to the thing paths. 'The table' is left to the table paths."""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from core.types import Answer

log = logging.getLogger(__name__)

# What follows the point cue (voice.intents.POINT_CUE, on normalize()d text) is the target.
TARGET = re.compile(r"\b(?:(?:point(?:ing)?|quite|pint|joint|paint)\s+(?:it\s+)?(?:to|at|towards?)|point(?:ing)?\s+out"
                    r"|show\s+me)\s+(.+)$")
FILLER = {'my', 'the', 'a', 'an', 'our', 'your', 'that', 'this', 'where', 'wheres', 'is', 'please', 'now',
          'for', 'me', 'us', 'area', 'spot', 'over', 'there'}
ALIASES = {'sofa': 'couch', 'settee': 'couch'}
NOT_ZONES = {frozenset({'table'})}       # the table has its own paths (the laser fit, the table view)


def _content(s: str) -> set[str]:
    words = re.sub(r"[^a-z0-9\s]", " ", s.lower().replace('_', ' ')).split()
    return {ALIASES.get(w, w) for w in words if w not in FILLER}


def _target_words(text: str) -> Optional[set[str]]:
    from voice.intents import normalize
    m = TARGET.search(normalize(text or ""))
    if m is None:
        return None
    return _content(m.group(1)) or None


def _centre(poly) -> tuple[float, float]:
    """The polygon's area centroid, or, when that falls outside (a concave outline), the point deepest inside it."""
    import cv2
    import numpy as np
    P = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    m = cv2.moments(P)
    if abs(m["m00"]) > 1e-6:
        c = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        if cv2.pointPolygonTest(P, c, False) >= 0:
            return float(c[0]), float(c[1])
    lo = P.reshape(-1, 2).min(axis=0)
    s = 200.0 / max(1.0, float((P.reshape(-1, 2).max(axis=0) - lo).max()))    # rasterised ~200 px across
    mask = np.zeros((202, 202), np.uint8)
    cv2.fillPoly(mask, [np.round((P - lo) * s).astype(np.int32)], 255)
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(x / s + lo[0]), float(y / s + lo[1])


def pick_zone(text: str, zones: Iterable) -> Optional[tuple]:
    """The (name, say, poly) zone a point question names as its whole target, else None. Every word said must be
    in the zone's name or spoken name; of several such zones, the one whose name is said in full; a tie is None."""
    said = _target_words(text)
    if not said or frozenset(said) in NOT_ZONES:
        return None
    fits = [z for z in zones if len(z[2]) >= 3 and said <= (_content(z[0]) | _content(z[1]))]
    if len(fits) > 1:
        fits = [z for z in fits if _content(z[0]) <= said or _content(z[1]) <= said] or fits
    return fits[0] if len(fits) == 1 else None


def answer_for_zone(text: str, zones: Iterable) -> Optional[Answer]:
    """"That's the couch." aimed at the couch zone, for 'point to the couch'; None when the question doesn't ask
    to point or its target isn't a zone."""
    from voice.intents import asks_to_point
    if not asks_to_point(text):
        return None
    z = pick_zone(text, list(zones))
    if z is None:
        return None
    name, say, poly = z
    u, v = _centre(poly)
    xs, ys = [float(p[0]) for p in poly], [float(p[1]) for p in poly]
    log.info("zone %s answers the point question %r", name, text)
    return Answer(f"That's {say}.",
                  action=f"room:{u:.0f},{v:.0f},{min(xs):.0f},{min(ys):.0f},{max(xs):.0f},{max(ys):.0f}")
