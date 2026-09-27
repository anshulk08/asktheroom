"""Where a thing on the table is, said by what is around it and from the user's seat: "just left of the
laptop", "in front of the pill bottle, nearer you", "between the laptop and the water bottle".

Landmarks are the other VISIBLE things on the table a listener can find at a glance: configured props,
things with a taught name, and things with a confident automatic guess (core/auto_name.py, GUESS_MIN; in
room mode nearly everything is a guess, and before this only taught names counted, so "where are the
batteries?" got a bare "on the table"). Bigger and easier-to-see objects win over small ones: a landmark's
score is its gap to the thing (edge to centre, cm) less a bonus for its footprint (box_cm) and for common
big objects (SALIENT), and only landmarks within LANDMARK_CM are used.

Directions are the user's (core/viewframe.View, viewer.front): the thing's position and the landmark's
box are turned into the viewer frame, where x runs to the user's right and y towards them. Positions stay
in the camera frame everywhere else. No landmark: the caller says the table area (View.area) or asks
Grok to describe the spot (voice/visual.VisualQA.describe_where), within a short deadline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.types import Status
from core.viewframe import View

LANDMARK_CM = 40.0      # a landmark whose edge is this close to the thing's centre is used
JUST_CM = 12.0          # 'just left of' closer than this
BETWEEN_CM = 30.0       # 'between A and B': both this close, on opposite sides
DEFAULT_AREA_CM2 = 40.0  # a landmark with no box: about a pill bottle's footprint
SIZE_WEIGHT = 10.0      # cm of gap a landmark e times bigger makes up for
SALIENT_CM = 6.0        # ... and this many for a common big, easily seen object
SALIENT = {"laptop", "notebook", "book", "box", "bottle", "water bottle", "can", "remote", "mug", "cup",
           "bowl", "plate", "phone", "tablet", "keyboard", "bag", "tub", "jar", "lamp", "monitor", "speaker",
           "tissue box", "pill bottle", "glass", "vase", "tray", "wallet"}
FLAT = {"laptop", "notebook", "book", "tray", "plate", "placemat", "paper", "papers", "magazine", "mat",
        "cutting board", "keyboard", "folder"}


@dataclass
class Landmark:
    name: str                           # entity
    said: str                           # 'the laptop'
    box: tuple                          # viewer-frame (x0, y0, x1, y1) cm
    centre: tuple                       # viewer-frame cm
    gap: float                          # cm from the thing's centre to the box
    score: float                        # lower is better


def spoken_name(name: str, cfg: dict) -> Optional[str]:
    """What a listener calls a landmark: a prop's or taught thing's name, else a thing's confident guess,
    else None (a new thing is no landmark)."""
    from voice.answers import UNNAMED, _dn
    if name.startswith("thing:"):
        label = (cfg.get("display_names") or {}).get(name, UNNAMED)
        if label and label != UNNAMED:
            return label
        return (cfg.get("thing_guesses") or {}).get(name)
    return _dn(cfg, name)


def _salient(said: str) -> bool:
    w = said.lower()
    return w in SALIENT or any(w.endswith(" " + s) for s in SALIENT)


def _view_box(view: View, pos, box_cm) -> tuple:
    """A camera-frame cm box (or just a centre) as an axis-aligned viewer-frame box."""
    if box_cm is None:
        x, y = view.to_view(pos)
        return (x, y, x, y)
    x1, y1, x2, y2 = box_cm
    pts = [view.to_view(p) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _gap(p, box) -> tuple[float, float]:
    """(gx, gy): the thing's centre p minus the nearest point of box (0, 0 inside it)."""
    return p[0] - min(max(p[0], box[0]), box[2]), p[1] - min(max(p[1], box[1]), box[3])


def candidates(obj: str, pos, world, cfg: dict, view: Optional[View] = None) -> list[Landmark]:
    """Landmarks for the thing obj at camera-frame pos, best first (only those within LANDMARK_CM)."""
    view = view or View.from_cfg(cfg)
    try:
        ents = world.state_json().get("entities", [])
    except Exception:
        return []
    own = spoken_name(obj, cfg)
    p = view.to_view(pos)
    out = []
    for d in ents:
        n = d.get("name")
        if not n or n == obj or d.get("status") != Status.VISIBLE.value or not d.get("pos_cm"):
            continue
        if (d.get("zone") or "table") != "table":
            continue
        said = spoken_name(n, cfg)
        if not said or (own and said.lower() == own.lower()):
            continue                             # 'the mug near the mug' helps nobody
        try:
            box_cm = world.get(n).box_cm
        except Exception:
            box_cm = None
        vb = _view_box(view, d["pos_cm"], box_cm)
        gx, gy = _gap(p, vb)
        gap = math.hypot(gx, gy)
        if gap > LANDMARK_CM:
            continue
        area = (box_cm[2] - box_cm[0]) * (box_cm[3] - box_cm[1]) if box_cm is not None else DEFAULT_AREA_CM2
        score = gap - SIZE_WEIGHT * math.log(max(area, 10.0) / DEFAULT_AREA_CM2) - (SALIENT_CM if _salient(said) else 0.0)
        out.append(Landmark(n, f"the {said}", vb, ((vb[0] + vb[2]) / 2, (vb[1] + vb[3]) / 2), gap, score))
    return sorted(out, key=lambda lm: lm.score)


def relation(p, lm: Landmark) -> str:
    """Where viewer-frame point p is from landmark lm: 'just left of the laptop', 'on the notebook'."""
    gx, gy = _gap(p, lm.box)
    if gx == 0 and gy == 0:
        return f"on {lm.said}" if lm.said[4:].lower() in FLAT else f"right by {lm.said}"
    just = "just " if math.hypot(gx, gy) < JUST_CM else ""
    if abs(gx) >= abs(gy):
        return f"{just}{'left' if gx < 0 else 'right'} of {lm.said}"
    if gy > 0:
        return f"{just}in front of {lm.said}, nearer you"
    return f"{just}behind {lm.said}"


def _between(p, lms: list[Landmark]) -> Optional[str]:
    """'between the laptop and the water bottle': two close landmarks on opposite sides of p."""
    close = [lm for lm in lms if lm.gap <= BETWEEN_CM][:4]
    for i, a in enumerate(close):
        for b in close[i + 1:]:
            va = (a.centre[0] - p[0], a.centre[1] - p[1])
            vb = (b.centre[0] - p[0], b.centre[1] - p[1])
            na, nb = math.hypot(*va), math.hypot(*vb)
            if na and nb and (va[0] * vb[0] + va[1] * vb[1]) / (na * nb) < -0.5:     # more than 120 deg apart
                return f"between {a.said} and {b.said}"
    return None


def phrase(obj: str, pos, world, cfg: dict) -> Optional[str]:
    """Where the thing is by its landmarks ('just left of the laptop'), or None when none is close."""
    if pos is None:
        return None
    view = View.from_cfg(cfg)
    lms = candidates(obj, pos, world, cfg, view)
    if not lms:
        return None
    p = view.to_view(pos)
    if lms[0].gap == 0:                          # on or against the best one: that says it
        return relation(p, lms[0])
    return _between(p, lms) or relation(p, lms[0])
