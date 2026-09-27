"""Shared data contracts (spec section 2). Agreed before module code; change only as a team.

Positions are table centimetres (origin at ArUco marker 0, x right, y down) unless a field says px.
Must stay Python 3.10 compatible (JetPack 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

Point = tuple[float, float]
PointCm = Point
BoxPx = tuple[int, int, int, int]            # x1, y1, x2, y2 pixels
BoxCm = tuple[float, float, float, float]    # x1, y1, x2, y2 table cm


@dataclass
class Frame:
    t: float                 # time.monotonic()
    wall: float              # time.time(), for display
    img: Optional[np.ndarray]  # BGR 1280x720; None in synthetic tests
    idx: int


@dataclass
class Detection:
    cls: str                 # one of config.objects + 'hand' (tracked hands: 'hand:3')
    conf: float
    box_px: tuple[int, int, int, int]            # x1, y1, x2, y2
    center_cm: Point
    box_cm: tuple[float, float, float, float]    # x1, y1, x2, y2
    # A 'thing' proposal lying mostly inside a person box (core/proposals.py): it may be a finger or a
    # carried object, so it never starts a new thing but may be an existing one. Says nothing of contact.
    occluded: bool = False


@dataclass
class Detections:
    t: float
    frame_idx: int
    items: list[Detection] = field(default_factory=list)
    hands: list[Detection] = field(default_factory=list)   # with track ids in cls, e.g. 'hand:3'


class Status(str, Enum):
    # str() and f-strings give the bare value on every Python version (3.12 changed the default)
    def __str__(self) -> str:
        return self.value

    VISIBLE = "VISIBLE"
    HELD = "HELD"
    UNDER = "UNDER"
    INSIDE = "INSIDE"
    GONE = "GONE"
    UNKNOWN = "UNKNOWN"


class EventType(str, Enum):
    def __str__(self) -> str:
        return self.value

    PICKED_UP = "PICKED_UP"
    PUT_BACK = "PUT_BACK"
    MOVED = "MOVED"
    COVERED = "COVERED"
    UNCOVERED = "UNCOVERED"
    PUT_INSIDE = "PUT_INSIDE"
    TAKEN_OUT = "TAKEN_OUT"
    EXITED_VIEW = "EXITED_VIEW"
    LOST_TRACK = "LOST_TRACK"
    CORRECTED = "CORRECTED"
    FOUND = "FOUND"          # stretch: search camera found a lost object
    APPEARED = "APPEARED"    # open world: a new thing:N was confirmed from class-agnostic proposals


EVENT_TYPES = [e.value for e in EventType]


@dataclass
class Entity:
    name: str
    kind: str                            # 'target' | 'container' | 'cover'
    status: Status = Status.UNKNOWN
    parent: Optional[str] = None         # entity name, 'hand:3', or 'unknown'
    pos_cm: Optional[Point] = None       # last observed centre
    box_cm: Optional[BoxCm] = None       # last observed box
    last_seen: Optional[float] = None    # wall time
    confidence: float = 1.0              # heuristic 0-1, not a probability
    candidates: list[str] = field(default_factory=list)  # alternative parents when ambiguous
    edge: Optional[str] = None           # 'left'|'right'|'top'|'bottom' when GONE
    pre_pickup_pos: Optional[Point] = None
    zone: str = "table"                  # 'table' or a floor/room surface name (stretch)
    held_since: Optional[float] = None   # time.monotonic() when the current HELD began
    # Open world (things only; empty for the configured objects):
    aliases: list[str] = field(default_factory=list)          # taught names, newest first
    maybe_same_as: list[tuple[str, float]] = field(default_factory=list)  # (earlier thing, look score)
    merged_into: Optional[str] = None    # set when this identity was folded into another one


@dataclass
class Event:
    t: float                             # time.monotonic()
    wall: float                          # time.time()
    obj: str
    type: str                            # an EventType member or its value
    from_cm: Optional[Point] = None
    to_cm: Optional[Point] = None
    parent: Optional[str] = None
    edge: Optional[str] = None
    confidence: float = 1.0
    snapshot: Optional[str] = None       # path to jpg


INTENT_KINDS = ["WHERE", "HISTORY", "HANDLED", "CHANGES", "RESET", "RECAL", "TEACH", "OTHER"]


@dataclass
class Intent:
    kind: str                            # one of INTENT_KINDS
    obj: Optional[str]                   # canonical object name, after synonyms (or a taught alias)
    raw: str                             # the question as heard
    name: Optional[str] = None           # spoken name not in the config ('charger'); TEACH: the new name


@dataclass
class Answer:
    text: str                            # what to speak
    point_at: Optional[str] = None       # entity name for the laser
    action: Optional[str] = None         # 'point' | 'sweep:left' | 'circle' | None
    target_cm: Optional[Point] = None    # a raw table position to point at when no entity fits (visual Q&A)
    # the proof pictures behind the answer (core/evidence.py): [{kind, snapshot_url, t, caption, box, ...}];
    # not part of equality, so an answer is the same answer with or without its pictures
    evidence: list = field(default_factory=list, compare=False)
    obj: Optional[str] = field(default=None, compare=False)   # the entity the answer is about, room ones included


def entity_json(e: Entity, resolved_cm: Optional[Point]) -> dict:
    """One entry of WorldState JSON 'entities'."""
    return {
        "name": e.name,
        "kind": e.kind,
        "status": e.status.value,
        "parent": e.parent,
        "pos_cm": list(e.pos_cm) if e.pos_cm else None,
        "resolved_cm": list(resolved_cm) if resolved_cm else None,
        "confidence": round(e.confidence, 3),
        "candidates": list(e.candidates),
        "edge": e.edge,
        "zone": e.zone,
        "last_seen": e.last_seen,
    }


# WorldState JSON (pushed to the dashboard at 5 Hz; embedded in the Grok prompt by voice/llm.py):
# {
#   "t": 1727290000.1, "online": true, "fps": 12.4,
#   "entities": [{"name": "keys", "kind": "target", "status": "INSIDE", "parent": "box",
#                 "pos_cm": [41.2, 29.0], "resolved_cm": [70.4, 38.1], "confidence": 0.85,
#                 "candidates": [], "edge": null, "zone": "table", "last_seen": 1727289980.4}],
#   "edges": [["keys", "INSIDE", "box"], ["box", "ON", "table"]],
#   "laser": {"on": true, "target": "box", "err_cm": 0.8}
# }
