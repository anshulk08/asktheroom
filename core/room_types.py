"""Room memory contracts (spec 0009, M0). Shared by core/room_zones.py (zones), core/room_view.py (table
view over the full camera frame), core/room.py (tracker, per-frame driver, CLI), core/room_world.py (the
World mixin: association, absence, return, place()) and voice/answers.py (room templates).

Room positions are full-frame pixels (the 1920x1080 camera frame), never table cm. Times are the same
clocks as the table pipeline: `t` is time.monotonic() (Frame.t), `wall` is time.time().
Python 3.10 (JetPack 6): no match statements, no 3.11+ stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

from core.types import BoxPx, Point, Status

TABLE = "table"


@dataclass
class RoomConfig:
    """config.yaml `room_memory:` (new section at the end). Every key optional; these are the defaults."""
    enabled: bool = False
    capture_size: tuple[int, int] = (1920, 1080)   # full camera frame, px
    zoom: int = 100                                # Brio zoom_absolute the room runs at
    ref_zoom: int = 160                            # zoom the table pipeline was tuned at (table view)
    table_view_rect: Optional[tuple[int, int, int, int]] = None   # full-frame px; per rig, config.local.yaml
    zones_path: str = "room_zones.json"
    ring_s: float = 1.0                            # FrameBuffer ring when room memory is on (1080p frames)
    room_every_n: int = 5                          # process one zone every N perception frames
    confirm_visits: int = 2                        # consecutive matched visits to confirm a track
    handoff_s: float = 120.0                       # a table departure authorises one acquisition this long
    table_fresh_s: float = 2.0                     # a table-VISIBLE prop seen this recently is on the table
    room_prop_conf: float = 0.45                   # class score a room detection needs
    absent_visits: int = 3                         # valid empty visits before UNKNOWN ...
    absent_min_s: float = 3.0                      # ... and at least this long since the last match
    fresh_visits: int = 2                          # present tense needs fewer valid misses than this ...
    fresh_s: float = 10.0                          # ... and a match this recent
    stale_min_s: float = 2.0                       # fresh_visits misses only count once this long unmatched
    lum_lo: int = 25                               # a box darker or brighter than this: not a valid visit
    lum_hi: int = 235
    hand_iou: float = 0.6                          # a prop box this much on a hand box is the hand
    blocker_overlap: float = 0.3                   # a blocker covering this much of a track box: invalid visit
    change_thr: int = 40                           # grey-level change between visits counted as change
    change_area_ratio: float = 1.5                 # a change blob this much bigger than the track box blocks it
    max_crop_px: int = 1280                        # zone crops with a longer side are resized down to this
    things: bool = True                            # also track unnamed objects (YOLOE) in zones; handoff by Grok name
    thing_name_wait_s: float = 45.0                # a confirmed thing track waits this long for its Grok answer
    name_match_min: float = 2.0                    # core.auto_name.match_score two guesses need (head noun shared)
    names_per_minute: int = 60                     # Grok calls for room crops, at most (only while a handoff is possible)
    room_every_n_hot: int = 2                      # ... every N frames instead while a handoff is hot (spec 0010 P0-3)
    hot_max_s: float = 30.0                        # a departure keeps the room hot (fast cadence) this long; the handoff
                                                   # itself stays open for handoff_s at the normal cadence
    handoff_min_dwell_s: float = 4.0               # a thing must have sat on the table this long before leaving to count
                                                   # as carried off (a foot at the table edge appears and vanishes in seconds)
    ignore_names: tuple = ("sock", "sneaker", "shoe", "foot", "feet", "hand", "arm", "sleeve", "leg", "knee",
                           "shirt", "jeans", "pants", "shorts", "fabric", "cloth", "person")
                                                   # a departed thing Grok named like this is a body part or clothing at
                                                   # the table edge, never carried off: no handoff, no hot mode
    confirm_visits_arrival: int = 1                # visits to confirm a track whose spot changed (an arrival)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RoomConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        for k in ("capture_size", "table_view_rect"):
            if kw.get(k) is not None:
                kw[k] = tuple(int(v) for v in kw[k])
        return cls(**kw)


@dataclass
class RoomObservation:
    """One known-prop detection in a zone crop, mapped back to full-frame px."""
    zone: str
    cls: str                  # a config object name (after the prompt -> object mapping)
    conf: float
    box_px: BoxPx             # full-frame px
    t: float                  # capture time, monotonic
    wall: float               # capture time, wall
    frame_idx: int


@dataclass
class RoomTrack:
    """A short track of one class in one zone, kept by RoomTracker across visits of that zone.
    World.room_update sets `role` and `entity` on the tracks it is given; the tracker keeps them."""
    tid: str                  # 'r:1', 'r:2', ...
    zone: str
    cls: str
    box_px: BoxPx
    first_seen: float         # monotonic capture time of its first observation
    first_wall: float
    last_seen: float          # monotonic capture time of its latest match
    last_wall: float
    hits: int = 1             # consecutive valid visits with a match
    misses: int = 0           # consecutive valid visits without a match (confirmed tracks)
    confirmed: bool = False
    role: str = "pending"     # 'pending' | 'assoc' | 'conflict' | 'ignored'
    entity: Optional[str] = None
    changed: bool = False          # its spot changed (frame difference) when it was first seen: it arrived,
                                   # rather than static clutter the detector flickers on (thing handoffs need it)
    guess: Optional[dict] = None   # cls 'thing' only: Grok's {name, also, confidence} for its crop, once named
    name_asked: bool = False       # its crop was queued for Grok (at most once)


@dataclass
class ZoneVisit:
    """What one processed zone visit changed. RoomTracker.visit makes it; World.room_update applies it."""
    zone: str
    say: str                  # the zone's spoken name, e.g. 'the bookshelf'
    t: float                  # capture time of the frame, monotonic
    wall: float
    frame_idx: int
    confirmed: list[RoomTrack] = field(default_factory=list)   # confirmed tracks matched this visit
    missed: list[RoomTrack] = field(default_factory=list)      # confirmed tracks with a valid miss (misses already +1)
    dropped: list[RoomTrack] = field(default_factory=list)     # removed this visit (after `missed` handling)
    crop: Optional[np.ndarray] = None                          # the zone crop (BGR), for event snapshots
    full: Optional[np.ndarray] = None                          # the whole camera frame, for answer evidence


@dataclass
class RoomState:
    """World-side room state of one entity (World._room[name]) while its zone is not the table."""
    zone: str
    say: str
    box_px: BoxPx
    track: Optional[str]      # the associated track id
    seen_t: float             # monotonic capture time of the last match
    seen_wall: float
    arrived_wall: float
    arrival_observed: bool
    misses: int = 0           # valid empty visits since the last match
    absent: bool = False      # absent_visits valid empty visits: UNKNOWN
    absent_t: Optional[float] = None
    table_pos_cm: Optional[Point] = None   # its last table position before it left (history only)
    tentative: bool = False   # an unnamed thing handed off by departure + Grok name match: answers hedge


@dataclass
class Conflict:
    """A confirmed room track of an entity's class that is not (and never becomes) that entity."""
    entity: str
    zone: str
    say: str
    track: str
    box_px: BoxPx
    seen_t: float
    seen_wall: float


@dataclass
class Place:
    """world.place(name): where to say something is. Table places keep today's templates."""
    kind: str                 # 'table' | 'room' | 'none'
    zone: str                 # 'table' or a room zone name
    say: str                  # spoken zone ('the table', 'the bookshelf')
    status: Status            # of `via`
    chain: list[str]          # the object, then its parents (as resolve() gives it)
    via: str                  # entity whose location is used: the object itself or its outermost container
    pos_cm: Optional[Point] = None      # table places only
    box_px: Optional[BoxPx] = None      # room places only
    observed_directly: bool = True      # via == the object itself
    fresh: bool = False                 # room: currently verified (see RoomConfig.fresh_*)
    absent: bool = False                # room: valid empty visits made it UNKNOWN
    arrived_wall: Optional[float] = None
    last_seen_wall: Optional[float] = None
    arrival_observed: bool = False
    conflicts: list[Conflict] = field(default_factory=list)   # fresh conflict sightings of the object itself
    tentative: bool = False             # room: handed off by a Grok name match, not a known class: hedge
