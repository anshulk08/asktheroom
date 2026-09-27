"""Room memory M0 (spec 0009): per-zone tracks of known props, the per-frame room driver, and the zone CLI.

RoomTracker keeps short tracks `r:N` per drawn zone across visits of that zone: confirmation after
`confirm_visits` matched visits (`confirm_visits_arrival`, 1, for a track whose spot changed when it
appeared: an arrival), and valid/invalid misses (a hand box, a large frame-difference blob or a too
dark/bright box over the spot makes a visit count for nothing). RoomMemory runs once per perception frame:
every `room_every_n`-th call (`room_every_n_hot`-th while the world has a handoff open) it crops the next
zone (round-robin) from the full 1080p frame, runs the already-loaded prop model backend on the crop, and
hands the tracker's ZoneVisit to `world.room_update`, which applies the association rules
(core/room_world.py). Positions here are full-frame px, never table cm.

Unnamed things (`room_memory.things`): the props model does not know most objects from a high corner, so
the same zone crop also goes through a class-agnostic YOLOE proposer (sharing the table's loaded model);
its boxes that are not a prop, a hand or someone carrying something become cls 'thing' observations and
are tracked like any class. Each confirmed thing track has one close-up named by Grok in the background
(RoomNamer), so the World can hand a table thing that left off to a zone thing with a matching name.

Speed (spec 0010 P0-3; a judge asks within ~5 s of putting the object down, far zones took 20-24 s): while
`world.room_handoff_hints` is non-empty the zones are visited every frame, an arrival confirms on its first
visit, its "is it one of these?" call goes to Grok before any open naming (static clutter is not sent at
all then), and the World decides the track the moment the name lands (RoomNamer.on_named), not on the
zone's next visit.

    python -m core.room --zone bookshelf --say "the bookshelf" --poly 100,80 600,80 600,400 100,400
    python -m core.room --list
    python -m core.room --delete-zone bookshelf
    python -m core.room --grab 0 --out full.jpg                      # one full frame from the camera
    python -m core.room --show full.jpg --out zones.jpg              # draw the zones for checking
    python -m core.room --measure-rect --full full.jpg --ref table160.jpg   # table_view_rect for config.local.yaml

Python 3.10 (JetPack 6).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from core import geom
from core.room_types import RoomConfig, RoomObservation, RoomTrack, ZoneVisit
from core.room_zones import Zone, Zones, view_version
from core.types import BoxPx, Event, Frame

log = logging.getLogger(__name__)

MATCH_IOU = 0.3             # spec 0009 section 3: a track match needs IoU >= 0.3 ...
MATCH_DIAG = 0.5            # ... or a centre within 0.5 track-box diagonals
DEDUPE_IOU = 0.5            # two boxes of one class this overlapped in one crop are one object (two prompts)
THING = "thing"             # cls of an unnamed object (a YOLOE proposal); never matches a prop's track
PROP_IOU = 0.5              # a proposal this overlapped with a prop observation is that prop
MARK_MIN_SIDE = 240         # px: the context patch around a boxed object for "is it one of these?"
NAME_MARGIN = 0.5           # a thing's close-up for Grok: its box grown by this per side (small, far objects
                            # need the surroundings to be recognisable: rig run Sat 26 Sep)
ERR_LOG_S = 10.0            # a failing proposer is logged at most this often
DILATE = np.ones((5, 5), np.uint8)


# ---------------------------------------------------------------------------------------------
# Tracker

class RoomTracker:
    """Per-zone tracks across visits of that zone. World.room_update sets role/entity; the tracker keeps them."""

    def __init__(self, cfg: RoomConfig):
        self.cfg = cfg
        self._tracks: dict[str, list[RoomTrack]] = {}
        self._n = 0

    def tracks(self, zone: Optional[str] = None) -> list[RoomTrack]:
        if zone is not None:
            return list(self._tracks.get(zone, []))
        return [tr for ts in self._tracks.values() for tr in ts]

    def reset(self) -> None:
        """Forget every track (a spoken reset). The id counter goes on: a naming job or an event log entry
        that still names an old 'r:N' must never point at a new track."""
        self._tracks = {}

    def visit(self, zone: str, say: str, obs: list[RoomObservation], blockers: list[BoxPx],
              changes: list[BoxPx], t: float, wall: float, frame_idx: int,
              lum: Optional[Callable[[BoxPx], float]] = None, crop=None) -> ZoneVisit:
        """Apply one processed visit of `zone`. `obs` are its known-prop detections (full px); `blockers`
        hand/person boxes; `changes` frame-difference boxes since the zone's previous visit; `lum(box)` the
        mean grey level in a box of this frame."""
        v = ZoneVisit(zone=zone, say=say, t=t, wall=wall, frame_idx=frame_idx, crop=crop)
        tracks = self._tracks.setdefault(zone, [])
        matched = self._match(tracks, obs)
        keep: list[RoomTrack] = []
        for tr in tracks:
            o = matched.get(tr.tid)
            if o is not None:
                if not tr.confirmed and _changed(o.box_px, changes):
                    tr.changed = True             # placed while a hand was over it: the change shows next visit
                tr.box_px = tuple(int(c) for c in o.box_px)
                tr.last_seen, tr.last_wall = t, wall
                tr.hits += 1
                tr.misses = 0
                tr.confirmed = tr.confirmed or tr.hits >= self._need(tr)
                if tr.confirmed:
                    v.confirmed.append(tr)
                keep.append(tr)
            elif not tr.confirmed and any(o.cls == tr.cls and geom.overlap_frac(o.box_px, tr.box_px) > 0
                                          for o in obs):
                v.dropped.append(tr)              # a same-class box on it went to another track: a duplicate
                                                  # (e.g. a partial box while a hand was still placing it)
            elif not self._valid(tr.box_px, blockers + [o.box_px for o in obs], changes, lum):
                keep.append(tr)                   # an invalid visit counts for nothing (another object
                                                  # on its spot is a blocker too: spec 0009, Absence)
            elif not tr.confirmed:
                v.dropped.append(tr)
            else:
                tr.misses += 1
                v.missed.append(tr)
                if tr.misses >= self.cfg.absent_visits and t - tr.last_seen >= self.cfg.absent_min_s:
                    v.dropped.append(tr)
                else:
                    keep.append(tr)
        used = {id(o) for o in matched.values()}
        for o in obs:
            if id(o) in used:
                continue
            self._n += 1
            tr = RoomTrack(tid=f"r:{self._n}", zone=zone, cls=o.cls, box_px=tuple(int(c) for c in o.box_px),
                           first_seen=t, first_wall=wall, last_seen=t, last_wall=wall, hits=1)
            tr.changed = _changed(tr.box_px, changes)
            tr.confirmed = tr.hits >= self._need(tr)
            if tr.confirmed:
                v.confirmed.append(tr)
            keep.append(tr)
        self._tracks[zone] = keep
        return v

    def _need(self, tr: RoomTrack) -> int:
        """Matched visits a track needs to confirm. An arrival (its spot changed when it was first seen, or
        on a later visit once the placing hand was gone) confirms on confirm_visits_arrival (1): the second
        visit of a far zone cost a second or more of the handoff (spec 0010 P0-3), and the change evidence
        is what tells the carried object from detector flicker on static clutter, which keeps
        confirm_visits."""
        return self.cfg.confirm_visits_arrival if tr.changed else self.cfg.confirm_visits

    @staticmethod
    def _match(tracks: list[RoomTrack], obs: list[RoomObservation]) -> dict[str, RoomObservation]:
        """Greedy per class: best IoU first (then nearest centre), each track and observation used once."""
        pairs = []
        for i, tr in enumerate(tracks):
            diag = math.hypot(tr.box_px[2] - tr.box_px[0], tr.box_px[3] - tr.box_px[1])
            for j, o in enumerate(obs):
                if o.cls != tr.cls:
                    continue
                ov = geom.iou(tr.box_px, o.box_px)
                d = geom.dist(geom.center(tr.box_px), geom.center(o.box_px))
                if ov >= MATCH_IOU or (d <= MATCH_DIAG * diag and _near_ok(tr, o)):
                    pairs.append((-ov, d, i, j))
        pairs.sort()
        out: dict[str, RoomObservation] = {}
        used_obs: set[int] = set()
        for _, _, i, j in pairs:
            if tracks[i].tid in out or j in used_obs:
                continue
            out[tracks[i].tid] = obs[j]
            used_obs.add(j)
        return out

    def _valid(self, box: BoxPx, blockers: list[BoxPx], changes: list[BoxPx],
               lum: Optional[Callable[[BoxPx], float]]) -> bool:
        """Whether this visit can count as a miss for a track at `box` (spec 0009 section 3, Absence)."""
        c = self.cfg
        if any(geom.overlap_frac(b, box) >= c.blocker_overlap for b in blockers):
            return False
        big = c.change_area_ratio * geom.area(box)
        if any(geom.area(ch) >= big and geom.overlap_frac(ch, box) >= c.blocker_overlap for ch in changes):
            return False
        if lum is not None:
            v = float(lum(box))
            if not (c.lum_lo <= v <= c.lum_hi):       # NaN (box off the crop) is invalid too
                return False
        return True


# ---------------------------------------------------------------------------------------------
# Grok names for thing tracks

class _Job:
    __slots__ = ("track", "img", "attempts", "due", "hints", "gen", "ctx")

    def __init__(self, track: RoomTrack, img: np.ndarray, due: float, hints=None, gen: int = 0, ctx=None):
        self.track, self.img, self.attempts, self.due, self.hints = track, img, 0, due, hints
        self.gen = gen                     # the reset generation it was queued in (RoomNamer.reset)
        self.ctx = ctx                     # the marked wider view for open naming, or None


class RoomNamer:
    """Names confirmed thing tracks in the background: one close-up per track to `name_fn` (Grok, via
    core.auto_name.AutoNamer._ask), the result set as `track.guess`. The perception thread only queues a
    copied crop. Calls only while `online()` (offline, jobs wait), at most `per_minute`; a failure (an
    exception or None) is retried once `retry_s` later, then given up. The queue keeps the newest
    `max_pending` jobs: a burst of junk boxes must not hold memory or starve later tracks. Verification
    jobs (with hints: "is it one of these?") go before open naming, newest first either way: the carried
    object is the one that matters. `on_named(track)`, when set, is called on the worker thread right after
    a guess lands (RoomMemory decides the track at once; spec 0010 P0-3)."""

    def __init__(self, name_fn: Callable[[np.ndarray], Optional[dict]], per_minute: int = 6,
                 online: Optional[Callable[[], bool]] = None, clock: Callable[[], float] = time.monotonic,
                 retry_s: float = 5.0, max_pending: int = 8, start: bool = True,
                 verify_fn: Optional[Callable[[np.ndarray, list], Optional[dict]]] = None,
                 on_named: Optional[Callable[[RoomTrack], None]] = None):
        self.name_fn = name_fn
        self.verify_fn = verify_fn         # (img, hints) -> guess: "is this one of these?" beats open naming
        self.on_named = on_named           # called with the track once its guess is set (the worker thread)
        self.per_minute = int(per_minute)
        self.online = online or (lambda: True)
        self.clock = clock
        self.retry_s = float(retry_s)
        self._lock = threading.Lock()
        self._jobs: deque = deque(maxlen=max(1, int(max_pending)))   # full: appending drops the oldest
        self._calls: deque = deque()       # clock() of recent calls (the per-minute cap)
        self._gen = 0                      # bumped by reset(): a job from an older generation never lands
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if start:
            self._thread = threading.Thread(target=self._run, name="room-name", daemon=True)
            self._thread.start()

    def submit(self, track: RoomTrack, img: np.ndarray, hints: Optional[list] = None,
               ctx: Optional[np.ndarray] = None) -> None:
        """hints: Grok names of things that just left the table (the handoff candidates), asked about
        directly when a verify_fn is set. ctx: a wider view with the track in a red box, sent with the
        close-up for open naming (name_fn(img, ctx))."""
        with self._lock:
            self._jobs.append(_Job(track, img, self.clock(), list(hints) if hints else None, self._gen, ctx))
        self._wake.set()

    def pending(self) -> int:
        with self._lock:
            return len(self._jobs)

    def reset(self) -> None:
        """Drop every queued job and disown the one in flight: after a spoken reset its tracks are gone,
        and a Grok answer for a pre-reset crop must not become a new track's name (the worker checks the
        generation after the call, so even an answer already on its way is thrown away)."""
        with self._lock:
            self._jobs.clear()
            self._gen += 1

    def step(self, now: Optional[float] = None) -> bool:
        """Try the next due job. True when a call was made."""
        now = self.clock() if now is None else now
        with self._lock:
            due = [j for j in reversed(self._jobs) if j.due <= now]              # newest first: the fresh arrival
            job = next((j for j in due if j.hints), due[0] if due else None)     # a verification first
            if job is None or not self.online():
                return False
            while self._calls and self._calls[0] <= now - 60.0:
                self._calls.popleft()
            if len(self._calls) >= self.per_minute:
                return False
            self._jobs.remove(job)
            self._calls.append(now)
        job.attempts += 1
        try:
            if job.hints and self.verify_fn is not None:
                g = self.verify_fn(job.img, job.hints)
            else:
                g = self.name_fn(job.img) if job.ctx is None else self.name_fn(job.img, job.ctx)
        except Exception as e:
            log.info("naming room track %s failed (attempt %d): %s", job.track.tid, job.attempts, e)
            g = None
        self._save(job, g)
        with self._lock:
            stale = job.gen != self._gen
        if stale:                          # reset while it was asked: neither applied nor retried
            log.info("room track %s in %s: named after a reset, dropped", job.track.tid, job.track.zone)
            return True
        if g:
            job.track.guess = g
            log.info("room track %s in %s looks like a %s", job.track.tid, job.track.zone, g.get("name"))
            if self.on_named is not None:
                try:
                    self.on_named(job.track)
                except Exception:              # a deciding bug must not cost the naming worker
                    log.warning("deciding room track %s on its name failed", job.track.tid, exc_info=True)
        else:
            log.info("room track %s in %s: no usable name (attempt %d)", job.track.tid, job.track.zone, job.attempts)
        if not g and job.attempts < 2:
            job.due = now + self.retry_s
            with self._lock:
                self._jobs.append(job)
        return True

    def _save(self, job, g) -> None:
        """Debug (env ASKROOM_ROOM_CROPS=dir): keep every room crop sent to Grok, named by track and reply,
        to see what it was asked about when a handoff's names don't match."""
        d = os.environ.get("ASKROOM_ROOM_CROPS")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            name = (g or {}).get("name") or "none"
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(name))[:40]
            cv2.imwrite(os.path.join(d, f"{job.track.tid.replace(':', '')}_{job.track.zone}_{job.attempts}_{safe}.jpg"),
                        job.img)
        except Exception:
            log.debug("saving the room crop failed", exc_info=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                made = self.step()
            except Exception:
                log.exception("room naming step failed")
                made = False
            if not made:
                self._wake.wait(0.2)
                self._wake.clear()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _changed(box, changes: list) -> bool:
    """Frame-difference evidence at box: a change blob covering at least a fifth of it."""
    return any(geom.overlap_frac(ch, box) >= 0.2 for ch in changes)


def _near_ok(tr: RoomTrack, o: RoomObservation) -> bool:
    """A match by centre distance alone. A prop's class keeps different objects apart; unnamed objects all
    share cls 'thing', so a small one next to a big one (the remote beside a bag of chips on the couch, rig
    run Sat 26 Sep) must not match the big one's track: the centre must be within half the *smaller*
    diagonal and the sizes within 3x."""
    if tr.cls != "thing":
        return True
    da = math.hypot(tr.box_px[2] - tr.box_px[0], tr.box_px[3] - tr.box_px[1])
    db = math.hypot(o.box_px[2] - o.box_px[0], o.box_px[3] - o.box_px[1])
    aa, ab = max(geom.area(tr.box_px), 1), max(geom.area(o.box_px), 1)
    return (geom.dist(geom.center(tr.box_px), geom.center(o.box_px)) <= MATCH_DIAG * min(da, db)
            and max(aa, ab) / min(aa, ab) <= 3.0)


# ---------------------------------------------------------------------------------------------
# Driver

class RoomMemory:
    """Called once per perception frame with the full camera frame; processes one zone every room_every_n
    calls (room_every_n_hot while a handoff is open)."""

    def __init__(self, cfg: RoomConfig, zones: Zones, backend, to_obj: dict[str, str], world,
                 table_rect: BoxPx, hand_conf: float = 0.35, proposer=None, namer: Optional[RoomNamer] = None):
        self.cfg = cfg
        self.zones = zones
        self.backend = backend                 # the table detector's backend: shared, never a second load
        self.to_obj = to_obj
        self.world = world
        self.table_rect = tuple(int(v) for v in table_rect)
        self.hand_conf = hand_conf             # a hand box needs this score to block (the table's hand cut-off)
        self.proposer = proposer               # propose(img, known, hands) -> [Proposal], or None: props only
        self.namer = namer                     # RoomNamer for confirmed thing tracks, or None: never named
        self.tracker = RoomTracker(cfg)
        self._prop_err_t = float("-inf")
        self._calls = 0
        self._since = 0                        # calls since the last processed zone (the cadence)
        self._next = 0
        self._last_idx = -1                    # frame_idx of the last processed zone (synthetic visits)
        self._prev: dict[str, np.ndarray] = {}  # zone -> grey crop of its previous visit (change evidence)
        self._extra: list[Event] = []          # events of visits decided off the perception thread (_on_named)
        self._extra_lock = threading.Lock()
        if namer is not None and getattr(namer, "on_named", None) is None:
            namer.on_named = self._on_named

    @classmethod
    def from_config(cls, cfg: dict, world, backend, table_rect: Optional[BoxPx], proposer=None,
                    name_fn: Optional[Callable[[np.ndarray], Optional[dict]]] = None,
                    online: Optional[Callable[[], bool]] = None,
                    verify_fn: Optional[Callable[[np.ndarray, list], Optional[dict]]] = None) -> Optional["RoomMemory"]:
        """RoomMemory from config.yaml's room_memory: section, or None (logged) when it can't run. With a
        `proposer` (and room_memory.things) zones also track unnamed things; with a `name_fn` too, their
        confirmed tracks are named by it in the background (names_per_minute, only while `online()`)."""
        rc = RoomConfig.from_dict(cfg.get("room_memory"))
        if not rc.enabled:
            log.info("room memory off: room_memory.enabled is false")
            return None
        if table_rect is None:
            log.warning("room memory off: no table_view_rect")
            return None
        try:
            zones = Zones.load(rc.zones_path)
        except FileNotFoundError:
            log.warning("room memory off: no zones file %s (draw zones: python -m core.room --zone ...)",
                        rc.zones_path)
            return None
        except (ValueError, KeyError, TypeError) as e:
            log.warning("room memory off: can't read zones file %s: %s", rc.zones_path, e)
            return None
        if not zones.zones:
            log.warning("room memory off: %s has no zones", rc.zones_path)
            return None
        want = view_version(rc.capture_size, rc.zoom, table_rect)
        if zones.view != want:
            log.warning("room memory off: %s was drawn at view %s, the camera view is now %s "
                        "(capture_size, zoom or table_view_rect changed); redraw the zones",
                        rc.zones_path, zones.view, want)
            return None
        from core.detect import class_list
        _, to_obj = class_list(cfg)
        ct = cfg.get("conf_threshold", 0.35)
        hand_conf = float(ct.get("hand", ct.get("default", 0.35))) if isinstance(ct, dict) else float(ct)
        if not rc.things:
            proposer = None
        namer = RoomNamer(name_fn, per_minute=rc.names_per_minute, online=online, verify_fn=verify_fn) \
            if proposer is not None and name_fn is not None else None
        log.info("room memory on: zones %s; things %s", ", ".join(f"{z.name} ({z.say})" for z in zones.zones.values()),
                 "off" if proposer is None else ("named by Grok" if namer is not None else "unnamed"))
        return cls(rc, zones, backend, to_obj, world, table_rect, hand_conf=hand_conf, proposer=proposer,
                   namer=namer)

    def stop(self) -> None:
        """Stops the naming worker (build's cleanup)."""
        if self.namer is not None:
            self.namer.stop()

    def reset(self) -> None:
        """A spoken reset, without a restart: forget every track, every queued or in-flight Grok name, and
        every zone's previous crop (the first visit after this takes fresh backgrounds, so nothing counts
        as 'changed' until something really arrives), then the World's room side too. Runs on the
        perception thread, like step(), so the next run starts exactly like a fresh start: World.reset()
        on the asking thread may already have cleared the room state, but one visit with the old tracks
        could have slipped in between (a pre-reset conflict or pending thing in a fresh world)."""
        self.tracker.reset()
        if self.namer is not None:
            self.namer.reset()
        self._prev.clear()
        fn = getattr(self.world, "room_reset", None)
        if callable(fn):
            fn()

    def step(self, full: Optional[Frame]) -> list[Event]:
        """One perception frame. Every room_every_n-th call with a frame (every room_every_n_hot-th while the
        world has a handoff open: `room_handoff_hints` non-empty, so the zone the object went to is seen
        within a frame or two instead of up to zones x room_every_n frames later; spec 0010 P0-3) processes
        the next zone, round-robin, and returns world.room_update's events; otherwise, or with no image or
        an empty crop, []. Events of visits decided off this thread since the last call (_on_named) come
        first. Hot mode costs one more prop + YOLOE pass per perception frame on the Jetson (10-13 fps with
        the cold cadence): not measured yet; measure fps on the rig with a handoff open."""
        self._calls += 1
        self._since += 1
        out = self._drain()
        names = list(self.zones.zones)
        if not names or full is None or full.img is None:
            return out
        hints = self._hints(full.t)
        every = (min(self.cfg.room_every_n_hot, self.cfg.room_every_n) if self._hot(full.t, hints)
                 else self.cfg.room_every_n)          # hot is never slower than cold
        if self._since < max(1, int(every)):
            return out
        self._since = 0
        self._last_idx = full.idx
        zone = self.zones.zones[names[self._next % len(names)]]
        self._next += 1
        visit = self._visit(zone, full, hints)
        if visit is None:
            return out
        return out + list(self.world.room_update(visit) or [])

    def _drain(self) -> list[Event]:
        with self._extra_lock:
            out, self._extra = self._extra, []
        return out

    def _on_named(self, track: RoomTrack) -> None:
        """RoomNamer's callback, on its worker thread: Grok named `track`. The World decides it now, with a
        synthetic visit of its zone holding just this track (the track's last sighting, no crop), instead
        of on the zone's next visit, which cold could be ~1.7 s away (spec 0010 P0-3). room_update takes the
        world lock; the tracker is only read (a track it dropped meanwhile is not decided: it would never be
        refreshed or missed again). The visit's events come out of the next step(). A failure is logged,
        never raised."""
        try:
            zone = self.zones.zones.get(track.zone)
            if zone is None or not track.confirmed or not any(t is track for t in self.tracker.tracks(track.zone)):
                return
            visit = ZoneVisit(zone=track.zone, say=zone.say, t=track.last_seen, wall=track.last_wall,
                              frame_idx=self._last_idx, confirmed=[track], crop=None)
            events = list(self.world.room_update(visit) or [])
            if events:
                with self._extra_lock:
                    self._extra.extend(events)
        except Exception:
            log.warning("deciding room track %s on its name failed", track.tid, exc_info=True)

    def _visit(self, zone: Zone, full: Frame, hints: Optional[list] = None) -> Optional[ZoneVisit]:
        """One processed visit of `zone`. `hints`: what `_hints(full.t)` said (None: the world can't say)."""
        img = full.img
        h, w = img.shape[:2]
        bx1, by1, bx2, by2 = zone.bbox()
        x1, y1, x2, y2 = max(0, bx1), max(0, by1), min(w, bx2), min(h, by2)
        if x2 <= x1 or y2 <= y1:
            log.debug("zone %s is outside the %dx%d frame", zone.name, w, h)
            return None
        crop = img[y1:y2, x1:x2]
        s = 1.0
        long_side = max(x2 - x1, y2 - y1)
        if long_side > self.cfg.max_crop_px:
            s = self.cfg.max_crop_px / long_side
            small = cv2.resize(crop, (max(1, round((x2 - x1) * s)), max(1, round((y2 - y1) * s))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = crop

        def to_full(b) -> BoxPx:
            return (int(round(x1 + b[0] / s)), int(round(y1 + b[1] / s)),
                    int(round(x1 + b[2] / s)), int(round(y1 + b[3] / s)))

        def to_small(b: BoxPx) -> BoxPx:
            return (int(math.floor((b[0] - x1) * s)), int(math.floor((b[1] - y1) * s)),
                    int(math.ceil((b[2] - x1) * s)), int(math.ceil((b[3] - y1) * s)))

        hands: list[BoxPx] = []
        hands_small: list[BoxPx] = []
        props: list[tuple[str, float, BoxPx, BoxPx]] = []
        for label, conf, box in self.backend.infer(small):
            obj = self.to_obj.get(label)
            if obj is None:
                continue
            if obj == "hand":
                if conf >= self.hand_conf:
                    hands.append(to_full(box))
                    hands_small.append(tuple(int(round(v)) for v in box))
            elif conf >= self.cfg.room_prop_conf:
                props.append((obj, float(conf), to_full(box), tuple(int(round(v)) for v in box)))
        obs: list[RoomObservation] = []
        known_small: list[BoxPx] = []
        for obj, conf, box, sbox in sorted(props, key=lambda p: -p[1]):
            if any(geom.iou(box, hb) >= self.cfg.hand_iou for hb in hands):
                continue                           # the hand itself, called a prop
            c = geom.center(box)
            if not zone.contains(c) or geom.contains_point(self.table_rect, c):
                continue
            if any(o.cls == obj and geom.iou(o.box_px, box) >= DEDUPE_IOU for o in obs):
                continue                           # the same object under a second prompt
            obs.append(RoomObservation(zone=zone.name, cls=obj, conf=round(conf, 3), box_px=box,
                                       t=full.t, wall=full.wall, frame_idx=full.idx))
            known_small.append(sbox)
        if self.cfg.things and self.proposer is not None:
            obs += self._things(zone, full, small, known_small, hands_small, hands, obs, to_full)

        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        changes: list[BoxPx] = []
        prev = self._prev.get(zone.name)
        if prev is not None and prev.shape == grey.shape:
            mask = (cv2.absdiff(grey, prev) > self.cfg.change_thr).astype(np.uint8) * 255
            mask = cv2.dilate(mask, DILATE, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                changes.append(to_full((cx, cy, cx + cw, cy + ch)))
        self._prev[zone.name] = grey

        def lum(box: BoxPx) -> float:
            a1, b1, a2, b2 = to_small(box)
            a1, b1 = max(0, a1), max(0, b1)
            a2, b2 = min(grey.shape[1], a2), min(grey.shape[0], b2)
            if a2 <= a1 or b2 <= b1:
                return float("nan")                # off this crop: no valid view of it
            return float(grey[b1:b2, a1:a2].mean())

        visit = self.tracker.visit(zone.name, zone.say, obs, hands, changes, full.t, full.wall, full.idx,
                                   lum=lum, crop=crop.copy())
        if self.namer is not None:
            for tr in visit.confirmed:
                # While a handoff is open only arrivals (their spot changed) are sent: static clutter never
                # arrived, so it can't be the carried object, and each junk call cost ~1 s of Grok's time
                # ahead of the one that matters (counter clutter, rig run Sat 26 Sep).
                if (tr.cls == THING and tr.guess is None and not tr.name_asked and hints != []
                        and (hints is None or tr.changed)):
                    verify = bool(hints and self.namer.verify_fn)
                    img = (_marked_close_up(visit.crop, tr.box_px, x1, y1) if verify
                           else _close_up(visit.crop, tr.box_px, x1, y1))
                    if img is not None:
                        tr.name_asked = True
                        ctx = None if verify else _marked_close_up(visit.crop, tr.box_px, x1, y1)
                        self.namer.submit(tr, img, hints, ctx=ctx)
        return visit

    def _hot(self, t: float, hints: Optional[list]) -> bool:
        """The fast cadence: the world's room_hot (a named departure within hot_max_s), else hints non-empty
        for a world without it."""
        fn = getattr(self.world, "room_hot", None)
        if callable(fn):
            try:
                return bool(fn(t))
            except Exception:
                log.debug("room_hot failed", exc_info=True)
        return bool(hints)

    def _hints(self, t: float) -> Optional[list]:
        """Grok names of things that left the table and could still be handed off, or None when the world
        can't say (then every new room thing is named, as before). [] means no handoff is possible: room
        clutter is not sent to Grok, which keeps the per-minute cap for the object that matters."""
        fn = getattr(self.world, "room_handoff_hints", None)
        if not callable(fn):
            return None
        try:
            return list(fn(t))
        except Exception:
            log.debug("room_handoff_hints failed", exc_info=True)
            return None

    def _things(self, zone: Zone, full: Frame, small: np.ndarray, known: list[BoxPx], hands_small: list[BoxPx],
                hands: list[BoxPx], props: list[RoomObservation], to_full) -> list[RoomObservation]:
        """Thing observations of this visit: proposer boxes on the (possibly resized) zone crop that are not
        carried (occluded: inside a person box), not a hand, not a prop of this visit, inside the zone and
        off the table view. A failing proposer costs the things of this visit, never the props."""
        try:
            proposals = list(self.proposer.propose(small, known, hands_small) or [])
        except Exception:
            now = time.monotonic()
            if now - self._prop_err_t >= ERR_LOG_S:
                log.warning("room proposer failed on zone %s; props only this visit", zone.name, exc_info=True)
                self._prop_err_t = now
            return []
        out: list[RoomObservation] = []
        for p in proposals:
            if getattr(p, "occluded", False):
                continue
            box = to_full(p.box_px)
            if any(geom.iou(box, hb) >= self.cfg.hand_iou for hb in hands):
                continue                           # the hand itself
            c = geom.center(box)
            if not zone.contains(c) or geom.contains_point(self.table_rect, c):
                continue
            if any(geom.iou(box, o.box_px) >= PROP_IOU for o in props):
                continue                           # the prop, boxed again
            out.append(RoomObservation(zone=zone.name, cls=THING, conf=round(float(p.conf), 3), box_px=box,
                                       t=full.t, wall=full.wall, frame_idx=full.idx))
        return out


def _close_up(crop: np.ndarray, box: BoxPx, ox: int, oy: int) -> Optional[np.ndarray]:
    """A copy of `box` (full px) cut from the native-resolution zone crop at (ox, oy), grown by NAME_MARGIN
    per side and clipped to the crop; None when too small to name."""
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    mx, my = NAME_MARGIN * (x2 - x1), NAME_MARGIN * (y2 - y1)
    a1, b1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    a2, b2 = min(w, int(math.ceil(x2 + mx))), min(h, int(math.ceil(y2 + my)))
    if a2 - a1 < 4 or b2 - b1 < 4:
        return None
    return crop[b1:b2, a1:a2].copy()


def _marked_close_up(crop: np.ndarray, box: BoxPx, ox: int, oy: int, min_side: int = MARK_MIN_SIDE
                     ) -> Optional[np.ndarray]:
    """For "is it one of these?": the object boxed in red inside a wider patch of the zone (at least
    min_side px, native resolution), so Grok sees its surroundings. A far, dark object cut out alone was
    "no usable name" on the counter and side table (trial runs, Sat 26 Sep); set-of-marks with context is
    what made visual questions work."""
    from core.crops import marked_view
    return marked_view(crop, (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy), min_side)


# ---------------------------------------------------------------------------------------------
# CLI

def _current_view(rc: RoomConfig) -> tuple[BoxPx, str]:
    from core.room_view import default_rect
    rect = rc.table_view_rect or default_rect(rc.capture_size, rc.zoom, rc.ref_zoom)
    return tuple(int(v) for v in rect), view_version(rc.capture_size, rc.zoom, rect)


def _parse_poly(pts: Sequence[str]) -> list[tuple[float, float]]:
    out = []
    for p in pts:
        x, y = p.split(",")
        out.append((float(x), float(y)))
    if len(out) < 3:
        raise ValueError("a zone needs at least 3 points")
    return out


def _load_zones(path: str) -> Optional[Zones]:
    try:
        return Zones.load(path)
    except FileNotFoundError:
        return None


def _read_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"can't read image {path}")
    return img


def main(argv=None) -> int:
    from core.config import load_config
    ap = argparse.ArgumentParser(description="room memory zones (spec 0009 M0)")
    ap.add_argument("--config", help="config.yaml path (default: the repo's, plus config.local.yaml)")
    ap.add_argument("--zones", help="zones file (default: room_memory.zones_path)")
    ap.add_argument("--zone", help="add or replace this zone")
    ap.add_argument("--say", help="its spoken name (default: 'the <zone>')")
    ap.add_argument("--poly", nargs="+", metavar="X,Y", help="polygon corners in full-frame px")
    ap.add_argument("--delete-zone", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="IMAGE", help="draw the zones on this full frame (with --out)")
    ap.add_argument("--measure-rect", action="store_true", help="fit table_view_rect (with --full, --ref)")
    ap.add_argument("--full", help="a full frame at the room zoom")
    ap.add_argument("--ref", help="a table frame at the reference zoom (160)")
    ap.add_argument("--grab", metavar="DEVICE", help="save one full frame from this camera (with --out)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    path = a.zones or rc.zones_path

    if a.zone:
        if not a.poly:
            ap.error("--zone needs --poly")
        try:
            poly = _parse_poly(a.poly)
        except ValueError as e:
            ap.error(f"--poly: {e}")
        rect, view = _current_view(rc)
        zones = _load_zones(path) or Zones(view, rc.capture_size, {})
        if zones.view != view:
            print(f"{path} was drawn at view {zones.view}, the configured view is {view} "
                  f"(capture_size, zoom or table_view_rect changed): delete the file and redraw every zone",
                  file=sys.stderr)
            return 1
        w, h = rc.capture_size
        if any(not (0 <= x <= w and 0 <= y <= h) for x, y in poly):
            print(f"warning: some points are outside the {w}x{h} frame", file=sys.stderr)
        zones.zones[a.zone] = Zone(a.zone, a.say or f"the {a.zone.replace('_', ' ')}", poly)
        zones.save(path)
        print(f"saved zone {a.zone} ({zones.zones[a.zone].say}) to {path}")
        return 0

    if a.delete_zone:
        zones = _load_zones(path)
        if zones is None or a.delete_zone not in zones.zones:
            print(f"no zone {a.delete_zone} in {path}", file=sys.stderr)
            return 1
        del zones.zones[a.delete_zone]
        zones.save(path)
        print(f"deleted zone {a.delete_zone} from {path}")
        return 0

    if a.list:
        zones = _load_zones(path)
        if zones is None:
            print(f"no zones file {path}")
            return 1
        _, view = _current_view(rc)
        state = "matches the config" if zones.view == view else f"MISMATCH: the config's view is {view}"
        print(f"{path}: view {zones.view} ({state}), frame {zones.size_px[0]}x{zones.size_px[1]}, "
              f"{len(zones.zones)} zone(s)")
        for z in zones.zones.values():
            print(f"  {z.name}: {z.say!r} bbox {list(z.bbox())} {len(z.poly)} points")
        return 0

    if a.show:
        if not a.out:
            ap.error("--show needs --out")
        img = _read_img(a.show)
        zones = _load_zones(path)
        if zones is None:
            print(f"no zones file {path}", file=sys.stderr)
            return 1
        if (img.shape[1], img.shape[0]) != zones.size_px:
            print(f"warning: the image is {img.shape[1]}x{img.shape[0]}, the zones were drawn on "
                  f"{zones.size_px[0]}x{zones.size_px[1]}", file=sys.stderr)
        rect, _ = _current_view(rc)
        cv2.rectangle(img, rect[:2], rect[2:], (255, 128, 0), 2)
        cv2.putText(img, "table view", (rect[0] + 6, rect[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 128, 0), 2)
        for z in zones.zones.values():
            pts = np.round(np.asarray(z.poly)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            x, y = pts[:, 0, 0].min(), pts[:, 0, 1].min()
            cv2.putText(img, f"{z.name}: {z.say}", (int(x) + 6, int(y) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
        cv2.imwrite(a.out, img)
        print(f"saved {a.out}")
        return 0

    if a.measure_rect:
        if not (a.full and a.ref):
            ap.error("--measure-rect needs --full and --ref")
        from core.room_view import default_rect, measure_rect
        full, ref = _read_img(a.full), _read_img(a.ref)
        init = rc.table_view_rect or default_rect((full.shape[1], full.shape[0]), rc.zoom, rc.ref_zoom)
        rect, corr = measure_rect(full, ref, tuple(init))
        print(f"room_memory: {{table_view_rect: [{', '.join(str(int(v)) for v in rect)}]}}")
        print(f"# ECC correlation {corr:.3f}")
        return 0

    if a.grab is not None:
        if not a.out:
            ap.error("--grab needs --out")
        from core.capture import open_camera
        dev = int(a.grab) if a.grab.isdigit() else a.grab
        cap = open_camera(dev, *rc.capture_size)
        try:
            img = None
            for _ in range(15):                    # let exposure settle; keep the last good frame
                ok, f = cap.read()
                if ok:
                    img = f
        finally:
            cap.release()
        if img is None:
            print(f"no frame from {a.grab}", file=sys.stderr)
            return 1
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.out, img)
        print(f"saved {a.out} ({img.shape[1]}x{img.shape[0]})")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


VERIFY_SYSTEM = ("You look at part of a room seen by a ceiling camera. One object is marked with a red box. The "
                 "person is looking for one of the objects listed. Say which one the object in the red box is, "
                 "or 'none' if it is none of them or you can't tell. Also say what the boxed object is in 1-3 "
                 "words. Reply with strict JSON.")
VERIFY_MIN_CONF = 0.7          # a "yes, it's the remote" below this is not a match
VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["match", "name", "confidence"],
    "properties": {"match": {"type": "string"}, "name": {"type": "string"}, "confidence": {"type": "number"}},
}


def make_verify_fn(namer) -> Callable[[np.ndarray, list], Optional[dict]]:
    """(img, hints) -> guess, through the auto-namer's Grok provider. Asking "is it one of these?" (the
    names of things that just left the table) is far more reliable than open naming for a small, far
    object: open naming called the remote on the couch a phone and an eyeglasses case (rig run). A match
    returns that hint's guess, so the World's name check passes; 'none' returns what Grok says it is.
    Speed: the call goes through the auto-namer's provider, built from visual_memory (grok-4.3 at
    reasoning_effort 'none', its lowest, timeout_s 8), which is what makes it ~0.6-1.5 s; narrate() takes
    no per-call effort, so nothing is set here. The crop goes at namer.c.crop_px (384) on its long side."""
    from core.auto_name import _jpeg, clean_name, match_score
    from core.narration import _parse_json

    def verify(img: np.ndarray, hints: list) -> Optional[dict]:
        named = [h for h in hints if isinstance(h, dict) and h.get("name")]
        if not named:
            return None
        listed = "; ".join(h["name"] + (f" (also: {', '.join(h.get('also') or [])})" if h.get("also") else "")
                           for h in named)
        reply = namer.provider.narrate(VERIFY_SYSTEM, [("text", f"Looking for: {listed}"),
                                                       ("image", _jpeg(img, namer.c.crop_px, namer.c.jpeg_quality)),
                                                       ("text", "Which one is the object in the red box, or none?")],
                                        VERIFY_SCHEMA)
        d = _parse_json(reply.text) or {}
        try:
            conf = min(1.0, max(0.0, float(d.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        match = clean_name(d.get("match")) or ""
        name = clean_name(d.get("name"))
        if conf >= VERIFY_MIN_CONF:
            for h in named:
                # Asked "is it one of these?", Grok leans to yes (it matched a keyboard on the stove to a
                # remote, rig run): its own description must fit the name too.
                if match == clean_name(h["name"]) and name and (match_score(name, h) >= 1.5
                                                                or match_score(h["name"], {"name": name}) >= 1.5):
                    return dict(h)
        return {"name": name, "also": [], "confidence": round(conf, 3)} if name else None

    return verify
