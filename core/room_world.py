"""Room memory on the World side (spec 0009 M0, section 3 Transitions). RoomTracker (core/room.py) hands
World.room_update one ZoneVisit per processed zone; these rules decide what its confirmed tracks mean for
the entities. Every change to a room entity is exactly one operation:

  Acquire    the entity has an unconsumed table departure within handoff_s, is GONE / UNKNOWN on the
             table, and the track was first seen after that departure: zone = the track's zone, VISIBLE,
             FOUND. The departure is consumed: it authorises at most one acquisition.
  Refresh    the entity's own associated track matched again: room timestamps only, no event.
  Absence    its own track missed absent_visits valid visits: UNKNOWN, zone kept, LOST_TRACK.
  Reacquire  UNKNOWN in zone Z; a track in Z at its last spot (IoU >= 0.3), first seen after the
             absence: VISIBLE again, FOUND.
  Return     confirmed table presence (World._observe after the presence flip, never one detection):
             room state and conflicts dropped, zone 'table', MOVED.
  Conflict   any other confirmed track of the entity's class: recorded as a sighting, entity unchanged.
             A conflict track never upgrades, by repetition or by a later departure.

A departure is EXITED_VIEW or LOST_TRACK emitted while the entity's zone is 'table' (hand-lost and
held-timeout losses are LOST_TRACK too). Room absence (LOST_TRACK in a room zone) is not one.

One instance per prop class is an M0 demo assumption, not identity evidence: a second key ring that first
appears in a zone after the keys leave the table would inherit them. The first-seen-after-departure and
consume-once rules only keep decoys that were already there, or later ones, from doing so.

Room positions are full-frame px (RoomState.box_px); pos_cm stays None for a room entity, and freshness
reads RoomState timestamps, never ent.last_seen (one table flicker rewrites that through _debounce).

The World class mixes this in; methods share its state (entities, lock, _bits, _present, _seen_t, ...).
"""
from __future__ import annotations

import time
from typing import Optional

from core import geom, relations
from core.room_types import TABLE, Conflict, Place, RoomConfig, RoomState, RoomTrack, ZoneVisit
from core.types import Entity, Event, EventType, Status

REACQUIRE_IOU = 0.3          # a new track this much over the last room box is the object at its spot
DEPARTURES = (EventType.EXITED_VIEW, EventType.LOST_TRACK)
TABLE_HIDDEN = (Status.UNDER, Status.INSIDE, Status.HELD)
OFF_TABLE = (Status.GONE, Status.UNKNOWN)
TABLE_SAY = 'the table'


class RoomRules:
    """Room association, absence, return and place(), mixed into World."""

    def _reset_room(self) -> None:
        self._room: dict[str, RoomState] = {}
        self._departures: dict[str, tuple[float, float]] = {}   # name -> (t, wall) of its unconsumed departure
        self._conflicts: dict[str, dict[str, Conflict]] = {}     # name -> track id -> Conflict
        self.room_cfg = RoomConfig.from_dict(getattr(self, '_room_raw', None))

    # ----- public API -----------------------------------------------------------------------------

    def room_update(self, visit: ZoneVisit) -> list[Event]:
        """Apply one zone visit. Tracks of a class that is not an entity are ignored."""
        with self.lock:
            out: list[Event] = []
            for trk in visit.confirmed:
                if trk.cls in self.entities:
                    out += self._room_confirmed(trk.cls, trk, visit)
            for trk in visit.missed:
                out += self._room_missed(trk, visit)
            for trk in visit.dropped:
                self._room_dropped(trk)
            return out

    def place(self, name: str, now: Optional[float] = None) -> Place:
        """Where to say `name` is (now: wall time). The chain's outermost entity decides: in a room zone,
        a room place with room freshness; otherwise today's table place (resolve_chain's position)."""
        now = time.time() if now is None else now
        with self.lock:
            if name not in self.entities:
                return Place(kind='none', zone='', say='', status=Status.UNKNOWN, chain=[], via=name)
            pos, chain = relations.resolve_chain(name, self.entities, self.cfg.max_nesting)
            via = chain[-1]
            vent = self.entities[via]
            fresh_s = self.room_cfg.fresh_s
            conflicts = [c for c in self._conflicts.get(name, {}).values() if now - c.seen_wall <= fresh_s]
            st = self._room.get(via)
            if vent.zone != TABLE and st is not None:
                fresh = (not st.absent and st.misses < self.room_cfg.fresh_visits
                         and now - st.seen_wall <= fresh_s)
                return Place(kind='room', zone=st.zone, say=st.say, status=vent.status, chain=chain, via=via,
                             box_px=st.box_px, observed_directly=via == name, fresh=fresh, absent=st.absent,
                             arrived_wall=st.arrived_wall, last_seen_wall=st.seen_wall,
                             arrival_observed=st.arrival_observed, conflicts=conflicts)
            say = TABLE_SAY if vent.zone == TABLE else f'the {vent.zone}'
            return Place(kind='table', zone=vent.zone, say=say, status=vent.status, chain=chain, via=via,
                         pos_cm=pos, observed_directly=via == name, conflicts=conflicts)

    def room_json(self) -> dict:
        """For state_json: name -> its room state, plus 'conflicts' (every recorded conflict sighting)."""
        out: dict = {n: {'zone': st.zone, 'say': st.say, 'box_px': list(st.box_px), 'seen_wall': st.seen_wall,
                         'absent': st.absent, 'arrival_observed': st.arrival_observed}
                     for n, st in self._room.items()}
        out['conflicts'] = [{'entity': c.entity, 'zone': c.zone, 'say': c.say, 'track': c.track,
                             'box_px': list(c.box_px), 'seen_wall': c.seen_wall}
                            for per in self._conflicts.values() for c in per.values()]
        return out

    # ----- hooks called by world.py ------------------------------------------------------------------

    def _room_departure(self, ev: Event) -> None:
        """_emit hook: an EXITED_VIEW / LOST_TRACK while the entity is on the table is its latest
        departure (a newer one replaces an unconsumed older one)."""
        if ev.type in DEPARTURES and self.entities[ev.obj].zone == TABLE:
            self._departures[ev.obj] = (ev.t, ev.wall)

    def _room_return(self, name: str, ent: Entity) -> list[Event]:
        """_observe hook, after it reset the zone to 'table': confirmed table presence of a room entity.
        _observe runs only after the table presence flip, so one flicker never gets here. The tracker keeps
        the old track; the next visit re-decides it against the table (ignored while the table sees it)."""
        self._room.pop(name, None)
        self._conflicts.pop(name, None)
        return [self._emit(name, EventType.MOVED, to_cm=ent.pos_cm)]

    # ----- transitions ---------------------------------------------------------------------------------

    def _room_confirmed(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        st = self._room.get(name)
        if st is not None and st.track == trk.tid:
            self._room_refresh(st, trk)
            return []
        if trk.role == 'conflict':            # never upgrades
            self._room_conflict(name, trk, visit)
            return []
        op = self._room_decide(name, trk, visit)
        if op == 'acquire':
            return self._room_acquire(name, trk, visit)
        if op == 'reacquire':
            return self._room_reacquire(name, trk, visit)
        if op == 'ignored':
            trk.role, trk.entity = 'ignored', None
            return []
        self._room_conflict(name, trk, visit)
        return []

    def _room_decide(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> str:
        """'acquire' | 'reacquire' | 'ignored' | 'conflict' for a confirmed track that is not the
        entity's own. The table wins while it has the object: seen there within table_fresh_s, the track
        is something else (re-decided next visit); believed hidden there (UNDER / INSIDE / HELD), the
        belief stands. Only a GONE / UNKNOWN entity can be acquired: one seen on the table again after its
        departure is not authorised by that old departure."""
        ent, rc = self.entities[name], self.room_cfg
        if ent.zone == TABLE:
            if ent.status == Status.VISIBLE and visit.t - self._seen_t.get(name, float('-inf')) <= rc.table_fresh_s:
                return 'ignored'
            dep = self._departures.get(name)
            if (ent.status in OFF_TABLE and dep is not None and visit.t - dep[0] <= rc.handoff_s
                    and trk.first_seen > dep[0]):
                return 'acquire'
            return 'conflict'
        st = self._room.get(name)
        if (st is not None and st.absent and visit.zone == st.zone and st.absent_t is not None
                and trk.first_seen > st.absent_t and geom.iou(trk.box_px, st.box_px) >= REACQUIRE_IOU):
            return 'reacquire'
        return 'conflict'

    def _room_acquire(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        del self._departures[name]            # consumed: one departure, one acquisition
        ent = self.entities[name]
        self._room[name] = RoomState(zone=visit.zone, say=visit.say, box_px=trk.box_px, track=trk.tid,
                                     seen_t=trk.last_seen, seen_wall=trk.last_wall, arrived_wall=trk.first_wall,
                                     arrival_observed=True, table_pos_cm=ent.pos_cm)
        return self._room_settle(name, ent, trk, visit)

    def _room_reacquire(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        st = self._room[name]
        st.box_px, st.track, st.seen_t, st.seen_wall = trk.box_px, trk.tid, trk.last_seen, trk.last_wall
        st.arrived_wall, st.arrival_observed = trk.first_wall, False
        st.misses, st.absent, st.absent_t = 0, False, None
        return self._room_settle(name, self.entities[name], trk, visit)

    def _room_settle(self, name: str, ent: Entity, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        """The entity is VISIBLE in the visit's zone. No table cm: pos_cm / box_cm are cleared, and the table
        debounce restarts so that only a new presence flip (Return) brings it back to the table."""
        ent.status, ent.zone, ent.parent, ent.candidates, ent.confidence, ent.edge = \
            Status.VISIBLE, visit.zone, None, [], 1.0, None
        ent.pos_cm, ent.box_cm, ent.held_since, ent.pre_pickup_pos = None, None, None, None
        self._bits[name].clear()
        self._present[name] = False
        self._confirmed.add(name)
        trk.role, trk.entity = 'assoc', name
        return [self._emit(name, EventType.FOUND, t=visit.t, wall=visit.wall, img=visit.crop)]

    def _room_refresh(self, st: RoomState, trk: RoomTrack) -> None:
        """Its own track matched again. An observation older than the one last applied is ignored."""
        if trk.last_seen < st.seen_t:
            return
        st.box_px, st.seen_t, st.seen_wall, st.misses = trk.box_px, trk.last_seen, trk.last_wall, 0

    def _room_conflict(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> None:
        trk.role = 'conflict'
        self._conflicts.setdefault(name, {})[trk.tid] = Conflict(
            entity=name, zone=visit.zone, say=visit.say, track=trk.tid, box_px=trk.box_px,
            seen_t=trk.last_seen, seen_wall=trk.last_wall)

    def _room_missed(self, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        """A valid visit without a match for the entity's own track. Blocked visits never reach here (the
        tracker leaves them out), so a person in front of the zone counts for nothing."""
        name = trk.cls
        st = self._room.get(name)
        if st is None or st.track != trk.tid:
            return []
        st.misses = trk.misses
        if st.misses < self.room_cfg.absent_visits or st.absent:
            return []
        st.absent, st.absent_t = True, visit.t
        ent = self.entities[name]
        ent.status, ent.parent, ent.candidates, ent.held_since = Status.UNKNOWN, None, [], None
        # zone stays the room zone, so _emit does not count this LOST_TRACK as a departure
        return [self._emit(name, EventType.LOST_TRACK, t=visit.t, wall=visit.wall, img=visit.crop)]

    def _room_dropped(self, trk: RoomTrack) -> None:
        st = self._room.get(trk.cls)
        if st is not None and st.track == trk.tid:
            st.track = None                   # released; the state stays (absent) for reacquire
        for name in list(self._conflicts):
            self._conflicts[name].pop(trk.tid, None)
            if not self._conflicts[name]:
                del self._conflicts[name]
