"""A stand-in World with the same read API (get, resolve, place, history, state_json).

Used by tests and by `--fake` dev runs of the server and voice loop until core/world.py exists.
"""
from __future__ import annotations

import time
from typing import Optional

from core.events import EventLog
from core.room_types import TABLE, Place
from core.types import Entity, Event, Point, Status, entity_json

MAX_DEPTH = 3


class FakeWorld:
    def __init__(self, entities: list[Entity], events: Optional[EventLog] = None):
        self.entities = {e.name: e for e in entities}
        self.events = events or EventLog(":memory:", "/tmp/askroom_fake_snaps")
        self.online = False
        self.fps = 0.0
        self.laser = {"on": False, "target": None, "err_cm": None}
        self.places: dict[str, Place] = {}

    def get(self, name: str) -> Entity:
        return self.entities[name]

    def resolve(self, name: str) -> tuple[Optional[Point], list[str]]:
        chain = [name]
        e = self.entities[name]
        for _ in range(MAX_DEPTH):
            if e.status == Status.VISIBLE or e.parent is None or e.parent not in self.entities:
                break
            e = self.entities[e.parent]
            chain.append(e.name)
        return e.pos_cm, chain

    def place(self, name: str, now: Optional[float] = None) -> Place:
        """Where to say `name` is: a place set with set_place, else the table place from resolve()."""
        if name in self.places:
            return self.places[name]
        pos, chain = self.resolve(name)
        via = chain[-1]
        return Place(kind=TABLE, zone=TABLE, say="the table", status=self.entities[via].status, chain=chain,
                     via=via, pos_cm=pos, observed_directly=via == name)

    def history(self, name: str, n: int = 3) -> list[Event]:
        return self.events.last(name, n)

    def state_json(self) -> dict:
        ents, edges = [], []
        for e in self.entities.values():
            ents.append(entity_json(e, self.resolve(e.name)[0]))
            if e.status in (Status.INSIDE, Status.UNDER, Status.HELD) and e.parent:
                edges.append([e.name, e.status.value, e.parent])
            elif e.status == Status.VISIBLE:
                edges.append([e.name, "ON", "table"])
        return {"t": time.time(), "online": self.online, "fps": self.fps,
                "entities": ents, "edges": edges, "laser": dict(self.laser)}

    # test helpers
    def set(self, name: str, **fields) -> None:
        e = self.entities[name]
        for k, v in fields.items():
            setattr(e, k, v)

    def set_place(self, name: str, place: Place) -> None:
        """Make place(name) return `place` (a room place, or a table place with conflicts)."""
        self.places[name] = place


def demo_world(events: Optional[EventLog] = None) -> FakeWorld:
    """The headline scene: keys inside the box, pill bottle under the notebook, phone carried off left."""
    now = time.time()
    ents = [
        Entity("keys", "target", Status.INSIDE, parent="box", pos_cm=(41.2, 29.0),
               last_seen=now - 95, confidence=0.85),
        Entity("pill_bottle", "target", Status.UNDER, parent="notebook", pos_cm=(20.0, 40.0),
               last_seen=now - 300, confidence=0.85),
        Entity("wallet", "target", Status.VISIBLE, pos_cm=(60.0, 15.0), last_seen=now, confidence=1.0),
        Entity("glasses", "target", Status.UNKNOWN, parent="unknown", pos_cm=(80.0, 50.0),
               last_seen=now - 600, confidence=0.4),
        Entity("phone", "target", Status.GONE, pos_cm=(3.0, 30.0), last_seen=now - 120,
               confidence=0.9, edge="left"),
        Entity("remote", "target", Status.HELD, parent="hand:2", pos_cm=(50.0, 45.0),
               last_seen=now - 2, confidence=0.9, pre_pickup_pos=(50.0, 45.0)),
        Entity("box", "container", Status.VISIBLE, pos_cm=(70.4, 38.1), last_seen=now, confidence=1.0),
        Entity("notebook", "cover", Status.VISIBLE, pos_cm=(20.5, 39.0), last_seen=now, confidence=1.0),
    ]
    w = FakeWorld(ents, events)
    t0 = time.monotonic()
    for ago, obj, typ, kw in [
        (400, "keys", "PICKED_UP", dict(from_cm=(30.0, 20.0), parent="hand:1", confidence=0.9)),
        (395, "keys", "PUT_INSIDE", dict(to_cm=(41.2, 29.0), parent="box", confidence=0.85)),
        (95, "box", "MOVED", dict(from_cm=(41.2, 29.0), to_cm=(70.4, 38.1), confidence=1.0)),
        (310, "pill_bottle", "PICKED_UP", dict(from_cm=(20.0, 40.0), parent="hand:1", confidence=0.9)),
        (305, "pill_bottle", "PUT_BACK", dict(to_cm=(20.0, 40.0), confidence=1.0)),
        (300, "pill_bottle", "COVERED", dict(to_cm=(20.0, 40.0), parent="notebook", confidence=0.85)),
        (600, "glasses", "LOST_TRACK", dict(from_cm=(80.0, 50.0), confidence=0.4)),
        (125, "phone", "PICKED_UP", dict(from_cm=(40.0, 30.0), parent="hand:1", confidence=0.9)),
        (120, "phone", "EXITED_VIEW", dict(from_cm=(3.0, 30.0), edge="left", confidence=0.9)),
        (2, "remote", "PICKED_UP", dict(from_cm=(50.0, 45.0), parent="hand:2", confidence=0.9)),
    ]:
        w.events.add(Event(t=t0 - ago, wall=now - ago, obj=obj, type=typ, **kw))
    return w
