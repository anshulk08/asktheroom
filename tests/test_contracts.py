import time

from core.config import display_name, load_config
from core.events import EventLog
from core.fakeworld import demo_world
from core.types import EVENT_TYPES, Event, Status


def test_config_loads():
    cfg = load_config()
    assert len(cfg["objects"]) == 8
    assert cfg["synonyms"]["pills"] == "pill_bottle"
    assert display_name(cfg, "pill_bottle") == "pill bottle"


def test_eventlog_roundtrip(tmp_path):
    log = EventLog(str(tmp_path / "e.db"), str(tmp_path / "snaps"))
    now = time.time()
    log.add(Event(t=1.0, wall=now, obj="keys", type="PICKED_UP", from_cm=(1.0, 2.0), parent="hand:1"))
    log.add(Event(t=2.0, wall=now + 1, obj="keys", type="MOVED", to_cm=(5.0, 6.0)))
    last = log.last("keys", 3)
    assert [e.type for e in last] == ["MOVED", "PICKED_UP"]
    assert last[1].from_cm == (1.0, 2.0) and last[1].to_cm is None
    assert log.last_of_type("keys", ["PICKED_UP"]).parent == "hand:1"
    assert len(log.since(now + 0.5)) == 1


def test_eventlog_1000_inserts_fast(tmp_path):
    log = EventLog(str(tmp_path / "e.db"), str(tmp_path / "snaps"))
    t0 = time.perf_counter()
    for i in range(1000):
        log.add(Event(t=i, wall=i, obj="keys", type=EVENT_TYPES[i % len(EVENT_TYPES)]))
    assert time.perf_counter() - t0 < 1.0


def test_demo_world_resolves_through_parent():
    w = demo_world()
    pos, chain = w.resolve("keys")
    assert chain == ["keys", "box"] and pos == (70.4, 38.1)
    assert w.get("wallet").status == Status.VISIBLE
    s = w.state_json()
    assert ["keys", "INSIDE", "box"] in s["edges"]
    assert w.history("keys", 1)[0].type == "PUT_INSIDE"
