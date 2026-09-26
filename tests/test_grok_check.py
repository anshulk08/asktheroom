"""Grok settle check (core/grok_check.py): one call per settled table, never during activity, offline,
over the cap or inside min_gap_s; a perception thread that never waits on it; verdict rows with table cm
and times; unnamed things named without ever replacing a taught name; phantoms recorded but the world
untouched; and the WHERE fallback in voice.visual that answers from a stored sighting, offline too."""
import json
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.grok_check import GrokCheck, from_config, same_thing
from core.narration import FakeProvider, ProviderError
from core.narration_store import med_claim
from core.types import Answer, Detection, Detections, Entity, Event, EventType, Frame, Status
from core.visual_memory import VisualConfig
from voice.intents import parse
from voice.visual import VisualQA

CFG = load_config()
T0 = datetime(2026, 9, 26, 10, 0, 0).timestamp()        # local 10:00 AM
W, H = 1280, 720                                        # the sim frame: 90 x 60 cm, 14.2 px/cm


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class ThingWorld(FakeWorld):
    """FakeWorld plus the open-world calls (thing_labels, find, bind_alias) and an update() to attach to."""
    def __init__(self, entities, events, labels=None):
        super().__init__(entities, events)
        self.labels = dict(labels or {})
        self.emit: list = []

    def thing_labels(self):
        return {n: self.labels.get(n) for n in self.entities if n.startswith("thing:")}

    def find(self, name):
        return name if name in self.entities else next((n for n, v in self.labels.items() if v == name), None)

    def bind_alias(self, entity, name, by=None):
        if entity not in self.entities or entity in self.labels:
            return False
        self.labels[entity] = name
        return True

    def update(self, dets, frame):
        out, self.emit = self.emit, []
        return out


@pytest.fixture
def log(tmp_path):
    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    yield lg
    lg.close()


def world(log, labels=None, wallet=Status.UNKNOWN):
    return ThingWorld([
        Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 15.0), box_cm=(17.0, 13.0, 23.0, 17.0)),
        Entity("box", "container", Status.VISIBLE, pos_cm=(70.0, 40.0), box_cm=(62.0, 34.0, 78.0, 46.0)),
        Entity("thing:3", "target", Status.VISIBLE, pos_cm=(45.0, 50.0), box_cm=(41.0, 47.0, 49.0, 53.0)),
        Entity("wallet", "target", wallet, pos_cm=(30.0, 30.0) if wallet == Status.VISIBLE else None),
        Entity("pill_bottle", "target", Status.UNKNOWN)], log, labels)


def reply(marks=(), unmarked=()):
    return json.dumps({"marks": [dict(mark=m, real=r, label=lab, confidence=c) for m, r, lab, c in marks],
                       "unmarked": [dict(label=lab, point={"x": x, "y": y}, confidence=c)
                                    for lab, x, y, c in unmarked]})


def checker(log, rep=None, w=None, online=True, clock=None, start=False, **kw):
    from server.sim import SimTable
    cfg = {**CFG, "grok_check": dict(enabled=True, **kw)}
    prov = rep if isinstance(rep, FakeProvider) else FakeProvider(rep if rep is not None else reply())
    g = GrokCheck(cfg, log, table=SimTable(CFG), provider=prov, online=lambda: online() if callable(online)
                  else online, clock=clock or Clock(T0), start=start)
    if w is not None:
        g.attach(w)
    return g, prov


def img():
    return np.full((H, W, 3), 170, np.uint8)


def feed(g, w, t, hands=False, events=()):
    """One perception frame through the attached world at T0 + t."""
    w.emit = list(events)
    hs = [Detection("hand:1", 0.9, (0, 0, 50, 50), (1.0, 1.0), (0.0, 0.0, 2.0, 2.0))] if hands else []
    w.update(Detections(t=t, frame_idx=int(t * 10), items=[], hands=hs), Frame(t=t, wall=T0 + t, img=img(), idx=0))


def run(g, w, clock, t0, t1, hands=False, step=0.1):
    t = t0
    while t < t1 - 1e-9:
        clock.t = T0 + t
        feed(g, w, round(t, 3), hands)
        t += step


# ---------------------------------------------------------------- when it calls

def test_one_call_at_start_up_and_one_per_settle_none_during_activity(log):
    w, clock = world(log), Clock(T0)
    g, prov = checker(log, w=w, clock=clock, quiet_s=1.5, min_gap_s=5)
    run(g, w, clock, 0.0, 1.4)
    assert g.run_once() is None and prov.calls == []            # not settled for quiet_s yet
    run(g, w, clock, 1.4, 1.6)
    assert g.run_once() is not None and len(prov.calls) == 1   # start-up check
    run(g, w, clock, 10.0, 14.0, hands=True)                   # someone moves things around
    assert g.run_once() is None and len(prov.calls) == 1
    run(g, w, clock, 14.0, 15.4)                               # hands gone, not yet quiet_s
    assert g.run_once() is None
    run(g, w, clock, 15.4, 15.8)
    assert g.run_once() is not None and len(prov.calls) == 2
    run(g, w, clock, 15.8, 30.0)                               # a quiet table asks nothing more
    assert g.run_once() is None and len(prov.calls) == 2


def test_a_world_event_alone_opens_and_settles_an_episode(log):
    w, clock = world(log), Clock(T0)
    g, prov = checker(log, w=w, clock=clock)
    run(g, w, clock, 0.0, 1.6)
    g.run_once()
    clock.t = T0 + 20
    feed(g, w, 20.0, events=[Event(T0 + 20, "keys", EventType.MOVED, None, None, None)])
    run(g, w, clock, 20.1, 21.7)
    assert g.run_once() is not None and len(prov.calls) == 2


def test_offline_capped_and_min_gap(log):
    w, clock, net = world(log), Clock(T0), {"up": False}
    g, prov = checker(log, w=w, clock=clock, online=lambda: net["up"], max_per_hour=2, min_gap_s=5)
    run(g, w, clock, 0.0, 1.6)
    assert g.run_once() is None and prov.calls == []           # offline: nothing sent, the frame waits
    net["up"] = True
    assert g.run_once() is not None and len(prov.calls) == 1   # back online: the waiting frame goes
    run(g, w, clock, 2.0, 2.3, hands=True)
    run(g, w, clock, 2.3, 4.0)                                 # settled again 2 s after the last call
    assert g.run_once() is None                                # inside min_gap_s: waits
    clock.t = T0 + 6.5
    assert g.run_once() is not None and len(prov.calls) == 2   # the same settled frame, once allowed
    run(g, w, clock, 10.0, 10.3, hands=True)
    run(g, w, clock, 10.3, 12.0)
    clock.t = T0 + 30
    assert g.run_once() is None and len(prov.calls) == 2       # hourly cap
    clock.t = T0 + 3700
    assert g.run_once() is not None and len(prov.calls) == 3   # an hour later it goes


def test_a_settled_frame_is_dropped_when_activity_began_again(log):
    w, clock = world(log), Clock(T0)
    g, prov = checker(log, w=w, clock=clock)
    run(g, w, clock, 0.0, 1.6)
    run(g, w, clock, 1.6, 2.0, hands=True)                     # the world's boxes no longer match that frame
    assert g.run_once() is None and prov.calls == [] and g.dropped == 1


def test_feed_never_raises_and_never_waits_on_a_stalled_call(log):
    gate = threading.Event()

    def slow(job):
        gate.wait(5)
        return reply()

    w, clock = world(log), Clock(T0)
    g, prov = checker(log, rep=FakeProvider(slow), w=w, clock=clock, start=True, min_gap_s=0)
    try:
        g.feed(None, None)                                     # broken input: swallowed
        t0 = time.perf_counter()
        for i in range(60):                                    # settles repeatedly while the call hangs
            clock.t = T0 + i
            feed(g, w, float(i), hands=(i % 4 < 2))
        assert time.perf_counter() - t0 < 0.5
        deadline = time.monotonic() + 2
        while not prov.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(prov.calls) == 1                            # one call in flight; the rest wait in one slot
    finally:
        gate.set()
        g.stop()


def test_a_failed_call_is_reported_and_stores_nothing(log):
    w, clock = world(log), Clock(T0)
    g, _ = checker(log, rep=FakeProvider([ProviderError("HTTP 503")]), w=w, clock=clock)
    s = g.check(img(), T0)
    assert "503" in s["error"] and g.store.rows() == []
    assert "failed" not in json.dumps(w.state_json()["grok_check"]["disclosure"])


# ---------------------------------------------------------------- what it records and does

def test_verdict_rows_with_table_cm_and_times(log):
    w = world(log)
    rep = reply(marks=[(1, True, "car keys", 0.9),         # keys: agree
                       (2, False, None, 0.9),              # box: phantom
                       (3, True, "red mug", 0.9),          # thing:3: named
                       (9, True, "ghost", 0.9)],           # not a mark: ignored
                unmarked=[("brown wallet", 0.5, 0.5, 0.8),  # (45, 30) cm
                          ("keys", 0.222, 0.25, 0.9),       # inside mark 1: a repeat, ignored
                          ("coaster", 0.9, 0.9, 0.3)])      # unsure
    g, prov = checker(log, rep, w=w)
    s = g.check(img(), T0 + 5, episode="ep1", t=5.0)
    texts = " ".join(p[1] for p in prov.calls[0].parts if p[0] == "text")
    assert "1 = keys" in texts and "2 = box" in texts and "3 = unnamed object" in texts
    rows = {r["grok_label"] or r["world_label"]: r for r in g.store.rows()}
    assert rows["car keys"]["verdict"] == "agree" and rows["car keys"]["entity"] == "keys"
    assert rows["car keys"]["x_cm"] == pytest.approx(20, abs=0.5) and rows["car keys"]["y_cm"] == pytest.approx(15, abs=0.5)
    assert rows["box"]["verdict"] == "phantom"
    assert rows["red mug"]["verdict"] == "named"
    assert rows["brown wallet"]["verdict"] == "unmarked"
    assert (rows["brown wallet"]["x_cm"], rows["brown wallet"]["y_cm"]) == pytest.approx((45, 30), abs=0.5)
    assert rows["coaster"]["verdict"] == "unsure"
    assert "ghost" not in rows and len(g.store.rows()) == 5
    assert not g._on_table(120.0, 30.0) and g._on_table(-3.0, 62.0)    # the sim frame is all table
    r = rows["car keys"]
    assert r["wall"] == T0 + 5 and r["t"] == 5.0 and r["episode"] == "ep1" and r["model"] == "fake" and r["latency_ms"] == 1
    assert s["agree"] == 1 and s["phantom"] == ["box"] and s["unmarked"] == ["brown wallet"]
    assert "phantom: box" in s["text"]


def test_a_phantom_verdict_leaves_the_world_as_it_was(log):
    w = world(log)
    before = json.dumps(w.state_json()["entities"], sort_keys=True)
    g, _ = checker(log, reply(marks=[(1, False, None, 0.95), (2, False, None, 0.95)]), w=w)
    g.check(img(), T0)
    assert json.dumps(w.state_json()["entities"], sort_keys=True) == before


def test_relabel_and_synonyms(log):
    w = world(log)
    g, _ = checker(log, reply(marks=[(1, True, "wallet", 0.9), (2, True, "cardboard box", 0.9)]), w=w)
    g.check(img(), T0)
    v = {r["entity"]: r["verdict"] for r in g.store.rows()}
    assert v == {"keys": "relabel", "box": "agree"}


def test_unnamed_things_take_the_label_but_never_over_a_taught_name(log):
    w = world(log)
    g, _ = checker(log, reply(marks=[(3, True, "red mug", 0.9)]), w=w)
    assert g.check(img(), T0)["bound"] == ["thing:3 = red mug"] and w.labels["thing:3"] == "red mug"

    w = world(log, labels={"thing:3": "charger"})              # taught: Grok disagrees, name stays
    g, _ = checker(log, reply(marks=[(3, True, "red mug", 0.9)]), w=w)
    g.check(img(), T0)
    assert w.labels["thing:3"] == "charger" and g.store.rows()[0]["verdict"] == "relabel"

    w = world(log)                                             # sure enough to record, not to name
    g, _ = checker(log, reply(marks=[(3, True, "red mug", 0.65)]), w=w)
    assert g.check(img(), T0)["bound"] == [] and "thing:3" not in w.labels

    w = world(log)                                             # a name something else answers to
    g, _ = checker(log, reply(marks=[(3, True, "keys", 0.9)]), w=w)
    assert g.check(img(), T0)["bound"] == [] and "thing:3" not in w.labels

    w = world(log)
    g, _ = checker(log, reply(marks=[(3, True, "red mug", 0.9)]), w=w, bind_names=False)
    assert g.check(img(), T0)["bound"] == [] and "thing:3" not in w.labels


def test_labels_are_safe_to_say(log):
    w = world(log)
    g, _ = checker(log, reply(marks=[(3, True, "the pills she took", 0.9)],
                              unmarked=[("a very long label of many words", 0.5, 0.5, 0.9)]), w=w)
    g.check(img(), T0)
    assert [r["verdict"] for r in g.store.rows()] == ["unsure"] and "thing:3" not in w.labels


def test_fake_provider_agrees_with_the_tracker(log):
    w = world(log)
    cfg = {**CFG, "grok_check": dict(enabled=True, provider="fake")}
    from server.sim import SimTable
    g = GrokCheck(cfg, log, table=SimTable(CFG), clock=Clock(T0), start=False).attach(w)
    s = g.check(img(), T0)
    assert s["agree"] == 2 and s["unsure"] == 1 and not s["phantom"]


def test_attach_status_and_disclosure(log):
    w, clock = world(log), Clock(T0)
    g, _ = checker(log, reply(marks=[(2, False, None, 0.9)]), w=w, clock=clock)
    st = w.state_json()["grok_check"]
    assert st["enabled"] and st["last"] is None and "still frame" in st["disclosure"]
    run(g, w, clock, 0.0, 1.6)
    g.run_once()
    st = w.state_json()["grok_check"]
    assert st["calls_last_hour"] == 1 and st["last"]["phantom"] == ["box"]
    json.dumps(st)                                             # the dashboard gets it as JSON


def test_from_config_is_off_by_default(log):
    assert from_config(CFG, log, world(log)) is None
    assert CFG["grok_check"]["enabled"] is False


def test_rows_older_than_keep_h_are_pruned_on_start(log):
    g, _ = checker(log)
    g.store.add([{"wall": T0 - 30 * 3600, "verdict": "unmarked", "grok_label": "mug"},
                  {"wall": T0 - 3600, "verdict": "unmarked", "grok_label": "cup"}])
    checker(log)
    assert [r["grok_label"] for r in g.store.rows()] == ["cup"]


def test_same_thing():
    assert same_thing("keys", "car keys") and same_thing("phone", "smartphone") and same_thing("glasses", "eyeglasses")
    assert same_thing("pill bottle", "Pill Bottle") and same_thing("box", "cardboard box")
    assert not same_thing("keys", "wallet") and not same_thing("mug", None) and not same_thing("", "mug")


# ---------------------------------------------------------------- answers from sightings

def sighted(log, w, label, x=0.5, y=0.5, conf=0.9, when=T0):
    g, _ = checker(log, reply(unmarked=[(label, x, y, conf)]), w=w, clock=Clock(T0 + 60))
    g.check(img(), when)
    return g


def qa(log, w, g):
    from server.sim import SimTable
    q = VisualQA(CFG, w, log, None, SimTable(CFG), provider=FakeProvider("{}"), online=lambda: True,
                 c=VisualConfig(enabled=True, embed="fake"), clock=Clock(T0 + 60))
    q.grok_check = g
    return q


def test_where_falls_back_to_a_sighting_for_a_never_placed_object(log):
    w = world(log)
    q = qa(log, w, sighted(log, w, "black wallet"))
    a = q.route(parse("Where is my wallet?", CFG), "Where is my wallet?", online=True)
    assert a.action is None and a.point_at is None and a.target_cm is None   # an old VLM sighting: spoken only
    assert "10:00 AM" in a.text and "wallet" in a.text
    a2 = q.route(parse("Where is my wallet?", CFG), "Where is my wallet?", online=False)
    assert a2 == a                                              # stored rows: the same offline


def test_the_world_model_answers_whenever_it_has_a_position(log):
    w = world(log, wallet=Status.VISIBLE)
    q = qa(log, w, sighted(log, w, "wallet"))
    assert q.route(parse("Where is my wallet?", CFG), "Where is my wallet?", online=True) is None


def test_no_sighting_no_checker_or_too_old_leaves_the_templates(log):
    w = world(log)
    q = qa(log, w, sighted(log, w, "phone"))                   # a sighting of something else
    assert q.route(parse("Where is my wallet?", CFG), "Where is my wallet?", online=True) is None
    q = qa(log, w, sighted(log, w, "wallet", when=T0 - 3600))  # older than sighting_max_age_s
    assert q.route(parse("Where is my wallet?", CFG), "Where is my wallet?", online=True) is None
    q = qa(log, w, None)
    assert q.route(parse("Where is my wallet?", CFG), "Where is my wallet?", online=True) is None


def test_offline_an_unknown_name_uses_a_sighting_online_grok_picks(log):
    w = world(log)
    q = qa(log, w, sighted(log, w, "red mug"))
    a = q.route(parse("Where is my red mug?", CFG), "Where is my red mug?", online=False)
    assert a.action is None and a.target_cm is None and "red mug" in a.text
    seen = []                                                  # online: a live Grok look, not the sighting
    q.pick = lambda t, said: seen.append(said) or Answer("P")
    q.look = lambda t, intent: seen.append(intent.name or intent.obj) or Answer("P")
    assert q.route(parse("Where is my red mug?", CFG), "Where is my red mug?", online=True) == Answer("P")
    assert seen == ["red mug"]


def test_a_pill_bottle_sighting_answer_stays_neutral(log):
    w = world(log)
    q = qa(log, w, sighted(log, w, "pill bottle"))
    a = q.route(parse("Where are my pills?", CFG), "Where are my pills?", online=True)
    assert "pill bottle" in a.text and not med_claim(a.text)


def test_build_fake_attaches_it_when_enabled_and_offline_it_sends_nothing(monkeypatch):
    import main
    import net
    import voice.understand
    monkeypatch.setattr(net.NetMonitor, "probe", lambda self: False)
    monkeypatch.setattr(voice.understand.Qwen, "health", lambda self, timeout=1.0: False)
    cfg = {**CFG, "grok_check": dict(CFG["grok_check"], enabled=True, provider="fake")}
    room, _ = main.build(cfg, fake=True, with_voice=False)
    try:
        deadline = time.monotonic() + 3
        while room.frames.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(2.0)                                         # past quiet_s: settled, but offline
        st = room.world.state_json()["grok_check"]
        assert st["enabled"] and st["calls_last_hour"] == 0 and st["last"] is None
    finally:
        room.shutdown()
