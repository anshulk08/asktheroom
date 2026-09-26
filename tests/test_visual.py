"""Visual memory: the frame archive (policy, pruning, disk cap, never blocking), text search with time
windows, live visual Q&A with box grounding and pointing, recall from the archive, routing between the
world model / narrations / VLM, the medication rule on VLM output, and the pointing hook in main.Room."""
import json
import os
import time
from datetime import datetime

import numpy as np
import pytest

from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.narration import FakeProvider
from core.narration_store import NarrationStore
from core.types import Answer, Detection, Detections, Entity, Event, Frame, Status
from core.visual_memory import FakeEmbedder, VisualArchive, VisualConfig, digest, letterbox, tiles
from voice.intents import parse
from voice.visual import ABSTAIN, OFFLINE, PILLS_SAFE, VisualQA, point_to_px, px_to_cm, query_phrase

CFG = load_config()
T0 = datetime(2026, 9, 25, 9, 0, 0).timestamp()        # local 9:00 AM
W, H = 1280, 720


def img(color=(180, 190, 200), block=None, at=(600, 300), size=120):
    im = np.zeros((H, W, 3), np.uint8)
    im[:] = color
    if block is not None:
        x, y = at
        im[y:y + size, x:x + size] = block
    return im


def frame(t, im):
    return Frame(t=t, wall=T0 + t, img=im, idx=int(t * 10))


def dets(t, hands=0):
    hs = [Detection("hand:1", 0.9, (0, 0, 50, 50), (1.0, 1.0), (0.0, 0.0, 2.0, 2.0))] * hands
    return Detections(t=t, frame_idx=0, items=[], hands=hs)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def vcfg(**kw):
    base = dict(enabled=True, embed="fake", archive_every_s=30.0, check_every_s=1.0, change_thr=0.3,
                keep_h=24.0, max_mb=500.0, tiles=(0, 0), min_sim=0.2, recall_frames=4, abstain_below=0.5)
    base.update(kw)
    return VisualConfig(**base)


@pytest.fixture
def log(tmp_path):
    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    yield lg
    lg.close()


def archive(log, world=None, **kw):
    c = vcfg(**kw)
    return VisualArchive(CFG, log, world, embedder=FakeEmbedder(), start=False, c=c, clock=Clock(T0 + 3600))


# ---------------------------------------------------------------- archive policy

def test_first_interval_and_change_policy(log):
    a = archive(log)
    same = img()
    for t in np.arange(0, 20, 0.5):                    # 20 s of an unchanged table
        a.feed(frame(float(t), same), dets(float(t)), [])
        a.drain()
    rows = a.store.window(T0 - 1, T0 + 1e4)
    assert [r.reason for r in rows] == ["first"]
    red = img(block=(0, 0, 255))
    for t in np.arange(20, 24, 0.5):                   # a hand puts something red down ...
        a.feed(frame(float(t), red), dets(float(t), 1), [])
        a.drain()
    assert len(a.store.window(T0 - 1, T0 + 1e4)) == 1   # ... nothing saved while the hand is there
    for t in np.arange(24, 26, 0.5):                   # hand gone: the settled view is saved once
        a.feed(frame(float(t), red), dets(float(t)), [])
        a.drain()
    rows = a.store.window(T0 - 1, T0 + 1e4)
    assert [r.reason for r in rows] == ["first", "change"] and not rows[1].hands
    for t in np.arange(26, 58, 0.5):
        a.feed(frame(float(t), red), dets(float(t)), [])
        a.drain()
    assert [r.reason for r in a.store.window(T0 - 1, T0 + 1e4)] == ["first", "change", "interval"]


def test_a_hand_passing_over_an_unchanged_table_saves_nothing(log):
    a = archive(log)
    base = img()
    a.feed(frame(0.0, base), dets(0.0), [])
    a.drain()
    for t in np.arange(1, 5, 1.0):
        a.feed(frame(float(t), img(block=(140, 170, 215))), dets(float(t), 1), [])   # the hand itself
        a.drain()
    for t in np.arange(5, 8, 1.0):
        a.feed(frame(float(t), base), dets(float(t)), [])
        a.drain()
    assert len(a.store.window(T0 - 1, T0 + 1e4)) == 1


def test_world_events_mark_a_change_even_without_pixels_moving(log):
    a = archive(log)
    a.feed(frame(0.0, img()), dets(0.0), [])
    a.drain()
    a.feed(frame(1.0, img()), dets(1.0, 1), [Event(t=1.0, wall=T0 + 1, obj="keys", type="PUT_INSIDE")])
    a.drain()
    a.feed(frame(2.0, img()), dets(2.0), [])
    a.drain()
    assert [r.reason for r in a.store.window(T0 - 1, T0 + 1e4)] == ["first", "change"]


def test_feed_is_rate_limited_and_never_blocks(log):
    a = archive(log, check_every_s=1.0)
    im = img()
    t0 = time.perf_counter()
    for i in range(300):
        a.feed(frame(i / 30, im), dets(i / 30), [])      # 10 s at 30 fps, worker never runs
    assert (time.perf_counter() - t0) / 300 < 0.001
    assert a._q.qsize() == 2 and a.skipped >= 5           # bounded: later checks are skipped
    a.feed(None, None, None)                              # never raises


def test_digest_records_what_the_world_believed(log):
    fw = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(10.0, 30.0)),
                    Entity("pill_bottle", "target", Status.UNDER, parent="notebook", pos_cm=(50.0, 30.0)),
                    Entity("notebook", "cover", Status.VISIBLE, pos_cm=(80.0, 30.0))], log)
    d = digest(fw, CFG)
    assert {"name": "keys", "area": "left"} in d["visible"] and {"name": "notebook", "area": "right"} in d["visible"]
    assert d["hidden"] == [{"name": "pill bottle", "status": "under", "parent": "notebook"}]
    a = archive(log, world=fw)
    a.feed(frame(0.0, img()), dets(0.0), [])
    a.drain()
    assert a.store.window(T0 - 1, T0 + 1)[0].digest == d


def test_disk_cap_and_age_prune_oldest_first(log):
    a = archive(log, max_mb=1e-9)
    st = a.store
    rows = []
    for i in range(5):
        p = os.path.join(a.root, f"{i}.jpg")
        os.makedirs(a.root, exist_ok=True)
        open(p, "wb").write(b"x" * 1000)
        rows.append(st.add(T0 + i, p, 1000, 1.0, False, "interval", {}))
    st.prune(keep_h=24, max_mb=0.0025, now=T0 + 10)     # room for two
    left = st.window(T0 - 1, T0 + 100)
    assert [r.id for r in left] == rows[3:] and not os.path.exists(os.path.join(a.root, "0.jpg"))
    st.prune(keep_h=1 / 3600, max_mb=100, now=T0 + 10)  # everything older than a second
    assert st.window(T0 - 1, T0 + 100) == []


def test_tiles_and_letterbox():
    im = img()
    ts = tiles(im, (3, 2))
    assert len(ts) == 7 and ts[0] is im
    assert all(t.shape[0] > H / 2 and t.shape[1] > W / 3 for t in ts[1:])   # overlapping
    assert letterbox(im).shape == (W, W, 3)


# ---------------------------------------------------------------- search

def fill(a, spec):
    """spec: [(minutes after 9:00, colour block or None)] -> archived + embedded rows."""
    ids = []
    for m, block in spec:
        im = img(color=block) if block is not None else img()     # the fake embedder sees mean colour
        fr = frame(m * 60.0, im)
        th = __import__("core.visual_memory", fromlist=["_thumb"])._thumb(im)
        ids.append(a._save(fr, th, 20.0, False, "change"))
    a.drain()
    return ids


def test_search_ranks_by_text_within_the_time_window(log):
    a = archive(log)
    fill(a, [(0, None), (10, (0, 0, 255)), (20, (0, 0, 255)), (40, (0, 0, 255)), (50, (255, 0, 0)),
             (300, (0, 0, 255))])
    hits = a.search("a red mug", T0, T0 + 3600, k=3)
    got = [round((r.t - T0) / 60) for r, _ in hits]
    assert set(got) == {10, 20, 40} and 300 not in got
    assert hits[0][1] > 0.5
    blue = a.search("a blue book", T0, T0 + 3600, k=1)
    assert round((blue[0][0].t - T0) / 60) == 50


def test_search_keeps_hits_apart_in_time(log):
    a = archive(log)
    fill(a, [(0, (0, 0, 255)), (0.2, (0, 0, 255)), (0.4, (0, 0, 255)), (30, (0, 0, 255))])
    hits = a.search("red", T0 - 1, T0 + 3600, k=2, min_gap_s=60)
    assert [round((r.t - T0) / 60) for r, _ in hits] == [0, 30]


def test_query_phrases():
    assert query_phrase("Was there a red mug here this morning?") == "a red mug"
    assert query_phrase("When did the papers show up?") == "papers"
    assert query_phrase("did I leave my glasses on the table?") == "my glasses"
    assert query_phrase("What was on the table before I left?") is None
    assert query_phrase("was there anything here earlier") is None


# ---------------------------------------------------------------- geometry

def test_point_to_px_fractions_and_0_1000():
    assert point_to_px({"x": 0.25, "y": 0.5}, (1280, 720)) == (320.0, 360.0)
    assert point_to_px({"x": 250, "y": 500}, (1280, 720)) == (320.0, 360.0)      # 0-1000 tolerated
    assert point_to_px({"x": -0.2, "y": 1.4}, (1280, 720)) == (0.0, 720.0)        # clamped
    for bad in (None, {"x": "a", "y": 1}, {"x": 0.5}, "nope", {"x": 5000, "y": 1}):
        assert point_to_px(bad, (1280, 720)) is None


def test_px_to_cm_with_a_homography_and_with_cm_to_px_only():
    from main import FlatTable
    from server.sim import SimTable
    ft = FlatTable(CFG)
    assert np.allclose(px_to_cm(ft, [(640, 360)]), [[45, 30]])
    st = SimTable(CFG)
    assert np.allclose(px_to_cm(st, [(640, 360), (0, 0), (1280, 720)]), [[45, 30], [0, 0], [90, 60]], atol=1e-3)
    assert px_to_cm(None, [(1, 1)]) is None


# ---------------------------------------------------------------- A: looking now

class Frames:
    def __init__(self, im):
        self.im = im

    def latest(self):
        return None if self.im is None else Frame(t=0.0, wall=T0, img=self.im, idx=1)


def look_reply(answer="Your mug is on the right.", conf=0.9, mark=None, point=None):
    return json.dumps({"answer": answer, "confidence": conf, "mark": mark, "point": point})


def qa(log, reply, world=None, im="default", online=True, **kw):
    from server.sim import SimTable
    world = world or FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 15.0),
                                       box_cm=(17.0, 13.0, 23.0, 17.0), last_seen=T0),
                                Entity("box", "container", Status.VISIBLE, pos_cm=(70.0, 40.0),
                                       box_cm=(62.0, 34.0, 78.0, 46.0), last_seen=T0)], log)
    prov = FakeProvider(reply)
    q = VisualQA(CFG, world, log, Frames(img() if im == "default" else im), SimTable(CFG), provider=prov,
                 online=lambda: online, c=vcfg(**kw), clock=Clock(T0 + 3600))
    return q, prov


def test_look_marks_visible_entities_and_points_at_the_chosen_mark(log):
    q, prov = qa(log, look_reply("Your keys are near the top left.", mark=1))
    a = q.look("where are the shiny things?")
    assert a == Answer("Your keys are near the top left.", point_at="keys", action="point")
    [call] = prov.calls
    images = [p for p in call.parts if p[0] == "image"]
    assert len(images) == 1                                  # the marked frame; no close-ups asked for
    texts = " ".join(p[1] for p in call.parts if p[0] == "text")
    assert "1 = keys" in texts and "2 = box" in texts
    s = call.system.lower()
    assert "numbered" in s and "mark" in s and "medication" in s and "beyond the table" in s
    assert "box_2d" not in s
    assert "never mention the marks" in s                   # the listener can't see them
    assert "not the image" in s                              # 'near the top of the image' means nothing aloud


def test_marks_skip_hidden_and_unboxed_entities(log):
    world = FakeWorld([Entity("keys", "target", Status.INSIDE, parent="box", pos_cm=(70.0, 40.0)),
                       Entity("box", "container", Status.VISIBLE, pos_cm=(70.0, 40.0), box_cm=(62.0, 34.0, 78.0, 46.0)),
                       Entity("wallet", "target", Status.VISIBLE, pos_cm=(30.0, 30.0))], log)
    q, prov = qa(log, look_reply(mark=1), world=world)
    q.look("what's on the table?")
    texts = " ".join(p[1] for p in prov.calls[0].parts if p[0] == "text")
    assert "1 = box" in texts and "keys" not in texts.split("Marks:")[1].split("\n")[0] and "2 =" not in texts


def test_look_never_points_at_an_untracked_spot(log):
    """Behaviour change: the laser points only at a verified tracked entity. A VLM point on no tracked
    thing (here (9, 30) cm, an untracked note) gets the spoken answer, a brief 'not sure exactly
    where', and no laser target (it used to aim at the raw table position)."""
    # (0.1, 0.5) of the 1280x720 sim frame = (128, 360) px = (9, 30) cm
    spot = {"x": 0.1, "y": 0.5}
    q, _ = qa(log, look_reply("Your note is on the left, by the edge. It's yellow.", point=spot))
    a = q.look("where is my note?")
    assert a.point_at is None and a.action is None and a.target_cm is None
    assert a.text == "Your note is on the left, by the edge. I'm not sure exactly where, so I won't point."
    q, _ = qa(log, look_reply("There is a note on the left.", point=spot))
    a = q.look("what's on the table?")
    assert a.action is None and a.text == "There is a note on the left. I'm not sure exactly where, so I won't point."
    q, _ = qa(log, look_reply("There is a note on the left. It says milk.", point=spot))
    a = q.look("what does the note say?")                       # two sentences of content: both kept
    assert a.action is None and a.target_cm is None and a.text == "There is a note on the left. It says milk."


def test_look_point_near_but_not_on_a_tracked_entity_is_not_attached_to_it(log):
    """A point 5.5 cm from the keys' centre but outside their box is not silently taken to mean the
    keys (it used to be, within 6 cm)."""
    # (20, 20.5) cm = (284.4, 246) px = (0.2222, 0.3417) of the frame; keys box (17, 13, 23, 17)
    q, _ = qa(log, look_reply("Something small is there.", point={"x": 0.2222, "y": 0.3417}))
    a = q.look("what is that small thing?")
    assert a.point_at is None and a.action is None and a.target_cm is None


def test_look_point_on_nested_or_overlapping_entities(log):
    """A point on keys lying on the notebook is the keys (the smallest box, inside the other); a point
    where two boxes merely overlap is ambiguous: no laser target."""
    world = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 15.0), box_cm=(17.0, 13.0, 23.0, 17.0),
                              last_seen=T0),
                       Entity("notebook", "cover", Status.VISIBLE, pos_cm=(20.0, 15.0),
                              box_cm=(10.0, 8.0, 30.0, 22.0), last_seen=T0),
                       Entity("wallet", "target", Status.VISIBLE, pos_cm=(31.0, 20.0),
                              box_cm=(28.0, 18.0, 34.0, 22.0), last_seen=T0)], log)
    q, _ = qa(log, look_reply("Your keys are there.", point={"x": 0.222, "y": 0.25}), world=world)
    assert q.look("where are the shiny things?").point_at == "keys"
    # (29, 20) cm = (412.4, 240) px: inside both the notebook and the wallet, neither inside the other
    q, _ = qa(log, look_reply("It's there.", point={"x": 412.4 / 1280, "y": 240 / 720}), world=world)
    a = q.look("where is the brown thing?")
    assert a.point_at is None and a.action is None and a.target_cm is None


def test_look_point_on_a_tracked_entity_follows_it(log):
    # keys at (20, 15) cm = (284, 180) px = (0.222, 0.25): a point there resolves to the entity
    q, _ = qa(log, look_reply("Your keys are there.", point={"x": 0.222, "y": 0.25}))
    assert q.look("where are the shiny things?").point_at == "keys"


def test_look_ignores_bad_marks_and_points(log):
    for kw in ({"mark": 99}, {"mark": "one"}, {"mark": 0}, {"point": {"x": 5000, "y": 1}}, {"point": "nope"}, {}):
        q, _ = qa(log, look_reply(**kw))
        a = q.look("what's on the table?")
        assert a.point_at is None and a.target_cm is None and a.action is None, kw


def test_look_abstains_when_unsure(log):
    q, _ = qa(log, look_reply("Maybe a mug?", conf=0.3))
    assert q.look("what's on the table?").text == ABSTAIN


def test_look_without_a_frame(log):
    q, prov = qa(log, look_reply(), im=None)
    assert q.look("what's on the table?").text == "I can't see the table right now." and prov.calls == []


def test_look_applies_the_medication_rule(log):
    q, _ = qa(log, look_reply("You took your pills. The bottle is on the right."))
    assert q.look("did I take my pills?").text == "The bottle is on the right."
    q, _ = qa(log, look_reply("You took your pills."))
    assert q.look("did I take my pills?").text == PILLS_SAFE


def test_look_sends_upscaled_crops_of_named_entities(log, monkeypatch):
    import core.crops as crops
    from core.crops import Crop, CropTrack

    class Store:
        def for_entity(self, ent):
            c = Crop(img=np.full((40, 30, 3), 90, np.uint8), box_px=(0, 0, 30, 40), box_cm=ent.box_cm, t=0.0, score=1)
            return CropTrack(ent.name, ent.name, (0, 0, 30, 40), ent.box_cm, 0.0, best=c)

    monkeypatch.setattr(crops, "active", lambda: Store())
    q, prov = qa(log, look_reply())
    q.look("are my keys next to the box?", parse("are my keys next to the box?", CFG))
    parts = prov.calls[0].parts
    texts = [p[1] for p in parts if p[0] == "text"]
    assert any("close-up of the keys" in t for t in texts) and any("close-up of the box" in t for t in texts)
    import cv2
    crop = cv2.imdecode(np.frombuffer([p for p in parts if p[0] == "image"][1][1], np.uint8), 1)
    assert max(crop.shape[:2]) == 384                          # upscaled from 40 px
    assert "keys: visible" in texts[-1] and "Question: are my keys next to the box?" in texts[-1]


def test_close_ups_say_when_they_were_taken(log, monkeypatch):
    """A close-up is evidence from its own moment: the prompt gives its age next to image 1, so an old
    view is never passed off as the table now."""
    import core.crops as crops
    from core.crops import Crop, CropTrack
    ages = {"keys": 0.0, "box": 42.0}

    class Store:
        def for_entity(self, ent):
            c = Crop(img=np.full((40, 30, 3), 90, np.uint8), box_px=(0, 0, 30, 40), box_cm=ent.box_cm,
                     t=0.0 - ages[ent.name], score=1)
            return CropTrack(ent.name, ent.name, (0, 0, 30, 40), ent.box_cm, c.t, best=c)

    monkeypatch.setattr(crops, "active", lambda: Store())
    q, prov = qa(log, look_reply())
    q.look("are my keys next to the box?", parse("are my keys next to the box?", CFG))
    texts = [p[1] for p in prov.calls[0].parts if p[0] == "text"]
    keys = next(t for t in texts if "close-up of the keys" in t)
    box = next(t for t in texts if "close-up of the box" in t)
    assert "same time as image 1" in keys
    assert "42 seconds before image 1" in box and "may have changed" in box
    assert "when it was taken" in prov.calls[0].system


def test_attach_binds_crops_to_the_things_the_world_saw(log, monkeypatch):
    """After every world.update the crop store learns which thing each view was (so close-ups follow
    identity, not position): a swap at the same spot between two frames keeps each thing's own view."""
    import core.crops as crops
    from core.crops import CropStore

    store = CropStore(recent_every_s=0, swap_de=1e9)            # no colour check: identity alone
    monkeypatch.setattr(crops, "active", lambda: store)

    class World:
        def __init__(self):
            self.ents = {}

        def update(self, dets, frame):
            for name, d in zip(self.names, dets.items):
                self.ents[name] = Entity(name, "target", Status.VISIBLE, pos_cm=d.center_cm, box_cm=d.box_cm,
                                         last_seen=frame.wall)
            return []

        def thing_labels(self):
            return {n: None for n in self.ents}

        def get(self, name):
            return self.ents[name]

        def state_json(self):
            return {"entities": []}

    world = World()
    q, _ = qa(log, look_reply(), world=FakeWorld([], log))
    q.attach(world)
    block = np.full((80, 80, 3), 60, np.uint8)
    block[::8] = 250
    im = img(block=block, at=(600, 300), size=80)
    for i, name in enumerate(["thing:1", "thing:1", "thing:2"]):
        d = Detection("thing", 0.8, (600, 300, 680, 380), (64.0, 34.0),
                      tuple(float(v) for v in (60, 30, 68, 38)))       # a new box object per view, as the detector's
        f = frame(1.0 + i / 10, im)
        store.update(im, [d], [], f.t)
        world.names = [name]
        world.update(Detections(t=f.t, frame_idx=f.idx, items=[d]), f)
    one = store.for_entity(world.get("thing:1"))
    assert one is not None and one.owner == "thing:1" and one.best.t <= 1.1 + 1e-9
    assert store.for_entity(world.get("thing:2")) is None       # nothing confirmed as thing:2 yet


# ---------------------------------------------------------------- B: recalling

def recall_reply(answer="A red mug was on the table from about 9:10 to 9:40.", conf=0.8):
    return json.dumps({"answer": answer, "confidence": conf, "frames": [1, 2]})


def test_recall_sends_matching_frames_with_times_and_context(log):
    q, prov = qa(log, recall_reply(), recall_frames=3)
    q.archive = archive(log)
    fill(q.archive, [(0, None), (10, (0, 0, 255)), (20, (0, 0, 255)), (40, (0, 0, 255)), (50, (255, 0, 0))])
    NarrationStore(log)
    st = NarrationStore(log)
    i = st.add_pending(T0 + 600, T0 + 640, None, {})
    st.mark_done(i, "You put a red mug down.", {"confidence": 0.9}, "fake", "f", 1)
    log.add(Event(t=1.0, wall=T0 + 1200, obj="keys", type="PICKED_UP", parent="hand:1"))
    a = q.recall("Was there a red mug here this morning?")
    assert a.text == "A red mug was on the table from about 9:10 to 9:40."
    parts = prov.calls[0].parts
    texts = " ".join(p[1] for p in parts if p[0] == "text")
    assert len([p for p in parts if p[0] == "image"]) == 3
    assert "Frame 1 at 9:10 AM" in texts and "Frame 3 at 9:40 AM" in texts
    assert "You put a red mug down." in texts and "keys PICKED_UP by hand" in texts


def test_recall_abstains_without_a_vlm_call_when_nothing_matches(log):
    q, prov = qa(log, recall_reply(), min_sim=0.99)
    q.archive = archive(log, min_sim=0.99)
    fill(q.archive, [(10, (255, 0, 0))])
    a = q.recall("Was there a red mug here this morning?")
    assert a.text == "I don't remember seeing a red mug this morning." and prov.calls == []


def test_recall_by_time_only_and_without_pictures(log):
    q, prov = qa(log, recall_reply("The table had your keys and a book on it."))
    q.archive = archive(log)
    assert q.recall("What was on the table this morning?").text == \
        "I don't have any saved pictures of the table this morning."
    fill(q.archive, [(m, None) for m in range(0, 120, 10)])
    a = q.recall("What was on the table this morning?")
    assert a.text == "The table had your keys and a book on it."
    assert len([p for p in prov.calls[0].parts if p[0] == "image"]) == 4


def test_recall_low_confidence_and_meds(log):
    q, _ = qa(log, recall_reply("Maybe.", conf=0.2))
    q.archive = archive(log)
    fill(q.archive, [(10, None)])
    assert q.recall("What was on the table this morning?").text == "I can't tell from the pictures I saved."
    q, _ = qa(log, recall_reply("You swallowed a pill at 9:10."))
    q.archive = archive(log)
    fill(q.archive, [(10, None)])
    assert q.recall("What was on the table this morning?").text == PILLS_SAFE


# ---------------------------------------------------------------- C: routing

@pytest.mark.parametrize("text,how", [
    ("What's on the table?", "look"), ("What does the note say?", "look"), ("Is the charger plugged in?", "look"),
    ("Where is my red mug?", "look"),
    ("Was there a red mug here this morning?", "recall"), ("When did the papers show up?", "recall"),
    ("What was on the table before I left?", "recall"),
    ("Where are my keys?", None), ("What happened to my keys?", None), ("What was I doing before lunch?", None),
    ("What happened while I was away?", None), ("Reset the table.", None),
])
def test_routing(log, text, how):
    q, _ = qa(log, look_reply())
    q.archive = archive(log)
    seen = []
    q.look = lambda t, i=None: seen.append("look") or Answer("L")
    q.recall = lambda t: seen.append("recall") or Answer("R")
    a = q.route(parse(text, CFG), text, online=True)
    assert (seen[0] if seen else None) == how, (text, seen)
    assert (a is None) == (how is None)


def test_unknown_name_with_a_narration_stays_with_narrations(log):
    q, _ = qa(log, look_reply())
    st = NarrationStore(log)
    i = st.add_pending(T0 + 3000, T0 + 3040, None, {})
    st.mark_done(i, "You stacked the papers.", {"confidence": 0.9}, "fake", "f", 1)
    assert q.route(parse("When did the papers show up?", CFG), "When did the papers show up?", True) is None


def test_offline_and_capped(log):
    q, prov = qa(log, look_reply(), max_per_hour=1)
    assert q.route(parse("What's on the table?", CFG), "What's on the table?", online=False).text == OFFLINE
    assert q.route(parse("Where is my red mug?", CFG), "Where is my red mug?", online=False) is None
    assert q.route(parse("What's on the table?", CFG), "What's on the table?", online=True) is not None
    assert q.route(parse("What's on the table?", CFG), "What's on the table?", online=True) is None   # capped
    assert len(prov.calls) == 1


@pytest.mark.parametrize("online", [True, False])
@pytest.mark.parametrize("text", ["What time is it?", "Tell me a joke.", "What's the weather?",
                                  "What did you say?"])
def test_questions_not_about_the_table_are_left_to_the_other_answerer(log, text, online):
    q, prov = qa(log, look_reply())
    q.archive = archive(log)
    seen = []
    q.look = lambda t, i=None: seen.append("look") or Answer("L")
    q.recall = lambda t: seen.append("recall") or Answer("R")
    assert q.route(parse(text, CFG), text, online=online) is None and seen == []


@pytest.mark.parametrize("text", ["Does the box have a lid?", "Does the calculator have batteries?",
                                  "How many pens are there?", "What colour is it?", "Is that my wallet?"])
def test_other_questions_about_tracked_things_or_the_view_still_look(log, text):
    world = FakeWorld([Entity("box", "container", Status.VISIBLE, pos_cm=(70.0, 40.0)),
                       Entity("thing:3", "thing", Status.VISIBLE, pos_cm=(30.0, 30.0))], log)
    world.thing_labels = lambda: {"thing:3": "calculator"}
    q, _ = qa(log, look_reply(), world=world)
    seen = []
    q.look = lambda t, i=None: seen.append("look") or Answer("L")
    assert q.route(parse(text, CFG), text, online=True) == Answer("L") and seen == ["look"]
    assert q.route(parse(text, CFG), text, online=False).text == OFFLINE


def test_pipeline_answers_non_visual_questions_with_other(log):
    from voice.pipeline import make_ask
    q, prov = qa(log, look_reply("A blue notebook and your keys."))

    class Net:
        online = True

    for online in (True, False):
        Net.online = online
        ask = make_ask(CFG, q.world, log, net=Net(), other=lambda *a, **k: Answer("grok"), visual=q)
        assert ask("What time is it?", "voice").text == "grok"
        assert ask("Tell me a joke.", "voice").text == "grok"
    assert prov.calls == []


def test_pipeline_gives_visual_the_first_say(log):
    from voice.pipeline import make_ask
    q, prov = qa(log, look_reply("A blue notebook and your keys."))

    class Net:
        online = True

    ask = make_ask(CFG, q.world, log, net=Net(), other=lambda *a, **k: Answer("grok"), visual=q)
    assert ask("What's on the table?", "voice").text == "A blue notebook and your keys."
    assert ask("Where are my keys?", "voice").text.startswith("Your keys are on the table")
    ask2 = make_ask(CFG, q.world, log, net=Net(), other=lambda *a, **k: Answer("grok"))
    assert ask2("What's on the table?", "voice").text == "grok"          # visual memory off: as before


def test_status_discloses_what_is_sent(log):
    q, _ = qa(log, look_reply())
    q.archive = archive(log)

    class W:
        def state_json(self):
            return {"entities": []}

    w = W()
    q.attach(w)
    st = w.state_json()["visual_memory"]
    assert st["enabled"] and "sent to fake" in st["disclosure"] and st["archive"]["frames"] == 0


def test_from_config_is_off_unless_enabled(log):
    """The code default is off (no frames leave the device unless config.yaml says so)."""
    from voice.visual import from_config
    assert VisualConfig.from_dict({}).enabled is False and VisualConfig.from_dict(None).enabled is False
    assert from_config(dict(CFG, visual_memory=dict(CFG["visual_memory"], enabled=False)), None, log) is None
    assert from_config({k: v for k, v in CFG.items() if k != "visual_memory"}, None, log) is None


# ---------------------------------------------------------------- main.Room points at a raw spot

def test_room_aims_at_target_cm(tmp_path):
    import main
    from core.fakeworld import demo_world

    class Laser:
        state = {"on": False, "target": None, "err_cm": None}

        def __init__(self):
            self.aimed = []

        def aim_object(self, name, pos):
            self.aimed.append((name, tuple(pos)))
            return 0.5

        def off(self):
            pass

    events = EventLog(":memory:", str(tmp_path))
    laser = Laser()
    room = main.Room(CFG, demo_world(events), events, None, None, laser, lambda t, s: Answer(t))
    assert room.aim(Answer("There.", action="point", target_cm=(12.5, 40.0))) == 0.5
    assert laser.aimed == [(None, (12.5, 40.0))]
    room.aim(Answer("Keys.", point_at="wallet", action="point"))
    assert laser.aimed[-1][0] == "wallet"
    room.shutdown()
    events.close()


# ---------------------------------------------------------------- the real embedder (when exported)

MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS, "mobileclip2_s0_text.onnx")),
                    reason="run scripts/export_mobileclip.py first")
def test_onnx_mobileclip_matches_text_to_the_right_frame():
    from core.visual_memory import OnnxClipEmbedder
    e = OnnxClipEmbedder(os.path.join(MODELS, "mobileclip2_s0_image.onnx"),
                         os.path.join(MODELS, "mobileclip2_s0_text.onnx"),
                         os.path.join(os.path.dirname(MODELS), "assets", "bpe_simple_vocab_16e6.txt.gz"),
                         providers=("cpu",))
    red, blue = img(block=(0, 0, 230), size=400), img(block=(230, 0, 0), size=400)
    E = e.image([red, blue])
    T = e.text(["a photo of a red square", "a photo of a blue square"])
    assert E.shape == (2, 512) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-3)
    s = E @ T.T
    assert s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0]


def test_clip_tokenizer_matches_open_clip():
    oc = pytest.importorskip("open_clip")
    from core.clip_tokenizer import ClipTokenizer
    import open_clip.tokenizer as T
    vocab = os.path.join(os.path.dirname(T.__file__), "bpe_simple_vocab_16e6.txt.gz")
    ours = ClipTokenizer(vocab)
    probe = ["a photo of a red mug", "Grandma's ring & keys!", "3:30 pm pill bottle"]
    assert (ours(probe) == oc.get_tokenizer("MobileCLIP2-S0")(probe).numpy()).all()


@pytest.mark.parametrize("said,spoken", [
    ("next to the camera, under mark 10.", "Next to the camera."),
    ("The keys are in the small yellow box at mark 9.", "The keys are in the small yellow box."),
    ("The calculator (mark 3) is on the left.", "The calculator is on the left."),
    ("Mark 2 is your mug, on the right.", "Your mug, on the right."),
    ("It's on the left, marked 4, by the cup.", "It's on the left, by the cup."),
])
def test_look_never_speaks_the_marks(log, said, spoken):
    q, _ = qa(log, look_reply(said, mark=1))
    assert q.look("where is it?").text == spoken
