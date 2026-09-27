"""Episode narration memory (core/narration.py, core/narration_store.py): segmentation, job assembly,
output validation, the medication rule, the offline queue, the cost cap, storage, and a synthetic
end-to-end run through the real World with a fake VLM provider."""
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from core.config import load_config
from core.events import EventLog
from core.narration import (SCHEMA, FakeProvider, Keyframe, NarrationConfig, NarrationError, Narrator,
                            ProviderError, Segmenter, build_job, event_line, from_config, select_frames,
                            validate)
from core.narration_store import NarrationStore, redact_meds, store_for
from core.types import Detection, Detections, Event, Frame

CFG = load_config()
# The latest local 2:02:00 PM already past: within the store's 24 h, so pending episodes aren't pruned.
_TWO = datetime.now().replace(hour=14, minute=2, second=0, microsecond=0)
WALL0 = (_TWO if _TWO <= datetime.now() else _TWO - timedelta(days=1)).timestamp()


def hand(i=1):
    return Detection(f"hand:{i}", 0.9, (100, 100, 200, 200), (10.0, 10.0), (5.0, 5.0, 15.0, 15.0))


def dets(t, hands=0):
    return Detections(t=t, frame_idx=int(t * 10), items=[], hands=[hand(i + 1) for i in range(hands)])


def frame(t, img=True):
    im = np.full((72, 128, 3), int(t * 10) % 255, np.uint8) if img else None
    return Frame(t=t, wall=WALL0 + t, img=im, idx=int(t * 10))


def ev(t, obj="keys", typ="PICKED_UP", **kw):
    return Event(t=t, wall=WALL0 + t, obj=obj, type=typ, **kw)


def ncfg(**kw):
    base = dict(enabled=True, provider="fake", quiet_s=4.0, max_episode_s=120.0, min_episode_s=2.0,
                min_hand_frames=2, keyframe_every_s=1.0, keyframes=24, max_images=8)
    base.update(kw)
    return NarrationConfig(**base)


def run(seg, t0, t1, hands=0, events=None, fps=10):
    """Feed the segmenter from t0 to t1; events maps a frame time (rounded) to its events."""
    out = []
    n = int(round((t1 - t0) * fps))
    for i in range(n):
        t = round(t0 + i / fps, 3)
        out.append(seg.update(t, WALL0 + t, hands > 0, (events or {}).get(t, [])))
    return out


def ended(steps):
    return [s.ended for s in steps if s.ended is not None]


# ---------------------------------------------------------------- segmentation

def test_hand_in_and_out_makes_one_episode_that_ends_after_quiet_s():
    seg = Segmenter(ncfg())
    steps = run(seg, 0.0, 1.0) + run(seg, 1.0, 6.0, hands=1) + run(seg, 6.0, 9.9)
    assert not ended(steps), "still inside quiet_s"
    steps = run(seg, 9.9, 11.0)
    [ep] = ended(steps)
    assert abs(ep.t_start - 1.0) < 0.2           # the second hand frame (min_hand_frames=2) opened it
    assert 5.9 <= ep.t_active <= 6.0
    ts = [k.t for k in ep.keyframes]
    assert ts == sorted(ts) and len(ts) >= 5     # about one a second
    assert all(b - a >= 0.99 for a, b in zip(ts, ts[1:]))
    assert sum(1 for k in ep.keyframes if k.t > ep.t_active) == 1   # one settled 'after' frame kept
    assert ep.worth_narrating


def test_quiet_s_is_respected_exactly():
    seg = Segmenter(ncfg(quiet_s=4.0))
    run(seg, 0.0, 3.0, hands=1)
    last_active = 2.9
    assert not ended(run(seg, 3.0, round(last_active + 3.9, 3)))
    assert ended(run(seg, round(last_active + 3.9, 3), round(last_active + 4.2, 3)))


def test_a_one_frame_hand_flicker_does_not_open_an_episode():
    seg = Segmenter(ncfg())
    s1 = seg.update(0.0, WALL0, True, [])
    s2 = seg.update(0.1, WALL0 + 0.1, False, [])
    assert s1.keep is None and s2.keep is None and seg.ep is None


def test_world_events_alone_open_an_episode():
    seg = Segmenter(ncfg())
    steps = run(seg, 0.0, 10.0, events={1.0: [ev(1.0, typ="LOST_TRACK")]})
    [ep] = ended(steps)
    assert ep.t_start == 1.0 and [e["type"] for e in ep.events] == ["LOST_TRACK"]
    assert ep.keyframes[0].event == "keys LOST_TRACK"
    assert ep.worth_narrating                     # short, but it has an event


def test_short_eventless_episode_is_not_worth_narrating():
    seg = Segmenter(ncfg(min_episode_s=2.0))
    steps = run(seg, 0.0, 1.0, hands=1) + run(seg, 1.0, 8.0)
    [ep] = ended(steps)
    assert not ep.worth_narrating


def test_max_episode_length_splits_and_the_next_episode_starts_at_once():
    seg = Segmenter(ncfg(max_episode_s=30.0))
    steps = run(seg, 0.0, 70.0, hands=1, fps=5)
    eps = ended(steps)
    assert len(eps) == 2
    assert eps[0].t_end - eps[0].t_start <= 30.0 + 1e-6
    assert abs(eps[1].t_start - eps[0].t_end) < 0.25
    assert seg.ep is not None                     # the third one is still running


def test_keyframes_are_thinned_evenly_and_event_frames_survive():
    seg = Segmenter(ncfg(keyframes=10, max_episode_s=200.0))
    events = {12.0: [ev(12.0, typ="PICKED_UP", parent="hand:1")],
              31.0: [ev(31.0, typ="PUT_INSIDE", parent="box")]}
    steps = run(seg, 0.0, 60.0, hands=1, events=events, fps=5)
    dropped = [k for s in steps for k in s.dropped]
    kfs = seg.ep.keyframes
    assert len(kfs) == 10 and len(dropped) > 30
    labels = [k.event for k in kfs if k.event]
    assert labels == ["keys PICKED_UP by hand", "keys PUT_INSIDE box"]
    assert kfs[0].t == 0.2                        # first frame kept (second hand frame)
    ts = [k.t for k in kfs][:-1]                  # the newest frame is always ~1 s after the one before
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert max(gaps) <= 3 * min(gaps), gaps       # roughly even, not all bunched at one end


# ---------------------------------------------------------------- job assembly

def test_select_frames_keeps_event_frames_first_and_last():
    kfs = [Keyframe(float(i), WALL0 + i, f"{i}.jpg", "keys PICKED_UP by hand" if i in (7, 13) else None)
           for i in range(24)]
    sel = select_frames(kfs, 8)
    ts = [k.t for k in sel]
    assert len(sel) == 8 and ts == sorted(ts)
    assert {0.0, 23.0, 7.0, 13.0} <= set(ts)
    assert select_frames(kfs[:5], 8) == kfs[:5]


def test_select_frames_with_more_events_than_slots_spreads_them():
    kfs = [Keyframe(float(i), WALL0 + i, f"{i}.jpg", "e") for i in range(20)]
    sel = select_frames(kfs, 6)
    assert len(sel) == 6 and sel[0].t == 0.0 and sel[-1].t == 19.0


def test_event_line_format_and_names():
    names = {"keys": "keys", "box": "box", "thing:3": "phone charger", "thing:4": None}
    t = WALL0 + 11
    assert event_line(dict(wall=t, obj="keys", type="PICKED_UP", parent="hand:1"), names) == \
        "14:02:11 keys PICKED_UP by hand"
    assert event_line(dict(wall=t, obj="keys", type="PUT_INSIDE", parent="box"), names) == \
        "14:02:11 keys PUT_INSIDE box"
    assert event_line(dict(wall=t, obj="thing:3", type="EXITED_VIEW", edge="left"), names) == \
        "14:02:11 phone charger EXITED_VIEW off the left edge"
    assert event_line(dict(wall=t, obj="thing:4", type="APPEARED"), names) == \
        "14:02:11 something new APPEARED"
    assert event_line(dict(wall=t, obj="pill_bottle", type="COVERED", parent="notebook"), names) == \
        "14:02:11 pill bottle COVERED by notebook"


def _episode_row(tmp_path, n=12, events=None):
    d = tmp_path / "ep"
    d.mkdir()
    kfs = []
    for i in range(n):
        name = f"{i}.jpg"
        (d / name).write_bytes(b"\xff\xd8jpeg" + bytes([i]))
        label = None
        for e in events or []:
            if abs(e["wall"] - (WALL0 + i)) < 0.01:
                label = f"{e['obj']} {e['type']}"
        kfs.append(dict(t=float(i), wall=WALL0 + i, name=name, event=label))
    return dict(t_start=WALL0, t_end=WALL0 + n - 1, frames_dir=str(d),
                episode=dict(keyframes=kfs, events=events or []))


def test_build_job_has_frames_events_names_and_rules(tmp_path):
    evs = [dict(wall=WALL0 + 3, obj="keys", type="PICKED_UP", parent="hand:1"),
           dict(wall=WALL0 + 7, obj="keys", type="PUT_INSIDE", parent="box")]
    row = _episode_row(tmp_path, 12, evs)
    job = build_job(row, CFG, labels={"thing:1": "phone charger"}, max_images=6)
    assert job.events_text == "14:02:03 keys PICKED_UP by hand; 14:02:07 keys PUT_INSIDE box"
    images = [p for p in job.parts if p[0] == "image"]
    assert len(images) == 6
    texts = " ".join(p[1] for p in job.parts if p[0] == "text")
    assert "Frame" in texts and "14:02:07" in texts and "keys PUT_INSIDE" in texts
    assert "pill bottle" in job.names and "phone charger" in job.names and "pill_bottle" not in job.names
    assert "phone charger" in texts
    s = job.system.lower()
    assert "never" in s and "medication" in s and "unclear" in s and "json" in s
    assert "overhead" in s


def test_build_job_skips_missing_frames(tmp_path):
    row = _episode_row(tmp_path, 5)
    os.remove(os.path.join(row["frames_dir"], "2.jpg"))
    job = build_job(row, CFG, max_images=8)
    assert len([p for p in job.parts if p[0] == "image"]) == 4


# ---------------------------------------------------------------- validation

NAMES = ["keys", "pill bottle", "wallet", "box", "notebook", "phone charger"]


def good(**over):
    d = {"summary": "You sorted some papers, then put your keys in the box.",
         "actions": [{"t": "14:02:11", "verb": "picked up", "objects": ["Keys"], "detail": "from the left"},
                     {"t": "later", "verb": "put", "objects": ["keys", "box", "mug"], "detail": "into the box"}],
         "objects_involved": ["keys", "box", "stapler"], "activity_tags": ["Tidying", "tidying", "keys"],
         "confidence": 1.7, "notes": ""}
    d.update(over)
    return d


def test_validate_normalizes_names_times_tags_and_confidence():
    out = validate(json.dumps(good()), NAMES)
    assert out["summary"] == "You sorted some papers, then put your keys in the box."
    assert out["actions"][0]["objects"] == ["keys"] and out["actions"][0]["t"] == "14:02:11"
    assert out["actions"][1]["t"] is None
    assert out["actions"][1]["objects"] == ["keys", "box", "unknown object"]
    assert out["objects_involved"] == ["keys", "box"]
    assert out["activity_tags"] == ["tidying", "keys"]
    assert out["confidence"] == 1.0


def test_validate_repairs_fences_prose_and_trailing_commas():
    raw = "Sure! Here it is:\n```json\n" + json.dumps(good(), indent=1)[:-1] + ",\n}\n```\nDone."
    assert validate(raw, NAMES)["summary"].startswith("You sorted")
    minimal = '{"summary": "You picked up the wallet.", "confidence": "0.4",}'
    out = validate(minimal, NAMES)
    assert out["actions"] == [] and out["confidence"] == 0.4 and out["notes"] == ""


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2]", '{"actions": []}', '{"summary": "  "}',
                                 '{"summary": 5}'])
def test_validate_rejects_unusable_output(raw):
    with pytest.raises(NarrationError):
        validate(raw, NAMES)


def test_validate_caps_summary_length():
    long = " ".join(f"You moved thing {i}." for i in range(10))
    assert len(validate(json.dumps(good(summary=long)), NAMES)["summary"].split(". ")) <= 3


# ---------------------------------------------------------------- the medication rule

@pytest.mark.parametrize("bad", [
    "You took your pills.", "You opened the pill bottle and took two pills.", "You missed your medication.",
    "Your medicine was taken at 2 PM.", "You swallowed something.", "You skipped your meds today.",
    "It looks like you had your dose.", "You took one from the pill bottle.", "You forgot your tablets.",
])
def test_medication_claims_are_redacted(bad):
    text, n = redact_meds(f"You sorted some papers. {bad} Then you put your keys in the box.")
    assert n == 1
    assert text == "You sorted some papers. Then you put your keys in the box."


@pytest.mark.parametrize("ok", ["You opened the pill bottle and put it back.",
                                "You picked up the pill bottle.", "You took the wallet off the table.",
                                "You moved the notebook over the pill bottle."])
def test_neutral_pill_bottle_sentences_are_kept(ok):
    assert redact_meds(ok) == (ok, 0)


def test_validate_enforces_the_medication_rule_everywhere():
    d = good(summary="You opened the pill bottle. You took your pills.",
             actions=[{"t": None, "verb": "took", "objects": ["pill bottle"], "detail": "swallowed a pill"},
                      {"t": None, "verb": "opened", "objects": ["pill bottle"], "detail": "twisted the cap"}],
             activity_tags=["medication taken", "tidying"], notes="Probably took the medicine.")
    out = validate(json.dumps(d), NAMES)
    blob = json.dumps(out).lower()
    for word in ("took", "taken", "swallow"):
        assert word not in blob, blob
    assert out["summary"] == "You opened the pill bottle."
    assert [a["verb"] for a in out["actions"]] == ["opened"]
    assert out["activity_tags"] == ["tidying"] and out["redacted"] >= 3


def test_summary_that_is_only_a_medication_claim_becomes_neutral():
    out = validate(json.dumps(good(summary="You took your pills.", objects_involved=["pill bottle"])), NAMES)
    assert out["summary"] == "You handled the pill bottle."
    out = validate(json.dumps(good(summary="You took your pills.", objects_involved=[])), NAMES)
    assert out["summary"] == "Unclear."


# ---------------------------------------------------------------- narrator: queue, retry, cap, storage

def reply(summary="You put your keys in the box.", conf=0.9, **kw):
    return json.dumps(good(summary=summary, confidence=conf, **kw))


class Clock:
    def __init__(self, t=WALL0 + 1000):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def log(tmp_path):
    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    yield lg
    lg.close()


def make(log, provider=None, online=True, clock=None, **kw):
    cfg = dict(CFG)
    cfg["narration"] = {**dict(enabled=True, provider="fake", quiet_s=4.0, min_hand_frames=2,
                               keyframe_every_s=1.0), **kw}
    net = {"on": online}
    n = Narrator(cfg, log, provider=provider or FakeProvider(reply()), online=lambda: net["on"],
                 clock=clock or Clock(), start=False)
    n.net = net
    return n


def episode(n, t0=0.0, secs=5.0, events=None):
    """One hand episode through feed(): hands for secs, then quiet until it closes."""
    fps = 10
    for i in range(int(secs * fps)):
        t = round(t0 + i / fps, 3)
        n.feed(frame(t), dets(t, 1), (events or {}).get(t, []))
    t = t0 + secs
    while n.seg.ep is not None:
        n.feed(frame(t), dets(t, 0), [])
        t = round(t + 0.1, 3)
    n.drain()
    return t


def test_feed_writes_downscaled_keyframes_and_queues_an_episode(log):
    n = make(log)
    big = np.zeros((720, 1280, 3), np.uint8)
    for i in range(30):
        t = i / 10
        n.feed(Frame(t=t, wall=WALL0 + t, img=big, idx=i), dets(t, 1), [])
    n.feed(Frame(t=3.0, wall=WALL0 + 3, img=big, idx=31), dets(3.0, 0), [])
    n.drain()
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(n.root) for f in fs]
    assert files and all(f.endswith(".jpg") for f in files)
    import cv2
    assert max(cv2.imread(files[0]).shape[:2]) == 640
    for i in range(60):
        n.feed(frame(3.1 + i / 10), dets(3.1 + i / 10, 0), [])
    n.drain()
    assert n.store.pending_count() == 1


def test_offline_episodes_stay_queued_and_are_narrated_when_online(log):
    prov = FakeProvider(reply())
    n = make(log, prov, online=False)
    episode(n)
    assert n.run_pending() == 0 and prov.calls == [] and n.store.pending_count() == 1
    n.net["on"] = True
    assert n.run_pending() == 1
    [row] = n.store.between(WALL0 - 10, WALL0 + 100)
    assert row.status == "done" and row.summary == "You put your keys in the box."
    assert row.provider == "fake" and row.latency_ms is not None
    assert n.status()["last_summary"] == "You put your keys in the box."


def test_pending_episodes_survive_a_restart(tmp_path):
    db, snaps = str(tmp_path / "e.db"), str(tmp_path / "snaps")
    log1 = EventLog(db, snaps)
    n1 = make(log1, online=False)
    episode(n1)
    n1.stop()
    log1.close()
    log2 = EventLog(db, snaps)
    prov = FakeProvider(reply())
    n2 = make(log2, prov)
    assert n2.store.pending_count() == 1
    assert n2.run_pending() == 1 and len(prov.calls) == 1
    assert len([p for p in prov.calls[0].parts if p[0] == "image"]) >= 3   # frames were on disk
    log2.close()


def test_api_failure_backs_off_then_retries(log):
    clock = Clock()
    prov = FakeProvider([ProviderError("503", retryable=True), reply()])
    n = make(log, prov, clock=clock, backoff_s=5.0)
    episode(n)
    assert n.run_pending() == 0 and len(prov.calls) == 1
    row = n.store.next_due(clock.t + 3600)
    assert row.attempts == 1 and row.next_try >= clock.t + 5.0
    assert n.run_pending() == 0 and len(prov.calls) == 1        # not before next_try
    clock.t += 6.0
    assert n.run_pending() == 1 and n.store.pending_count() == 0


def test_bad_output_fails_after_max_attempts(log):
    clock = Clock()
    prov = FakeProvider("not json at all")
    n = make(log, prov, clock=clock, max_attempts=3, backoff_s=1.0)
    episode(n)
    for _ in range(5):
        n.run_pending()
        clock.t += 1000
    assert len(prov.calls) == 3 and n.store.pending_count() == 0
    assert n.store.counts().get("failed") == 1


def test_queue_is_bounded_and_drops_the_oldest(log):
    n = make(log, online=False, max_queued=2)
    t = 0.0
    dirs = []
    for _ in range(4):
        t = episode(n, t0=t + 1.0)
        dirs.append(n.last_dir)
    assert n.store.pending_count() == 2
    assert not os.path.exists(dirs[0]) and not os.path.exists(dirs[1])
    assert os.path.exists(dirs[2]) and os.path.exists(dirs[3])


def test_cost_cap_per_hour(log):
    clock = Clock()
    prov = FakeProvider(reply())
    n = make(log, prov, clock=clock, max_per_hour=2)
    t = 0.0
    for _ in range(3):
        t = episode(n, t0=t + 1.0)
    assert n.run_pending() == 2 and n.store.pending_count() == 1
    clock.t += 1800
    assert n.run_pending() == 0
    clock.t += 1801
    assert n.run_pending() == 1


def test_max_images_per_call(log):
    prov = FakeProvider(reply())
    n = make(log, prov, max_images=3)
    episode(n, secs=12.0)
    n.run_pending()
    assert len([p for p in prov.calls[0].parts if p[0] == "image"]) == 3


def test_short_eventless_episode_is_discarded_with_its_frames(log):
    n = make(log, min_episode_s=2.0)
    episode(n, secs=0.5)
    assert n.store.pending_count() == 0
    assert not any(fs for _, _, fs in os.walk(n.root))


def test_feed_never_raises_and_is_cheap(log):
    n = make(log)
    big = np.zeros((720, 1280, 3), np.uint8)
    t0 = time.perf_counter()
    for i in range(300):
        t = i / 15
        n.feed(Frame(t=t, wall=WALL0 + t, img=big, idx=i), dets(t, 1), [])
    per = (time.perf_counter() - t0) / 300
    assert per < 0.002, per
    n.feed(None, None, None)                    # garbage in: logged, never raised
    n.feed(frame(99.0), "not dets", [object()])


def test_threaded_narrator_end_to_end(tmp_path):
    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    cfg = dict(CFG)
    cfg["narration"] = dict(enabled=True, provider="fake", quiet_s=1.0)
    prov = FakeProvider(reply())
    n = Narrator(cfg, lg, provider=prov)
    try:
        for i in range(45):
            t = i / 10
            n.feed(frame(t), dets(t, 1 if i < 25 else 0), [])
        assert n.idle(timeout=5.0)
        assert len(prov.calls) == 1 and n.status()["queued"] == 0
    finally:
        n.stop()
        lg.close()


def test_from_config_is_off_by_default_and_discloses_when_on(log):
    assert from_config(CFG, log) is None
    cfg = dict(CFG)
    cfg["narration"] = dict(CFG.get("narration") or {}, enabled=True, provider="fake")
    n = from_config(cfg, log, start=False)
    st = n.status()
    assert st["enabled"] is True and "sent to" in st["disclosure"] and "fake" in st["disclosure"]
    n.stop()


def test_attach_feeds_from_world_update_and_adds_state(log):
    class W:
        def update(self, d, f):
            return [ev(d.t)] if d.t == 0.5 else []

        def state_json(self):
            return {"entities": []}

        def thing_labels(self):
            return {"thing:1": "mug"}

    w = W()
    n = make(log)
    n.attach(w)
    for i in range(10):
        w.update(dets(i / 10, 1), frame(i / 10))
    assert n.seg.ep is not None and n.seg.ep.events[0]["type"] == "PICKED_UP"
    assert w.state_json()["narration"]["enabled"] is True
    assert n.labels() == {"thing:1": "mug"}


# ---------------------------------------------------------------- storage

def test_store_round_trip_and_queries(log):
    st = NarrationStore(log)
    i = st.add_pending(WALL0, WALL0 + 30, "/nope", {"keyframes": [], "events": []})
    st.mark_done(i, "You put your keys in the box.", {"summary": "You put your keys in the box.",
                                                       "objects_involved": ["keys", "box"],
                                                       "activity_tags": ["tidying"], "confidence": 0.8},
                 "claude", "claude-haiku-4-5", 1500)
    j = st.add_pending(WALL0 + 600, WALL0 + 620, "/nope", {})
    st.mark_done(j, "You charged your phone with the phone charger.", {"confidence": 0.4}, "fake", "f", 10)
    [r] = st.between(WALL0 + 10, WALL0 + 20)
    assert r.id == i and r.data["activity_tags"] == ["tidying"] and r.model == "claude-haiku-4-5"
    assert [x.id for x in st.between(WALL0 - 5, WALL0 + 700)] == [i, j]
    assert st.latest(1)[0].id == j
    assert [x.id for x in st.search(["chargers"])] == [j]          # plural stems to 'charger'
    assert [x.id for x in st.search(["tidying"])] == [i]           # tags are searched too
    assert [x.id for x in st.search(["keys", "charger"])] == [j, i]
    assert st.search(["stove"]) == []
    assert store_for(log) is not None


def test_store_redacts_on_write(log):
    st = NarrationStore(log)
    i = st.add_pending(WALL0, WALL0 + 1, "/nope", {})
    st.mark_done(i, "You opened the pill bottle. You took your pills.", {}, "fake", "f", 1)
    assert st.get(i).summary == "You opened the pill bottle."


def test_store_for_does_not_create_the_table_when_reading(tmp_path):
    lg = EventLog(":memory:", str(tmp_path / "s"))
    assert store_for(lg, create=False) is None
    assert store_for(None) is None and store_for(object()) is None
    NarrationStore(lg)
    assert store_for(lg, create=False) is not None
    lg.close()


def test_prune_removes_old_frames_and_keeps_text(log, tmp_path):
    st = NarrationStore(log)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    for d in (old_dir, new_dir):
        d.mkdir()
        (d / "1.jpg").write_bytes(b"x")
    now = time.time()
    a = st.add_pending(now - 30 * 3600, now - 30 * 3600 + 10, str(old_dir), {})
    st.mark_done(a, "You read a book.", {}, "fake", "f", 1)
    b = st.add_pending(now - 60, now - 50, str(new_dir), {})
    c = st.add_pending(now - 26 * 3600, now - 26 * 3600, "/gone", {})
    st.prune(24, now=now)
    assert not old_dir.exists() and new_dir.exists()
    assert st.get(a).summary == "You read a book." and st.get(a).frames_dir is None
    assert st.get(b).status == "pending" and st.get(c).status == "dropped"


# ---------------------------------------------------------------- providers (no network)

class _Blk:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Usage:
    input_tokens, output_tokens = 1234, 210


class _Resp:
    stop_reason = "end_turn"
    usage = _Usage()

    def __init__(self, text):
        self.content = [_Blk(text)]


class _Msgs:
    def __init__(self, text=None, exc=None):
        self.kw, self.text, self.exc = None, text, exc

    def create(self, **kw):
        self.kw = kw
        if self.exc:
            raise self.exc
        return _Resp(self.text)


def test_claude_provider_request_shape():
    from core.narration import ClaudeProvider
    p = ClaudeProvider(NarrationConfig(provider="claude"))
    msgs = _Msgs(reply())
    p._client = type("C", (), {"messages": msgs})()
    out = p.narrate("SYS", [("text", "hello"), ("image", b"\xff\xd8abc")], SCHEMA)
    kw = msgs.kw
    assert kw["model"] == "claude-haiku-4-5" and kw["system"] == "SYS"
    content = kw["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1]["type"] == "image" and content[1]["source"]["media_type"] == "image/jpeg"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert out.usage == {"input_tokens": 1234, "output_tokens": 210}
    assert json.loads(out.text)["summary"]


@pytest.mark.parametrize("status,retry", [(None, True), (429, True), (529, True), (500, True),
                                          (401, True), (400, False), (404, False)])
def test_provider_errors_are_classified(status, retry):
    from core.narration import ClaudeProvider

    class E(Exception):
        status_code = status

    p = ClaudeProvider(NarrationConfig(provider="claude"))
    p._client = type("C", (), {"messages": _Msgs(exc=E("boom"))})()
    with pytest.raises(ProviderError) as ei:
        p.narrate("S", [("text", "x")], SCHEMA)
    assert ei.value.retryable is retry


def test_openai_compat_provider_uses_data_urls_and_env_key(monkeypatch):
    from core.narration import OpenAICompatProvider

    class Choice:
        def __init__(self, t):
            self.message = type("M", (), {"content": t})()

    class Comp:
        kw = None

        def create(self, **kw):
            Comp.kw = kw
            r = type("R", (), {})()
            r.choices = [Choice(reply())]
            r.usage = type("U", (), {"prompt_tokens": 900, "completion_tokens": 150})()
            return r

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    p = OpenAICompatProvider(NarrationConfig(provider="openai_compat", model="grok-4.3"))
    with pytest.raises(ProviderError):
        p.narrate("S", [("text", "x")], SCHEMA)          # no key: stays queued
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    p._client = type("C", (), {"chat": type("Ch", (), {"completions": Comp()})()})()
    out = p.narrate("SYS", [("text", "hi"), ("image", b"\xff\xd8abc")], SCHEMA)
    msgs = Comp.kw["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    img = msgs[1]["content"][1]
    assert img["type"] == "image_url" and img["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert Comp.kw["model"] == "grok-4.3" and out.usage == {"input_tokens": 900, "output_tokens": 150}


# ---------------------------------------------------------------- end to end through the real World

def test_sim_story_end_to_end_with_the_real_world(tmp_path):
    """server/sim.py's scripted story, frames and all, through the real World with a narrator attached:
    each activity becomes an episode whose job carries its own world events and frames."""
    from core.world import World
    from eval.synth import FPS
    from server.sim import Painter, story
    from voice.answers import answer
    from voice.intents import parse

    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    world = World(CFG, lg)
    jobs = []

    def fake(job):
        jobs.append(job)
        first = job.events_text.split("; ")[0] if job.events_text else "nothing"
        return json.dumps({"summary": f"You handled things ({first}).", "actions": [],
                           "objects_involved": [], "activity_tags": ["test"], "confidence": 0.8,
                           "notes": ""})

    cfg = dict(CFG)
    cfg["narration"] = dict(enabled=True, provider="fake", quiet_s=2.5, min_hand_frames=2)
    n = Narrator(cfg, lg, provider=FakeProvider(fake), start=False)
    n.attach(world)
    scene, steps = story(CFG, 0)
    painter = Painter(CFG)
    k = 0
    for i, (snap, d) in enumerate(zip(scene.snaps, scene.render())):
        while k + 1 < len(steps) and steps[k + 1].start <= i:
            k += 1
        t = 1000.0 + i / FPS
        img = painter.draw(snap, steps[k].caption)
        world.update(Detections(t=t, frame_idx=i, items=d.items, hands=d.hands),
                     Frame(t=t, wall=WALL0 + t - 1000.0, img=img, idx=i))
        n.drain()                                   # what the frames worker does between frames
    n.close_episode()
    n.drain()
    n.run_pending()
    rows = n.store.between(WALL0 - 1, WALL0 + 3600)
    assert len(rows) >= 5, [r.summary for r in rows]
    alltext = " | ".join(j.events_text for j in jobs)
    assert "keys PUT_INSIDE box" in alltext and "phone EXITED_VIEW off the table on your right" in alltext
    keys_job = next(j for j in jobs if "keys PUT_INSIDE box" in j.events_text)
    assert "box MOVED" not in keys_job.events_text          # the box move is its own episode
    for j in jobs:
        assert 1 <= len([p for p in j.parts if p[0] == "image"]) <= 8
    assert all(r.status == "done" for r in rows)
    # and the answers use it
    now = rows[-1].t_end + 5
    a = answer(parse("what was I doing?", CFG), world, lg, CFG, now=now)
    assert "You handled things" not in a.text and "you handled things" in a.text, a.text
    n.stop()
    lg.close()


# ---------------------------------------------------------------- Grok is the default provider

def test_default_provider_is_grok():
    from core.narration import OpenAICompatProvider, make_provider
    p = make_provider(NarrationConfig())
    assert isinstance(p, OpenAICompatProvider) and p.model == "grok-4.3" and p.name == "grok"
    assert NarrationConfig().reasoning_effort == "low"


def test_grok_asks_for_json_schema_and_drops_what_the_api_rejects(monkeypatch):
    from core.narration import OpenAICompatProvider

    class Bad(Exception):
        status_code = 400

    class Comp:
        def __init__(self):
            self.calls = []

        def create(self, **kw):
            self.calls.append(kw)
            if "reasoning_effort" in kw:
                raise Bad("Argument not supported: reasoning_effort with image input")
            r = type("R", (), {})()
            r.choices = [type("C", (), {"message": type("M", (), {"content": reply()})()})()]
            r.usage = None
            return r

    monkeypatch.setenv("XAI_API_KEY", "k")
    comp = Comp()
    p = OpenAICompatProvider(NarrationConfig(reasoning_effort="none"))
    p._client = type("Cl", (), {"chat": type("Ch", (), {"completions": comp})()})()
    out = p.narrate("S", [("text", "x")], SCHEMA)
    assert json.loads(out.text)["summary"]
    first, second = comp.calls
    assert first["reasoning_effort"] == "none" and "reasoning_effort" not in second
    assert second["response_format"]["type"] == "json_schema"
    assert second["response_format"]["json_schema"]["strict"] is True
    p.narrate("S", [("text", "x")], SCHEMA)
    assert len(comp.calls) == 3 and "reasoning_effort" not in comp.calls[2]     # remembered


def test_grok_other_errors_are_classified_not_retried_inline(monkeypatch):
    from core.narration import OpenAICompatProvider

    class RateLimited(Exception):
        status_code = 429

    class Comp:
        n = 0

        def create(self, **kw):
            Comp.n += 1
            raise RateLimited("slow down")

    monkeypatch.setenv("XAI_API_KEY", "k")
    p = OpenAICompatProvider(NarrationConfig())
    p._client = type("Cl", (), {"chat": type("Ch", (), {"completions": Comp()})()})()
    with pytest.raises(ProviderError) as ei:
        p.narrate("S", [("text", "x")], SCHEMA)
    assert ei.value.retryable and Comp.n == 1
