import json

import pytest

from core.config import load_config
from core.types import Detection, Detections, Entity, Status
from eval import replay as replay_mod
from eval.baselines import CurrentFrame, LastSeen, NearestObject, prediction
from eval.report import collect, render
from eval.replay import WorldAdapter, replay, run_system, score
from eval.synth import CORE, STRETCH_SYNTH, generate
from eval.trial import (CATEGORIES, Trial, check_category, dets_from_json, dets_to_json,
                        load_trials, make_truth, parse_truth, read_detections, write_detections)

CFG = load_config()
SIZE = {"box": (20, 15), "notebook": (25, 18)}


def det(cls, x, y, conf=0.9):
    w, h = SIZE.get(cls, (6, 4))
    return Detection(cls, conf, (int(x * 14), int(y * 12), int(x * 14) + 10, int(y * 12) + 10),
                     (x, y), (x - w / 2, y - h / 2, x + w / 2, y + h / 2))


def frames(*scenes):
    """scenes: lists of Detection (one list per frame) -> Detections stream."""
    return [Detections(t=1000 + i / 10, frame_idx=i, items=list(s), hands=[]) for i, s in enumerate(scenes)]


def run(system, stream, obj):
    for d in stream:
        system.update(d)
    return system.predict(obj)


# ------------------------------------------------------------------ JSON / truth

def test_detections_json_roundtrip(tmp_path):
    d = Detections(t=1234.5678, frame_idx=7,
                   items=[Detection("keys", 0.91234, (10, 20, 30, 40), (4.25, 5.5), (1.0, 2.0, 7.5, 9.0))],
                   hands=[Detection("hand:3", 0.8, (0, 0, 5, 5), (1.0, 1.0), (0.0, 0.0, 2.0, 2.0))])
    j = json.loads(json.dumps(dets_to_json(d)))
    back = dets_from_json(j)
    assert back == Detections(t=1234.5678, frame_idx=7,
                              items=[Detection("keys", 0.912, (10, 20, 30, 40), (4.25, 5.5), (1.0, 2.0, 7.5, 9.0))],
                              hands=d.hands)
    p = tmp_path / "d.jsonl"
    assert write_detections(p, [d, d]) == 2
    assert [x.frame_idx for x in read_detections(p)] == [7, 7]


def test_parse_truth_semantics():
    assert parse_truth("notebook", CFG) == make_truth("UNDER", parent="notebook")
    assert parse_truth("box", CFG) == make_truth("INSIDE", parent="box")
    assert parse_truth("left", CFG) == make_truth("GONE", edge="left")
    assert parse_truth("40,22", CFG) == make_truth("VISIBLE", pos_cm=(40, 22))
    assert parse_truth("visible", CFG)["pos_cm"] is None
    assert parse_truth("held", CFG)["status"] == "HELD"
    with pytest.raises(ValueError):
        parse_truth("sofa", CFG)
    assert check_category("covered", parse_truth("notebook", CFG)) is None
    assert "usually" in check_category("covered", parse_truth("box", CFG))


def test_truth_resolved_fills_pos_from_final_detections(tmp_path):
    t = Trial.create(tmp_path, 3, "inside_box_moved", "keys", parse_truth("box", CFG), "q?", "file")
    t.write_detections(frames([det("box", 20, 20)], [det("box", 60, 30)], [det("box", 60, 31)]))
    tr = t.truth_resolved(last_n=2)
    assert tr["pos_cm"] == [60.0, 30.5] and tr["pos_source"] == "detections"
    assert load_trials(tmp_path)[0].id == 3


# ------------------------------------------------------------------ baselines

def test_current_frame_visible_debounce_and_threshold():
    s = frames([det("keys", 10, 10)], [det("keys", 10.5, 10)], [])  # missed in the final frame only
    assert run(CurrentFrame(CFG), s, "keys")["status"] == "VISIBLE"
    s = frames([det("keys", 10, 10)], [], [], [])
    assert run(CurrentFrame(CFG), s, "keys") == prediction("UNKNOWN")
    s = frames([det("keys", 10, 10, conf=0.2)])
    assert run(CurrentFrame(CFG), s, "keys")["status"] == "UNKNOWN"


# keys seen at (30, 20), dropped into the box at (32, 22), then the box moved to (75, 45)
BOX_MOVED = frames(*[[det("keys", 30, 20), det("box", 32, 22), det("notebook", 70, 10)]] * 3,
                   *[[det("box", 32, 22), det("notebook", 70, 10)]] * 5,
                   *[[det("box", 75, 45), det("notebook", 70, 10)]] * 5)
BOX_TRUTH = make_truth("INSIDE", parent="box", pos_cm=(75, 45))


def test_last_seen_answers_stale_spot_and_is_wrong_for_moved_box():
    p = run(LastSeen(CFG), BOX_MOVED, "keys")
    assert p["status"] == "UNKNOWN" and p["resolved_cm"] == [30.0, 20.0]
    ok, err = score(p, BOX_TRUTH)
    assert not ok and err == pytest.approx(51.48, abs=0.01)


def test_nearest_object_inside_static_box_is_right():
    s = BOX_MOVED[:8]
    p = run(NearestObject(CFG), s, "keys")
    assert p["status"] == "INSIDE" and p["parent"] == "box" and p["resolved_cm"] == [32.0, 22.0]
    assert score(p, make_truth("INSIDE", parent="box", pos_cm=(32, 22))) == (True, 0.0)


def test_nearest_object_box_moved_far_is_wrong():
    # documented decision: compares with where the box is NOW, so a far move defeats it
    p = run(NearestObject(CFG), BOX_MOVED, "keys")
    assert p["parent"] is None and p["resolved_cm"] == [30.0, 20.0]
    assert score(p, BOX_TRUTH)[0] is False


def test_nearest_object_box_moved_a_little_is_right_and_points_at_new_spot():
    s = frames(*[[det("keys", 30, 20), det("box", 32, 22)]] * 3,
               *[[det("box", 32, 22)]] * 3, *[[det("box", 40, 24)]] * 3)
    p = run(NearestObject(CFG), s, "keys")
    assert p["parent"] == "box" and p["resolved_cm"] == [40.0, 24.0]


def test_nearest_object_covered_and_nearest_wins_and_threshold():
    s = frames(*[[det("keys", 30, 20), det("notebook", 60, 20), det("box", 50, 45)]] * 3,
               *[[det("notebook", 31, 21), det("box", 50, 45)]] * 4)
    p = run(NearestObject(CFG), s, "keys")
    assert (p["status"], p["parent"]) == ("UNDER", "notebook")
    far = frames(*[[det("keys", 10, 10), det("box", 60, 40)]] * 3, *[[det("box", 60, 40)]] * 4)
    assert run(NearestObject(CFG), far, "keys")["parent"] is None


# ------------------------------------------------------------------ scoring

def test_scoring_rules():
    inside = make_truth("INSIDE", parent="box", pos_cm=(50, 30))
    assert score(prediction("INSIDE", parent="box", resolved_cm=(51, 30)), inside) == (True, 1.0)
    assert score(prediction("UNDER", parent="notebook", resolved_cm=(20, 30)), inside)[0] is False
    gone = make_truth("GONE", edge="left")
    assert score(prediction("GONE", edge="left"), gone) == (True, None)
    assert score(prediction("GONE", edge="right"), gone)[0] is False
    assert score(prediction("UNKNOWN", pos_cm=(1, 30), resolved_cm=(1, 30)), gone)[0] is False
    vis = make_truth("VISIBLE", pos_cm=(40, 22))
    assert score(prediction("VISIBLE", resolved_cm=(43, 26)), vis) == (True, 5.0)
    assert score(prediction("VISIBLE", resolved_cm=(46, 22)), vis) == (False, 6.0)
    assert score(prediction("UNKNOWN"), vis) == (False, None)
    held = make_truth("HELD")
    assert score(prediction("HELD", parent="hand:2", resolved_cm=(5, 5)), held) == (True, None)
    assert score(prediction("VISIBLE", resolved_cm=(5, 5)), held)[0] is False


# ------------------------------------------------------------------ report

def _res(system, cat, correct, skipped=False, err=1.0):
    return {"system": system, "trial_id": 1, "category": cat, "correct": correct,
            "skipped": skipped, "reason": "not built" if skipped else None, "error": None,
            "laser_err_cm": None if skipped else err, "predict_ms": 0.01, "update_ms_median": 0.001}


def test_report_table_rendering():
    results = {
        "full": [_res("full", "covered", False, skipped=True), _res("full", "inside", False, skipped=True)],
        "last_seen": [_res("last_seen", "covered", False, err=12.0), _res("last_seen", "inside", True, err=2.0)],
        "nearest_object": [_res("nearest_object", "covered", True), _res("nearest_object", "inside", True)],
    }
    md = render(results)
    assert "| category | n | full | nearest_object | last_seen |" in md
    assert "| covered | 1 | skipped | 1/1 (100%) | 0/1 (0%) |" in md
    assert "| **overall** | 2 | **skipped** | **2/2 (100%)** | **1/2 (50%)** |" in md
    assert "**hidden**" in md
    assert "| median laser error (cm) | skipped | 1.0 | 7.0 |" in md
    assert "`full` skipped on 2 trial(s): not built" in md


# ------------------------------------------------------------------ synth + replay end to end

@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("trials")
    generate(str(d), CFG, per_category=2, seed=1, categories=CORE + STRETCH_SYNTH)
    return d


def test_synth_writes_every_category_with_consistent_truth(synth_dir):
    trials = load_trials(synth_dir)
    assert {t.category for t in trials} == set(CORE + STRETCH_SYNTH)
    for t in trials:
        assert check_category(t.category, t.truth) is None, (t.id, t.truth)
        dets = list(t.detections())
        assert len(dets) > 40
        assert all(d.frame_idx == i for i, d in enumerate(dets))
        for d in dets:
            for it in d.items:
                assert it.cls in CFG["objects"]
                assert 0 <= it.center_cm[0] <= 91 and 0 <= it.center_cm[1] <= 61
            assert all(h.cls.startswith("hand:") for h in d.hands)


def test_replay_over_synth_with_full_world(synth_dir):
    out = replay(str(synth_dir), ["full", "nearest_object", "last_seen", "current_frame"], CFG,
                 verbose=False)
    assert not any(r["skipped"] for r in out["full"]), [r["reason"] for r in out["full"]]
    by = {s: {} for s in out}
    for s, rs in out.items():
        for r in rs:
            by[s].setdefault(r["category"], []).append(r["correct"])
    for cat in ("covered", "inside", "inside_box_moved", "carried_away"):
        assert not any(by["current_frame"][cat]) and not any(by["last_seen"][cat])
    for cat in ("visible", "moved", "put_back", "uncovered"):
        assert all(by["current_frame"][cat]), cat
    assert all(by["nearest_object"]["inside"]) and all(by["nearest_object"]["covered"])
    assert not any(by["nearest_object"]["inside_box_moved"])
    for cat in ("inside", "inside_box_moved", "carried_away"):
        assert all(by["full"][cat]), cat
    t = load_trials(synth_dir)[0]
    assert set(t.results()) == {"full", "nearest_object", "last_seen", "current_frame"}
    md = render(collect(str(synth_dir)))
    assert "| inside_box_moved | 2 | 2/2 (100%) |" in md and "dropped (stretch)" in md


def test_full_world_gets_covered_trials_right(synth_dir):
    """The hand sliding the notebook touches the object; the cover rule must still win."""
    out = replay(str(synth_dir), ["full"], CFG, verbose=False)
    covered = [r["correct"] for r in out["full"] if r["category"] == "covered"]
    assert covered and all(covered)


class _StubWorld:
    """Stands in for a finished core.world.World: remembers the last box position."""

    def __init__(self, cfg, events):
        self.box = None
        self.frames = 0

    def update(self, dets, frame):
        assert frame.img is None and frame.idx == dets.frame_idx
        self.frames += 1
        for d in dets.items:
            if d.cls == "box":
                self.box = d.center_cm
        return []

    def get(self, name):
        return Entity(name, "target", Status.INSIDE, parent="box", pos_cm=(30.0, 20.0))

    def resolve(self, name):
        return self.box, [name, "box"]


def test_world_adapter_scores_when_world_exists(monkeypatch, tmp_path):
    import core.world
    monkeypatch.setattr(core.world, "World", _StubWorld)
    t = Trial.create(tmp_path, 1, "inside_box_moved", "keys", BOX_TRUTH, "q?", "synth")
    t.write_detections(BOX_MOVED)
    r = run_system("full", CFG, t, list(t.detections()))
    assert not r["skipped"] and r["correct"] and r["laser_err_cm"] == 0.0
    assert r["prediction"]["parent"] == "box" and r["frames"] == len(BOX_MOVED)


def test_system_errors_are_counted_wrong_not_fatal(monkeypatch, tmp_path):
    class Boom(_StubWorld):
        def update(self, dets, frame):
            raise KeyError("oops")
    import core.world
    monkeypatch.setattr(core.world, "World", Boom)
    t = Trial.create(tmp_path, 1, "inside", "keys", BOX_TRUTH, "q?", "synth")
    t.write_detections(BOX_MOVED[:3])
    r = run_system("full", CFG, t, list(t.detections()))
    assert not r["skipped"] and not r["correct"] and "KeyError" in r["error"]


def test_categories_cover_spec_minimums():
    assert sum(n for n, _ in CATEGORIES.values()) >= 48
    assert set(CORE) <= set(CATEGORIES)
