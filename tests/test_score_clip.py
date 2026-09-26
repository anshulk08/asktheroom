"""eval/score_clip.py on synthetic guided clips: tests/synth.Scene draws the frames (written as a real
clip: video.mp4 + frames.json + meta.json + truth.json) and a stub detector returns the scene's own
detections, so the replay runs Room.perceive -> World and the real ask pipeline with no models."""
import json

import pytest

from core.config import Config, load_config
from eval import score_clip
from eval.clip import load_clip, write_video
from tests.synth import Scene

FPS = 10
H_10PX_PER_CM = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]      # synth: 10 px per cm, origin at pixel (0, 0)
BOX_AT, NB_AT = (100, 50), (30, 50)


def clip_config() -> dict:
    cfg = load_config()
    cfg["table_tag"] = dict(cfg["table_tag"], enabled=False)
    cfg["table"] = dict(cfg["table"], size_cm=[128, 72])
    return cfg


class Take:
    """Films a synthetic scene: rendered frames -> video.mp4, their times -> frames.json, and the
    scene's detections kept per frame for the stub detector."""

    def __init__(self, tmp_path, name="take"):
        self.dir = tmp_path / name
        self.dir.mkdir()
        self.cfg = clip_config()
        self.scene = Scene(Config.from_dict(self.cfg), fps=FPS, t0=-1.0 / FPS, render=True)
        self.imgs, self.dets, self.t = [], [], []

    @property
    def now(self) -> float:
        return round(self.scene.t, 6)

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds * FPS)):
            d, f = self.scene.step()
            self.imgs.append(f.img)
            self.dets.append(d)
            self.t.append(round(f.t, 6))

    def put(self, key, at, hid=1):
        """A hand sets a thing down at `at` and leaves."""
        self.scene.hand(hid, *at)
        self.scene.thing(key, *at)
        self.run(0.5)
        self.scene.hand_off(hid)

    def write(self, truth: dict) -> str:
        write_video(self.dir / "video.mp4", self.imgs, fps=FPS)
        (self.dir / "frames.json").write_text(json.dumps({"wall": [1.7e9 + t for t in self.t], "t": self.t}))
        meta = {"clip": self.dir.name, "camera": {"device": "synthetic", "controls": {}},
                "table_cal": {"H": H_10PX_PER_CM, "markers_px": {}, "size_cm": [128, 72], "t": 0},
                "config": self.cfg, "git": "test", "models": {"detect": "stub", "proposals": {"kind": "stub",
                                                                                              "model": None}},
                "recorded_at": "2026-09-26T12:00:00", "clock_offset_s": 0.0}
        (self.dir / "meta.json").write_text(json.dumps(meta))
        truth = dict({"steps": [], "commands": [], "questions": [], "checkpoints": []}, **truth)
        (self.dir / "truth.json").write_text(json.dumps(truth))
        return str(self.dir)

    def stub(self):
        return Stub(self.dets)


class Stub:
    """The detector: the scene's detections for the frame index, timed like the replayed frame."""
    last_ms = 1.0

    def __init__(self, dets):
        self.dets = dets
        self.resets = 0

    def detect(self, frame):
        d = self.dets[frame.idx]
        d.t, d.frame_idx = frame.t, frame.idx
        return d

    def reset_proposals(self):
        self.resets += 1


def score(take, truth, **kw):
    clip = load_clip(take.write(truth))
    return score_clip.score_clip(clip, detector=take.stub(), **kw)


def crit(report, name):
    return next(c for c in report["criteria"] if c["name"] == name)


def at_rest(*props, parent=None):
    return {p: {"state": "on_table", "parent": parent} for p in props}


PROPS = {"A": "charger", "BOX": "box", "NB": "notebook"}


# ----- the four required cases -------------------------------------------------------------------------

def test_a_still_table_has_no_false_births_and_keeps_every_identity(tmp_path):
    take = Take(tmp_path)
    take.scene.place("box", *BOX_AT)
    take.scene.place("notebook", *NB_AT)
    take.scene.thing("A", 60, 20)
    take.run(20)
    r = score(take, {"props": PROPS, "checkpoints": [{"t": 10, "expect": at_rest("A", "BOX", "NB")},
                                                     {"t": 19, "expect": at_rest("A", "BOX", "NB")}]})
    assert r["false_births"] == [] and r["identity_changes"] == [] and r["false_disappearances"] == []
    assert r["mapping"]["A"][0]["entity"] == "thing:1" and len(r["mapping"]["A"]) == 1
    assert r["mapping"]["BOX"][0]["entity"] == "box" and r["mapping"]["NB"][0]["entity"] == "notebook"
    assert [(e["entity"], e["prop"]) for e in r["initial_scene"]["entities"]] == [("thing:1", "A")]
    assert r["guessed"] == ["A"]              # 'charger' is no configured name and truth has no positions
    assert r["checkpoint_accuracy"] == 1.0
    assert r["roles"]["BOX"]["rate"] == pytest.approx(1.0) and r["roles"]["NB"]["rate"] == pytest.approx(1.0)
    assert crit(r, "false births")["result"] == "PASS" and crit(r, "questions")["result"] == "n/a"
    assert r["pass"] is True
    assert r["replay"]["frames_decoded"] == 200 and r["replay"]["errors"] == 0


def test_placement_teach_and_question_point_at_the_one_identity(tmp_path):
    take = Take(tmp_path)
    take.scene.place("box", *BOX_AT)
    take.run(3)
    t_place = take.now
    take.put("A", (40, 30))
    take.run(3.5)
    t_teach = take.now
    take.run(2)
    t_ask = take.now
    take.run(2)
    r = score(take, {"props": {"A": "charger", "BOX": "box"},
                     "steps": [{"t": t_place, "event": "place", "obj": "A", "parent": None, "note": ""}],
                     "commands": [{"t": t_teach, "text": "this is my charger"}],
                     "questions": [{"t": t_ask, "text": "where is my charger?", "expect_prop": "A",
                                    "expect_parent": None}],
                     "checkpoints": [{"t": t_ask + 1, "expect": at_rest("A", "BOX")}]})
    assert [m["entity"] for m in r["mapping"]["A"]] == ["thing:1"]
    assert r["commands"][0]["point_at"] == "thing:1" and "charger" in r["commands"][0]["answer"]
    assert r["commands"][0]["overheard_ignored"] is True       # always-on mic alone would have dropped it
    q = r["questions"][0]
    assert q["point_at"] == "thing:1" and q["expected_entity"] == "thing:1" and q["correct"] is True
    assert r["question_accuracy"] == 1.0
    (p,) = r["placements"]
    assert p["entity"] == "thing:1" and p["missed"] is False and 0 < p["delay_s"] < 3
    assert r["missed_placements"] == 0 and r["identity_changes"] == [] and r["false_births"] == []
    assert r["pass"] is True


def test_an_identity_change_on_an_untouched_prop_is_caught(tmp_path):
    """The thing vanishes with no hand (the detector lost it) and comes back 6 cm off: the world makes
    thing:2, which now stands where A is, so A's identity changed although nobody touched it."""
    take = Take(tmp_path)
    take.scene.place("box", *BOX_AT)
    take.scene.thing("A", 50, 30)
    take.run(5)
    take.scene.remove("A")
    take.run(3)
    take.scene.thing("A", 56, 30)
    take.run(5)
    r = score(take, {"props": {"A": "charger", "BOX": "box"},
                     "checkpoints": [{"t": 3, "expect": at_rest("A")}, {"t": 12.5, "expect": at_rest("A")}]})
    assert [m["entity"] for m in r["mapping"]["A"]] == ["thing:1", "thing:2"]
    (ch,) = r["identity_changes"]
    assert (ch["prop"], ch["from"], ch["to"], ch["excused"]) == ("A", "thing:1", "thing:2", False)
    assert {e["type"] for e in r["false_disappearances"]} <= {"LOST_TRACK", "COVERED"}
    assert r["false_disappearances"][0]["prop"] == "A" and r["false_disappearances"][0]["entity"] == "thing:1"
    assert r["false_births"] == []                              # thing:2 is A, not a phantom
    assert crit(r, "identity changes")["result"] == "FAIL" and r["pass"] is False


def test_a_phantom_thing_while_a_hand_waves_is_a_false_birth(tmp_path):
    take = Take(tmp_path)
    take.scene.place("box", *BOX_AT)
    take.scene.thing("A", 20, 20)
    take.run(5)
    t_wave = take.now
    for x in (60, 70, 80):
        take.scene.hand(1, x, 60)
        take.run(0.3)
    take.scene.hand_off(1)
    take.scene.thing("ghost", 110, 15)         # a glare patch the proposer keeps for 2.5 s
    take.run(2.5)
    take.scene.remove("ghost")
    take.run(4)
    r = score(take, {"props": {"A": "charger", "BOX": "box"},
                     "steps": [{"t": t_wave, "event": "wave", "obj": None, "parent": None,
                                "note": "wave over the table"}],
                     "checkpoints": [{"t": 14, "expect": at_rest("A", "BOX")}]})
    (fb,) = r["false_births"]
    assert fb["entity"] == "thing:2" and fb["pos_cm"] == pytest.approx([110, 15], abs=1)
    assert r["false_births_per_min"] == pytest.approx(60 / r["replay"]["video_s"], rel=0.01)
    assert r["identity_changes"] == [] and r["checkpoint_accuracy"] == 1.0
    assert crit(r, "false births")["result"] == "FAIL"


# ----- the initial scene, re-births, and props described instead of named ----------------------------

def test_static_clutter_is_the_initial_scene_and_props_match_configured_names_only_by_description(tmp_path):
    """Like data/clips/still_1: clutter on the table from the start (one piece admitted at once, one that
    flickers until 5 s, so the world admits it late though its box was there from the first frames) is
    the initial scene, not false births. 'wallet' is the configured wallet by name; a 'lego tub
    (container stand-in)' is whatever stands where it is: here the detector calls it the box."""
    take = Take(tmp_path)
    take.scene.place("wallet", 20, 20)
    take.scene.place("notebook", *NB_AT)
    take.scene.place("box", 100, 50)
    take.scene.thing("speaker", 110, 12, w=14, h=10)
    take.scene.thing("cable", 60, 62, w=20, h=3)
    for i in range(50):
        take.scene.miss("cable", i % 10 >= 3)
        take.run(0.1)
    take.scene.miss("cable", False)
    take.run(8)
    r = score(take, {"props": {"A": "wallet", "NB": "notebook", "BOX": "lego tub (container stand-in)"},
                     "scene": "static clutter also in view from the start: speaker box, cables",
                     "checkpoints": [{"t": 12, "expect": at_rest("A", "NB", "BOX")}]})
    assert r["false_births"] == [] and r["identity_changes"] == []
    assert r["initial_scene"]["count"] == 2
    assert {e["entity"] for e in r["initial_scene"]["entities"]} == {"thing:1", "thing:2"}
    late = next(e for e in r["initial_scene"]["entities"] if e["t"] > 3)
    assert late["pos_cm"] == pytest.approx([60, 62], abs=1)
    assert {p: r["mapping"][p][0]["entity"] for p in ("A", "NB", "BOX")} == {"A": "wallet", "NB": "notebook",
                                                                            "BOX": "box"}
    assert r["guessed"] == ["BOX"]
    assert r["roles"]["BOX"]["kind"] == "container" and r["roles"]["BOX"]["rate"] == pytest.approx(1.0)
    assert r["roles"]["NB"]["kind"] == "cover" and "A" not in r["roles"]
    assert r["checkpoint_accuracy"] == 1.0 and r["pass"] is True


def test_clutter_that_comes_back_as_a_new_identity_is_an_identity_change_not_the_scene(tmp_path):
    take = Take(tmp_path)
    take.scene.place("box", *BOX_AT)
    take.scene.thing("speaker", 30, 20, w=14, h=10)
    take.run(5)
    take.scene.remove("speaker")
    take.run(3)
    take.scene.thing("speaker", 36, 20, w=14, h=10)
    take.run(4)
    r = score(take, {"props": {"BOX": "box"}})
    assert r["initial_scene"]["count"] == 1 and r["false_births"] == []
    (ch,) = r["identity_changes"]
    assert (ch["prop"], ch["from"], ch["to"]) == ("scene", "thing:1", "thing:2")
    assert r["pass"] is False


# ----- the shell game: hands at the handling steps, the box in its role, the answer through the chain ---

def test_put_inside_the_box_sees_the_hands_and_answers_inside_the_box(tmp_path):
    take = Take(tmp_path)
    take.scene.place("box", 80, 40)
    take.scene.thing("A", 10, 50)
    take.run(2)
    t_teach = take.now
    take.run(1)
    t_pick = take.now
    take.scene.hand(1, 10, 50)
    take.run(0.3)
    take.scene.remove("A")
    take.run(0.3)
    t_in = take.now
    for x, y in [(30, 46), (50, 43), (70, 41)]:
        take.scene.hand(1, x, y)
        take.run(0.1)
    take.scene.hand(1, 80, 40)
    take.run(0.6)
    take.scene.hand(1, 100, 60)
    take.run(0.5)
    take.scene.hand_off(1)
    take.run(3)
    t_q = take.now
    take.run(1)
    r = score(take, {"props": {"A": "charger", "BOX": "box"},
                     "steps": [{"t": t_pick, "event": "pickup", "obj": "A", "parent": None, "note": ""},
                               {"t": t_in, "event": "put_inside", "obj": "A", "parent": "BOX", "note": ""}],
                     "commands": [{"t": t_teach, "text": "this is my charger"}],
                     "questions": [{"t": t_q, "text": "where is my charger?", "expect_prop": "A",
                                    "expect_parent": "BOX"}],
                     "checkpoints": [{"t": t_q, "expect": {"A": {"state": "inside", "parent": "BOX"},
                                                           "BOX": {"state": "on_table", "parent": None}}}]})
    assert [s["event"] for s in r["hand_steps"]] == ["pickup", "put_inside"]
    assert all(s["hands_seen"] and s["near"] for s in r["hand_steps"])
    assert r["roles"]["BOX"]["rate"] == pytest.approx(1.0) and r["roles"]["BOX"]["kind"] == "container"
    assert r["checkpoint_accuracy"] == 1.0
    q = r["questions"][0]
    assert q["correct"] is True and q["parent_ok"] is True and "inside the box" in q["answer"]
    assert r["false_disappearances"] == [] and r["identity_changes"] == []
    assert r["pass"] is True


# ----- replay pacing ------------------------------------------------------------------------------

def test_replay_takes_the_newest_frame_each_time_the_capped_loop_is_ready(tmp_path):
    """30 fps camera, 15 fps perception cap: the live loop takes the newest frame when it is ready, so
    every other frame, not the first frame after each 1/15 s (which would be about 10 fps). The last
    frame is still the newest one when the loop is next ready, so it is taken too."""
    from eval.clip import load_clip as load
    take = Take(tmp_path)
    take.scene.place("box", *BOX_AT)
    take.run(3)
    d = take.write({"props": {"BOX": "box"}})
    times = [round(i * 0.033 + (0.002 if i % 2 else 0.0), 4) for i in range(len(take.t))]   # ~30 fps, jittery
    (take.dir / "frames.json").write_text(json.dumps({"wall": [1.7e9 + t for t in times], "t": times}))
    trace = score_clip.replay_clip(load(d), detector=take.stub())
    got = [s.t for s in trace.samples]
    assert got[:3] == [0.0, times[2], times[4]] and got[-2:] == [times[28], times[29]]   # the last one too
    assert trace.frames_decoded == 30


# ----- the command line ----------------------------------------------------------------------------

def test_command_line_prints_pass_fail_lines_and_writes_json(tmp_path, monkeypatch, capsys):
    take = Take(tmp_path)
    take.scene.place("box", *BOX_AT)
    take.scene.thing("A", 60, 20)
    take.run(4)
    d = take.write({"props": {"A": "charger", "BOX": "box"}, "checkpoints": [{"t": 3, "expect": at_rest("A")}]})
    monkeypatch.setattr(score_clip, "make_detector", lambda cfg, table: take.stub())
    out = tmp_path / "r.json"
    assert score_clip.main([d, "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "PASS  false births" in text and "OVERALL: PASS" in text
    assert json.loads(out.read_text())["pass"] is True
