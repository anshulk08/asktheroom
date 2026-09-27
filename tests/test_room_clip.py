"""Room-demo clips: recorded at the app's capture size (room memory on: the full camera frame) and replayed
through core.room_view.TableView exactly as main.open_frames cuts the table view; 1280x720 clips unchanged."""
import json

import numpy as np
import pytest

from core.config import load_config
from core.room_zones import view_version
from core.types import Detections, Frame
from eval import raw_record, score_clip
from eval.clip import FrameSlot, clip_view, load_clip, table_frames, write_video

RIG_RECT = [0, 980, 817, 1440]


def room_meta(capture=(2560, 1440), rect=RIG_RECT, size=None, enabled=True, out=(1280, 720)):
    cfg = {"frame_size_px": list(out),
           "room_memory": {"enabled": enabled, "capture_size": list(capture), "table_view_rect": rect}}
    meta = {"config": cfg}
    if size is not None:
        meta["view"] = {"size_px": list(size)}
    return meta


def test_clip_view_is_the_rigs_rect_at_capture_size_and_none_for_table_clips():
    assert clip_view(room_meta(size=(2560, 1440))) == ((0, 980, 817, 1440), (1280, 720))
    assert clip_view(room_meta(), size_px=(2560, 1440)) == ((0, 980, 817, 1440), (1280, 720))
    assert clip_view(room_meta(enabled=False), size_px=(1280, 720)) is None
    assert clip_view({}, size_px=(1280, 720)) is None                         # old clips: no config view
    assert clip_view(room_meta(size=(1280, 720))) is None                     # resized before the cut existed
    # a clip recorded at 1080p with the 1440p rect: the rect scales with the frame (x0.75)
    assert clip_view(room_meta(size=(1920, 1080))) == ((0, 735, 613, 1080), (1280, 720))
    # no measured rect: main.open_frames' centred default
    got = clip_view(room_meta(rect=None, capture=(1920, 1080), size=(1920, 1080)))
    assert got is not None and got[1] == (1280, 720) and got[0][2] - got[0][0] == 1200


def test_table_frames_cut_like_the_live_table_view():
    full = np.zeros((360, 640, 3), np.uint8)
    full[180:, :320] = (0, 0, 255)                   # the table view region is red, the rest black
    frames = [Frame(t=0.1 * i, wall=100 + 0.1 * i, img=full, idx=i) for i in range(3)]
    out = list(table_frames(frames, ((0, 180, 320, 360), (1280, 720))))
    assert [f.idx for f in out] == [0, 1, 2] and [f.t for f in out] == [f.t for f in frames]
    assert out[0].img.shape == (720, 1280, 3) and out[0].img[..., 2].min() == 255
    assert list(table_frames(frames, None))[0].img is full
    slot = FrameSlot()
    slot.cur = frames[1]
    assert slot.at(5.0) is frames[1] and slot.latest() is frames[1]


def test_recorder_captures_at_the_app_size_and_records_the_view_and_zones(tmp_path):
    cfg = load_config()
    assert raw_record.capture_size(cfg) == tuple(cfg["frame_size_px"])
    zones = {"view": "x", "size_px": [2560, 1440], "zones": {"couch": {"say": "the couch", "poly": [[0, 0], [9, 0], [9, 9]]}}}
    (tmp_path / "z.json").write_text(json.dumps(zones))
    cfg["room_memory"] = dict(cfg.get("room_memory") or {}, enabled=True, capture_size=[2560, 1440],
                              table_view_rect=RIG_RECT, zones_path=str(tmp_path / "z.json"))
    assert raw_record.capture_size(cfg) == (2560, 1440)
    meta = raw_record.meta_for(cfg, "/dev/video0", {}, "abc")
    assert meta["view"] == {"size_px": [2560, 1440], "room_memory": True, "table_view_rect": RIG_RECT,
                            "out_size": list(cfg["frame_size_px"])}
    assert meta["room_zones"] == zones
    assert clip_view(meta) == ((0, 980, 817, 1440), (1280, 720))


class ViewStub:
    """A detector that finds nothing and remembers what it was shown."""
    last_ms = 1.0

    def __init__(self, backend=None):
        self.shapes, self.red = [], []
        if backend is not None:
            self.backend = backend

    def detect(self, frame):
        self.shapes.append(frame.img.shape)
        self.red.append(float(frame.img[..., 2].mean()))
        return Detections(t=frame.t, frame_idx=frame.idx)

    def reset_proposals(self):
        pass


def write_room_clip(d, capture=(640, 360), rect=(0, 180, 320, 360), zones=True, n=12):
    d.mkdir()
    full = np.zeros((capture[1], capture[0], 3), np.uint8)
    full[rect[1]:rect[3], rect[0]:rect[2]] = (0, 0, 255)
    write_video(d / "video.mp4", [full] * n, fps=10)
    t = [round(0.1 * i, 3) for i in range(n)]
    (d / "frames.json").write_text(json.dumps({"t": t, "wall": [1.7e9 + x for x in t]}))
    cfg = load_config()
    cfg["table_tag"] = dict(cfg["table_tag"], enabled=False)
    cfg["table"] = dict(cfg["table"], size_cm=[128, 72])
    cfg["room_memory"] = dict(cfg.get("room_memory") or {}, enabled=True, capture_size=list(capture),
                              table_view_rect=list(rect), zoom=100, things=False)
    z = {"view": view_version(capture, 100, rect), "size_px": list(capture),
         "zones": {"couch": {"say": "the couch", "poly": [[400, 0], [640, 0], [640, 170], [400, 170]]}}}
    meta = {"clip": d.name, "config": cfg, "view": {"size_px": list(capture)},
            "table_cal": {"H": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], "markers_px": {}, "size_cm": [128, 72], "t": 0},
            "room_zones": z if zones else None}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "truth.json").write_text(json.dumps({"props": {}, "steps": []}))
    return str(d)


def test_replay_of_a_room_clip_gives_the_table_pipeline_the_live_cut_and_runs_room_memory(tmp_path):
    clip = load_clip(write_room_clip(tmp_path / "room_1"))
    assert clip.size_px == (640, 360) and clip.view() == ((0, 180, 320, 360), (1280, 720))
    det = ViewStub(backend=score_clip.NoBoxes())
    trace = score_clip.replay_clip(clip, detector=det, max_fps=100)
    assert det.shapes and all(s == (720, 1280, 3) for s in det.shapes)
    assert min(det.red) > 250                      # only the red table view, never the black room
    assert trace.view == [[0, 180, 320, 360], [1280, 720]] and trace.room is True


def test_room_memory_stays_off_without_recorded_zones(tmp_path):
    clip = load_clip(write_room_clip(tmp_path / "room_2", zones=False))
    trace = score_clip.replay_clip(clip, detector=ViewStub(backend=score_clip.NoBoxes()), max_fps=100)
    assert trace.view is not None and trace.room is False


def test_a_1280x720_clip_replays_uncut(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    img = np.full((720, 1280, 3), 200, np.uint8)
    write_video(d / "video.mp4", [img] * 5, fps=10)
    (d / "frames.json").write_text(json.dumps({"t": [0.1 * i for i in range(5)], "wall": [0.1 * i for i in range(5)]}))
    cfg = load_config()
    cfg["table_tag"] = dict(cfg["table_tag"], enabled=False)
    (d / "meta.json").write_text(json.dumps({"config": cfg, "table_cal": {"H": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]],
                                                                         "markers_px": {}, "size_cm": [128, 72], "t": 0}}))
    clip = load_clip(d)
    assert clip.view() is None
    det = ViewStub()
    trace = score_clip.replay_clip(clip, detector=det, max_fps=100)
    assert det.shapes and det.shapes[0] == (720, 1280, 3) and trace.view is None and trace.room is False


def test_hands_off_detector_has_no_fixed_class_model():
    cfg = load_config()
    cfg["proposals"] = dict(cfg["proposals"], enabled=True, kind="change")
    det = score_clip.make_hands_off_detector(cfg, None)
    assert isinstance(det.backend, score_clip.NoBoxes) and det.proposer is not None


def test_replay_config_takes_the_yoloe_model_override(tmp_path):
    import argparse
    ap = argparse.ArgumentParser()
    score_clip.add_replay_args(ap)
    a = ap.parse_args(["--yoloe-model", "models/yoloe-26s-seg-pf.pt", "--hands-off"])
    clip = load_clip(write_room_clip(tmp_path / "room_3"))
    cfg = score_clip.replay_config(clip, a)
    assert cfg["proposals"]["yoloe"]["model"] == "models/yoloe-26s-seg-pf.pt" and a.hands_off
    assert cfg["room_memory"]["table_view_rect"] == [0, 180, 320, 360]


@pytest.mark.parametrize("name", ["room_still", "room_clutter", "room_couch", "room_carry", "room_move", "room_remove",
                                  "room_straight", "room_block", "room_keys_off", "room_return"])
def test_room_scenarios_become_truth_with_segments_and_zones(name):
    from eval.guided import CLIPS, truth_from
    c = CLIPS[name]
    assert c["room"] and 60 <= c["seconds"] <= 180
    cues = [1000.0 + s["at"] for s in c["steps"]]
    t = truth_from(c, cues, first_frame_wall=1000.0, clock_offset_s=0.0)
    assert all(s["seg"] in ("still", "people") for s in t["steps"])
    placed = {s["obj"] for s in t["steps"] if s["event"] == "place"}
    assert placed, name                                        # every room clip binds props with place cues
    for s in t["steps"]:
        if s["event"] in ("carry_to", "place_room"):
            assert s["zone"] in ("couch", "side_table", "counter", "floor")
        if s["event"] == "place_room":
            assert s["obj"] not in placed                      # straight into the room, never on the table
    if name == "room_clutter":
        assert t["scene_objects"] == ["laptop", "cable pile"]
    if name in ("room_keys_off", "room_return", "room_carry"):
        back = [s for s in t["steps"] if s.get("expect_same")]
        assert back and all(s["event"] == "putdown" for s in back)
    if name == "room_block":
        b0, b1 = [s["t"] for s in t["steps"] if s["event"] in ("block", "unblock")]
        assert 10 <= b1 - b0 <= 20


def test_set_overrides_any_config_key_and_mode_is_permanence_mode(tmp_path):
    import argparse
    ap = argparse.ArgumentParser()
    score_clip.add_replay_args(ap)
    a = ap.parse_args(["--set", "permanence.mode=registry", "--set", "proposals.yoloe.conf=0.25",
                       "--set", "room_memory.zones_path=elsewhere.json", "--mode", "registry2"])
    clip = load_clip(write_room_clip(tmp_path / "room_4"))
    recorded = json.loads((tmp_path / "room_4" / "meta.json").read_text())["config"]
    cfg = score_clip.replay_config(clip, a)
    assert cfg["permanence"] == {"mode": "registry2"}               # --mode applies after --set
    assert cfg["proposals"]["yoloe"]["conf"] == 0.25
    assert cfg["proposals"]["yoloe"]["model"] == recorded["proposals"]["yoloe"]["model"]   # siblings kept
    assert clip.meta["config"]["proposals"]["yoloe"].get("conf") == recorded["proposals"]["yoloe"].get("conf")
    assert score_clip.overrides(a)[-1] == "permanence.mode=registry2"
    assert score_clip.set_key({"a": {"b": 1}}, "a.c=[1, 2]") == {"a": {"b": 1, "c": [1, 2]}}
    assert score_clip.set_key({}, "x.y=true") == {"x": {"y": True}}
    with pytest.raises(ValueError):
        score_clip.set_key({"a": 1}, "a.b=2")
    with pytest.raises(ValueError):
        score_clip.set_key({}, "novalue")


def test_replay_from_args_records_the_overrides_on_the_trace(tmp_path, monkeypatch):
    import argparse
    ap = argparse.ArgumentParser()
    score_clip.add_replay_args(ap)
    a = ap.parse_args(["--mode", "registry"])
    clip = load_clip(write_room_clip(tmp_path / "room_5"))
    seen = {}

    def fake_replay(clip, cfg, **kw):
        seen["cfg"] = cfg
        return score_clip.Trace()
    monkeypatch.setattr(score_clip, "replay_clip", fake_replay)
    trace = score_clip.replay_from_args(clip, a)
    assert seen["cfg"]["permanence"]["mode"] == "registry" and trace.overrides == ["permanence.mode=registry"]


class FakeNamer:
    """An injected naming provider (core.narration's narrate API), as WS3's cached provider would be."""

    def __init__(self, name="coffee mug"):
        self.name, self.calls = name, 0

    def narrate(self, system, parts, schema):
        from types import SimpleNamespace
        self.calls += 1
        return SimpleNamespace(text=json.dumps({"name": self.name, "also": [], "confidence": 0.9}))


def test_names_hook_attaches_auto_name_with_an_injected_provider_on_clip_time(tmp_path):
    from eval import scorecard
    from tests.test_score_clip import Take
    take = Take(tmp_path)
    take.run(1)
    take.put("A", (60, 30))
    take.run(4)
    truth = {"props": {"A": "mug"}, "steps": [{"t": 1.0, "event": "place", "obj": "A"}]}
    clip = load_clip(take.write(truth))
    fake = FakeNamer()
    trace = score_clip.replay_clip(clip, detector=take.stub(), names=fake)
    assert fake.calls >= 1 and "names: FakeNamer" in trace.detector
    named = {n: v for s in trace.samples for n, v in (s.names or {}).items()}
    assert named["thing:1"]["guess"]["name"] == "coffee mug"
    r = scorecard.scorecard(trace, clip)
    assert r["naming"]["props"]["A"] == {"entity": "thing:1", "guess": "coffee mug", "fits": True}
    assert score_clip.replay_clip(clip, detector=take.stub()).samples[-1].names == {}     # off by default


def test_load_provider_passes_what_the_factory_takes(tmp_path):
    import sys
    import types
    mod = types.ModuleType("fake_naming_mod")
    mod.make = lambda cfg=None, clip_dir=None: ("made", cfg["x"], clip_dir)
    mod.bare = lambda: "bare"
    sys.modules["fake_naming_mod"] = mod
    clip = load_clip(write_room_clip(tmp_path / "room_6"))
    assert score_clip.load_provider("fake_naming_mod:make", {"x": 1}, clip) == ("made", 1, str(clip.dir))
    assert score_clip.load_provider("fake_naming_mod:bare", {}, clip) == "bare"
    assert score_clip.load_provider("grok", {}, clip) is None
