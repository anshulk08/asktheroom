"""eval/clip.py: a guided recording's files (video.mp4 + frames.json + meta.json + truth.json) and
its frames with the recorded times; eval/replay_video.py uses the same frames and the embedder."""
import json

import cv2
import numpy as np
import pytest

from core.config import load_config
from core.types import Detections
from eval import replay_video
from eval.clip import clip_frames, load_clip, write_video

CFG = load_config()


def frames_of(n, w=64, h=48):
    return [np.full((h, w, 3), 20 * i, np.uint8) for i in range(n)]


def test_frames_carry_the_recorded_times_not_the_nominal_fps(tmp_path):
    write_video(tmp_path / "video.mp4", frames_of(5), fps=30)
    t = [0.0, 0.05, 0.2, 0.21, 0.5]
    wall = [1.7e9 + x for x in t]
    (tmp_path / "frames.json").write_text(json.dumps({"wall": wall, "t": t}))
    got = list(clip_frames(tmp_path / "video.mp4"))
    assert [f.t for f in got] == t and [f.wall for f in got] == wall
    assert [f.idx for f in got] == [0, 1, 2, 3, 4]
    assert got[0].img.shape == (48, 64, 3)


def test_without_frames_json_the_times_come_from_the_video_fps(tmp_path):
    write_video(tmp_path / "v.mp4", frames_of(3), fps=10)
    got = list(clip_frames(tmp_path / "v.mp4"))
    assert [f.t for f in got] == pytest.approx([0.0, 0.1, 0.2])


def test_load_clip_reads_every_file_and_defaults_the_truth_lists(tmp_path):
    write_video(tmp_path / "video.mp4", frames_of(2), fps=10)
    (tmp_path / "frames.json").write_text(json.dumps({"wall": [5.0, 5.1], "t": [0.0, 0.1]}))
    (tmp_path / "meta.json").write_text(json.dumps({"clip": "c1", "config": {"a": 1}, "table_cal": {"H": [[1]]}}))
    (tmp_path / "truth.json").write_text(json.dumps({"props": {"A": "wallet"}, "steps": [{"t": 1, "event": "place",
                                                                                           "obj": "A"}]}))
    c = load_clip(tmp_path)
    assert c.name == "c1" and c.meta["config"] == {"a": 1} and c.t == [0.0, 0.1]
    assert c.truth["props"] == {"A": "wallet"} and c.truth["commands"] == [] and c.truth["questions"] == []
    assert c.truth["checkpoints"] == [] and c.truth["steps"][0]["parent"] is None
    assert c.duration_s == pytest.approx(0.1)


def test_replay_video_gives_the_world_the_configured_embedder(monkeypatch):
    """main.build makes World(cfg, events, embed=make_embedder(cfg)); the replay must too."""
    made = []
    monkeypatch.setattr(replay_video, "make_embedder", lambda cfg: made.append(cfg) or (lambda img, box: None))
    from core.types import Frame
    frames = [Frame(t=i / 10, wall=100 + i / 10, img=None, idx=i) for i in range(3)]
    r = replay_video.replay(frames, lambda f: Detections(t=f.t, frame_idx=f.idx), CFG)
    assert made == [CFG] and r["frames"] == 3


def test_replay_video_main_uses_the_frames_json_times(tmp_path, monkeypatch):
    write_video(tmp_path / "video.mp4", frames_of(4), fps=30)
    (tmp_path / "frames.json").write_text(json.dumps({"wall": [9.0, 9.5, 10.0, 10.5], "t": [0.0, 0.5, 1.0, 1.5]}))
    seen = {}

    def fake_replay(frames, det, cfg, *a, **k):
        seen["t"] = [f.t for f in frames]
        return {"frames": 4, "video_s": 1.5, "fps": 99.0, "detect_ms_median": None, "detected": {},
                "touched": [], "untouched": [], "events": {}, "false_disappearances": [], "pass": True}

    class Det:
        def __init__(self, *a, **k):
            pass

    import core.detect
    monkeypatch.setattr(replay_video, "replay", fake_replay)
    monkeypatch.setattr(core.detect, "Detector", Det)
    monkeypatch.setattr(core.detect, "UltralyticsBackend", lambda *a, **k: None)
    assert replay_video.main(["--video", str(tmp_path / "video.mp4")]) == 0
    assert seen["t"] == [0.0, 0.5, 1.0, 1.5]


def test_write_video_round_trips_through_opencv(tmp_path):
    write_video(tmp_path / "x.mp4", frames_of(3), fps=10)
    cap = cv2.VideoCapture(str(tmp_path / "x.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()
