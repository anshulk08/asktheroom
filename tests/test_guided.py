"""eval/guided.py: clip scripts are well formed and cue times become clip-relative ground truth."""
from eval.guided import CLIPS, truth_from


def test_every_clip_is_well_formed():
    events = {"hands_out", "hand_in", "wave", "rest_arm", "place", "pickup", "putdown", "put_inside", "cover",
              "uncover", "move", "exit_edge", None}
    for name, c in CLIPS.items():
        assert c["props"] and c["setup"] and c["steps"], name
        ts = [s["at"] for s in c["steps"]]
        assert ts == sorted(ts) and c["seconds"] > ts[-1], name
        for s in c["steps"]:
            assert s.get("event") in events, (name, s)
            assert s.get("obj") in (None, *c["props"]), (name, s)
        for q in c.get("questions", []):
            assert q["expect_prop"] in c["props"]


def test_cue_times_become_seconds_since_the_first_frame():
    clip = {"props": {"A": "wallet"}, "seconds": 20, "setup": "x",
            "steps": [{"at": 0, "say": "hands out", "event": "hands_out"},
                      {"at": 4, "say": "put A down", "event": "place", "obj": "A"}],
            "commands": [{"at": 8, "text": "this is my brown wallet"}],
            "questions": [{"at": 12, "text": "where is my brown wallet?", "expect_prop": "A"}],
            "checkpoints": [{"at": 10, "expect": {"A": {"state": "on_table"}}}]}
    # the Mac spoke cue 0 at mac time 1000.0 and cue 1 at 1004.3; the Jetson clock runs 2.5 s ahead of
    # the Mac; the first frame was at Jetson time 1001.0
    cues = [1000.0, 1004.3]
    t = truth_from(clip, cues, first_frame_wall=1001.0, clock_offset_s=2.5)
    assert [round(s["t"], 2) for s in t["steps"]] == [1.5, 5.8]
    start = 1000.0 + 2.5 - 1001.0                       # clip time of the scripted zero
    assert round(t["commands"][0]["t"], 2) == round(start + 8, 2)
    assert round(t["questions"][0]["t"], 2) == round(start + 12, 2)
    assert round(t["checkpoints"][0]["t"], 2) == round(start + 10, 2)
    assert t["props"] == {"A": "wallet"} and t["steps"][1]["obj"] == "A"


def test_recording_uses_the_remote_copy_and_stops_only_the_app_on_the_camera(monkeypatch):
    """ASKROOM_REMOTE_DIR: record in a scratch copy on the Jetson, not the teammate's ~/askroom; stop only a
    container with /dev/v4l mounted (the app), never a replay; set the Brio's controls."""
    import pytest

    from eval import guided
    calls = []

    class Stop(Exception):
        pass

    def popen(args, **kw):
        calls.append(args[-1])
        raise Stop

    monkeypatch.setattr(guided, "REMOTE", "askroom_rig")
    monkeypatch.setattr(guided, "ssh", lambda cmd, timeout=60: calls.append(cmd) or "")
    monkeypatch.setattr(guided, "clock_offset", lambda: 0.0)
    monkeypatch.setattr(guided, "camera_controls", lambda: {})
    monkeypatch.setattr(guided.subprocess, "run", lambda *a, **kw: type("R", (), {"stdout": "abc123\n"})())
    monkeypatch.setattr(guided.subprocess, "Popen", popen)
    with pytest.raises(Stop):
        guided.run_clip("still", "brio_still_1")
    stop, mkdir, rec = calls
    assert "--filter volume=/dev/v4l" in stop and "camera_setup.sh 166" in stop and " 80 10 160 3200" in stop
    assert "~/askroom_rig/data/clips/brio_still_1" in mkdir and "~/askroom/" not in mkdir
    assert rec.startswith("cd ~/askroom_rig && ")
