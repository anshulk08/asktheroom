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
