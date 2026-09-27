"""eval/conf_sweep.py: detection rates per class per cut-off, without a model."""
from core.types import Frame
from eval.conf_sweep import best_confs, paced, report, sweep


def test_paced_takes_frames_as_a_capped_live_loop_would():
    frames = [Frame(t=i / 30, wall=0.0, img=None, idx=i) for i in range(30)]
    assert len(list(paced(frames, 15))) == 15 and len(list(paced(frames, 0))) == 30


def test_best_conf_per_object_maps_prompt_labels_back():
    raw = [("key ring", 0.3, None), ("keys", 0.5, None), ("hand", 0.2, None), ("laptop", 0.9, None)]
    assert best_confs(raw, {"key ring": "keys", "keys": "keys", "hand": "hand"}) == {"keys": 0.5, "hand": 0.2}


def test_rates_per_cut_and_hands_at_steps():
    per_frame = [(0.0, {"notebook": 0.5, "hand": 0.18}), (0.5, {"notebook": 0.25}), (3.0, {"hand": 0.4})]
    r = sweep(per_frame, ["notebook"], [{"t": 0.2}, {"t": 3.2}, {"t": 6.0}], cuts=(0.15, 0.3))
    assert r["rates"]["notebook"] == {0.15: round(2 / 3, 3), 0.3: round(1 / 3, 3)}
    assert r["hand_at_steps"] == {0.15: 2, 0.3: 1} and r["steps"] == 3
    assert "hand at steps" in report(r)
