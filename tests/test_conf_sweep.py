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


def test_the_go_no_go_gate_names_each_class_below_the_bar():
    from eval.conf_sweep import gate
    per_frame = [(0.0, {"notebook": 0.7, "keys": 0.65}), (0.1, {"notebook": 0.8, "keys": 0.3}), (0.2, {"notebook": 0.61})]
    r = sweep(per_frame, ["notebook", "keys", "box"], [])
    assert gate(r, ["notebook"], 0.6, 0.8) == []
    fails = gate(r, ["notebook", "keys", "box"], 0.6, 0.8)
    assert [f.split()[0] for f in fails] == ["keys", "box"] and "at 0.6" in fails[0]


def test_image_frames_come_in_name_order(tmp_path):
    import cv2
    import numpy as np
    from eval.conf_sweep import image_frames
    for n in ("scene-0002", "scene-0000", "scene-0001"):
        cv2.imwrite(str(tmp_path / f"{n}.jpg"), np.full((4, 4, 3), int(n[-1]) * 50, np.uint8))
    fs = list(image_frames(tmp_path))
    assert [f.idx for f in fs] == [0, 1, 2] and [int(f.img[0, 0, 0]) // 50 for f in fs] == [0, 1, 2]
