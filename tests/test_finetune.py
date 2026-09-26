import cv2
import numpy as np
import pytest
import yaml

from core.config import load_config
from core.detect import class_list
from scripts.finetune import autolabel, common, extract, train

CFG = load_config()


def scene(seed: int, obj_x: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), 150, np.uint8)
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)      # sensor noise
    cv2.rectangle(img, (obj_x, 120), (obj_x + 80, 200), (20, 40, 200), -1)
    return img


def test_class_names_are_objects_then_hand():
    names = common.class_names(CFG)
    assert names == list(CFG["objects"]) + ["hand"] and len(names) == 9


def test_trial_of_handles_underscores():
    assert common.trial_of(common.frame_name("17", 120)) == "17"
    assert common.trial_of(common.frame_name("covered_3", 5)) == "covered_3"


def test_dhash_ignores_noise_but_sees_motion():
    a, b, c = scene(0), scene(1), scene(0, obj_x=60)
    assert common.hamming(common.dhash(a), common.dhash(b)) <= 6
    assert common.hamming(common.dhash(a), common.dhash(c)) > 6


def test_distinct_drops_still_frames():
    frames = [(i, scene(i)) for i in range(10)] + [(10 + i, scene(i, obj_x=60 + 50 * i)) for i in range(5)]
    kept = extract.distinct(frames, 6)
    assert kept[0][0] == 0 and 3 <= len(kept) <= 7


def test_thin_hits_target_and_keeps_every_video():
    per = {"1": list(range(300)), "2": list(range(200)), "3": list(range(5))}
    out = extract.thin(per, 100)
    assert set(out) == {"1", "2", "3"} and 95 <= sum(map(len, out.values())) <= 105
    assert out["1"][0] == 0 and out["1"][-1] == 299
    assert extract.thin(per, 1000) is per


def test_extract_end_to_end(tmp_path):
    for t in ("1", "2"):
        d = tmp_path / "trials" / t
        d.mkdir(parents=True)
        w = cv2.VideoWriter(str(d / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
        for i in range(40):
            w.write(scene(i, obj_x=40 + 12 * i))
        w.release()
    assert extract.main(["--trials", str(tmp_path / "trials"), "--data", str(tmp_path / "ft"),
                         "--every-s", "0.2"]) == 0
    imgs = sorted((tmp_path / "ft" / "images").glob("*.jpg"))
    assert imgs and {common.trial_of(p.stem) for p in imgs} == {"1", "2"}


def test_labels_map_prompts_to_object_classes():
    _, to_obj = class_list(CFG)
    names = common.class_names(CFG)
    raw = [("medicine bottle", 0.6, (100, 100, 140, 180)),
           ("pill bottle", 0.4, (0, 0, 10, 10)),            # weaker duplicate: dropped
           ("hand", 0.5, (200, 200, 300, 300)), ("hand", 0.3, (400, 50, 500, 150)),
           ("tv remote", 0.1, (0, 0, 50, 50)),              # under conf
           ("banana", 0.9, (0, 0, 50, 50))]                 # not ours
    lines = autolabel.to_labels(raw, to_obj, names, 640, 360, 0.2)
    cls = [names[int(ln.split()[0])] for ln in lines]
    assert cls == ["pill_bottle", "hand", "hand"]
    c, cx, cy, w, h = map(float, lines[0].split())
    assert (cx, cy, w, h) == pytest.approx((120 / 640, 140 / 360, 40 / 640, 80 / 360), abs=1e-5)


def test_split_holds_out_whole_trials():
    stems = [common.frame_name(t, i) for t in ("1", "2", "3", "4", "5") for i in range(20)]
    tr, va = train.split_by_trial(stems, 0.2)
    trials_tr, trials_va = {common.trial_of(s) for s in tr}, {common.trial_of(s) for s in va}
    assert trials_va and not trials_tr & trials_va and len(tr) + len(va) == 100
    assert 15 <= len(va) <= 40
    tr, va = train.split_by_trial(stems, val_trials=["3"])
    assert {common.trial_of(s) for s in va} == {"3"}
    with pytest.raises(ValueError):
        train.split_by_trial(stems, val_trials=["9"])
    with pytest.raises(ValueError):
        train.split_by_trial([common.frame_name("1", i) for i in range(5)])


def test_write_split_yaml(tmp_path):
    names = common.class_names(CFG)
    p = train.write_split(tmp_path, names, ["1_000001"], ["2_000001"])
    d = yaml.safe_load(p.read_text())
    assert d["names"][8] == "hand" and d["val"] == "val.txt"
    assert (tmp_path / "val.txt").read_text().strip().endswith("images/2_000001.jpg")


def test_public_hand_frames_always_train_and_never_validate():
    from scripts.finetune.train import split_by_trial
    stems = ([f"cap{g}_wallet-{k:02d}" for g in range(4) for k in range(2)] + [f"synth_{i:05d}" for i in range(10)]
             + [f"pubhand_{i}" for i in range(30)])
    for seed in range(8):
        tr, va = split_by_trial(stems, seed=seed)
        assert va and not any(s.startswith(("pubhand_", "synth_")) for s in va), seed
        assert sum(s.startswith("pubhand_") for s in tr) == 30


def test_negatives_are_counted(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "cap0_empty-00.txt").write_text("")
    (tmp_path / "labels" / "cap0_mug-00.txt").write_text("\n")
    (tmp_path / "labels" / "cap0_keys-00.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert train.count_negatives(tmp_path, ["cap0_empty-00", "cap0_mug-00", "cap0_keys-00"]) == 2
