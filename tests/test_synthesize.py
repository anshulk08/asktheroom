"""scripts/finetune/synthesize.py box bookkeeping, and capture -> synthesize -> split end to end with a fake camera."""
import shutil

import numpy as np
import pytest

from core.config import load_config
from core.types import Frame
from scripts.finetune import capture, synthesize, train
from scripts.finetune.bglabel import Params, read_yolo
from scripts.finetune.common import class_names, trial_of
from scripts.finetune.synthesize import Cut, Paste, compose, place_hand, rotate

NAMES = class_names(load_config())
W, H = 640, 360


def square(size, color=(0, 0, 255), label_rows=None):
    bgra = np.zeros((size[1], size[0], 4), np.uint8)
    bgra[:, :, :3] = color
    bgra[:, :, 3] = 255
    lab = np.ones(bgra.shape[:2], bool)
    if label_rows is not None:
        lab[label_rows:] = False
    return bgra, lab


def paste(cls, x, y, size=(20, 20), **kw):
    bgra, lab = square(size, **kw)
    return Paste(cls, bgra, lab, x, y)


def test_partial_occlusion_shrinks_the_covered_box():
    img, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, 10, 10), paste(1, 20, 10, color=(0, 255, 0))])
    assert boxes == [(0, (10, 10, 20, 30)), (1, (20, 10, 40, 30))]
    assert tuple(img[15, 15]) == (0, 0, 255) and tuple(img[15, 25]) == (0, 255, 0)


def test_mostly_covered_object_is_dropped():
    _, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, 10, 10), paste(1, 14, 10)])
    assert boxes == [(1, (14, 10, 34, 30))]                     # 4/20 of the first showing < 30%


def test_visibility_threshold_is_a_parameter():
    _, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, 10, 10), paste(1, 14, 10)], min_visible=0.1)
    assert boxes[0] == (0, (10, 10, 14, 30))


def test_arm_occludes_but_only_the_hand_is_labelled():
    arm = paste(8, 5, 0, size=(10, 40), label_rows=10)
    _, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, 0, 0), arm])
    assert boxes == [(0, (0, 0, 20, 20)), (8, (5, 0, 15, 10))]  # half the object shows, on both sides


def test_arm_hand_fully_covering_an_object_drops_it():
    _, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, 50, 50, size=(8, 8)), paste(8, 45, 45, size=(20, 60))])
    assert [c for c, _ in boxes] == [8]


def test_pastes_off_the_frame_are_clipped_or_dropped():
    _, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, -5, 50), paste(1, -15, 10), paste(2, 95, 95)])
    assert boxes == [(0, (0, 50, 15, 70))]                      # 75% in; 25% in; 6% in


def test_transparent_pixels_neither_paint_nor_occlude():
    ring = paste(1, 10, 10, size=(30, 30))
    ring.bgra[5:25, 5:25, 3] = 0
    ring.label[5:25, 5:25] = False
    _, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, 17, 17, size=(16, 16)), ring])
    assert boxes[0] == (0, (17, 17, 33, 33))                    # seen through the hole


def test_rotation_keeps_the_label_area():
    bgra, lab = square((40, 20))
    out, rl = rotate(bgra, lab, 90)
    assert abs(out.shape[0] - 40) <= 1 and abs(out.shape[1] - 20) <= 1
    assert abs(int(rl.sum()) - 800) < 60
    out, rl = rotate(bgra, lab, 45, scale=1.1)
    assert abs(int(rl.sum()) - 800 * 1.21) < 80


def test_hands_stay_on_an_edge_and_can_reach_an_object():
    bgra, lab = square((60, 160))
    cut = Cut("hand", bgra, (0, 0, 60, 70), (300, 200), "bottom")
    for seed in range(40):
        p = place_hand(np.random.default_rng(seed), cut, 8, (W, H))
        h, w = p.bgra.shape[:2]
        assert p.y + h >= H or p.y <= 0                         # attached to the bottom (or mirrored to the top)
        assert -w // 3 <= p.x <= W - 2 * w // 3
    p = place_hand(np.random.default_rng(1), cut, 8, (W, H), toward=(100, 200))
    ys, xs = np.nonzero(p.label)
    assert abs(p.x + xs.mean() - 100) < 2


# ---------------------------------------------------------------- capture -> synthesize -> split

class Table:
    """A fake overhead camera: whatever is on `items` (rectangles), on a grainy table."""

    def __init__(self):
        rng = np.random.default_rng(0)
        base = np.full((H, W, 3), (118, 112, 124), np.float32) + np.linspace(-15, 15, W, dtype=np.float32)[None, :, None]
        self.grain = [np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8) for _ in range(8)]
        self.items, self.arm_x, self.idx = [], None, 0

    def wait_new(self, after_idx, timeout=1.0):
        self.idx += 1
        img = self.grain[self.idx % len(self.grain)].copy()
        for (x1, y1, x2, y2), color in self.items:
            img[y1:y2, x1:x2] = color
        if self.arm_x is not None:                              # a hand sweeping in from the bottom
            x = 80 + (self.idx * 7) % 420
            img[140:H, x:x + 45] = (120, 150, 200)
        return Frame(t=self.idx / 30, wall=0.0, img=img, idx=self.idx)


POSES = {"keys": [((100, 100, 150, 130), (40, 40, 40))], "wallet": [((300, 150, 380, 200), (30, 60, 20))]}


def scripted(cam):
    seen = {}

    def ask(prompt):
        seen[prompt] = seen.get(prompt, 0) + 1
        cam.arm_x = None
        if prompt.startswith("Clear"):
            cam.items = []
        elif prompt.startswith("Hands"):
            cam.items, cam.arm_x = [], 0
        elif prompt.startswith("["):
            name, k = prompt[1:].split()[0], int(prompt.split()[1].split("/")[0]) - 1
            (x1, y1, x2, y2), color = POSES[name][0]
            dx = 60 * k
            cam.items = [((x1 + dx, y1 + 20 * k, x2 + dx, y2 + 20 * k), color)]
            if name == "wallet" and k == 1 and seen[prompt] == 1:
                cam.items.append(((500, 250, 540, H), (120, 150, 200)))    # arm left in view: retake
        return ""
    return ask


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    data = tmp_path_factory.mktemp("ft")
    cam, said = Table(), []
    cap = capture.Capture(cam, data, NAMES, poses=4, params=Params(hand_len_px=120), ask=scripted(cam), say=said.append)
    got = cap.run(only=["keys", "wallet", "hand"], hand_seconds=3)
    return data, got, said


def test_capture_labels_every_pose_and_retakes_a_bad_one(captured):
    data, got, said = captured
    assert got["keys"] == 4 and got["wallet"] == 4 and got["hand"] >= 5
    assert any("retake" in s for s in said)
    lab = read_yolo(data / "labels" / "cap1_keys-01.txt", W, H)
    assert lab[0][0] == NAMES.index("keys")
    assert all(abs(a - b) <= 3 for a, b in zip(lab[0][1], (160, 120, 210, 150)))
    hand = read_yolo(data / "labels" / "cap0_hand-00.txt", W, H)
    assert hand[0][0] == NAMES.index("hand") and abs(hand[0][1][1] - 140) <= 2 and hand[0][1][3] - hand[0][1][1] <= 122
    assert (data / "qa_capture.jpg").exists() and (data / "data.yaml").exists()
    assert len(list((data / "backgrounds").glob("*.jpg"))) == 6
    assert (data / "labels" / "cap0_empty-00.txt").read_text() == ""


def test_synthesis_is_seeded_and_leaves_the_holdout_unseen(captured, tmp_path):
    data = captured[0]
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        shutil.copytree(data, d)
    synthesize.synthesize(a, NAMES, n=12, seed=3)
    synthesize.synthesize(b, NAMES, n=12, seed=3)
    for i in range(12):
        s = f"synth_{i:05d}"
        assert (a / "labels" / f"{s}.txt").read_text() == (b / "labels" / f"{s}.txt").read_text()
        assert (a / "images" / f"{s}.jpg").read_bytes() == (b / "images" / f"{s}.jpg").read_bytes()
    labels = [read_yolo(a / "labels" / f"synth_{i:05d}.txt", W, H) for i in range(12)]
    assert sum(len(x) for x in labels) >= 12
    assert {c for x in labels for c, _ in x} <= {NAMES.index(n) for n in ("keys", "wallet", "hand")}
    cuts = [c for cs in synthesize.load_cuts(a).values() for c in cs]
    assert len(cuts) >= 6 and not any(trial_of(c.stem) == capture.HOLDOUT for c in cuts)
    assert len(synthesize.load_cuts(a, holdout=None)["keys"]) == 4
    synthesize.synthesize(b, NAMES, n=12, seed=4)
    assert any((a / "labels" / f"synth_{i:05d}.txt").read_text() != (b / "labels" / f"synth_{i:05d}.txt").read_text()
               for i in range(12))


def test_synthetic_images_never_land_in_validation(captured, tmp_path):
    data = tmp_path / "d"
    shutil.copytree(captured[0], data)
    synthesize.synthesize(data, NAMES, n=10, seed=0)
    stems = sorted(p.stem for p in (data / "labels").glob("*.txt"))
    tr, va = train.split_by_trial(stems, val_trials=[capture.HOLDOUT])
    assert va and all(trial_of(s) == capture.HOLDOUT for s in va)
    assert sum(s.startswith("synth_") for s in tr) == 10
    for seed in range(5):
        tr, va = train.split_by_trial(stems, seed=seed)
        assert not any(s.startswith("synth_") for s in va) and len(tr) + len(va) == len(stems)
