"""scripts/finetune/synthesize.py box bookkeeping, and capture -> synthesize -> split end to end with a fake camera."""
import json
import re
import shutil

import cv2
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


# ---------------------------------------------------------------- distractors: unknown objects as negatives

PROMPT_RE = re.compile(r"\[(\w+) (\d+)/(\d+)\] Place the (.+?) alone on the table, (.+?); hands away")
DISTRACTOR_POSES = {"mug": ((200, 220, 250, 270), (200, 60, 60)), "cup": ((420, 60, 470, 100), (60, 200, 200))}


def scripted_with_distractors(cam, prompts):
    base = scripted(cam)

    def ask(prompt):
        prompts.append(prompt)
        m = PROMPT_RE.match(prompt)
        if m and m.group(1) in DISTRACTOR_POSES:
            k = int(m.group(2)) - 1
            (x1, y1, x2, y2), color = DISTRACTOR_POSES[m.group(1)]
            cam.arm_x, cam.items = None, [((x1 + 40 * k, y1 + 15 * k, x2 + 40 * k, y2 + 15 * k), color)]
            return ""
        return base(prompt)
    return ask


@pytest.fixture(scope="module")
def captured_neg(tmp_path_factory):
    data = tmp_path_factory.mktemp("ftneg")
    cam, said, prompts = Table(), [], []
    cap = capture.Capture(cam, data, NAMES, poses=4, params=Params(hand_len_px=120),
                          ask=scripted_with_distractors(cam, prompts), say=said.append,
                          distractors=["mug"], distractor_poses=4)
    got = cap.run(only=["keys", "hand", "mug"], hand_seconds=3)
    return data, got, prompts


def test_distractors_are_captured_as_negatives_with_marked_cutouts(captured_neg):
    data, got, prompts = captured_neg
    assert got["keys"] == 4 and got["mug"] == 4 and got["hand"] >= 5
    for k in range(4):
        st = capture.stem("mug", k)
        assert (data / "images" / f"{st}.jpg").exists()
        assert (data / "labels" / f"{st}.txt").read_text() == ""          # a negative for every class
        meta = json.loads((data / "cutouts" / f"{st}.json").read_text())
        assert meta["name"] == "mug" and meta["distractor"] is True
    assert "distractor" not in json.loads((data / "cutouts" / "cap0_keys-00.json").read_text())
    mug = [PROMPT_RE.match(p) for p in prompts if p.startswith("[mug")]
    assert [m.group(2, 3, 4) for m in mug] == [(str(k), "4", "mug") for k in range(1, 5)]


def test_held_out_distractor_images_validate_as_negatives(captured_neg):
    data = captured_neg[0]
    stems = sorted(p.stem for p in (data / "labels").glob("*.txt"))
    tr, va = train.split_by_trial(stems, val_trials=[capture.HOLDOUT])
    assert capture.stem("mug", 3) in va and capture.stem("mug", 0) in tr
    assert train.count_negatives(data, va) == 2                            # the empty table and the mug


def test_distractor_names_must_be_new_words(tmp_path):
    for bad in (["keys"], ["hand"], ["empty"], ["air pods"], ["mug", "mug"], ["mug-2"]):
        with pytest.raises(ValueError):
            capture.Capture(Table(), tmp_path, NAMES, distractors=bad)
    assert capture.main(["--data", str(tmp_path), "--distractors", "wallet"]) == 2
    assert capture.main(["--data", str(tmp_path), "--only", "mug"]) == 2    # not a distractor this run


def test_only_can_pick_one_distractor(tmp_path):
    cam, prompts = Table(), []
    cap = capture.Capture(cam, tmp_path, NAMES, ask=scripted_with_distractors(cam, prompts), say=lambda s: None,
                          distractors=["mug", "cup"], distractor_poses=2)
    assert cap.run(only=["mug"]) == {"mug": 2}
    assert not any(p.startswith("[cup") for p in prompts)


def test_clearing_a_name_leaves_longer_names_alone(tmp_path):
    for sub, ext in (("images", "jpg"), ("labels", "txt")):
        (tmp_path / sub).mkdir()
        for st in ("cap0_pill-bottle-00", "cap0_pill-00", "cap1_pill-01"):
            (tmp_path / sub / f"{st}.{ext}").write_text("")
    (tmp_path / "cutouts").mkdir()
    for st in ("cap0_hand-00-0", "cap0_hand-00-1", "cap1_hand-101", "cap0_hand-x-00"):
        (tmp_path / "cutouts" / f"{st}.json").write_text("{}")
    capture._clear(tmp_path, "pill")
    capture._clear(tmp_path, "hand")
    assert sorted(p.stem for p in (tmp_path / "images").glob("*")) == ["cap0_pill-bottle-00"]
    assert sorted(p.stem for p in (tmp_path / "cutouts").glob("*")) == ["cap0_hand-x-00"]


def cut(name, color, size=(40, 30), distractor=False):
    bgra, _ = square(size, color=color)
    return synthesize.Cut(name, bgra, (0, 0, size[0], size[1]), (100, 100), None, f"cap0_{name}-00", distractor)


def test_distractor_paste_occludes_but_gets_no_box():
    p = synthesize.place_object(np.random.default_rng(0), cut("mug", (255, 0, 0), distractor=True),
                                synthesize.NO_BOX, (0, 0, W, H))
    assert p.cls == synthesize.NO_BOX and not p.label.any() and (p.bgra[:, :, 3] > 0).any()
    mug = Paste(synthesize.NO_BOX, *square((20, 20), color=(255, 0, 0)), 20, 10)
    mug.label[:] = False
    _, boxes = compose(np.zeros((100, 100, 3), np.uint8), [paste(0, 10, 10), mug])
    assert boxes == [(0, (10, 10, 20, 30))]                              # half the object shows; no mug box


def blue(img):
    return int(((img[:, :, 0] > 170) & (img[:, :, 1] < 60) & (img[:, :, 2] < 60)).sum())


def test_p_distractor_mixes_distractors_in_and_zero_leaves_them_out():
    bg = [np.full((H, W, 3), 120, np.uint8)]
    cuts = {"keys": [cut("keys", (0, 0, 255))], "mug": [cut("mug", (255, 0, 0), distractor=True)]}
    region = (0, 0, W, H)
    run = lambda p: [synthesize.make_image(np.random.default_rng([7, i]), bg, cuts, NAMES, region, p_hand=0,
                                           p_distractor=p) for i in range(60)]
    assert not any(blue(img) for img, _ in run(0.0))
    mixed = run(0.6)
    with_mug = [boxes for img, boxes in mixed if blue(img) > 200]
    assert 15 <= len(with_mug) <= 50
    assert any(b == [] for b in with_mug)                                # distractor-only scenes
    assert any(b for b in with_mug)                                      # next to a labelled object
    assert all(c == NAMES.index("keys") for _, boxes in mixed for c, _ in boxes)
    img, boxes = synthesize.make_image(np.random.default_rng(1), bg, {"mug": cuts["mug"]}, NAMES, region, p_hand=0)
    assert boxes == [] and blue(img) > 200                               # distractors alone still paste


def test_distractors_synthesize_seeded_without_boxes(captured_neg, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        shutil.copytree(captured_neg[0], d)
    for d in (a, b):
        synthesize.synthesize(d, NAMES, n=12, seed=5, p_distractor=0.7)
    for i in range(12):
        s = f"synth_{i:05d}"
        assert (a / "images" / f"{s}.jpg").read_bytes() == (b / "images" / f"{s}.jpg").read_bytes()
    labels = [read_yolo(a / "labels" / f"synth_{i:05d}.txt", W, H) for i in range(12)]
    assert {c for x in labels for c, _ in x} <= {NAMES.index("keys"), NAMES.index("hand")}
    cuts = synthesize.load_cuts(a)
    assert len(cuts["mug"]) == 3 and all(c.distractor for c in cuts["mug"])
    assert not any(c.distractor for c in cuts["keys"])


def test_without_distractors_the_mix_knob_changes_nothing(captured, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        shutil.copytree(captured[0], d)
    synthesize.synthesize(a, NAMES, n=6, seed=2, p_distractor=0.0)
    synthesize.synthesize(b, NAMES, n=6, seed=2, p_distractor=0.9)
    for i in range(6):
        s = f"synth_{i:05d}.jpg"
        assert (a / "images" / s).read_bytes() == (b / "images" / s).read_bytes()


def write_cut(data, name, distractor=None):
    (data / "cutouts").mkdir(parents=True, exist_ok=True)
    (data / "backgrounds").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(data / "backgrounds" / "empty-00.jpg"), np.full((H, W, 3), 120, np.uint8))
    st = f"cap0_{name}-00"
    cv2.imwrite(str(data / "cutouts" / f"{st}.png"), square((20, 20))[0])
    meta = {"name": name, "stem": st, "origin": [100, 100], "frame": [W, H], "edge": None, "box": [0, 0, 20, 20]}
    if distractor is not None:
        meta["distractor"] = distractor
    (data / "cutouts" / f"{st}.json").write_text(json.dumps(meta))


def test_unknown_cutouts_still_fail_unless_marked_distractor(tmp_path):
    write_cut(tmp_path / "ok", "mug", distractor=True)
    assert synthesize.synthesize(tmp_path / "ok", NAMES, n=2) == 2
    write_cut(tmp_path / "unknown", "mug")
    with pytest.raises(ValueError):
        synthesize.synthesize(tmp_path / "unknown", NAMES, n=2)
    write_cut(tmp_path / "clash", "phone", distractor=True)             # a distractor can't be a class
    with pytest.raises(ValueError):
        synthesize.synthesize(tmp_path / "clash", NAMES, n=2)


def test_capture_opens_the_camera_as_the_detector_sees_it():
    """With room memory on (spec 0009) the capture source is the 1080p frame cut to table_view_rect and resized
    to 1280x720 (what core.detect gets on the rig); off, the plain FrameBuffer. --full-frame skips the cut."""
    from core.room_view import TableView
    from scripts.finetune import capture as cap
    made = []

    class Buf:
        def __init__(self, src, **kw):
            made.append((src, kw))
            self.src, self.kw = src, kw

    cfg = {"frame_size_px": [1280, 720],
           "room_memory": {"enabled": True, "capture_size": [1920, 1080], "table_view_rect": [0, 735, 613, 1080],
                           "ring_s": 1.0}}
    src = cap.open_source(cfg, "/dev/video9", make_buffer=Buf)
    assert isinstance(src, TableView) and src.rect == (0, 735, 613, 1080) and src.out_size == (1280, 720)
    assert made[0][0] == "/dev/video9" and made[0][1]["ring_s"] == 1.0 and callable(made[0][1]["opener"])
    plain = cap.open_source(cfg, 0, make_buffer=Buf, full_frame=True)
    assert isinstance(plain, Buf) and plain.kw.get("opener") is not None      # still 1080p, uncut
    off = cap.open_source({"room_memory": {"enabled": False}}, 0, make_buffer=Buf)
    assert isinstance(off, Buf) and off.kw == {}


def test_max_rot_bounds_a_cutout_rotation_for_the_corner_camera():
    """Overhead, any rotation of a cutout is a real pose; from the corner camera an object never lies upside
    down, so --max-rot keeps pastes within +-max_rot degrees (0: the cutout's own orientation, only scaled/mirrored)."""
    for seed in range(5):
        p = synthesize.place_object(np.random.default_rng(seed), cut("wallet", (0, 255, 0), size=(60, 20)), 2,
                                    (0, 0, W, H), max_rot=0)
        ys, xs = np.nonzero(p.label)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        assert 2.6 <= w / h <= 3.4, (seed, w, h)                      # 60x20 stays 3:1, never rotated
    p = synthesize.place_object(np.random.default_rng(1), cut("wallet", (0, 255, 0), size=(60, 20)), 2,
                                (0, 0, W, H))
    ys, xs = np.nonzero(p.label)                                       # default: the old 0-360 draw
    assert (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) < 2.6


def test_train_parser_takes_flipud_for_the_corner_camera():
    a = train.build_parser().parse_args(["--flipud", "0"])
    assert a.flipud == 0.0 and train.build_parser().parse_args([]).flipud == 0.5
