"""scripts/finetune/bglabel.py on rendered tabletops with known boxes."""
import cv2
import numpy as np
import pytest

from scripts.finetune.bglabel import (Background, Stability, contact_sheet, cutout, hand_part, label_hands,
                                      label_object, read_yolo, save_sample, yolo_line)

W, H = 1280, 720
TABLE = (118, 112, 124)
SKIN = (120, 150, 200)


def table(seed: int, sigma: float = 5.0) -> np.ndarray:
    """A dim, grainy table with a lamp gradient, like the rig's camera."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), TABLE, np.float32) + np.linspace(-20, 20, W, dtype=np.float32)[None, :, None]
    img += rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def bg():
    return Background([table(i) for i in range(12)])


def put(img, box, color):
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = color
    return img


def near(a, b, tol=4):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_threshold_follows_the_camera_noise():
    quiet = Background([table(i, sigma=2) for i in range(6)])
    grainy = Background([table(i, sigma=25) for i in range(6)])
    assert quiet.thresh == quiet.p.min_thresh
    assert grainy.thresh > quiet.thresh
    assert not label_object(grainy, table(50, sigma=25)).ok      # grain alone is not an object


def test_one_object_gets_its_box_and_mask(bg):
    lab = label_object(bg, put(table(99), (500, 300, 620, 380), (40, 45, 50)))
    assert lab.ok, lab.reason
    assert near(lab.box, (500, 300, 620, 380))
    assert abs(int(lab.mask.sum()) - 120 * 80) < 0.1 * 120 * 80


def test_low_contrast_object_is_found(bg):
    lab = label_object(bg, put(table(98), (200, 200, 260, 250), (150, 140, 150)))
    assert lab.ok and near(lab.box, (200, 200, 260, 250))


def test_ring_shaped_object_is_filled(bg):
    img = table(97)
    cv2.circle(img, (700, 400), 40, (30, 30, 30), 6)          # glasses rim / key ring
    lab = label_object(bg, img)
    assert lab.ok and near(lab.box, (657, 357, 744, 444), 5)
    assert lab.mask[400, 700]                                   # the hole is part of the cutout


def test_nearby_pieces_are_one_object(bg):
    img = put(put(table(96), (400, 400, 440, 430), (40, 40, 40)), (452, 405, 500, 425), (60, 60, 200))
    lab = label_object(bg, img)
    assert lab.ok and near(lab.box, (400, 400, 500, 430))


def test_empty_table_is_rejected(bg):
    lab = label_object(bg, table(95))
    assert not lab.ok and "nothing" in lab.reason


def test_hand_left_in_frame_is_rejected(bg):
    img = put(table(94), (500, 300, 620, 380), (40, 45, 50))
    put(img, (800, 450, 900, H), SKIN)                          # arm reaching in from the bottom
    lab = label_object(bg, img)
    assert not lab.ok


def test_arm_alone_touching_the_edge_is_rejected(bg):
    lab = label_object(bg, put(table(93), (800, 450, 900, H), SKIN))
    assert not lab.ok and "edge" in lab.reason


def test_two_separate_objects_are_rejected(bg):
    img = put(put(table(92), (200, 200, 300, 280), (40, 40, 40)), (900, 400, 990, 470), (40, 40, 200))
    lab = label_object(bg, img)
    assert not lab.ok and "separate" in lab.reason


def test_small_crumb_next_to_an_object_is_ignored(bg):
    img = put(put(table(91), (200, 200, 300, 280), (40, 40, 40)), (900, 400, 912, 410), (40, 40, 200))
    lab = label_object(bg, img)
    assert lab.ok and near(lab.box, (200, 200, 300, 280))


def test_lighting_change_is_rejected_as_too_large(bg):
    img = np.clip(table(90).astype(int) + 60, 0, 255).astype(np.uint8)
    lab = label_object(bg, img)
    assert not lab.ok and "large" in lab.reason


def test_tiny_change_is_rejected(bg):
    lab = label_object(bg, put(table(89), (600, 300, 622, 322), (20, 20, 20)))
    assert not lab.ok and "small" in lab.reason


def test_hand_box_is_the_end_of_an_arm_from_the_bottom(bg):
    [h] = label_hands(bg, put(table(88), (600, 250, 700, H), SKIN))
    assert h.edge == "bottom"
    assert near(h.box, (600, 250, 700, 250 + h_len(bg)))
    assert h.mask[H - 1, 650]                                   # the arm stays in the mask for pasting


def h_len(bg):
    return bg.p.hand_len_px


def test_hand_from_the_left_and_two_hands(bg):
    img = put(put(table(87), (0, 100, 500, 200), SKIN), (1000, 400, W, 480), SKIN)
    hands = sorted(label_hands(bg, img), key=lambda l: l.box[0])
    assert [h.edge for h in hands] == ["left", "right"]
    assert near(hands[0].box, (500 - h_len(bg), 100, 500, 200))
    assert near(hands[1].box, (1000, 400, 1000 + h_len(bg), 480))


def test_no_hand_on_an_empty_table(bg):
    assert label_hands(bg, table(86)) == []


def test_hand_part_without_an_edge_is_the_whole_blob():
    m = np.zeros((50, 50), bool)
    m[10:20, 10:20] = True
    assert (hand_part(m, None, 5) == m).all()
    top = hand_part(np.ones((50, 50), bool), "top", 5)
    assert top[45:].all() and not top[:45].any()


def test_stability_waits_for_the_scene_to_settle():
    s = Stability(hold_s=0.7)
    t, fps = 0.0, 30
    for i in range(20):                                         # a hand moving across
        assert not s.update(put(table(i), (40 * i, 300, 40 * i + 150, 450), SKIN), t)
        t += 1 / fps
    still = put(table(200), (500, 300, 620, 380), (40, 45, 50))
    results = []
    for i in range(30):                                         # object alone, only grain changes
        img = np.clip(still.astype(int) + np.random.default_rng(i).normal(0, 4, still.shape), 0, 255)
        results.append(s.update(img.astype(np.uint8), t))
        t += 1 / fps
    assert not any(results[:20]) and all(results[22:])


def test_yolo_line_round_trips(tmp_path):
    assert yolo_line(3, (100, 200, 300, 400), 1000, 800) == "3 0.200000 0.375000 0.200000 0.250000"
    img = np.zeros((720, 1280, 3), np.uint8)
    save_sample(tmp_path, "cap0_keys-00", img, [(0, (100, 200, 300, 400)), (8, (0, 500, 250, 720))])
    save_sample(tmp_path, "cap0_empty-00", img, [])
    assert (tmp_path / "images" / "cap0_keys-00.jpg").exists()
    assert read_yolo(tmp_path / "labels" / "cap0_keys-00.txt", 1280, 720) == [(0, (100, 200, 300, 400)),
                                                                               (8, (0, 500, 250, 720))]
    assert (tmp_path / "labels" / "cap0_empty-00.txt").read_text() == ""


def test_cutout_alpha_is_the_mask():
    img = table(5)
    m = np.zeros((H, W), bool)
    m[100:150, 200:260] = True
    m[100, 200] = False
    bgra, (x, y) = cutout(img, m)
    assert (x, y) == (200, 100) and bgra.shape == (50, 60, 4)
    assert bgra[0, 0, 3] == 0 and bgra[1, 1, 3] == 255
    assert (bgra[1, 1, :3] == img[101, 201]).all()


def test_contact_sheet_tiles_thumbnails():
    img = table(1)
    sheet = contact_sheet([(img, [("keys", (100, 100, 300, 300))], "a")] * 7, cols=3, thumb_w=320)
    assert sheet.shape == (3 * 180, 3 * 320, 3)
