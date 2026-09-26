"""eval/replay_video.py with a scripted fake detector: false disappearances are counted only for
objects no hand touched."""
from core.config import load_config
from core.types import Detection, Detections, Frame
from eval.replay_video import replay

CFG = load_config()
BOXES = {"wallet": (100, 100, 200, 160), "keys": (600, 300, 660, 340), "phone": (900, 500, 980, 560)}


def det(cls, box):
    x1, y1, x2, y2 = box
    return Detection(cls=cls, conf=0.9, box_px=box, center_cm=((x1 + x2) / 20, (y1 + y2) / 20),
                     box_cm=(x1 / 10, y1 / 10, x2 / 10, y2 / 10))


class Scripted:
    """wallet blinks out for 2 s with no hand near it; a hand takes the keys; the phone stays put."""
    last_ms = 5.0

    def detect(self, f: Frame) -> Detections:
        i = f.idx
        items = [det("phone", BOXES["phone"])]
        if not 40 <= i < 60:
            items.append(det("wallet", BOXES["wallet"]))
        if i < 50:
            items.append(det("keys", BOXES["keys"]))
        hands = [det("hand", (590, 280, 700, 420))] if 30 <= i < 56 else []
        return Detections(t=f.t, frame_idx=i, items=items, hands=hands)


def frames(n=100, fps=10):
    return [Frame(t=i / fps, wall=1000.0 + i / fps, img=None, idx=i) for i in range(n)]


def test_untouched_object_that_vanishes_is_a_false_disappearance():
    r = replay(frames(), Scripted(), CFG)
    assert r["frames"] == 100 and r["video_s"] == 9.9
    assert r["touched"] == ["keys"] and r["untouched"] == ["phone", "wallet"]
    assert r["false_disappearances"] and {e["obj"] for e in r["false_disappearances"]} == {"wallet"}
    assert r["detected"]["phone"] == 1.0 and r["detected"]["wallet"] == 0.8 and r["detected"]["keys"] == 0.5
    assert r["detected"]["glasses"] == 0.0
    assert r["events"]["keys"].get("PICKED_UP") == 1
    assert r["detect_ms_median"] == 5.0
    assert r["pass"] is False


def test_named_untouched_objects_override_the_hand_rule():
    r = replay(frames(), Scripted(), CFG, untouched=["phone"])
    assert r["untouched"] == ["phone"] and r["false_disappearances"] == [] and r["pass"] is True


def test_fps_cap_drops_frames_by_video_time():
    r = replay(frames(90, fps=30), Scripted(), CFG, max_fps=10)
    assert r["frames"] == 30


def test_slow_processing_fails_the_fps_bar():
    r = replay(frames(20), lambda f: Detections(t=f.t, frame_idx=f.idx, items=[det("phone", BOXES["phone"])]),
               CFG, min_fps=1e9)
    assert r["false_disappearances"] == [] and r["pass"] is False
