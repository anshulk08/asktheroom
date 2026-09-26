"""core/room_view.py (spec 0009 M0): the table view cut from the full 1080p frame, and its ECC measurement."""
import cv2
import numpy as np
import pytest

from core.room_view import TableView, cut, default_rect, measure_rect
from core.types import Frame

RECT = (360, 202, 1560, 877)


def full_img(val=0):
    img = np.full((1080, 1920, 3), val, np.uint8)
    img[RECT[1]:RECT[3], RECT[0]:RECT[2]] = 200          # the table region is bright
    return img


def test_default_rect_for_1080p_is_1200x675_centred():
    x1, y1, x2, y2 = default_rect((1920, 1080))
    assert (x2 - x1, y2 - y1) == (1200, 675)
    assert abs((x1 + x2) / 2 - 960) <= 0.5
    assert abs((y1 + y2) / 2 - 540) <= 0.5
    assert all(isinstance(v, int) for v in (x1, y1, x2, y2))


def test_default_rect_same_zoom_is_the_whole_width():
    x1, y1, x2, y2 = default_rect((1280, 720), zoom=160, ref_zoom=160)
    assert (x1, y1, x2, y2) == (0, 0, 1280, 720)


def test_cut_shape_and_content():
    out = cut(full_img(), RECT)
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
    assert int(out.min()) == 200                           # only the table region
    small = cut(full_img(), (0, 0, 1920, 1080), out_size=(640, 360))   # downscale
    assert small.shape == (360, 640, 3)


class FakeSource:
    """The FrameBuffer read API over a fixed list of full frames."""

    def __init__(self, frames):
        self.frames = frames
        self.fps = 29.5
        self.failures = 3
        self.stopped = False

    def latest(self):
        return self.frames[-1] if self.frames else None

    def at(self, t):
        return min(self.frames, key=lambda f: abs(f.t - t)) if self.frames else None

    def wait_new(self, after_idx, timeout=1.0):
        f = self.latest()
        return f if f is not None and f.idx > after_idx else None

    def stop(self):
        self.stopped = True


def make_view():
    frames = [Frame(t=10.0 + i, wall=1000.0 + i, img=full_img(i), idx=i + 1) for i in range(3)]
    return TableView(FakeSource(frames), RECT), frames


def test_latest_at_wait_new_give_table_view_frames():
    tv, frames = make_view()
    f = tv.latest()
    assert f.img.shape == (720, 1280, 3)
    assert (f.t, f.wall, f.idx) == (frames[-1].t, frames[-1].wall, frames[-1].idx)
    g = tv.at(11.1)
    assert g.idx == 2 and g.t == 11.0 and g.img.shape == (720, 1280, 3)
    h = tv.wait_new(0, timeout=0.1)
    assert h.idx == 3 and h.img.shape == (720, 1280, 3)
    assert tv.wait_new(3, timeout=0.01) is None


def test_cut_is_cached_per_idx():
    tv, _ = make_view()
    assert tv.latest().img is tv.latest().img
    assert tv.at(12.0).img is tv.latest().img


def test_full_frames_come_from_the_source():
    tv, frames = make_view()
    assert tv.latest_full() is frames[-1]
    assert tv.full_at(10.2) is frames[0]
    assert tv.latest_full().img.shape == (1080, 1920, 3)


def test_none_img_stays_none_and_empty_source_gives_none():
    tv = TableView(FakeSource([Frame(t=1.0, wall=2.0, img=None, idx=7)]), RECT)
    f = tv.latest()
    assert f.img is None and f.idx == 7
    empty = TableView(FakeSource([]), RECT)
    assert empty.latest() is None and empty.at(1.0) is None and empty.latest_full() is None


def test_attributes_delegate_to_the_source():
    tv, _ = make_view()
    assert tv.fps == 29.5
    assert tv.failures == 3
    tv.stop()
    assert tv.source.stopped
    with pytest.raises(AttributeError):
        tv.no_such_attribute


def textured(w=1920, h=1080):
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(900):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(8, 60))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.circle(img, c, r, col, -1)
    return cv2.GaussianBlur(img, (0, 0), 3)


def test_measure_rect_recovers_a_known_rect_from_20px_off():
    full = textured()
    ref = cut(full, RECT)                                   # what the camera would give at zoom 160
    init = (RECT[0] + 20, RECT[1] - 20, RECT[2] + 20, RECT[3] - 20)
    rect, cc = measure_rect(full, ref, init)
    assert all(isinstance(v, int) for v in rect)
    assert max(abs(a - b) for a, b in zip(rect, RECT)) <= 1, rect
    assert cc > 0.9
