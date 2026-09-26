"""Synthetic detection sequences for world-model tests (task W4).

A scripted tabletop: 10 px per cm with the table origin at pixel (0, 0), so the default
128 x 72 cm table fills a 1280 x 720 frame. By default frames carry no image (img=None), which
turns off the image rules. Scene(render=True) also draws a BGR frame: a flat table colour, each
object a distinct block texture at its box, hands a skin-toned texture on top. Objects can be drawn
without being detected (miss) and unknown things painted that are never detected (overlay).

Open-world things (Scene.thing) arrive as class-agnostic proposals, cls 'thing', keyed by a scene
key the world never sees. Scene.embed stands in for an appearance embedder: each thing has a look
vector (look(key) by default, or one passed in, e.g. look_like() for a controlled similarity).
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from core.config import Config
from core.types import BoxCm, Detection, Detections, Event, Frame

PX_PER_CM = 10
LOOK_DIM = 32
THING_CONF = 0.5
DEFAULT_SIZE_CM = {'target': (6.0, 4.0), 'container': (20.0, 15.0), 'cover': (25.0, 18.0)}
HAND_SIZE_CM = (12.0, 12.0)
TABLE_BGR = (200, 210, 220)            # light table: objects stand out in grey
SKIN_BGR = (140, 170, 215)
TEXTURE_BLOCK_PX = 8
DRAW_ORDER = {'container': 0, 'target': 1, 'cover': 2}    # covers lie on top of what they cover


def _box(x: float, y: float, w: float, h: float) -> BoxCm:
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def _px(box: BoxCm) -> tuple[int, int, int, int]:
    return tuple(int(round(v * PX_PER_CM)) for v in box)


def _det(cls: str, box: BoxCm, conf: float = 0.9, shift_px: tuple[int, int] = (0, 0)) -> Detection:
    x1, y1, x2, y2 = _px(box)
    dx, dy = shift_px
    return Detection(cls=cls, conf=conf, box_px=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                     center_cm=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box_cm=box)


def texture(key: str, h: int, w: int) -> np.ndarray:
    """Deterministic h x w BGR block texture for key (same key and size -> same pixels)."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    b = TEXTURE_BLOCK_PX
    blocks = rng.integers(30, 225, (-(-h // b), -(-w // b), 3), dtype=np.uint8)
    return np.repeat(np.repeat(blocks, b, axis=0), b, axis=1)[:h, :w]


def skin(key: str, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    b = TEXTURE_BLOCK_PX
    noise = rng.integers(-20, 21, (-(-h // b), -(-w // b), 1))
    tex = np.clip(np.array(SKIN_BGR) + np.repeat(np.repeat(noise, b, axis=0), b, axis=1), 0, 255)
    return tex[:h, :w].astype(np.uint8)


def look(key: str) -> np.ndarray:
    """Deterministic unit 'appearance embedding' for a scene key (unrelated keys: cosine ~0)."""
    rng = np.random.default_rng(zlib.crc32(('look:' + key).encode()))
    v = rng.normal(size=LOOK_DIM)
    return v / np.linalg.norm(v)


def look_like(key: str, like: str, sim: float) -> np.ndarray:
    """Unit vector whose cosine with look(like) is exactly sim (its own part comes from key)."""
    a, b = look(like), look(key)
    b = b - (b @ a) * a
    return sim * a + math.sqrt(max(0.0, 1 - sim * sim)) * b / np.linalg.norm(b)


class Scene:
    def __init__(self, cfg: Config, fps: float = 10.0, t0: float = 1000.0, render: bool = False,
                 jitter_px: int = 0):
        self.cfg = cfg
        self.dt = 1.0 / fps
        self.t = t0
        self.idx = 0
        self.objects: dict[str, BoxCm] = {}
        self.hands: dict[int, BoxCm] = {}
        self.render = render
        self.jitter_px = jitter_px                 # detection boxes wobble by up to this many px
        self.missed: set[str] = set()              # drawn but not detected
        self.overlays: dict[str, BoxCm] = {}       # drawn, never detected
        self.things: dict[str, BoxCm] = {}         # open-world objects: detected as cls 'thing'
        self.looks: dict[str, np.ndarray] = {}     # thing key -> look vector (default look(key))
        self._tex: dict[tuple, np.ndarray] = {}

    def place(self, name: str, x: float, y: float, w: float | None = None, h: float | None = None) -> None:
        dw, dh = DEFAULT_SIZE_CM[self.cfg.kind_of(name)]
        self.objects[name] = _box(x, y, w or dw, h or dh)

    def remove(self, name: str) -> None:
        self.objects.pop(name, None)
        self.things.pop(name, None)

    def thing(self, key: str, x: float, y: float, w: float = 6.0, h: float = 4.0, look=None) -> None:
        """Place (or move) an open-world object; the detector reports it only as a 'thing' proposal."""
        self.things[key] = _box(x, y, w, h)
        if look is not None:
            self.looks[key] = look

    def embed(self, img, box_px) -> np.ndarray | None:
        """Fake appearance embedder: the look of the thing whose pixel box is centred nearest box_px."""
        cx, cy = (box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2
        best, bd = None, 5.0 + self.jitter_px * 1.5
        for key, box in self.things.items():
            x1, y1, x2, y2 = _px(box)
            d = math.hypot((x1 + x2) / 2 - cx, (y1 + y2) / 2 - cy)
            if d <= bd:
                best, bd = key, d
        if best is None:
            return None
        return self.looks.get(best, look(best))

    def hand(self, hid: int, x: float, y: float) -> None:
        self.hands[hid] = _box(x, y, *HAND_SIZE_CM)

    def hand_off(self, hid: int) -> None:
        self.hands.pop(hid, None)

    def miss(self, name: str, missed: bool = True) -> None:
        """Detector misses a placed object (it is still drawn), or detects it again."""
        (self.missed.add if missed else self.missed.discard)(name)

    def overlay(self, key: str, x: float, y: float, w: float, h: float) -> None:
        """Paint an undetectable thing (its own texture) centred at x, y."""
        self.overlays[key] = _box(x, y, w, h)

    def step(self) -> tuple[Detections, Frame]:
        self.t += self.dt
        self.idx += 1
        shift = self._jitter()
        dets = Detections(
            t=self.t, frame_idx=self.idx,
            items=[_det(n, b, shift_px=shift) for n, b in self.objects.items() if n not in self.missed]
            + [_det('thing', b, THING_CONF, shift) for k, b in self.things.items() if k not in self.missed],
            hands=[_det(f'hand:{i}', b) for i, b in self.hands.items()],
        )
        return dets, Frame(t=self.t, wall=self.t, img=self.draw() if self.render else None, idx=self.idx)

    def _jitter(self) -> tuple[int, int]:
        j = self.jitter_px
        if not j:
            return (0, 0)
        return ((self.idx * 3) % (2 * j + 1) - j, (self.idx * 5) % (2 * j + 1) - j)

    def draw(self) -> np.ndarray:
        w, h = self.cfg.frame_size_px
        img = np.empty((h, w, 3), np.uint8)
        img[:] = TABLE_BGR
        order = {k: DRAW_ORDER[self.cfg.kind_of(k)] for k in self.objects}
        order.update({k: DRAW_ORDER['target'] for k in self.things})    # things lie under covers
        objs = sorted({**self.objects, **self.things}.items(), key=lambda kv: order[kv[0]])
        for key, box in objs + list(self.overlays.items()):
            self._paint(img, box, key, texture)
        for i, box in self.hands.items():
            self._paint(img, box, f'hand:{i}', skin)
        return img

    def _paint(self, img: np.ndarray, box: BoxCm, key: str, make) -> None:
        x1, y1, x2, y2 = _px(box)
        h, w = img.shape[:2]
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return
        size = (y2 - y1, x2 - x1)
        tex = self._tex.get((key, size))
        if tex is None:
            tex = self._tex[(key, size)] = make(key, *size)
        img[cy1:cy2, cx1:cx2] = tex[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]

    def run(self, world, seconds: float) -> list[Event]:
        events: list[Event] = []
        for _ in range(math.ceil(seconds / self.dt - 1e-9)):
            events += world.update(*self.step())
        return events
