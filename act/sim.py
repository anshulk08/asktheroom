"""Simulated pan-tilt laser rig + overhead camera (for tests and `--sim` runs). Owner: A.

Ground truth: a pan-tilt head ~120 cm above the table, off-centre, on a bracket pitched toward the
table, with a small roll misalignment and slightly different µs/deg per axis; the dot is the ray's
intersection with the table plane, so pulses -> cm is genuinely nonlinear. Servos have backlash
(dead band) and pulse jitter. The camera renders 1280x720 BGR frames at `fps`; a frame stamped t
shows the world as it was at t - latency_s (pipeline lag), like a real USB/CSI capture thread.
"""
from __future__ import annotations

import bisect
import copy
import math
import time
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np

from act.actuator import Clock, FakeActuator, SimClock
from core.types import Frame

W, H = 1280, 720


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


class HeadGeometry:
    """Pulses -> table cm for a pan-tilt head at `pos` (cm, z up) aimed near the table centre."""

    def __init__(self, pos=(62.0, -18.0, 118.0), table=(90.0, 60.0), us_per_deg=(9.6, 10.3),
                 offset_deg=(1.5, -2.0), roll_deg=2.0):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.k = us_per_deg
        self.off = offset_deg
        aim = np.array([table[0] / 2, table[1] / 2, 0.0]) - self.pos
        aim /= np.linalg.norm(aim)
        pitch, yaw = math.asin(-aim[2]), math.atan2(-aim[0], aim[1])
        self.R = _rz(yaw) @ _rx(-pitch) @ _ry(math.radians(roll_deg))

    def hit(self, pan: float, tilt: float) -> tuple[float, float]:
        th = math.radians((pan - 1500) / self.k[0] + self.off[0])
        al = math.radians((tilt - 1500) / self.k[1] + self.off[1])
        d = self.R @ _rz(th) @ _rx(-al) @ np.array([0.0, 1.0, 0.0])
        if d[2] >= -1e-6:
            return (math.inf, math.inf)          # pointing at or above the horizon
        s = -self.pos[2] / d[2]
        return float(self.pos[0] + s * d[0]), float(self.pos[1] + s * d[1])

    def pulses_for(self, xy, p0=(1500.0, 1500.0)) -> tuple[float, float]:
        """Inverse by damped Newton (ground truth, no backlash)."""
        p = np.asarray(p0, dtype=np.float64)
        xy = np.asarray(xy, dtype=np.float64)
        for _ in range(60):
            f = np.asarray(self.hit(*p)) - xy
            if np.linalg.norm(f) < 1e-9:
                break
            J = np.empty((2, 2))
            for i in range(2):
                dp = np.zeros(2)
                dp[i] = 0.01
                J[:, i] = (np.asarray(self.hit(*(p + dp))) - np.asarray(self.hit(*(p - dp)))) / 0.02
            p = p - np.clip(np.linalg.solve(J, f), -100, 100)
        return float(p[0]), float(p[1])


class _Servo:
    """Dead band / backlash of width band_us plus Gaussian jitter on each write."""

    def __init__(self, x0: float, band_us: float, noise_us: float, rng: np.random.Generator):
        self.x, self.h, self.noise, self.rng = x0, band_us / 2, noise_us, rng

    def command(self, c: float) -> float:
        if c > self.x + self.h:
            self.x = c - self.h
        elif c < self.x - self.h:
            self.x = c + self.h
        return self.x + (self.rng.normal(0.0, self.noise) if self.noise > 0 else 0.0)


class SimActuator(FakeActuator):
    """FakeActuator whose writes drive the simulated servos; keeps a timestamped state history."""

    def __init__(self, rig: "SimRig", cfg: dict):
        super().__init__(cfg, clock=rig.clock, realtime=rig.realtime)
        self.rig = rig
        b = rig.backlash_deg
        self._servos = (_Servo(self.pan, b * rig.geom.k[0], rig.pulse_noise_us, rig.rng),
                        _Servo(self.tilt, b * rig.geom.k[1], rig.pulse_noise_us, rig.rng))
        t0 = self.clock.now() - 10.0
        self.pose_t, self.pose = [t0], [(self.pan, self.tilt)]
        self.laser_t, self.laser_v = [t0], [False]

    def _hw_pulses(self, pan: float, tilt: float, t: float) -> None:
        super()._hw_pulses(pan, tilt, t)
        self.pose_t.append(t)
        self.pose.append((self._servos[0].command(pan), self._servos[1].command(tilt)))

    def _hw_laser(self, on: bool, t: float) -> None:
        super()._hw_laser(on, t)
        self.laser_t.append(t)
        self.laser_v.append(on)

    def state_at(self, t: float) -> tuple[float, float, bool]:
        """Physical (pan, tilt, laser) at time t."""
        self.poll()   # apply a pending laser timeout (recorded at its deadline) before reading
        i = max(0, bisect.bisect_right(self.pose_t, t) - 1)
        j = max(0, bisect.bisect_right(self.laser_t, t) - 1)
        return self.pose[i][0], self.pose[i][1], self.laser_v[j]


class SimTable:
    """Fixed homography between table cm and image px (a slightly perspective overhead view)."""

    def __init__(self, size_cm=(90.0, 60.0)):
        w, h = size_cm
        cm = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        px = np.float32([[168, 92], [1128, 76], [1150, 668], [140, 650]])
        self.size_cm = (float(w), float(h))
        self.H = cv2.getPerspectiveTransform(cm, px)
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _apply(M: np.ndarray, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, M).reshape(-1, 2)

    def px_to_cm(self, pts: np.ndarray) -> np.ndarray:
        return self._apply(self.Hinv, pts)

    def cm_to_px(self, pts: np.ndarray) -> np.ndarray:
        return self._apply(self.H, pts)


class SimCamera:
    """FrameBuffer-like: latest() and at(t). Frames are rendered lazily and cached."""

    def __init__(self, rig: "SimRig", fps: float = 30.0, history_s: float = 2.0):
        self.rig, self.period, self.history_s = rig, 1.0 / fps, history_s
        self.w, self.h = getattr(rig, "size_px", (W, H))
        self._cache: "OrderedDict[int, Frame]" = OrderedDict()
        self._base = self._render_base()
        rng = np.random.default_rng(rig.seed + 1)
        self._noise = [rng.normal(0, rig.noise_sigma, (self.h, self.w, 3)).astype(np.float32)
                       for _ in range(4)]

    def _k_now(self) -> int:
        return int(math.floor(self.rig.clock.now() / self.period + 1e-9))

    def latest(self) -> Optional[Frame]:
        return self._frame(self._k_now())

    def at(self, t: float) -> Optional[Frame]:
        k, now = int(round(t / self.period)), self._k_now()
        if k > now or (now - k) * self.period > self.history_s:
            return None
        return self._frame(k)

    def _frame(self, k: int) -> Frame:
        f = self._cache.get(k)
        if f is None:
            t = k * self.period
            f = Frame(t=t, wall=time.time(), img=self._render(t - self.rig.latency_s, k), idx=k)
            self._cache[k] = f
            while len(self._cache) > 16:
                self._cache.popitem(last=False)
        return f

    # -- rendering
    def _render_base(self) -> np.ndarray:
        rig, tab = self.rig, self.rig.table
        rng = np.random.default_rng(rig.seed)
        img = np.empty((H, W, 3), np.float32)
        img[:] = (58, 60, 64)                                           # floor
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        low = cv2.resize(rng.normal(0, 1, (9, 16)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        grain = np.sin(xx * 0.045 + 2.5 * low + 0.3 * np.sin(yy * 0.01)) * 9 + low * 8
        wood = np.stack([95 + grain, 128 + grain * 1.2, 165 + grain * 1.4], axis=-1)
        w, h = tab.size_cm
        quad = tab.cm_to_px([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.int32)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, quad, 1)
        img[mask > 0] = wood[mask > 0]

        def poly(cm_pts, color):
            cv2.fillPoly(img, [tab.cm_to_px(cm_pts).astype(np.int32)], color)

        def rect(x, y, rw, rh, color):
            poly([[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]], color)

        for mx, my in [(0, 0), (w, 0), (w, h), (0, h)]:                 # ArUco-ish corner markers
            rect(mx - 2.5, my - 2.5, 5, 5, (15, 15, 15))
            rect(mx - 1.2, my - 1.2, 2.4, 2.4, (235, 235, 235))
        rect(60, 30, 20, 15, (70, 110, 150))       # cardboard box
        rect(10, 32, 21, 15, (225, 228, 230))      # notebook
        rect(35, 8, 7, 14, (40, 30, 25))           # phone
        rect(48, 44, 16, 4, (35, 35, 38))          # remote
        rect(20, 10, 9, 6, (40, 40, 150))          # dim red wallet (must not fool the HSV fallback)
        c = tab.cm_to_px([[74, 12]])[0]
        cv2.ellipse(img, (int(c[0]), int(c[1])), (40, 22), 20, 0, 360, (60, 140, 60), -1)  # green cup
        c = tab.cm_to_px([[44, 30]])[0]
        cv2.circle(img, (int(c[0]), int(c[1])), 12, (160, 160, 165), -1)                    # keys
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        img *= (1.0 - 0.18 * r2)[..., None]                               # lens vignette
        return img

    def _render(self, t_state: float, k: int) -> np.ndarray:
        rig = self.rig
        rng = np.random.default_rng((rig.seed * 1_000_003 + k) & 0x7FFFFFFF)
        g = 1.0 + 0.02 * math.sin(t_state * 0.7) + rng.normal(0, 0.006)   # drift + flicker
        img = self._base * g + self._noise[k % 4]
        pan, tilt, on = rig.act.state_at(t_state)
        if on and not rig.blocked:
            x, y = rig.geom.hit(pan, tilt)
            w, h = rig.table.size_cm
            if -1.0 <= x <= w + 1 and -1.0 <= y <= h + 1:
                px, py = rig.table.cm_to_px([[x, y]])[0]
                self._draw_dot(img, px, py, rng.uniform(0.85, 1.0))
        if on and rig.ghost_cm is not None:        # a glint of the beam elsewhere, brighter than the dot
            px, py = rig.table.cm_to_px([rig.ghost_cm])[0]
            self._draw_dot(img, px, py, 1.3)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_dot(img: np.ndarray, px: float, py: float, amp: float, r: int = 18,
                  scale: float = 1.0) -> None:
        x0, y0 = int(px) - r, int(py) - r
        x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
        if cx0 >= cx1 or cy0 >= cy1:
            return
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        halo = np.exp(-d2 / (2 * (4.0 * scale) ** 2)) * 230 * amp
        bloom = np.exp(-d2 / (2 * (9.0 * scale) ** 2)) * 50 * amp
        core = np.exp(-d2 / (2 * (1.6 * scale) ** 2)) * 255 * amp
        roi = img[cy0:cy1, cx0:cx1]
        roi[..., 2] += halo + bloom + core
        roi[..., 1] += 0.15 * halo + core
        roi[..., 0] += 0.15 * halo + core


class SimRig:
    """Simulated head + camera + table. `rig.act`, `rig.frames`, `rig.table`, `rig.cfg` plug into Laser.

    realtime=False (default) uses a SimClock: sleeps are free, so tests run fast.
    """

    def __init__(self, cfg: Optional[dict] = None, *, seed: int = 0, realtime: bool = False,
                 latency_s: float = 0.08, fps: float = 30.0, backlash_deg: float = 1.0,
                 pulse_noise_us: float = 1.0, noise_sigma: float = 3.0,
                 geom: Optional[HeadGeometry] = None):
        self.seed, self.realtime = seed, realtime
        self.rng = np.random.default_rng(seed)
        self.clock: Clock = Clock() if realtime else SimClock()
        self.latency_s, self.backlash_deg = latency_s, backlash_deg
        self.pulse_noise_us, self.noise_sigma = pulse_noise_us, noise_sigma
        self.blocked = False                 # something between laser and table
        self.ghost_cm: Optional[tuple[float, float]] = None   # a second, brighter dot while the laser is on
        size = tuple(((cfg or {}).get("table") or {}).get("size_cm", (90, 60)))
        self.table = SimTable(size)
        self.geom = geom or HeadGeometry(table=self.table.size_cm)
        self.cfg = copy.deepcopy(cfg) if cfg else {"table": {"size_cm": list(size)}, "laser_timeout_s": 10}
        # The "measured" limits: pulse range that covers the table, plus a small margin.
        self.cfg["servo_limits"] = self.table_limits()
        self.cfg["camera_latency_s"] = latency_s
        self.cfg["actuator"] = "fake"
        self.act = SimActuator(self, self.cfg)
        self.frames = SimCamera(self, fps=fps)

    def table_limits(self, margin: float = 0.03) -> dict:
        w, h = self.table.size_cm
        edge = [(x, y) for x in np.linspace(0, w, 10) for y in (0, h)] + \
               [(x, y) for y in np.linspace(0, h, 7) for x in (0, w)]
        p = np.array([self.geom.pulses_for(e) for e in edge])
        lo, hi = p.min(axis=0), p.max(axis=0)
        m = margin * (hi - lo)
        return {"pan": [round(lo[0] - m[0]), round(hi[0] + m[0])],
                "tilt": [round(lo[1] - m[1]), round(hi[1] + m[1])]}

    def true_dot_cm(self) -> tuple[float, float]:
        """Where the (physical) beam currently lands, ignoring camera latency."""
        pan, tilt, _ = self.act.state_at(self.clock.now())
        return self.geom.hit(pan, tilt)

    def make_laser(self, cal_path: str):
        from act.laser import Laser
        return Laser(self.act, self.frames, self.table, cal_path, cfg=self.cfg, clock=self.clock)


# ---------------------------------------------------------------- room scene (room pointing, spec 0006)

class RoomScene:
    """A small room of axis-aligned boxes (cm, z up) seen by a pinhole camera high on the front wall.

    Every surface is a box, so one slab raycast serves the camera (what is visible at a pixel) and the
    laser (where the beam lands). Boxes: name -> (lo xyz, hi xyz, BGR colour, reflectance).
    """

    def __init__(self, size_px=(640, 360), f_px=460.0, cam=(0.0, 0.0, 230.0), pitch_deg=25.0):
        self.w, self.h = size_px
        self.f, self.c = f_px, np.array([size_px[0] / 2, size_px[1] / 2])
        self.cam = np.asarray(cam, dtype=np.float64)
        p = math.radians(pitch_deg)
        self.Rc = np.array([[1.0, 0, 0],                               # rows: camera x, y (down), z (forward)
                            [0, -math.sin(p), -math.cos(p)],
                            [0, math.cos(p), -math.sin(p)]])
        self.boxes: dict[str, tuple] = {
            "floor": ((-250, 0, -2), (250, 720, 0), (58, 62, 70), 0.8),
            "back_wall": ((-250, 700, 0), (250, 720, 260), (150, 160, 170), 1.0),
            "left_wall": ((-270, 0, 0), (-250, 720, 260), (140, 150, 160), 1.0),
            "right_wall": ((250, 0, 0), (270, 720, 260), (140, 150, 160), 1.0),
            "table": ((-90, 260, 72), (0, 340, 75), (95, 128, 165), 0.9),
            "shelf": ((70, 430, 137), (150, 470, 140), (120, 140, 150), 0.9),
            "coffee_table": ((20, 560, 42), (120, 640, 45), (80, 100, 120), 0.9),
            "bottle": ((100, 442, 140), (110, 452, 168), (60, 150, 70), 0.7),
            "backpack": ((-170, 480, 0), (-125, 510, 45), (120, 60, 40), 0.5),
        }

    def project(self, P) -> Optional[tuple[float, float]]:
        d = self.Rc @ (np.asarray(P, dtype=np.float64) - self.cam)
        if d[2] <= 1e-6:
            return None
        return float(self.f * d[0] / d[2] + self.c[0]), float(self.f * d[1] / d[2] + self.c[1])

    def ray(self, uv) -> np.ndarray:
        """World direction of the camera ray through pixel uv."""
        d = np.array([(uv[0] - self.c[0]) / self.f, (uv[1] - self.c[1]) / self.f, 1.0])
        d = self.Rc.T @ d
        return d / np.linalg.norm(d)

    def cast(self, o, d) -> tuple[float, Optional[str]]:
        """Nearest hit of the ray o + s*d (s > 0): (s, box name), or (inf, None)."""
        o, d = np.asarray(o, dtype=np.float64), np.asarray(d, dtype=np.float64)
        best, name = math.inf, None
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            for k, (lo, hi, _, _) in self.boxes.items():
                t1, t2 = (np.asarray(lo) - o) * inv, (np.asarray(hi) - o) * inv
                tn = np.nanmax(np.minimum(t1, t2))
                tf = np.nanmin(np.maximum(t1, t2))
                if tf >= max(tn, 1e-6) and tn < best:
                    best, name = max(tn, 0.0), k
        return best, name

    def cast_many(self, o, D) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised cast for rays D (N,3) from one origin: (distance (N,), box index (N,), -1 = none)."""
        o = np.asarray(o, dtype=np.float64)
        best = np.full(len(D), np.inf)
        idx = np.full(len(D), -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / D
            for i, (lo, hi, _, _) in enumerate(self.boxes.values()):
                t1, t2 = (np.asarray(lo) - o) * inv, (np.asarray(hi) - o) * inv
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tf >= np.maximum(tn, 1e-6)) & (tn < best)
                best[hit], idx[hit] = np.maximum(tn[hit], 0.0), i
        return best, idx

    def visible(self, P, eps: float = 0.5) -> bool:
        """The camera sees point P (nothing nearer on the line of sight)."""
        v = np.asarray(P, dtype=np.float64) - self.cam
        dist = float(np.linalg.norm(v))
        s, _ = self.cast(self.cam, v / dist)
        return s >= dist - eps

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """(BGR float image, box-index image) of the empty-laser room."""
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float64)
        D = np.stack([(xx - self.c[0]) / self.f, (yy - self.c[1]) / self.f, np.ones_like(xx)], -1)
        D = np.einsum("nj,jk->nk", D.reshape(-1, 3), self.Rc)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        dist, idx = self.cast_many(self.cam, D)
        cols = np.array([b[2] for b in self.boxes.values()] + [(20, 20, 20)], dtype=np.float32)
        img = cols[idx].reshape(self.h, self.w, 3)
        shade = np.clip(1.25 - dist / 900.0, 0.55, 1.1).astype(np.float32).reshape(self.h, self.w, 1)
        return img * shade, idx.reshape(self.h, self.w)

    def box_px(self, name: str, ids: np.ndarray) -> Optional[tuple[float, float, float, float]]:
        """Visible image box (x1, y1, x2, y2) of a scene box."""
        i = list(self.boxes).index(name)
        ys, xs = np.nonzero(ids == i)
        if len(xs) == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


class RoomHead:
    """Pan-tilt head at the camera plus `offset` cm (the pivot baseline b), aimed into the room.
    Same angle convention as HeadGeometry; hit3d() raycasts the beam into the scene."""

    def __init__(self, scene: RoomScene, offset=(3.0, 0.0, 0.0), aim=(0.0, 450.0, 40.0),
                 us_per_deg=(9.6, 10.3), offset_deg=(1.5, -2.0), roll_deg=2.0):
        self.scene = scene
        self.pos = scene.cam + np.asarray(offset, dtype=np.float64)
        self.k, self.off = us_per_deg, offset_deg
        a = np.asarray(aim, dtype=np.float64) - self.pos
        a /= np.linalg.norm(a)
        pitch, yaw = math.asin(-a[2]), math.atan2(-a[0], a[1])
        self.R = _rz(yaw) @ _rx(-pitch) @ _ry(math.radians(roll_deg))

    def direction(self, pan: float, tilt: float) -> np.ndarray:
        th = math.radians((pan - 1500) / self.k[0] + self.off[0])
        al = math.radians((tilt - 1500) / self.k[1] + self.off[1])
        return self.R @ _rz(th) @ _rx(-al) @ np.array([0.0, 1.0, 0.0])

    def hit3d(self, pan: float, tilt: float) -> tuple[Optional[np.ndarray], Optional[str]]:
        d = self.direction(pan, tilt)
        s, name = self.scene.cast(self.pos, d)
        return (None, None) if name is None else (self.pos + s * d, name)

    def dot_px(self, pan: float, tilt: float) -> Optional[tuple[float, float]]:
        """Where the camera sees the dot, or None (no surface, occluded from the camera, off-frame)."""
        P, _ = self.hit3d(pan, tilt)
        if P is None or not self.scene.visible(P):
            return None
        uv = self.scene.project(P)
        if uv is None or not (0 <= uv[0] < self.scene.w and 0 <= uv[1] < self.scene.h):
            return None
        return uv


class RoomCamera(SimCamera):
    """SimCamera over a RoomScene: fresh noise per frame (so averaging pairs helps), dot size and
    brightness falling with distance and scaled by the surface's reflectance."""

    def _render_base(self) -> np.ndarray:
        img, self.ids = self.rig.scene.render()
        return img

    def _render(self, t_state: float, k: int) -> np.ndarray:
        rig = self.rig
        rng = np.random.default_rng((rig.seed * 1_000_003 + k) & 0x7FFFFFFF)
        g = 1.0 + 0.02 * math.sin(t_state * 0.7) + rng.normal(0, 0.006)
        img = self._base * g + self._noise[int(rng.integers(len(self._noise)))]
        pan, tilt, on = rig.act.state_at(t_state)
        if on and not rig.blocked:
            P, name = rig.geom.hit3d(pan, tilt)
            uv = rig.geom.dot_px(pan, tilt) if P is not None else None
            if uv is not None:
                dist = float(np.linalg.norm(P - rig.scene.cam))
                refl = rig.scene.boxes[name][3]
                amp = rig.dot_gain * refl * min(1.0, (300.0 / dist) ** 2) * rng.uniform(0.85, 1.0)
                self._draw_dot(img, uv[0], uv[1], max(amp, 0.0), r=10,
                               scale=float(np.clip(150.0 / dist, 0.35, 0.6)))
        return np.clip(img, 0, 255).astype(np.uint8)


class RoomRig:
    """Simulated room: RoomScene + RoomHead (pivot `b_cm` from the lens) + RoomCamera + SimActuator.
    Plugs into Laser like SimRig (table=None: room aiming is pixel-space only)."""

    def __init__(self, *, b_cm: float = 3.0, seed: int = 0, latency_s: float = 0.05, fps: float = 30.0,
                 backlash_deg: float = 1.0, pulse_noise_us: float = 1.0, noise_sigma: float = 3.0,
                 dot_gain: float = 1.0, cfg: Optional[dict] = None):
        self.seed, self.realtime = seed, False
        self.rng = np.random.default_rng(seed)
        self.clock: Clock = SimClock()
        self.latency_s, self.backlash_deg = latency_s, backlash_deg
        self.pulse_noise_us, self.noise_sigma, self.dot_gain = pulse_noise_us, noise_sigma, dot_gain
        self.blocked = False
        self.scene = RoomScene()
        self.size_px = (self.scene.w, self.scene.h)
        self.geom = RoomHead(self.scene, offset=(0.6 * b_cm, 0.0, -0.8 * b_cm))
        self.cfg = copy.deepcopy(cfg) if cfg else {"laser_timeout_s": 60}
        self.cfg["servo_limits"] = {"pan": [1130, 1830], "tilt": [1310, 1770]}   # covers the view
        self.cfg["camera_latency_s"] = latency_s
        self.cfg["actuator"] = "fake"
        self.act = SimActuator(self, self.cfg)   # type: ignore[arg-type]
        self.frames = RoomCamera(self, fps=fps)
        self.frames._noise = [np.random.default_rng(seed + 7 + i).normal(0, noise_sigma, (self.scene.h,
                              self.scene.w, 3)).astype(np.float32) for i in range(16)]

    def true_dot_px(self) -> Optional[tuple[float, float]]:
        """Where the camera would see the (physical) beam now, ignoring latency; None if unseen."""
        pan, tilt, _ = self.act.state_at(self.clock.now())
        return self.geom.dot_px(pan, tilt)

    def box_px(self, name: str) -> Optional[tuple[float, float, float, float]]:
        return self.scene.box_px(name, self.frames.ids)

    def surface_at(self, uv) -> Optional[str]:
        x, y = int(np.clip(uv[0], 0, self.scene.w - 1)), int(np.clip(uv[1], 0, self.scene.h - 1))
        i = int(self.frames.ids[y, x])
        return None if i < 0 else list(self.scene.boxes)[i]

    def make_laser(self, cal_path: str = ""):
        from act.laser import Laser
        return Laser(self.act, self.frames, None, cal_path, cfg=self.cfg, clock=self.clock)  # type: ignore[arg-type]
