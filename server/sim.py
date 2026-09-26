"""Simulated camera + the real world model behind the dashboard. No hardware needed.

    python -m server.sim            # open http://localhost:8000
    python -m server.sim --check    # no server: print each step's events and the final beliefs

A scripted tabletop story (eval/synth.py's Scene) plays in real time at 10 fps. Each frame, its
noisy detections go through core.world.World, exactly as the detector's would on the Jetson. The
video shows what a camera would see (hidden objects are not drawn); the dashboard's graph, timeline
and markers show what the world model believes. Typed questions go through the real intent parser
and answer templates (Grok only for OTHER, and only when online); the answer's target gets a
simulated laser marker.
"""
from __future__ import annotations

import argparse
import logging
import random
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.types import Detections, Frame
from eval.synth import FPS, IMG_H, IMG_W, SIZES, Scene

log = logging.getLogger("askroom.sim")

LAYOUT = {"box": (20, 30), "keys": (45, 12), "wallet": (60, 12), "phone": (78, 14),
          "glasses": (15, 50), "remote": (38, 50), "pill_bottle": (60, 38), "notebook": (76, 44)}

TABLE_BGR = (178, 196, 212)
SKIN_BGR = (140, 170, 215)
OBJ_BGR = {"keys": (60, 200, 230), "pill_bottle": (70, 110, 240), "wallet": (60, 80, 120),
           "glasses": (200, 200, 90), "phone": (45, 45, 45), "remote": (95, 95, 95),
           "box": (80, 140, 190), "notebook": (235, 232, 225)}
DRAW_ORDER = {"container": 0, "target": 1, "cover": 2}


@dataclass
class Step:
    start: int          # first frame index of this step
    caption: str
    expect: str         # what the world model should conclude, for --check


def story(cfg: dict, seed: int = 0) -> tuple[Scene, list[Step]]:
    """The demo script. Each step records the frame it starts at so captions can follow along."""
    s = Scene(cfg, random.Random(seed), dict(LAYOUT))
    steps: list[Step] = []

    def step(caption: str, expect: str) -> None:
        steps.append(Step(len(s.snaps), caption, expect))

    step("All objects on the table", "everything VISIBLE")
    s.wait(3.0)

    step("Keys picked up and dropped into the box", "keys INSIDE box")
    s.hand_in(LAYOUT["keys"]); s.grab("keys")
    s.hand_to(LAYOUT["box"]); s.put_in("keys", "box"); s.hand_out(); s.wait(3.0)

    step("The box is moved (keys still inside)", "keys INSIDE box, resolved to the box's new spot")
    s.hand_in(LAYOUT["box"]); s.grab("box")
    s.hand_to((38, 24), speed_cm_s=20); s.release(); s.hand_out(); s.wait(3.0)

    step("Notebook slid over the pill bottle (hand on the notebook's far edge)",
         "pill_bottle UNDER notebook")
    s.hand_in((84, 44), edge="right"); s.grab("notebook")
    s.hand_to((68, 38), speed_cm_s=20); s.release(); s.hand_out("right"); s.wait(3.5)

    step("Wallet picked up, then put back in the same spot", "wallet VISIBLE (PUT_BACK)")
    s.hand_in(LAYOUT["wallet"], edge="top"); s.grab("wallet")
    s.hand_to((60, 24)); s.wait(0.8); s.hand_to((60.5, 12.5)); s.release(); s.hand_out("top")
    s.wait(2.5)

    step("Remote moved to a new spot", "remote VISIBLE (MOVED)")
    s.hand_in(LAYOUT["remote"]); s.grab("remote")
    s.hand_to((12, 35)); s.release(); s.hand_out("left"); s.wait(2.5)

    step("Phone carried off the right edge", "phone GONE (right)")
    s.hand_in(LAYOUT["phone"], edge="right"); s.grab("phone"); s.hand_out("right"); s.wait(3.0)

    step("Done. Ask: where are my keys? / where are my pills? / what changed?", "")
    s.wait(1.0)
    return s, steps


# ------------------------------------------------------------------------------ drawing

class SimTable:
    """Table cm -> frame px, the same mapping eval/synth uses for box_px (frame == table)."""

    def __init__(self, cfg: dict):
        self.W, self.H = cfg["table"]["size_cm"]

    def cm_to_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        return np.stack([pts[:, 0] * IMG_W / self.W, pts[:, 1] * IMG_H / self.H], axis=1)


def _texture(key: str, h: int, w: int, base, amp: int = 28) -> np.ndarray:
    """Base colour with a fixed blocky texture per object, so image rules see real structure."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    b = 8
    noise = rng.integers(-amp, amp + 1, (-(-h // b), -(-w // b), 1))
    tex = np.repeat(np.repeat(noise, b, axis=0), b, axis=1)[:h, :w]
    return np.clip(np.array(base) + tex, 0, 255).astype(np.uint8)


class Painter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.table = SimTable(cfg)
        bg = _texture("table", IMG_H, IMG_W, TABLE_BGR, amp=6)
        for x in range(0, int(self.table.W) + 1, 10):     # faint 10 cm grid
            px = int(x * IMG_W / self.table.W)
            cv2.line(bg, (px, 0), (px, IMG_H), (160, 176, 190), 1)
        for y in range(0, int(self.table.H) + 1, 10):
            py = int(y * IMG_H / self.table.H)
            cv2.line(bg, (0, py), (IMG_W, py), (160, 176, 190), 1)
        self.bg = bg

    def _rect(self, c, size) -> tuple[int, int, int, int]:
        (x, y), (w, h) = c, size
        p = self.table.cm_to_px([[x - w / 2, y - h / 2], [x + w / 2, y + h / 2]])
        return int(p[0, 0]), int(p[0, 1]), int(p[1, 0]), int(p[1, 1])

    def _paint(self, img, name, c, size, base) -> None:
        x1, y1, x2, y2 = self._rect(c, size)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(IMG_W, x2), min(IMG_H, y2)
        if x2 <= x1 or y2 <= y1:
            return
        img[y1:y2, x1:x2] = _texture(name, y2 - y1, x2 - x1, base)
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (30, 30, 30), 1)

    def draw(self, snap: dict, caption: str) -> np.ndarray:
        img = self.bg.copy()
        kinds = self.cfg["objects"]
        visible = [(o, c) for o, c in snap["pos"].items()
                   if c is not None and o not in snap["hidden"] and o not in snap["carry"]]
        for o, c in sorted(visible, key=lambda oc: DRAW_ORDER.get(kinds.get(oc[0]), 1)):
            self._paint(img, o, c, SIZES.get(o, (8, 8)), OBJ_BGR.get(o, (200, 200, 200)))
        if snap["hand"] is not None:
            for o in snap["carry"]:
                if snap["pos"].get(o) is not None:
                    self._paint(img, o, snap["pos"][o], SIZES.get(o, (8, 8)), OBJ_BGR.get(o, (200, 200, 200)))
            x1, y1, x2, y2 = self._rect(snap["hand"], SIZES["hand"])
            cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2),
                        0, 0, 360, SKIN_BGR, -1)
        cv2.rectangle(img, (0, IMG_H - 44), (IMG_W, IMG_H), (40, 40, 40), -1)
        cv2.putText(img, caption, (16, IMG_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2,
                    cv2.LINE_AA)
        return img


# ------------------------------------------------------------------------------ playback

class SimCamera:
    """Plays the story in real time, feeding the world; the dashboard reads latest()/latest_dets()."""

    def __init__(self, cfg: dict, world, loop: bool = True, hold_s: float = 25.0, seed: int = 0,
                 speed: float = 1.0):
        self.cfg, self.world, self.loop, self.hold_s, self.seed = cfg, world, loop, hold_s, seed
        self.speed = speed
        self.painter = Painter(cfg)
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._dets: Optional[Detections] = None
        self._stop = threading.Event()
        self.idx = 0
        self._story_t = 0.0             # story clock; never runs backwards across loops

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def latest_dets(self) -> Optional[Detections]:
        with self._lock:
            return self._dets

    def start(self) -> "SimCamera":
        threading.Thread(target=self._run, name="sim-camera", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        run = 0
        while not self._stop.is_set():
            scene, steps = story(self.cfg, self.seed + run)
            dets_all = scene.render()
            if run:
                self.world.reset()
            self._play(scene, steps, dets_all)
            run += 1
            if not self.loop or self._stop.wait(self.hold_s):
                return

    def _play(self, scene: Scene, steps: list[Step], dets_all: list[Detections]) -> None:
        period = 1.0 / FPS / self.speed
        next_t = time.monotonic()
        t0 = max(next_t, self._story_t)
        k = 0
        for i, (snap, d) in enumerate(zip(scene.snaps, dets_all)):
            if self._stop.is_set():
                return
            while k + 1 < len(steps) and steps[k + 1].start <= i:
                k += 1
                log.info("step %d: %s", k, steps[k].caption)
            # The world gets story time, so --speed changes only the playback pace: feeding it
            # sped-up real time would shrink every dwell and window threshold by the same factor.
            now_t, now_wall = t0 + i / FPS, time.time()
            self._story_t = now_t + 1.0 / FPS
            self.idx += 1
            dets = Detections(t=now_t, frame_idx=self.idx, items=d.items, hands=d.hands)
            frame = Frame(t=now_t, wall=now_wall, img=self.painter.draw(snap, steps[k].caption), idx=self.idx)
            for ev in self.world.update(dets, frame):
                log.info("  event %-12s %-12s parent=%s edge=%s conf=%.2f",
                         ev.type, ev.obj, ev.parent, ev.edge, ev.confidence)
            self.world.fps = FPS * self.speed
            with self._lock:
                self._frame, self._dets = frame, dets
            next_t += period
            self._stop.wait(max(0.0, next_t - time.monotonic()))


# ------------------------------------------------------------------------------ entry points

def check(cfg: dict, seed: int = 0, pixels: bool = True) -> None:
    """Headless: run the story through the world as fast as possible and print what it concluded."""
    from core.world import World
    world = World(cfg)
    scene, steps = story(cfg, seed)
    painter = Painter(cfg)
    k = 0
    for i, (snap, d) in enumerate(zip(scene.snaps, scene.render())):
        while k + 1 < len(steps) and steps[k + 1].start <= i:
            k += 1
            prev = steps[k - 1]
            print(f"   -> expected: {prev.expect}" if prev.expect else "")
            print(f"\nstep {k}: {steps[k].caption}")
        if i == 0:
            print(f"step 0: {steps[0].caption}")
        t = 1000.0 + i / FPS
        frame = Frame(t=t, wall=t, idx=i, img=painter.draw(snap, steps[k].caption) if pixels else None)
        for ev in world.update(Detections(t=t, frame_idx=i, items=d.items, hands=d.hands), frame):
            extra = f" parent={ev.parent}" if ev.parent else (f" edge={ev.edge}" if ev.edge else "")
            print(f"   event {ev.type.value if hasattr(ev.type, 'value') else ev.type:12s} {ev.obj}{extra}"
                  f"  conf={ev.confidence:.2f}")
    print("\nfinal beliefs:")
    for name in world.cfg.names():
        e = world.get(name)
        pos, chain = world.resolve(name)
        where = f" -> points at {tuple(round(v, 1) for v in pos)}" if pos else ""
        print(f"   {name:12s} {e.status.value:8s} parent={e.parent}  conf={e.confidence:.2f}"
              f"  chain={'>'.join(chain)}{where}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Ask the Room: real world model on a simulated camera")
    ap.add_argument("--check", action="store_true", help="headless: print events and final beliefs")
    ap.add_argument("--no-pixels", action="store_true", help="--check without images (skips image rules)")
    ap.add_argument("--once", action="store_true", help="play the story once, then hold the final state")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    from core.config import load_config
    cfg = load_config(args.config)
    if args.check:
        check(cfg, args.seed, pixels=not args.no_pixels)
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import tempfile

    import uvicorn

    from core.events import EventLog
    from core.types import Answer
    from core.world import World
    from net import NetMonitor
    from server.app import create_app
    from voice.pipeline import make_ask

    snap = tempfile.mkdtemp(prefix="askroom_sim_snaps_")
    events = EventLog(":memory:", snap)
    world = World(cfg, events)
    cam = SimCamera(cfg, world, loop=not args.once, seed=args.seed, speed=args.speed).start()

    net = NetMonitor(cfg)
    net.start()
    base_ask = make_ask(cfg, world, events, net=net)

    def ask_fn(text: str, source: str) -> Answer:
        world.online = bool(getattr(net, "online", False))
        ans = base_ask(text, source)
        if ans.point_at:
            world.laser = {"on": True, "target": ans.point_at, "err_cm": None}
            threading.Timer(cfg.get("laser_timeout_s", 10),
                            lambda: setattr(world, "laser", {"on": False, "target": None, "err_cm": None})).start()
        return ans

    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]
    app = create_app(cfg, world, events, frames=cam, ask_fn=ask_fn, table=cam.painter.table)
    log.info("sim dashboard on http://localhost:%s", port)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
