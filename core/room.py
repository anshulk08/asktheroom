"""Room memory M0 (spec 0009): per-zone tracks of known props, the per-frame room driver, and the zone CLI.

RoomTracker keeps short tracks `r:N` per drawn zone across visits of that zone: confirmation after
`confirm_visits` matched visits, and valid/invalid misses (a hand box, a large frame-difference blob or a
too dark/bright box over the spot makes a visit count for nothing). RoomMemory runs once per perception
frame: every `room_every_n`-th call it crops the next zone (round-robin) from the full 1080p frame, runs the
already-loaded prop model backend on the crop, and hands the tracker's ZoneVisit to `world.room_update`,
which applies the association rules (core/room_world.py). Positions here are full-frame px, never table cm.

    python -m core.room --zone bookshelf --say "the bookshelf" --poly 100,80 600,80 600,400 100,400
    python -m core.room --list
    python -m core.room --delete-zone bookshelf
    python -m core.room --grab 0 --out full.jpg                      # one full frame from the camera
    python -m core.room --show full.jpg --out zones.jpg              # draw the zones for checking
    python -m core.room --measure-rect --full full.jpg --ref table160.jpg   # table_view_rect for config.local.yaml

Python 3.10 (JetPack 6).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from core import geom
from core.room_types import RoomConfig, RoomObservation, RoomTrack, ZoneVisit
from core.room_zones import Zone, Zones, view_version
from core.types import BoxPx, Event, Frame

log = logging.getLogger(__name__)

MATCH_IOU = 0.3             # spec 0009 section 3: a track match needs IoU >= 0.3 ...
MATCH_DIAG = 0.5            # ... or a centre within 0.5 track-box diagonals
DEDUPE_IOU = 0.5            # two boxes of one class this overlapped in one crop are one object (two prompts)
DILATE = np.ones((5, 5), np.uint8)


# ---------------------------------------------------------------------------------------------
# Tracker

class RoomTracker:
    """Per-zone tracks across visits of that zone. World.room_update sets role/entity; the tracker keeps them."""

    def __init__(self, cfg: RoomConfig):
        self.cfg = cfg
        self._tracks: dict[str, list[RoomTrack]] = {}
        self._n = 0

    def tracks(self, zone: Optional[str] = None) -> list[RoomTrack]:
        if zone is not None:
            return list(self._tracks.get(zone, []))
        return [tr for ts in self._tracks.values() for tr in ts]

    def visit(self, zone: str, say: str, obs: list[RoomObservation], blockers: list[BoxPx],
              changes: list[BoxPx], t: float, wall: float, frame_idx: int,
              lum: Optional[Callable[[BoxPx], float]] = None, crop=None) -> ZoneVisit:
        """Apply one processed visit of `zone`. `obs` are its known-prop detections (full px); `blockers`
        hand/person boxes; `changes` frame-difference boxes since the zone's previous visit; `lum(box)` the
        mean grey level in a box of this frame."""
        v = ZoneVisit(zone=zone, say=say, t=t, wall=wall, frame_idx=frame_idx, crop=crop)
        tracks = self._tracks.setdefault(zone, [])
        matched = self._match(tracks, obs)
        keep: list[RoomTrack] = []
        for tr in tracks:
            o = matched.get(tr.tid)
            if o is not None:
                tr.box_px = tuple(int(c) for c in o.box_px)
                tr.last_seen, tr.last_wall = t, wall
                tr.hits += 1
                tr.misses = 0
                tr.confirmed = tr.confirmed or tr.hits >= self.cfg.confirm_visits
                if tr.confirmed:
                    v.confirmed.append(tr)
                keep.append(tr)
            elif not self._valid(tr.box_px, blockers + [o.box_px for o in obs], changes, lum):
                keep.append(tr)                   # an invalid visit counts for nothing (another object
                                                  # on its spot is a blocker too: spec 0009, Absence)
            elif not tr.confirmed:
                v.dropped.append(tr)
            else:
                tr.misses += 1
                v.missed.append(tr)
                if tr.misses >= self.cfg.absent_visits:
                    v.dropped.append(tr)
                else:
                    keep.append(tr)
        used = {id(o) for o in matched.values()}
        for o in obs:
            if id(o) in used:
                continue
            self._n += 1
            tr = RoomTrack(tid=f"r:{self._n}", zone=zone, cls=o.cls, box_px=tuple(int(c) for c in o.box_px),
                           first_seen=t, first_wall=wall, last_seen=t, last_wall=wall, hits=1)
            tr.confirmed = tr.hits >= self.cfg.confirm_visits
            if tr.confirmed:
                v.confirmed.append(tr)
            keep.append(tr)
        self._tracks[zone] = keep
        return v

    @staticmethod
    def _match(tracks: list[RoomTrack], obs: list[RoomObservation]) -> dict[str, RoomObservation]:
        """Greedy per class: best IoU first (then nearest centre), each track and observation used once."""
        pairs = []
        for i, tr in enumerate(tracks):
            diag = math.hypot(tr.box_px[2] - tr.box_px[0], tr.box_px[3] - tr.box_px[1])
            for j, o in enumerate(obs):
                if o.cls != tr.cls:
                    continue
                ov = geom.iou(tr.box_px, o.box_px)
                d = geom.dist(geom.center(tr.box_px), geom.center(o.box_px))
                if ov >= MATCH_IOU or d <= MATCH_DIAG * diag:
                    pairs.append((-ov, d, i, j))
        pairs.sort()
        out: dict[str, RoomObservation] = {}
        used_obs: set[int] = set()
        for _, _, i, j in pairs:
            if tracks[i].tid in out or j in used_obs:
                continue
            out[tracks[i].tid] = obs[j]
            used_obs.add(j)
        return out

    def _valid(self, box: BoxPx, blockers: list[BoxPx], changes: list[BoxPx],
               lum: Optional[Callable[[BoxPx], float]]) -> bool:
        """Whether this visit can count as a miss for a track at `box` (spec 0009 section 3, Absence)."""
        c = self.cfg
        if any(geom.overlap_frac(b, box) >= c.blocker_overlap for b in blockers):
            return False
        big = c.change_area_ratio * geom.area(box)
        if any(geom.area(ch) >= big and geom.overlap_frac(ch, box) >= c.blocker_overlap for ch in changes):
            return False
        if lum is not None:
            v = float(lum(box))
            if not (c.lum_lo <= v <= c.lum_hi):       # NaN (box off the crop) is invalid too
                return False
        return True


# ---------------------------------------------------------------------------------------------
# Driver

class RoomMemory:
    """Called once per perception frame with the full camera frame; processes one zone every room_every_n calls."""

    def __init__(self, cfg: RoomConfig, zones: Zones, backend, to_obj: dict[str, str], world,
                 table_rect: BoxPx, hand_conf: float = 0.35):
        self.cfg = cfg
        self.zones = zones
        self.backend = backend                 # the table detector's backend: shared, never a second load
        self.to_obj = to_obj
        self.world = world
        self.table_rect = tuple(int(v) for v in table_rect)
        self.hand_conf = hand_conf             # a hand box needs this score to block (the table's hand cut-off)
        self.tracker = RoomTracker(cfg)
        self._calls = 0
        self._next = 0
        self._prev: dict[str, np.ndarray] = {}  # zone -> grey crop of its previous visit (change evidence)

    @classmethod
    def from_config(cls, cfg: dict, world, backend, table_rect: Optional[BoxPx]) -> Optional["RoomMemory"]:
        """RoomMemory from config.yaml's room_memory: section, or None (logged) when it can't run."""
        rc = RoomConfig.from_dict(cfg.get("room_memory"))
        if not rc.enabled:
            log.info("room memory off: room_memory.enabled is false")
            return None
        if table_rect is None:
            log.warning("room memory off: no table_view_rect")
            return None
        try:
            zones = Zones.load(rc.zones_path)
        except FileNotFoundError:
            log.warning("room memory off: no zones file %s (draw zones: python -m core.room --zone ...)",
                        rc.zones_path)
            return None
        except (ValueError, KeyError, TypeError) as e:
            log.warning("room memory off: can't read zones file %s: %s", rc.zones_path, e)
            return None
        if not zones.zones:
            log.warning("room memory off: %s has no zones", rc.zones_path)
            return None
        want = view_version(rc.capture_size, rc.zoom, table_rect)
        if zones.view != want:
            log.warning("room memory off: %s was drawn at view %s, the camera view is now %s "
                        "(capture_size, zoom or table_view_rect changed); redraw the zones",
                        rc.zones_path, zones.view, want)
            return None
        from core.detect import class_list
        _, to_obj = class_list(cfg)
        ct = cfg.get("conf_threshold", 0.35)
        hand_conf = float(ct.get("hand", ct.get("default", 0.35))) if isinstance(ct, dict) else float(ct)
        log.info("room memory on: zones %s", ", ".join(f"{z.name} ({z.say})" for z in zones.zones.values()))
        return cls(rc, zones, backend, to_obj, world, table_rect, hand_conf=hand_conf)

    def step(self, full: Optional[Frame]) -> list[Event]:
        """One perception frame. Every room_every_n-th call processes the next zone and returns
        world.room_update's events; otherwise, or with no image or an empty crop, []."""
        self._calls += 1
        names = list(self.zones.zones)
        if not names or self._calls % max(1, int(self.cfg.room_every_n)) != 0:
            return []
        if full is None or full.img is None:
            return []
        zone = self.zones.zones[names[self._next % len(names)]]
        self._next += 1
        visit = self._visit(zone, full)
        if visit is None:
            return []
        return list(self.world.room_update(visit) or [])

    def _visit(self, zone: Zone, full: Frame) -> Optional[ZoneVisit]:
        img = full.img
        h, w = img.shape[:2]
        bx1, by1, bx2, by2 = zone.bbox()
        x1, y1, x2, y2 = max(0, bx1), max(0, by1), min(w, bx2), min(h, by2)
        if x2 <= x1 or y2 <= y1:
            log.debug("zone %s is outside the %dx%d frame", zone.name, w, h)
            return None
        crop = img[y1:y2, x1:x2]
        s = 1.0
        long_side = max(x2 - x1, y2 - y1)
        if long_side > self.cfg.max_crop_px:
            s = self.cfg.max_crop_px / long_side
            small = cv2.resize(crop, (max(1, round((x2 - x1) * s)), max(1, round((y2 - y1) * s))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = crop

        def to_full(b) -> BoxPx:
            return (int(round(x1 + b[0] / s)), int(round(y1 + b[1] / s)),
                    int(round(x1 + b[2] / s)), int(round(y1 + b[3] / s)))

        def to_small(b: BoxPx) -> BoxPx:
            return (int(math.floor((b[0] - x1) * s)), int(math.floor((b[1] - y1) * s)),
                    int(math.ceil((b[2] - x1) * s)), int(math.ceil((b[3] - y1) * s)))

        hands: list[BoxPx] = []
        props: list[tuple[str, float, BoxPx]] = []
        for label, conf, box in self.backend.infer(small):
            obj = self.to_obj.get(label)
            if obj is None:
                continue
            if obj == "hand":
                if conf >= self.hand_conf:
                    hands.append(to_full(box))
            elif conf >= self.cfg.room_prop_conf:
                props.append((obj, float(conf), to_full(box)))
        obs: list[RoomObservation] = []
        for obj, conf, box in sorted(props, key=lambda p: -p[1]):
            if any(geom.iou(box, hb) >= self.cfg.hand_iou for hb in hands):
                continue                           # the hand itself, called a prop
            c = geom.center(box)
            if not zone.contains(c) or geom.contains_point(self.table_rect, c):
                continue
            if any(o.cls == obj and geom.iou(o.box_px, box) >= DEDUPE_IOU for o in obs):
                continue                           # the same object under a second prompt
            obs.append(RoomObservation(zone=zone.name, cls=obj, conf=round(conf, 3), box_px=box,
                                       t=full.t, wall=full.wall, frame_idx=full.idx))

        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        changes: list[BoxPx] = []
        prev = self._prev.get(zone.name)
        if prev is not None and prev.shape == grey.shape:
            mask = (cv2.absdiff(grey, prev) > self.cfg.change_thr).astype(np.uint8) * 255
            mask = cv2.dilate(mask, DILATE, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                changes.append(to_full((cx, cy, cx + cw, cy + ch)))
        self._prev[zone.name] = grey

        def lum(box: BoxPx) -> float:
            a1, b1, a2, b2 = to_small(box)
            a1, b1 = max(0, a1), max(0, b1)
            a2, b2 = min(grey.shape[1], a2), min(grey.shape[0], b2)
            if a2 <= a1 or b2 <= b1:
                return float("nan")                # off this crop: no valid view of it
            return float(grey[b1:b2, a1:a2].mean())

        return self.tracker.visit(zone.name, zone.say, obs, hands, changes, full.t, full.wall, full.idx,
                                  lum=lum, crop=crop.copy())


# ---------------------------------------------------------------------------------------------
# CLI

def _current_view(rc: RoomConfig) -> tuple[BoxPx, str]:
    from core.room_view import default_rect
    rect = rc.table_view_rect or default_rect(rc.capture_size, rc.zoom, rc.ref_zoom)
    return tuple(int(v) for v in rect), view_version(rc.capture_size, rc.zoom, rect)


def _parse_poly(pts: Sequence[str]) -> list[tuple[float, float]]:
    out = []
    for p in pts:
        x, y = p.split(",")
        out.append((float(x), float(y)))
    if len(out) < 3:
        raise ValueError("a zone needs at least 3 points")
    return out


def _load_zones(path: str) -> Optional[Zones]:
    try:
        return Zones.load(path)
    except FileNotFoundError:
        return None


def _read_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"can't read image {path}")
    return img


def main(argv=None) -> int:
    from core.config import load_config
    ap = argparse.ArgumentParser(description="room memory zones (spec 0009 M0)")
    ap.add_argument("--config", help="config.yaml path (default: the repo's, plus config.local.yaml)")
    ap.add_argument("--zones", help="zones file (default: room_memory.zones_path)")
    ap.add_argument("--zone", help="add or replace this zone")
    ap.add_argument("--say", help="its spoken name (default: 'the <zone>')")
    ap.add_argument("--poly", nargs="+", metavar="X,Y", help="polygon corners in full-frame px")
    ap.add_argument("--delete-zone", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="IMAGE", help="draw the zones on this full frame (with --out)")
    ap.add_argument("--measure-rect", action="store_true", help="fit table_view_rect (with --full, --ref)")
    ap.add_argument("--full", help="a full frame at the room zoom")
    ap.add_argument("--ref", help="a table frame at the reference zoom (160)")
    ap.add_argument("--grab", metavar="DEVICE", help="save one full frame from this camera (with --out)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    path = a.zones or rc.zones_path

    if a.zone:
        if not a.poly:
            ap.error("--zone needs --poly")
        try:
            poly = _parse_poly(a.poly)
        except ValueError as e:
            ap.error(f"--poly: {e}")
        rect, view = _current_view(rc)
        zones = _load_zones(path) or Zones(view, rc.capture_size, {})
        if zones.view != view:
            print(f"{path} was drawn at view {zones.view}, the configured view is {view} "
                  f"(capture_size, zoom or table_view_rect changed): delete the file and redraw every zone",
                  file=sys.stderr)
            return 1
        w, h = rc.capture_size
        if any(not (0 <= x <= w and 0 <= y <= h) for x, y in poly):
            print(f"warning: some points are outside the {w}x{h} frame", file=sys.stderr)
        zones.zones[a.zone] = Zone(a.zone, a.say or f"the {a.zone.replace('_', ' ')}", poly)
        zones.save(path)
        print(f"saved zone {a.zone} ({zones.zones[a.zone].say}) to {path}")
        return 0

    if a.delete_zone:
        zones = _load_zones(path)
        if zones is None or a.delete_zone not in zones.zones:
            print(f"no zone {a.delete_zone} in {path}", file=sys.stderr)
            return 1
        del zones.zones[a.delete_zone]
        zones.save(path)
        print(f"deleted zone {a.delete_zone} from {path}")
        return 0

    if a.list:
        zones = _load_zones(path)
        if zones is None:
            print(f"no zones file {path}")
            return 1
        _, view = _current_view(rc)
        state = "matches the config" if zones.view == view else f"MISMATCH: the config's view is {view}"
        print(f"{path}: view {zones.view} ({state}), frame {zones.size_px[0]}x{zones.size_px[1]}, "
              f"{len(zones.zones)} zone(s)")
        for z in zones.zones.values():
            print(f"  {z.name}: {z.say!r} bbox {list(z.bbox())} {len(z.poly)} points")
        return 0

    if a.show:
        if not a.out:
            ap.error("--show needs --out")
        img = _read_img(a.show)
        zones = _load_zones(path)
        if zones is None:
            print(f"no zones file {path}", file=sys.stderr)
            return 1
        if (img.shape[1], img.shape[0]) != zones.size_px:
            print(f"warning: the image is {img.shape[1]}x{img.shape[0]}, the zones were drawn on "
                  f"{zones.size_px[0]}x{zones.size_px[1]}", file=sys.stderr)
        rect, _ = _current_view(rc)
        cv2.rectangle(img, rect[:2], rect[2:], (255, 128, 0), 2)
        cv2.putText(img, "table view", (rect[0] + 6, rect[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 128, 0), 2)
        for z in zones.zones.values():
            pts = np.round(np.asarray(z.poly)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            x, y = pts[:, 0, 0].min(), pts[:, 0, 1].min()
            cv2.putText(img, f"{z.name}: {z.say}", (int(x) + 6, int(y) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
        cv2.imwrite(a.out, img)
        print(f"saved {a.out}")
        return 0

    if a.measure_rect:
        if not (a.full and a.ref):
            ap.error("--measure-rect needs --full and --ref")
        from core.room_view import default_rect, measure_rect
        full, ref = _read_img(a.full), _read_img(a.ref)
        init = rc.table_view_rect or default_rect((full.shape[1], full.shape[0]), rc.zoom, rc.ref_zoom)
        rect, corr = measure_rect(full, ref, tuple(init))
        print(f"room_memory: {{table_view_rect: [{', '.join(str(int(v)) for v in rect)}]}}")
        print(f"# ECC correlation {corr:.3f}")
        return 0

    if a.grab is not None:
        if not a.out:
            ap.error("--grab needs --out")
        from core.capture import open_camera
        dev = int(a.grab) if a.grab.isdigit() else a.grab
        cap = open_camera(dev, *rc.capture_size)
        try:
            img = None
            for _ in range(15):                    # let exposure settle; keep the last good frame
                ok, f = cap.read()
                if ok:
                    img = f
        finally:
            cap.release()
        if img is None:
            print(f"no frame from {a.grab}", file=sys.stderr)
            return 1
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.out, img)
        print(f"saved {a.out} ({img.shape[1]}x{img.shape[0]})")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
