"""Replay raw full camera frames through the object registry (spec 0011, core/permanence.py) offline, and write
the tracker-agnostic track WS2's scorecard reads (`askroom-track/1`, docs/track-format.md on ws/identity-demo).

Input: a guided clip directory (video.mp4 at the full camera size, frames.json, meta.json with the view, the
zones and the table calibration, truth.json), or a directory of raw full frames (JPEG/PNG, sorted by name; the
wall time is the number in the file name when there is one, else --fps). Like the live app, each frame processes
one view (--sweep: every view of every frame).

References: --refs <dir>/<name>/*.jpg (enrolled from labelled stills of the same props: the fair setting), and
--enroll NAME IMAGE X1 Y1 X2 Y2. --teach-on-cues instead registers each prop at its truth 'place' cue (the new
arrival on the table after the cue), as if someone said "this is my wallet"; the header says so.

Needs ultralytics (YOLOE) and onnxruntime (DINOv2): run it with the training venv, e.g.
  ~/asktheroom/train-venv/bin/python -m eval.permanence_replay --frames data/clips/room_keys_off_1 \\
      --refs data/registry --track data/clips/room_keys_off_1/track-registry.jsonl
Grok questions need XAI_API_KEY (and --verify); without them only the appearance rules run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import geom  # noqa: E402
from core.permanence import (CARRIED, HIDDEN, LAST_SEEN, UNKNOWN, VISIBLE, Permanence,  # noqa: E402
                             PermanenceConfig, Places, Region, make_refinder, r_box, ref_patch,
                             yoloe_detect)

TRACK_STATE = {VISIBLE: "visible", HIDDEN: "hidden", CARRIED: "carried", LAST_SEEN: "last_seen"}
TRACK_EVENT = {"FOUND": "found", "PICKED_UP": "picked_up", "LOST_TRACK": "lost", "MOVED": "moved"}


def frames_from(src: Path, fps: float):
    """(t since the first frame, wall, image) from a clip directory or a frame directory."""
    if (src / "video.mp4").exists():
        times = json.loads((src / "frames.json").read_text()) if (src / "frames.json").exists() else {}
        ts, walls = times.get("t") or [], times.get("wall") or []
        cap = cv2.VideoCapture(str(src / "video.mp4"))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                return
            t = float(ts[i]) if i < len(ts) else i / fps
            yield t, (float(walls[i]) if i < len(walls) else t), img
            i += 1
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    t0 = None
    for i, f in enumerate(files):
        m = re.search(r"(\d{9,})", f.stem)
        wall = float(m.group(1)) / (1000.0 if len(m.group(1)) >= 13 else 1.0) if m else i / fps
        t0 = wall if t0 is None else t0
        yield wall - t0, wall, cv2.imread(str(f))


def zones_of(d) -> list[Region]:
    """Regions from a room_zones.json dict ({"zones": {name: {say, poly}}}) or a list of zone dicts."""
    if not d:
        return []
    z = d.get("zones", d) if isinstance(d, dict) else d
    items = z.items() if isinstance(z, dict) else ((q.get("name"), q) for q in z)
    return [Region(n, q.get("say", n), q["poly"]) for n, q in items]


def load_zones(path: str | None) -> list[Region]:
    return zones_of(json.loads(Path(path).read_text())) if path else []


def scaled(regions: list[Region], k: float) -> list[Region]:
    return [Region(r.name, r.say, [[x * k, y * k] for x, y in r.poly]) for r in regions]


class TableCm:
    """Full-frame px -> table cm through the clip's table view and calibration (None without them)."""

    def __init__(self, meta: dict, rect, out=(1280, 720)):
        self.table, self.rect, self.out = None, rect, out
        cal = meta.get("table_cal")
        if not cal or rect is None:
            return
        from core.config import load_config
        from core.table import Table
        cfg = meta.get("config") or load_config()        # the recording's table settings (tag mode, size)
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(cal, f)
        f.close()
        try:
            t = Table(cfg, cal_path=f.name)
            self.table = t if getattr(t, "ok", False) else None
        except Exception:
            self.table = None
        finally:
            os.unlink(f.name)

    def __call__(self, box):
        if self.table is None or box is None:
            return None
        x1, y1, x2, y2 = self.rect
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        tv = ((cx - x1) * self.out[0] / (x2 - x1), (cy - y1) * self.out[1] / (y2 - y1))
        cm = np.asarray(self.table.px_to_cm([tv]), float).reshape(-1)
        return [round(float(cm[0]), 1), round(float(cm[1]), 1)]


def annotate(img: np.ndarray, p: Permanence) -> np.ndarray:
    out = img.copy()
    for r in p.places.regions:
        cv2.polylines(out, [np.array(r.poly, np.int32).reshape(-1, 1, 2)], True, (0, 200, 0), 3)
    for ks in p._cands.values():
        for k in ks:
            cv2.rectangle(out, k.box[:2], k.box[2:], (0, 160, 255) if not k.in_person else (120, 120, 120), 1)
    for ps in p._people.values():
        for b in ps:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 2)
    for o in p.objects.values():
        if o.box is not None:
            cv2.rectangle(out, o.box[:2], o.box[2:], (0, 0, 255), 4)
            cv2.putText(out, f"{o.name}: {o.state} ({o.say})", (o.box[0], max(30, o.box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return out


def teach_on_cues(p: Permanence, truth: dict, t: float, wall: float, img, taught: set, settle_s: float = 4.0) -> None:
    """At each truth 'place' cue (+ settle_s), register the prop as the newest unclaimed arrival on the table."""
    props = truth.get("props") or {}
    rect = p.places.table_rect
    for s in truth.get("steps") or []:
        obj = s.get("obj")
        if s.get("event") != "place" or obj not in props or obj in taught or t < float(s["t"]) + settle_s:
            continue
        claimed = [o.box for o in p.objects.values() if o.box is not None and o.state in (VISIBLE, HIDDEN)]
        cue_wall = wall - (t - float(s["t"]))
        new = [k for ks in p._cands.values() for k in ks if k.first_wall >= cue_wall - 0.5 and not k.in_person
               and (rect is None or rect[0] <= geom.center(k.box)[0] <= rect[2] and rect[1] <= geom.center(k.box)[1] <= rect[3])
               and not any(geom.iou(k.box, b) >= 0.5 for b in claimed)]
        if new:
            k = max(new, key=lambda q: (q.first_wall, q.conf))
            if p.teach(str(props[obj]), img, k.box):
                taught.add(obj)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", required=True, help="clip directory (video.mp4 ...) or frame directory")
    ap.add_argument("--refs", default=None, help="registry reference crops: <dir>/<name>/*.jpg")
    ap.add_argument("--enroll", nargs=6, action="append", default=[], metavar=("NAME", "IMAGE", "X1", "Y1", "X2", "Y2"),
                    help="add a reference crop of NAME from IMAGE's box (repeatable)")
    ap.add_argument("--teach-on-cues", action="store_true", help="register each prop at its truth 'place' cue")
    ap.add_argument("--zones", default=None, help="room_zones.json (default: the clip's meta.json)")
    ap.add_argument("--zones-scale", type=float, default=1.0)
    ap.add_argument("--table-rect", type=float, nargs=4, default=None, help="default: the clip's meta.json view")
    ap.add_argument("--yoloe", default=str(Path.home() / "asktheroom/ws8-data/models/yoloe-26s-seg-pf.pt"))
    ap.add_argument("--reid", default=str(Path.home() / "asktheroom/ws8-data/models/dinov2_s14.onnx"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fps", type=float, default=2.0, help="frame rate when file names carry no time")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame (the live loop's rate)")
    ap.add_argument("--sweep", action="store_true", help="every view on every processed frame")
    ap.add_argument("--config", default=None, help="JSON of permanence: overrides")
    ap.add_argument("--verify", action="store_true", help="ask Grok (needs XAI_API_KEY)")
    ap.add_argument("--track", default=None, help="write askroom-track/1 JSONL here")
    ap.add_argument("--out", default=None, help="JSONL of per-frame object states (debug)")
    ap.add_argument("--annotate", default=None, help="write annotated frames here (every processed frame)")
    a = ap.parse_args(argv)

    src = Path(a.frames)
    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    truth = json.loads((src / "truth.json").read_text()) if (src / "truth.json").exists() else {}
    from ultralytics import YOLO
    from core.embed import Embedder, ReidConfig
    model = YOLO(a.yoloe)
    if hasattr(model, "to"):
        model.to(a.device)
    emb = Embedder(ReidConfig(enabled=True, model=a.reid, providers=["cpu"], threads=8, background=False))
    emb.load()
    raw = {"mode": "registry", "zoom": "zones", **(json.loads(a.config) if a.config else {})}
    c = PermanenceConfig.from_dict(raw)
    regions = scaled(load_zones(a.zones), a.zones_scale) if a.zones else zones_of(meta.get("room_zones"))
    view = meta.get("view") or {}
    rect = tuple(a.table_rect) if a.table_rect else tuple(view.get("table_view_rect") or view.get("rect") or ()) or None
    first = next(frames_from(src, a.fps))[2]
    h, w = first.shape[:2]
    places = Places(regions, rect, c.table_say, (w, h), c.near_px)
    zoom = [r_box(r) for r in regions] if c.zoom == "zones" else list(c.zoom or [])
    clock = {"wall": 0.0}
    p = Permanence(c, yoloe_detect(model, c.imgsz, c.conf), emb.batch, places=places,
                   refind=make_refinder({}, c) if a.verify else None, zoom_boxes=zoom,
                   clock=lambda: clock["wall"])
    if a.refs:
        p.enroll_dir(a.refs)
    for name, image, *box in a.enroll:
        img = cv2.imread(image)
        box = tuple(int(float(v)) for v in box)
        ok = p.add_ref(name, img[box[1]:box[3], box[0]:box[2]].copy(), (emb.batch(img, [box]) or [None])[0],
                       ref_patch(img, box))
        print(f"enrolled {name}: {ok}", file=sys.stderr)
    to_cm = TableCm(meta, rect)
    track = open(a.track, "w") if a.track else None
    out = open(a.out, "w") if a.out else None
    if track:
        track.write(json.dumps({"type": "header", "format": "askroom-track/1", "clip": meta.get("clip", src.name),
                                "tracker": "registry", "detector": f"yoloe {Path(a.yoloe).name} + dinov2",
                                "overrides": ["permanence.mode=registry"] + (["teach_on_cues"] if a.teach_on_cues else [])
                                + ([f"permanence={json.dumps(raw)}"] if a.config else []),
                                "view": [list(rect), [1280, 720]] if rect else None, "room_memory": False,
                                "refs": "teach_on_cues" if a.teach_on_cues else (a.refs or "enroll")}) + "\n")
    if a.annotate:
        Path(a.annotate).mkdir(parents=True, exist_ok=True)
    born, taught, n, ms, t0w = {}, set(), 0, [], time.perf_counter()
    for i, (t, wall, img) in enumerate(frames_from(src, a.fps)):
        if img is None or i % max(1, a.every):
            continue
        clock["wall"] = wall
        t0 = time.perf_counter()
        evs = p.sweep(img, wall) if a.sweep else p.step(img, wall)
        ms.append(1000 * (time.perf_counter() - t0))
        if a.teach_on_cues and truth:
            teach_on_cues(p, truth, t, wall, img, taught)
        ents = []
        for o in p.objects.values():
            if o.state == UNKNOWN:
                continue
            born.setdefault(o.name, t)
            zone = o.zone if o.zone and ":" not in o.zone else None
            e = {"id": o.name, "state": TRACK_STATE.get(o.state, "unknown"), "zone": zone, "place": o.say,
                 "table_cm": to_cm(o.box) if zone == "table" else None}
            if o.state == HIDDEN:
                e["hidden_in"] = "occluded"
            ents.append(e)
        if track:
            for ev in evs:
                track.write(json.dumps({"type": "event", "t": round(t, 3), "id": ev.obj,
                                        "event": TRACK_EVENT.get(ev.type, ev.type.lower())}) + "\n")
            track.write(json.dumps({"type": "frame", "t": round(t, 3), "entities": ents}) + "\n")
        if out:
            out.write(json.dumps({"t": t, "entities": ents}) + "\n")
        for ev in evs:
            print(f"{t:7.2f} {ev.obj} {ev.type} -> {p.objects[ev.obj].say}", file=sys.stderr)
        if a.annotate:
            cv2.imwrite(str(Path(a.annotate) / f"{n:05d}.jpg"), annotate(img, p))
        n += 1
    if track:
        for name, b in born.items():
            track.write(json.dumps({"type": "entity", "id": name, "kind": "registered", "born_t": round(b, 3),
                                    "merged_into": None, "names": [name.replace("_", " ")]}) + "\n")
        track.close()
    if out:
        out.close()
    summary = {"frames": n, "step_ms_median": float(np.median(ms)) if ms else None, "views": len(p._views),
               "wall_s": round(time.perf_counter() - t0w, 1), "entities": sorted(p.objects),
               "states": {k: o.state for k, o in p.objects.items()}, "places": {k: o.say for k, o in p.objects.items()},
               "taught": sorted(taught), "candidates_last": sum(len(v) for v in p._cands.values()),
               "people_last": sum(len(v) for v in p._people.values())}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
