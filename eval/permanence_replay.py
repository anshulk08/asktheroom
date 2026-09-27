"""Replay raw full camera frames through the object registry (spec 0011, core/permanence.py) offline.

Input: a directory of raw full frames (JPEG/PNG, sorted by name; the wall time is the number in the file name
when there is one, else --fps), or a guided clip directory (eval/clip.py: video.mp4 + frames.json). Output: one
JSON line per object per frame (the tracker-agnostic rows WS2's scorecard reads: t, entity, state, place, table
cm), then a summary line, and optionally annotated frames (candidates, people, found objects, places).

Needs ultralytics (YOLOE) and onnxruntime (DINOv2): run it with the training venv, e.g.
  ~/asktheroom/train-venv/bin/python -m eval.permanence_replay --frames stills/ --refs data/registry \\
      --zones room_zones.json --table-rect 0 980 817 1440 --out run.jsonl --annotate out/
Grok questions need XAI_API_KEY (and --verify); without them only the appearance rules run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.permanence import (Permanence, PermanenceConfig, Places, Region, grok_verify, r_box,  # noqa: E402
                             yoloe_detect)


def frames_from(src: Path, fps: float):
    """(wall, image) pairs from a frame directory or a clip directory."""
    if (src / "video.mp4").exists():
        walls = json.loads((src / "frames.json").read_text()).get("wall") if (src / "frames.json").exists() else None
        cap = cv2.VideoCapture(str(src / "video.mp4"))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                return
            yield (walls[i] if walls and i < len(walls) else i / fps), img
            i += 1
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i, f in enumerate(files):
        m = re.search(r"(\d{9,})", f.stem)
        wall = float(m.group(1)) / (1000.0 if len(m.group(1)) >= 13 else 1.0) if m else i / fps
        yield wall, cv2.imread(str(f))


def load_zones(path: str | None) -> list[Region]:
    if not path:
        return []
    d = json.loads(Path(path).read_text())
    return [Region(n, z.get("say", n), z["poly"]) for n, z in (d.get("zones") or {}).items()]


def scaled(regions: list[Region], k: float) -> list[Region]:
    return [Region(r.name, r.say, [[x * k, y * k] for x, y in r.poly]) for r in regions]


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", required=True, help="frame directory or clip directory")
    ap.add_argument("--refs", default=None, help="registry reference crops: <dir>/<name>/*.jpg")
    ap.add_argument("--enroll", nargs=6, action="append", default=[], metavar=("NAME", "IMAGE", "X1", "Y1", "X2", "Y2"),
                    help="add a reference crop of NAME from IMAGE's box (repeatable)")
    ap.add_argument("--zones", default=None, help="room_zones.json (full-frame px)")
    ap.add_argument("--zones-scale", type=float, default=1.0, help="scale zone polygons (drawn at another size)")
    ap.add_argument("--table-rect", type=float, nargs=4, default=None)
    ap.add_argument("--yoloe", default=str(Path.home() / "asktheroom/ws8-data/models/yoloe-26s-seg-pf.pt"))
    ap.add_argument("--reid", default=str(Path.home() / "asktheroom/ws8-data/models/dinov2_s14.onnx"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fps", type=float, default=2.0, help="frame rate when file names carry no time")
    ap.add_argument("--config", default=None, help="JSON of permanence: overrides")
    ap.add_argument("--verify", action="store_true", help="ask Grok (needs XAI_API_KEY)")
    ap.add_argument("--out", default=None, help="JSONL of per-frame object states")
    ap.add_argument("--annotate", default=None, help="write annotated frames here")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    from core.embed import Embedder, ReidConfig
    model = YOLO(a.yoloe)
    model.to(a.device) if hasattr(model, "to") else None
    emb = Embedder(ReidConfig(enabled=True, model=a.reid, providers=["cpu"], threads=8, background=False))
    emb.load()
    raw = {"mode": "registry", "zoom": "zones", **(json.loads(a.config) if a.config else {})}
    c = PermanenceConfig.from_dict(raw)
    regions = scaled(load_zones(a.zones), a.zones_scale)
    first = next(frames_from(Path(a.frames), a.fps))[1]
    h, w = first.shape[:2]
    places = Places(regions, tuple(a.table_rect) if a.table_rect else None, c.table_say, (w, h), c.near_px)
    zoom = [r_box(r) for r in regions] if c.zoom == "zones" else list(c.zoom or [])
    p = Permanence(c, yoloe_detect(model, c.imgsz, c.conf), emb.batch, places=places,
                   verify=grok_verify({}, c.verify_timeout_s) if a.verify else None, zoom_boxes=zoom)
    if a.refs:
        p.enroll_dir(a.refs)
    for name, image, *box in a.enroll:
        img = cv2.imread(image)
        box = tuple(int(float(v)) for v in box)
        ok = p.add_ref(name, img[box[1]:box[3], box[0]:box[2]].copy(), (emb.batch(img, [box]) or [None])[0])
        print(f"enrolled {name}: {ok}", file=sys.stderr)
    out = open(a.out, "w") if a.out else None
    if a.annotate:
        Path(a.annotate).mkdir(parents=True, exist_ok=True)
    n, ms = 0, []
    for wall, img in frames_from(Path(a.frames), a.fps):
        t0 = time.perf_counter()
        evs = p.sweep(img, wall)
        ms.append(1000 * (time.perf_counter() - t0))
        for ev in evs:
            print(f"{wall:.1f} {ev.obj} {ev.type}", file=sys.stderr)
        rows = [{"t": wall, "entity": o.name, "state": o.state, "place": o.say, "table_cm": None,
                 "box_px": list(o.box) if o.box else None, "how": o.how} for o in p.objects.values()]
        if out:
            for r in rows:
                out.write(json.dumps(r) + "\n")
        if a.annotate:
            cv2.imwrite(str(Path(a.annotate) / f"{n:05d}.jpg"), annotate(img, p))
        n += 1
    summary = {"frames": n, "sweep_ms_median": float(np.median(ms)) if ms else None, "views": len(p._views),
               "entities": sorted(p.objects), "states": {k: o.state for k, o in p.objects.items()},
               "places": {k: o.say for k, o in p.objects.items()},
               "candidates_last": sum(len(v) for v in p._cands.values()),
               "people_last": sum(len(v) for v in p._people.values())}
    if out:
        out.write(json.dumps({"summary": summary}) + "\n")
        out.close()
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
