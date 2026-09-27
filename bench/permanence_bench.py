"""What the object registry's slow loop (spec 0011, core/permanence.py) costs on this machine: per-view time
(YOLOE + DINOv2), sweep time and memory, on a saved raw full frame, with the models config.yaml (and
config.local.yaml) name: the YOLOE the proposer uses and the reid: model (its TensorRT engine once built).

Two cases: a still room (candidates keep their embeddings) and the worst case (every candidate embedded on
every view: a busy room). Run it on the rig with the app stopped, or with it running under nice and a memory
guard (it aborts below --min-avail-mb of MemAvailable):
  nice -n 10 python -m bench.permanence_bench --image data/room/full_1440.jpg --steps 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mem_available_mb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("inf")


def rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    except Exception:
        return float("nan")


def _mb(v: float):
    return round(v) if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True, help="a raw full camera frame (e.g. 2560x1440)")
    ap.add_argument("--steps", type=int, default=60, help="views per case")
    ap.add_argument("--min-avail-mb", type=float, default=1500)
    a = ap.parse_args(argv)

    from core.config import load_config
    from core.permanence import Permanence, PermanenceConfig, load_places, r_box, registry_embedder, yoloe_detect
    from core.proposals import YOLOEProposer
    cfg = load_config()
    if mem_available_mb() < a.min_avail_mb:
        print(f"MemAvailable {mem_available_mb():.0f} MB < {a.min_avail_mb:.0f}: not starting")
        return 2
    img = cv2.imread(a.image)
    if img is None:
        print(f"can't read {a.image}")
        return 2
    t0 = time.perf_counter()
    model = YOLOEProposer((cfg.get("proposals") or {}).get("yoloe")).model
    embed = registry_embedder(cfg)
    if embed is None:
        print("no re-ID model (reid.model)")
        return 2
    load_s = time.perf_counter() - t0
    out = {"image": list(img.shape[:2][::-1]), "load_s": round(load_s, 1), "mem_avail_start_mb": _mb(mem_available_mb())}
    for case, extra in (("still", {}), ("busy", {"reuse_iou": 2.0})):
        c = PermanenceConfig.from_dict({**(cfg.get("permanence") or {}), "mode": "registry", "verify": False, **extra})
        places = load_places(cfg, c)
        zoom = [r_box(r) for r in places.regions] if c.zoom == "zones" else list(c.zoom or [])
        n_emb, ms, per_view = [], [], []

        def counted(im, boxes):
            n_emb.append(len(boxes))
            return embed(im, boxes)

        p = Permanence(c, yoloe_detect(model, c.imgsz, c.conf), counted, places=places, zoom_boxes=zoom)
        p.register("probe")
        p.add_ref("probe", img[:64, :64].copy())
        for i in range(a.steps):
            if mem_available_mb() < a.min_avail_mb:
                out["aborted"] = f"MemAvailable below {a.min_avail_mb} MB at step {i}"
                break
            t1 = time.perf_counter()
            p.step(img, 1000.0 + i * 0.1)
            ms.append(1000 * (time.perf_counter() - t1))
            per_view.append(p.embedded)
        views = len(p._views)
        out[case] = {"views": views, "view_ms_median": round(float(np.median(ms[views:] or ms)), 1),
                     "view_ms_p90": round(float(np.percentile(ms[views:] or ms, 90)), 1),
                     "sweep_ms": round(float(np.median(ms[views:] or ms)) * views, 0),
                     "embeddings_per_view": round(float(np.mean(per_view[views:] or per_view or [0])), 1),
                     "candidates": sum(len(v) for v in p._cands.values())}
    out["rss_max_mb"] = _mb(rss_mb())
    out["mem_avail_end_mb"] = _mb(mem_available_mb())
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
