"""How often the detector names each object and hand in a clip, at each confidence cut-off: can a
threshold alone recover a class, or does it need retraining?

    python -m eval.conf_sweep data/clips/<id>                          # the configured detector
    python -m eval.conf_sweep data/clips/<id> --detect-model models/askroom-yolo26s-brio.engine
    python -m eval.conf_sweep data/clips/<id> --json out/sweep_<id>.json

On the Jetson, inside the app's container (scripts/dock.sh), with the app stopped: it loads the detector.

Frames are taken at --max-fps by video time (main.perception_max_fps, as eval.score_clip replays). For
each class it reports the share of frames with a box at least each cut-off sure (the class's best box in
the frame), and for hands also how many of the clip's truth steps had a hand that sure within step_s of
the step (a hand the world never sees can't hold, cover or put anything inside).
"""
from __future__ import annotations

import argparse
import json
from typing import Iterable, Optional

from core.types import Frame

CUTS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)


def paced(frames: Iterable[Frame], max_fps: float) -> Iterable[Frame]:
    """The frames a live loop capped at max_fps would take, by video time."""
    last = None
    for f in frames:
        if max_fps <= 0 or last is None or f.t - last >= 1.0 / max_fps - 1e-6:
            last = f.t
            yield f


def best_confs(raw, to_obj: dict) -> dict[str, float]:
    """The surest box per object (prompt labels mapped back) in one frame's raw detections."""
    out: dict[str, float] = {}
    for label, conf, _ in raw:
        obj = to_obj.get(label)
        if obj is not None and conf > out.get(obj, 0.0):
            out[obj] = float(conf)
    return out


def sweep(per_frame: list[tuple[float, dict]], objects: list[str], steps: list[dict],
          cuts=CUTS, step_s: float = 1.0) -> dict:
    """per_frame: (t, {obj: best conf}) per processed frame. Rates per class per cut, and hands at steps."""
    n = max(len(per_frame), 1)
    rates = {obj: {c: round(sum(1 for _, b in per_frame if b.get(obj, 0.0) >= c) / n, 3) for c in cuts}
             for obj in objects + ["hand"]}
    at_steps = {}
    for c in cuts:
        hit = 0
        for s in steps:
            t = float(s.get("t", 0.0))
            hit += any(b.get("hand", 0.0) >= c for ft, b in per_frame if abs(ft - t) <= step_s)
        at_steps[c] = hit
    return {"frames": len(per_frame), "cuts": list(cuts), "rates": rates,
            "hand_at_steps": at_steps, "steps": len(steps)}


def report(r: dict) -> str:
    cuts = r["cuts"]
    lines = [f"{r['frames']} frames; share of frames with a box at least this sure",
             "class         " + " ".join(f"{c:>5}" for c in cuts)]
    for obj, row in r["rates"].items():
        lines.append(f"{obj:13s} " + " ".join(f"{100 * row[c]:5.0f}" for c in cuts))
    lines.append(f"hand at steps " + " ".join(f"{r['hand_at_steps'][c]:>5}" for c in cuts)
                 + f"   (of {r['steps']} steps, within ±step_s)")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    from core.config import load_config
    from core.detect import UltralyticsBackend, class_list
    from eval.clip import load_clip
    ap = argparse.ArgumentParser(description="detector hit rate per class per confidence cut-off")
    ap.add_argument("clip")
    ap.add_argument("--config")
    ap.add_argument("--detect-model")
    ap.add_argument("--max-fps", type=float)
    ap.add_argument("--step-s", type=float, default=1.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg.setdefault("detect", {})["min_conf"] = min(CUTS)
    clip = load_clip(a.clip)
    backend = UltralyticsBackend(cfg, a.detect_model)
    _, to_obj = class_list(cfg)
    fps = a.max_fps if a.max_fps is not None else float((cfg.get("main") or {}).get("perception_max_fps", 15))
    per_frame = [(f.t, best_confs(backend.infer(f.img), to_obj)) for f in paced(clip.frames(), fps)]
    r = sweep(per_frame, list(cfg.get("objects") or {}), clip.truth["steps"], step_s=a.step_s)
    r.update(clip=clip.name, model=backend.info)
    print(f"{clip.name}: {backend.info['path']}")
    print(report(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
