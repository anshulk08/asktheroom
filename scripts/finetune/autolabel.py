"""Pre-label the extracted frames with YOLO-World (spec P7 step 2), for hand-fixing afterwards.

    python scripts/finetune/autolabel.py                     # data/finetune/images -> labels/
    python scripts/finetune/autolabel.py --model models/yolov8s-worldv2.pt --conf 0.15

Uses the same prompts and label -> object mapping as core.detect (class_list), so a prompt like
'medicine bottle' becomes pill_bottle. Output classes are exactly the config objects in order, then
hand. Keeps the best box per object (each exists once on the table) and every hand box. Writes
YOLO txt labels (class cx cy w h, normalized) and data.yaml with the class names.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # run as a script or imported by tests
from common import DEFAULT_DATA, ROOT, class_names, write_data_yaml  # noqa: E402

Box = tuple[float, float, float, float]


def to_labels(raw: list[tuple[str, float, Box]], to_obj: dict[str, str], names: list[str],
              w: int, h: int, conf: float) -> list[str]:
    """(label, conf, xyxy px) detections -> YOLO lines. Best box per object, all hands."""
    idx = {n: i for i, n in enumerate(names)}
    best: dict[str, tuple[float, Box]] = {}
    hands: list[Box] = []
    for label, c, box in raw:
        obj = to_obj.get(label)
        if obj is None or c < conf:
            continue
        if obj == "hand":
            hands.append(box)
        elif obj not in best or c > best[obj][0]:
            best[obj] = (c, box)
    out = []
    for obj, box in [(o, b) for o, (_, b) in best.items()] + [("hand", b) for b in hands]:
        x1, y1, x2, y2 = (max(0.0, min(v, lim)) for v, lim in zip(box, (w, h, w, h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(f"{idx[obj]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                   f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def main(argv=None) -> int:
    import cv2

    from core.config import load_config
    from core.detect import UltralyticsBackend, class_list
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", help="default: config detect.world_pt_baked")
    ap.add_argument("--conf", type=float, default=0.2, help="lower = more boxes to delete, fewer to draw")
    ap.add_argument("--overwrite", action="store_true", help="replace labels already fixed by hand")
    a = ap.parse_args(argv)

    cfg = load_config()
    model = a.model or str(ROOT / (cfg.get("detect") or {}).get("world_pt_baked", "models/yolov8s-worldv2-askroom.pt"))
    cfg = dict(cfg, detect={**(cfg.get("detect") or {}), "half": False, "min_conf": min(a.conf, 0.05)})
    backend = UltralyticsBackend(cfg, model)
    _, to_obj = class_list(cfg)
    names = class_names(cfg)
    data = Path(a.data)
    labels = data / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    imgs = sorted((data / "images").glob("*.jpg"))
    counts = dict.fromkeys(names, 0)
    skipped = 0
    for i, p in enumerate(imgs, 1):
        out = labels / f"{p.stem}.txt"
        if out.exists() and not a.overwrite:
            skipped += 1
            continue
        img = cv2.imread(str(p))
        lines = to_labels(backend.infer(img), to_obj, names, img.shape[1], img.shape[0], a.conf)
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        for ln in lines:
            counts[names[int(ln.split()[0])]] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(imgs)}")
    write_data_yaml(data, names)
    print(f"labelled {len(imgs) - skipped} frames ({skipped} kept as they were) in {labels}")
    for n in names:
        print(f"  {n:12s} {counts[n]:5d} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
