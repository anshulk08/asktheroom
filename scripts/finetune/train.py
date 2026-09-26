"""Fine-tune YOLO11s on the labelled overhead frames (spec P7 step 4). Run on a cloud GPU.

    python scripts/finetune/train.py --data data/finetune                 # 20% of videos held out
    python scripts/finetune/train.py --val-trials 3 11 17 --epochs 100
    python scripts/finetune/train.py --split-only                         # just write the split

Colab: upload data/finetune (images/, labels/) and this folder, then
    !pip install ultralytics
    !python scripts/finetune/train.py --data data/finetune --device 0

Validation holds out whole videos, never random frames: neighbouring frames of one video are near
copies, so a random split would score the model on frames it has effectively trained on.
Output: runs/askroom/<name>/weights/best.pt. Copy it to models/ on the Jetson and build the engine in
the container (see README.md: `core.detect --export` calls YOLO-World's set_classes, which a YOLO11
model doesn't have), then point detect.model at the .engine.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # run as a script or imported by tests
from common import DEFAULT_DATA, class_names, trial_of  # noqa: E402


TRAIN_ONLY = {"synth", "pubhand"}   # composites and public hand frames: always trained on, never validated on


def split_by_trial(stems: list[str], val_frac: float = 0.2, val_trials=None,
                   seed: int = 0) -> tuple[list[str], list[str]]:
    """Frame stems -> (train, val), whole trials to one side. With no val_trials, shuffles the trials
    (seeded) and moves them to val until about val_frac of the frames are there (at least one).
    Synthetic composites (trial 'synth') always go to train and don't count toward val_frac."""
    synth = sorted(s for s in stems if trial_of(s) in TRAIN_ONLY)
    stems = [s for s in stems if trial_of(s) not in TRAIN_ONLY]
    by = defaultdict(list)
    for s in stems:
        by[trial_of(s)].append(s)
    trials = sorted(by)
    if val_trials:
        val = {str(t) for t in val_trials}
        missing = val - set(trials)
        if missing:
            raise ValueError(f"no frames from trials {sorted(missing)}")
    else:
        if len(trials) < 2:
            raise ValueError("need frames from at least two videos to hold one out")
        order = trials[:]
        random.Random(seed).shuffle(order)
        val, n = set(), 0
        for t in order:
            if n >= val_frac * len(stems) or len(val) == len(trials) - 1:
                break
            val.add(t)
            n += len(by[t])
    tr = [s for t in trials if t not in val for s in sorted(by[t])]
    va = [s for t in trials if t in val for s in sorted(by[t])]
    return tr + synth, va


def write_split(data: Path, names: list[str], train: list[str], val: list[str]) -> Path:
    import yaml
    data = Path(data).resolve()
    for name, stems in (("train", train), ("val", val)):
        (data / f"{name}.txt").write_text("".join(f"{data / 'images' / s}.jpg\n" for s in stems))
    p = data / "dataset.yaml"
    p.write_text(yaml.safe_dump({"path": str(data), "train": "train.txt", "val": "val.txt",
                                 "names": dict(enumerate(names))}, sort_keys=False))
    return p


def main(argv=None) -> int:
    from core.config import load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for the first GPU; default lets ultralytics pick")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-trials", nargs="*", help="hold out exactly these trial ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="askroom-yolo11s")
    ap.add_argument("--split-only", action="store_true")
    a = ap.parse_args(argv)

    data = Path(a.data)
    labelled = {p.stem for p in (data / "labels").glob("*.txt")}
    stems = sorted(p.stem for p in (data / "images").glob("*.jpg") if p.stem in labelled)
    if not stems:
        print(f"no labelled frames in {data}/images + {data}/labels", file=sys.stderr)
        return 1
    train, val = split_by_trial(stems, a.val_frac, a.val_trials, a.seed)
    names = class_names(load_config())
    ds = write_split(data, names, train, val)
    print(f"train {len(train)} frames / {len({trial_of(s) for s in train})} videos, "
          f"val {len(val)} frames / videos {sorted({trial_of(s) for s in val})}  -> {ds}")
    if a.split_only:
        return 0

    from ultralytics import YOLO
    model = YOLO(a.model)
    # Overhead camera: up/down flips are as plausible as left/right. No mosaic in the last epochs so
    # the model finishes on whole-table views like the ones it will see.
    res = model.train(data=str(ds), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                      project="runs/askroom", name=a.name, seed=a.seed, flipud=0.5, fliplr=0.5,
                      close_mosaic=10, patience=30, exist_ok=True)
    best = Path(res.save_dir) / "weights" / "best.pt"
    got = list(YOLO(str(best)).names.values())
    if got != names:
        print(f"WARNING: model classes {got} != {names}; core/detect.py maps classes by name")
    print(f"best weights: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
