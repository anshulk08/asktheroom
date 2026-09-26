"""Shared by the fine-tuning scripts (spec P7): class order, frame naming, near-duplicate hashing.

Dataset layout (under --data, default data/finetune):
    images/<trial>_<frame>.jpg     from extract.py
    labels/<trial>_<frame>.txt     YOLO format from autolabel.py, then hand-fixed
    data.yaml                      class names, for Label Studio / Roboflow / training
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data" / "finetune"


def class_names(cfg: dict) -> list[str]:
    """Exactly the object names in config order, then 'hand'. core/detect.py maps a fine-tuned model's
    class names to objects by name, so these must match config.yaml objects."""
    return list(cfg.get("objects") or {}) + ["hand"]


def frame_name(trial: str, idx: int) -> str:
    return f"{trial}_{idx:06d}"


def trial_of(stem: str) -> str:
    """'17_000120' -> '17' (trial ids may contain underscores; the frame number never does)."""
    return stem.rsplit("_", 1)[0]


def dhash(img: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash of a BGR or grey image: robust to noise and exposure, changes when
    objects or hands move. A neighbour must be 1 grey level brighter to set a bit, so flat table
    areas hash to 0 instead of to sensor noise."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = cv2.resize(g.astype(np.float32), (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] - s[:, :-1] > 1.0).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def write_data_yaml(data: Path, names: list[str], train="images", val="images") -> Path:
    import yaml
    p = Path(data) / "data.yaml"
    p.write_text(yaml.safe_dump({"path": str(Path(data).resolve()), "train": train, "val": val,
                                 "names": dict(enumerate(names))}, sort_keys=False))
    return p
