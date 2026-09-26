"""Live Grok check of automatic naming (core/auto_name.py): crops of a desk photo, one real call each,
printing the guessed name, alternatives, confidence and latency. Not part of the test suite.

    set -a && . ./.env && set +a && .venv/bin/python scripts/auto_name_smoke.py
    .venv/bin/python scripts/auto_name_smoke.py --image photo.jpg --box 50 165 160 272 --box ...

The default boxes are objects in the stock desk photo experiments/openset/frames/proxy_getty_a.jpg
(pixels of that 612 x 459 image). Does nothing (exit 0) when XAI_API_KEY is not set.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auto_name import AutoNameConfig, AutoNamer  # noqa: E402
from core.config import ROOT, load_config  # noqa: E402

DEFAULT_BOXES = {                     # proxy_getty_a.jpg
    "coffee cup lid": (50, 165, 160, 272),
    "camera": (230, 90, 322, 238),
    "wallet": (412, 92, 540, 208),
    "car key": (220, 265, 340, 365),
    "wooden spoon": (65, 338, 195, 368),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", default=str(ROOT / "experiments/openset/frames/proxy_getty_a.jpg"))
    ap.add_argument("--box", nargs=4, type=int, action="append", metavar=("X1", "Y1", "X2", "Y2"))
    a = ap.parse_args(argv)
    if not os.environ.get("XAI_API_KEY"):
        print("XAI_API_KEY not set; skipping live auto-name smoke test.")
        return 0
    import cv2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"can't read {a.image}")
    boxes = {f"box {i + 1}": tuple(b) for i, b in enumerate(a.box)} if a.box else DEFAULT_BOXES
    namer = AutoNamer(load_config(), world=None, online=lambda: True, c=AutoNameConfig(enabled=True), start=False)
    print(f"{namer.provider.name} {namer.provider.model}")
    lat = []
    for what, (x1, y1, x2, y2) in boxes.items():
        mx, my = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
        crop = img[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx]
        t0 = time.perf_counter()
        try:
            g = namer._ask(crop)
        except Exception as ex:                       # the key is never printed, only the error class
            g = f"failed: {type(ex).__name__}"
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        print(f"[{ms:6.0f} ms] {what:>14}: {g}")
    lat.sort()
    print(f"\nmedian {lat[len(lat) // 2]:.0f} ms, max {lat[-1]:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
