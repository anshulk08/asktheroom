"""Self-contained ChangeProposer timing (no tests/ imports), for the Jetson:
copy core/{__init__,geom,types,proposals}.py next to it and run with the container's python."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if len(Path(__file__).resolve().parents) > 2:
    sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
from core.proposals import ChangeProposer  # noqa: E402

H, W = 720, 1280
rng = np.random.default_rng(0)
base = np.full((H, W, 3), (200, 210, 220), np.float32)
grain = [rng.normal(0, 3, (H, W, 3)).astype(np.float32) for _ in range(6)]


def frame(i, things=()):
    img = base + grain[i % 6]
    for (x1, y1, x2, y2), c in things:
        img[y1:y2, x1:x2] = c
    return np.clip(img, 0, 255).astype(np.uint8)


things = [((300, 400, 380, 470), (40, 80, 160)), ((800, 200, 900, 260), (30, 30, 30)),
          ((600, 300, 690, 380), (180, 60, 60)), ((1000, 500, 1080, 560), (90, 160, 90))]
empty = [frame(i) for i in range(6)]
full = [frame(i, things) for i in range(6)]
print('cv2 threads', cv2.getNumThreads())
for work in (320, 480, 640):
    p = ChangeProposer({'work_px': work, 'ref_frames': 10})
    for i in range(10):
        p.propose(empty[i % 6], [], [])
    ms = []
    for i in range(100):
        t = time.perf_counter()
        props = p.propose(full[i % 6], [(1000, 500, 1080, 560)], [(380, 300, 500, 420)])
        ms.append(1000 * (time.perf_counter() - t))
    print(f'work_px {work}: median {np.median(ms):.2f} ms, p95 {np.percentile(ms, 95):.2f} ms, {len(props)} proposals')
