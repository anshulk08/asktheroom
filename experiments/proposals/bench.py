"""ChangeProposer ms per frame on 1280x720 synthetic frames (3 unknowns, a known object, a hand),
per working resolution. usage: .venv/bin/python experiments/proposals/bench.py"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.proposals import ChangeProposer  # noqa: E402
from tests.test_proposals import Table  # noqa: E402

for work in (320, 427, 480, 640):
    tab = Table()
    p = ChangeProposer({'work_px': work, 'ref_frames': 10})
    frames = [tab.frame() for _ in range(10)]
    for f in frames:
        p.propose(f, [], [])
    tab.things.update({'a': (300, 400, 380, 470), 'b': (800, 200, 900, 260), 'c': (600, 300, 690, 380),
                       'keys': (1000, 500, 1080, 560)})
    tab.hands = [(380, 300, 500, 420)]
    frames = [tab.frame() for _ in range(8)]
    known, hands = [(1000, 500, 1080, 560)], [(380, 300, 500, 420)]
    ms = []
    for i in range(200):
        t = time.perf_counter()
        props = p.propose(frames[i % 8], known, hands)
        ms.append(1000 * (time.perf_counter() - t))
    print(f"work_px {work}: median {np.median(ms):.2f} ms, p95 {np.percentile(ms, 95):.2f} ms, "
          f"{len(props)} proposals")
