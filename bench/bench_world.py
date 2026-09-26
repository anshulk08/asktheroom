"""Time World.update on rendered 1280x720 frames: 8 objects, 1 hand, 300 updates."""
import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from core.config import Config
from core.world import World
from tests.synth import Scene

LAYOUT = {'keys': (15, 15), 'pill_bottle': (35, 15), 'wallet': (55, 15), 'glasses': (75, 15),
          'phone': (95, 15), 'remote': (115, 15), 'box': (30, 50), 'notebook': (90, 50)}

def run(label, keys_missed_from=None, n=300):
    cfg = Config.load(sys.argv[1] + '/config.yaml')
    s, w = Scene(cfg, render=True), World(cfg)
    for name, at in LAYOUT.items():
        s.place(name, *at)
    batches = []
    for i in range(n + 60):                                  # 60 warm-up updates
        for k, name in enumerate(LAYOUT):                    # detector misses ~1 batch in 4
            s.miss(name, (i + k) % 4 == 0 or (keys_missed_from is not None and name == 'keys' and i >= keys_missed_from))
        s.hand(1, 60 + 10 * ((i // 20) % 3), 34)
        batches.append(s.step())
    normal, refresh, events = [], [], []
    for i, (d, f) in enumerate(batches):
        before = w._bg_t
        t0 = time.perf_counter()
        events += w.update(d, f)
        dt = (time.perf_counter() - t0) * 1000
        if i >= 60:
            (refresh if w._bg_t != before else normal).append(dt)
    def stats(x):
        x = np.array(x); return f'n={len(x):3d} mean={x.mean():6.3f} ms  p95={np.percentile(x, 95):6.3f} ms  max={x.max():6.3f} ms'
    print(f'{label}\n  normal : {stats(normal)}\n  refresh: {stats(refresh)}\n  events: {[(e.obj, e.type.value) for e in events]}')

run('A. static scene, per-object detector flicker 1 in 4')
run('B. as A, plus keys missed by the detector from update 100 on (pixels still there)', keys_missed_from=100)
