"""ChangeProposer on the live dashboard stream (read-only: http://localhost:8080/frame.jpg, the Jetson
app keeps the camera). Captures a reference, then runs for --seconds, logging every proposal and
saving annotated frames (green = proposal, red = rejected region with its reason, grey = masked).

    .venv/bin/python experiments/proposals/live_sanity.py --seconds 60 --tag empty
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.proposals import ChangeProposer  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'
LEGEND = [0, 0, 240, 175]        # dashboard overlay, top-left of the 960x540 stream


def grab(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)


def draw(img, props, p):
    out = img.copy()
    x1, y1, x2, y2 = LEGEND
    cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), 2)
    for (bx1, by1, bx2, by2), why in p.debug.get('rejected', []):
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 255), 1)
        cv2.putText(out, why, (bx1, max(10, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for q in props:
        bx1, by1, bx2, by2 = q.box_px
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(out, f'thing {q.conf:.2f}', (bx1, max(10, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    fg = p.debug.get('fg')
    if fg is not None:
        m = cv2.resize(fg.astype(np.uint8) * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        out[m > 0] = (0.6 * out[m > 0] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:8080/frame.jpg')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--fps', type=float, default=5)
    ap.add_argument('--tag', default='live')
    ap.add_argument('--work', type=int, default=480)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    p = ChangeProposer({'ref_frames': 10, 'work_px': a.work, 'ignore_px': [LEGEND]})
    log, saved, t_end, last = [], 0, None, None
    while True:
        t0 = time.monotonic()
        try:
            img = grab(a.url)
        except Exception as ex:
            print('grab failed:', ex)
            time.sleep(0.5)
            continue
        props = p.propose(img, [], [])
        if p.ready and t_end is None:
            t_end = time.monotonic() + a.seconds
            p.save_reference(str(OUT / f'{a.tag}_reference.png'))
            cv2.imwrite(str(OUT / f'{a.tag}_reference_frame.jpg'), img)
        if t_end is not None and p.debug:
            rec = {'t': round(time.monotonic(), 2), 'ms': round(p.last_ms, 2), 'noise': p.debug['noise'],
                   'offset': [float(v) for v in p.debug['offset']], 'changed': p.debug['changed_frac'],
                   'props': [[list(q.box_px), q.conf] for q in props],
                   'rejected': [[list(b), r] for b, r in p.debug['rejected']]}
            log.append(rec)
            if props and saved < 12:
                cv2.imwrite(str(OUT / f'{a.tag}_prop_{saved:02d}.jpg'), draw(img, props, p))
                saved += 1
            last = (img, props)
            if time.monotonic() >= t_end:
                break
        time.sleep(max(0.0, 1.0 / a.fps - (time.monotonic() - t0)))
    if last is not None:
        cv2.imwrite(str(OUT / f'{a.tag}_last.jpg'), draw(last[0], last[1], p))
    (OUT / f'{a.tag}_log.json').write_text(json.dumps(log, indent=0))
    n = len(log)
    with_props = sum(bool(r['props']) for r in log)
    ms = [r['ms'] for r in log]
    print(f'{n} frames after the reference; {with_props} with proposals '
          f'({sum(len(r["props"]) for r in log)} total); proposer median {np.median(ms):.2f} ms at '
          f'{img.shape[1]}x{img.shape[0]}; noise L/C median {np.median([r["noise"][0] for r in log]):.2f}/'
          f'{np.median([r["noise"][1] for r in log]):.2f}; changed frac max {max(r["changed"] for r in log):.3f}')
    reasons = {}
    for r in log:
        for _, why in r['rejected']:
            reasons[why] = reasons.get(why, 0) + 1
    print('rejected regions by reason:', reasons)
    for r in log:
        if r['props']:
            print('  props', r['props'])


if __name__ == '__main__':
    main()
