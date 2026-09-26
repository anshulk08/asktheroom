"""Pull 400-600 varied frames from the trial recordings for labelling (spec P7 step 1).

    python scripts/finetune/extract.py                       # trials/*/video.mp4 -> data/finetune/images
    python scripts/finetune/extract.py --target 500 --every-s 0.25

Samples each video every --every-s, drops frames whose dHash is within --min-dist bits of a frame
already kept from the same video (the table sitting still), then thins evenly across videos to
--target. Frames are named <trial>_<frame>.jpg so train.py can hold out whole videos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # run as a script or imported by tests
from common import DEFAULT_DATA, ROOT, dhash, frame_name, hamming  # noqa: E402


def distinct(frames, min_dist: int) -> list:
    """Keep (idx, img) frames whose hash differs from every kept one by more than min_dist bits."""
    kept, hashes = [], []
    for idx, img in frames:
        h = dhash(img)
        if all(hamming(h, k) > min_dist for k in hashes):
            kept.append((idx, img))
            hashes.append(h)
    return kept


def thin(per_video: dict[str, list], target: int) -> dict[str, list]:
    """Evenly subsample each video so the total is about target, keeping every video represented."""
    total = sum(len(v) for v in per_video.values())
    if total <= target:
        return per_video
    out = {}
    for vid, items in per_video.items():
        n = max(1, round(len(items) * target / total))
        pick = sorted({int(round(i * (len(items) - 1) / max(n - 1, 1))) for i in range(n)})
        out[vid] = [items[i] for i in pick]
    return out


def sample(video: Path, every_s: float):
    """Yield (frame index, BGR image) every every_s seconds."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every_s * fps)))
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                return
            if i % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    yield i, img
            i += 1
    finally:
        cap.release()


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default=str(ROOT / "trials"))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--every-s", type=float, default=0.5)
    ap.add_argument("--min-dist", type=int, default=6, help="dHash bits; higher keeps fewer, more varied frames")
    a = ap.parse_args(argv)

    videos = sorted(Path(a.trials).glob("*/video.mp4"))
    if not videos:
        print(f"no videos in {a.trials}/*/video.mp4", file=sys.stderr)
        return 1
    per_video = {}
    for v in videos:
        kept = distinct(sample(v, a.every_s), a.min_dist)
        per_video[v.parent.name] = kept
        print(f"  {v.parent.name:>8s}: {len(kept)} distinct frames")
    per_video = thin(per_video, a.target)
    out = Path(a.data) / "images"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for vid, items in per_video.items():
        for idx, img in items:
            cv2.imwrite(str(out / f"{frame_name(vid, idx)}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    print(f"wrote {n} frames from {len(per_video)} videos to {out}")
    if n < 400:
        print("fewer than 400: lower --every-s or --min-dist, or record more trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
