"""Record a raw camera clip for replay (eval/guided.py drives it; eval/score_clip.py scores it).

    python -m eval.raw_record --out data/clips/<id> --device /dev/v4l/by-id/... --seconds 40

Runs in the Jetson container (scripts/dock.sh: OpenCV with ffmpeg). Writes <out>/video.mp4 (raw
1280x720 frames, no overlay), <out>/frames.json ({"wall": [...], "t": [...]}: per-frame wall time and
seconds since the first frame) and <out>/meta.json (camera device and controls, calibration, the
effective config, models, git commit). <out>/READY appears when the first frame is written, so the
driver can start its cues; <out>/STOP ends the recording early.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import datetime
import json
import time
from pathlib import Path

W, H = 1280, 720


def meta_for(cfg: dict, device: str, controls: dict, git: str, clock_offset_s: float = 0.0) -> dict:
    """What a replay needs to reproduce the recording's production setup."""
    p = (cfg.get("proposals") or {})
    kind = p.get("kind", "change") if p.get("enabled") else None
    cal_path = Path((cfg.get("paths") or {}).get("table_cal", "table_cal.json"))
    try:
        table_cal = json.loads(cal_path.read_text())
    except (OSError, ValueError):
        table_cal = None
    return {"camera": {"device": device, "controls": controls}, "table_cal": table_cal, "config": cfg,
            "git": git, "models": {"detect": (cfg.get("detect") or {}).get("model"),
                                   "proposals": {"kind": kind,
                                                 "model": (p.get(kind) or {}).get("model") if kind else None}},
            "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "clock_offset_s": clock_offset_s}


def record(out: Path, device, seconds: float, meta: dict, fps: float = 30.0) -> dict:
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(device if not str(device).isdigit() else int(device), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    t_end = time.time() + 3.0
    while time.time() < t_end:                      # the camera settles for a few seconds after opening
        cap.read()
    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    walls: list[float] = []
    stop = out / "STOP"
    t_stop = None
    while True:
        ok, img = cap.read()
        now = time.time()
        if not ok:
            continue
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        writer.write(img)
        walls.append(now)
        if len(walls) == 1:
            t_stop = now + seconds
            (out / "READY").write_text(f"{now:.6f}\n")
        if now >= t_stop or stop.exists():
            break
    writer.release()
    cap.release()
    frames = {"wall": walls, "t": [round(w - walls[0], 6) for w in walls]}
    (out / "frames.json").write_text(json.dumps(frames))
    (out / "meta.json").write_text(json.dumps(dict(meta, clip=out.name), indent=1, default=str))
    return {"frames": len(walls), "seconds": walls[-1] - walls[0], "fps": (len(walls) - 1) / max(1e-6, walls[-1] - walls[0])}


def main(argv=None) -> int:
    from core.config import load_config
    from core.table import apply_saved_size
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--controls", default="{}", help="JSON of the camera controls read on the host")
    ap.add_argument("--git", default="")
    ap.add_argument("--clock-offset", type=float, default=0.0)
    a = ap.parse_args(argv)
    cfg = apply_saved_size(load_config())
    meta = meta_for(cfg, a.device, json.loads(a.controls), a.git, a.clock_offset)
    r = record(Path(a.out), a.device, a.seconds, meta)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
