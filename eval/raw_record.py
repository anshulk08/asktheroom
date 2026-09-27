"""Record a raw camera clip for replay (eval/guided.py drives it; eval/score_clip.py scores it).

    python -m eval.raw_record --out data/clips/<id> --device /dev/v4l/by-id/... --seconds 40

Runs in the Jetson container (scripts/dock.sh: OpenCV with ffmpeg). Records at the size the live app
captures (capture_size): frame_size_px (1280x720) normally, room_memory.capture_size (e.g. 2560x1440) when
room memory is on, where the app cuts the table view (table_view_rect) out of every full frame; --size
overrides it. Writes <out>/video.mp4 (raw frames at that size, no overlay, no cut), <out>/frames.json
({"wall": [...], "t": [...]}: per-frame wall time and seconds since the first frame) and <out>/meta.json
(camera device and controls, calibration, the effective config (saved tracked-area size and tabletop
outline applied, as main.build does), the view (frame size, table_view_rect), the room zones file, models,
git commit). eval/clip.clip_view reads the view back so a replay cuts exactly like the app.
<out>/READY appears when the first frame is written, so the driver can start its cues; <out>/STOP ends the
recording early. Frames are encoded on a writer thread; a frame the writer can't keep up with is dropped
(and counted in meta.json), never written with a wrong time.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import datetime
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

W, H = 1280, 720


def capture_size(cfg: dict) -> tuple[int, int]:
    """The frame size the live app captures (main.open_frames): room_memory.capture_size with room memory
    on, else frame_size_px."""
    from core.room_types import RoomConfig
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    if rc.enabled:
        return int(rc.capture_size[0]), int(rc.capture_size[1])
    w, h = cfg.get("frame_size_px") or (W, H)
    return int(w), int(h)


def _zones(cfg: dict) -> Optional[dict]:
    """The room zones file the app would load (room memory on), else None."""
    from core.room_types import RoomConfig
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    if not rc.enabled:
        return None
    try:
        return json.loads(Path(rc.zones_path).read_text())
    except (OSError, ValueError):
        return None


def meta_for(cfg: dict, device: str, controls: dict, git: str, clock_offset_s: float = 0.0,
             size_px: Optional[tuple[int, int]] = None) -> dict:
    """What a replay needs to reproduce the recording's production setup."""
    p = (cfg.get("proposals") or {})
    kind = p.get("kind", "change") if p.get("enabled") else None
    cal_path = Path((cfg.get("paths") or {}).get("table_cal", "table_cal.json"))
    try:
        table_cal = json.loads(cal_path.read_text())
    except (OSError, ValueError):
        table_cal = None
    rm = cfg.get("room_memory") or {}
    size = tuple(size_px) if size_px else capture_size(cfg)
    return {"camera": {"device": device, "controls": controls}, "table_cal": table_cal, "config": cfg,
            "git": git, "models": {"detect": (cfg.get("detect") or {}).get("model"),
                                   "proposals": {"kind": kind,
                                                 "model": (p.get(kind) or {}).get("model") if kind else None}},
            "view": {"size_px": [int(size[0]), int(size[1])],
                     "room_memory": bool(rm.get("enabled")),
                     "table_view_rect": rm.get("table_view_rect") if rm.get("enabled") else None,
                     "out_size": list(cfg.get("frame_size_px") or (W, H))},
            "room_zones": _zones(cfg),
            "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "clock_offset_s": clock_offset_s}


def record(out: Path, device, seconds: float, meta: dict, fps: float = 30.0,
           size: tuple[int, int] = (W, H), queue_n: int = 90) -> dict:
    import cv2

    from core.capture import open_camera
    w, h = int(size[0]), int(size[1])
    out.mkdir(parents=True, exist_ok=True)
    cap = open_camera(device if not str(device).isdigit() else int(device), w, h, int(round(fps)))
    t_end = time.time() + 3.0
    while time.time() < t_end:                      # the camera settles for a few seconds after opening
        cap.read()
    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    q: "queue.Queue" = queue.Queue(maxsize=queue_n)   # ~3 s of frames: 1440p mp4v can lag the camera

    def write() -> None:
        while True:
            img = q.get()
            if img is None:
                return
            writer.write(img)
    th = threading.Thread(target=write, name="clip-writer", daemon=True)
    th.start()
    walls: list[float] = []
    dropped = 0
    stop = out / "STOP"
    t_stop = None
    while True:
        ok, img = cap.read()
        now = time.time()
        if not ok:
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        try:
            q.put_nowait(img)
        except queue.Full:
            dropped += 1                            # frames.json only lists written frames
            continue
        walls.append(now)
        if len(walls) == 1:
            t_stop = now + seconds
            (out / "READY").write_text(f"{now:.6f}\n")
        if now >= t_stop or stop.exists():
            break
    cap.release()
    q.put(None)
    th.join()
    writer.release()
    frames = {"wall": walls, "t": [round(x - walls[0], 6) for x in walls]}
    (out / "frames.json").write_text(json.dumps(frames))
    rec = {"size_px": [w, h], "fps_asked": fps, "dropped": dropped}
    (out / "meta.json").write_text(json.dumps(dict(meta, clip=out.name, record=rec), indent=1, default=str))
    return {"frames": len(walls), "seconds": walls[-1] - walls[0], "size": f"{w}x{h}", "dropped": dropped,
            "fps": (len(walls) - 1) / max(1e-6, walls[-1] - walls[0])}


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None) -> int:
    from core.config import load_config
    from core.table import apply_saved_size
    from core.table_area import apply_saved_area
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--controls", default="{}", help="JSON of the camera controls read on the host")
    ap.add_argument("--git", default="")
    ap.add_argument("--clock-offset", type=float, default=0.0)
    ap.add_argument("--size", type=_size, help="WxH instead of the app's capture size (e.g. 1280x720)")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)
    cfg = apply_saved_area(apply_saved_size(load_config()))       # main.build's order
    size = a.size or capture_size(cfg)
    meta = meta_for(cfg, a.device, json.loads(a.controls), a.git, a.clock_offset, size_px=size)
    r = record(Path(a.out), a.device, a.seconds, meta, fps=a.fps, size=size)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
