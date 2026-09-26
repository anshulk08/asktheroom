"""A guided recording ("clip") as the recorder writes it, and its frames with the recorded times.

    <clip_dir>/
      video.mp4     raw 1280x720 frames as captured, no overlay (mp4v)
      frames.json   {"wall": [Unix s per frame], "t": [s since the first frame per frame]}
      meta.json     {"clip", "camera", "table_cal", "config", "git", "models", "recorded_at", "clock_offset_s"}
      truth.json    {"props": {id: what}, "steps": [...], "commands": [...], "questions": [...],
                     "checkpoints": [...]}  (every t is seconds since the first frame)

clip_frames() replays video.mp4 with frames.json's times (Frame.t = t, Frame.wall = wall), so the
world's timers run on the times the camera actually delivered, dropped frames and all; without a
frames.json it falls back to the video's nominal fps. eval/score_clip.py replays and scores clips.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from core.types import Frame

PathLike = Union[str, Path]
TRUTH_LISTS = ("steps", "commands", "questions", "checkpoints")


def load_times(video: PathLike) -> Optional[dict]:
    """frames.json beside the video ({"t": [...], "wall": [...]}), else None."""
    p = Path(video).with_name("frames.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    t = [float(x) for x in d.get("t") or []]
    wall = [float(x) for x in d.get("wall") or []] or [0.0] * len(t)
    return {"t": t, "wall": wall}


def clip_frames(video: PathLike, times: Optional[dict] = None) -> Iterator[Frame]:
    """Every frame of the video in order, idx 0, 1, ... (the frames.json index), timed by frames.json.
    Frames past the end of frames.json (a recorder cut short) continue at the video's fps."""
    import cv2
    times = times if times is not None else load_times(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"can't open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ts, walls = (times or {}).get("t") or [], (times or {}).get("wall") or []
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            if i < len(ts):
                t, wall = ts[i], walls[i]
            else:
                t0 = ts[-1] + 1.0 / fps if ts else 0.0
                t = t0 + (i - len(ts)) / fps
                wall = (walls[-1] - ts[-1] + t) if walls else t
            yield Frame(t=t, wall=wall, img=img, idx=i)
            i += 1
    finally:
        cap.release()


def write_video(path: PathLike, imgs: Iterable, fps: float = 30.0) -> None:
    """BGR frames -> an mp4v file, as the recorder writes video.mp4 (tests build synthetic clips)."""
    import cv2
    writer = None
    try:
        for img in imgs:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"can't open video writer for {path}")
            writer.write(img)
    finally:
        if writer is not None:
            writer.release()


@dataclass
class Clip:
    dir: Path
    meta: dict
    truth: dict
    t: list = field(default_factory=list)
    wall: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.meta.get("clip") or self.dir.name)

    @property
    def video(self) -> Path:
        return self.dir / "video.mp4"

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def frames(self) -> Iterator[Frame]:
        return clip_frames(self.video, {"t": self.t, "wall": self.wall})


def load_clip(clip_dir: PathLike) -> Clip:
    d = Path(clip_dir)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    truth = json.loads((d / "truth.json").read_text()) if (d / "truth.json").exists() else {}
    truth = dict(truth, props=dict(truth.get("props") or {}))
    for k in TRUTH_LISTS:
        truth[k] = sorted((dict(x) for x in truth.get(k) or []), key=lambda x: float(x.get("t", 0.0)))
    for s in truth["steps"]:
        s.setdefault("obj", None)
        s.setdefault("parent", None)
    times = load_times(d / "video.mp4") or {"t": [], "wall": []}
    return Clip(dir=d, meta=meta, truth=truth, t=times["t"], wall=times["wall"])
