"""A guided recording ("clip") as the recorder writes it, and its frames with the recorded times.

    <clip_dir>/
      video.mp4     raw frames as captured, no overlay (mp4v): 1280x720, or with room memory on the full
                    camera frame (room_memory.capture_size, e.g. 2560x1440), not yet cut to the table view
      frames.json   {"wall": [Unix s per frame], "t": [s since the first frame per frame]}
      meta.json     {"clip", "camera", "table_cal", "config", "git", "models", "recorded_at", "clock_offset_s",
                     "view", "room_zones", "record"}  (the last three since the room-demo recorder)
      truth.json    {"props": {id: what}, "steps": [...], "commands": [...], "questions": [...],
                     "checkpoints": [...]}  (every t is seconds since the first frame)

clip_frames() replays video.mp4 with frames.json's times (Frame.t = t, Frame.wall = wall), so the
world's timers run on the times the camera actually delivered, dropped frames and all; without a
frames.json it falls back to the video's nominal fps. eval/score_clip.py replays and scores clips.

clip_view() says which table view the live app cut from these frames (main.open_frames: table_view_rect
of the full frame, resized to frame_size_px by core.room_view.TableView), or None when the frames already
are the table view (every 1280x720 clip). A replay puts the full frames behind that same TableView
(FrameSlot), so the table pipeline gets the cut the app got and room memory gets the full frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

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


def video_size(video: PathLike) -> Optional[tuple[int, int]]:
    """(width, height) of the video's frames, or None when it can't be opened."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w > 0 and h > 0 else None
    finally:
        cap.release()


def clip_view(meta: dict, size_px: Optional[Sequence[int]] = None
              ) -> Optional[tuple[tuple[int, int, int, int], tuple[int, int]]]:
    """(table_view_rect in the clip's frame px, out_size) the live app cut from the recorded frames, or None
    when they are the table view already. Decided by the recorded config (a property of the recording, not
    of the replay's config), as main.open_frames does: room memory on means the camera ran at
    room_memory.capture_size and TableView cut table_view_rect (or default_rect when unmeasured) and resized
    it to frame_size_px. size_px: the clip's frame size (meta.view.size_px, else the caller's). Frames of
    another size than capture_size get the rect scaled to them; frames already at out_size that are not the
    capture size are a clip resized before the cut existed (older recorder): None."""
    from core.room_types import RoomConfig
    from core.room_view import default_rect
    cfg = meta.get("config") or {}
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    if not rc.enabled:
        return None
    out = tuple(int(v) for v in (cfg.get("frame_size_px") or (1280, 720)))
    rect = rc.table_view_rect
    if rect is None:
        rect = tuple(default_rect(rc.capture_size, rc.zoom, rc.ref_zoom, out))
    view = meta.get("view") or {}
    size = tuple(int(v) for v in (view.get("size_px") or size_px or rc.capture_size))
    cap = tuple(int(v) for v in rc.capture_size)
    if size == out and size != cap:
        return None
    if size != cap:
        sx, sy = size[0] / cap[0], size[1] / cap[1]
        rect = (round(rect[0] * sx), round(rect[1] * sy), round(rect[2] * sx), round(rect[3] * sy))
    return tuple(int(v) for v in rect), (out[0], out[1])


class FrameSlot:
    """The FrameSource API over the one frame a replay is on: latest() / at(t) / wait_new() return it. Put
    core.room_view.TableView in front of it and the replay reads frames exactly as the live app does:
    view.at(t) is the table-view cut, view.full_at(t) the full frame (Room._room_step)."""

    fps = 30.0

    def __init__(self) -> None:
        self.cur: Optional[Frame] = None

    def latest(self) -> Optional[Frame]:
        return self.cur

    def at(self, t: float) -> Optional[Frame]:
        return self.cur

    def wait_new(self, after_idx: int, timeout: float = 1.0) -> Optional[Frame]:
        return self.cur

    def stop(self) -> None:
        pass


def table_frames(frames: Iterable[Frame], view) -> Iterator[Frame]:
    """frames cut to the table view as the live app cuts them (view: clip_view's result, None: as they are)."""
    if view is None:
        yield from frames
        return
    from core.room_view import TableView
    slot = FrameSlot()
    tv = TableView(slot, view[0], view[1])
    for f in frames:
        slot.cur = f
        yield tv.at(f.t)


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

    @property
    def size_px(self) -> Optional[tuple[int, int]]:
        """The recorded frame size: meta.view.size_px, else the video's."""
        v = (self.meta.get("view") or {}).get("size_px")
        return (int(v[0]), int(v[1])) if v else video_size(self.video)

    def view(self) -> Optional[tuple[tuple[int, int, int, int], tuple[int, int]]]:
        """clip_view for this clip: the table-view cut the live app made, or None."""
        return clip_view(self.meta, self.size_px)

    def frames(self) -> Iterator[Frame]:
        """The recorded frames as captured (full frames when the app cut a table view from them)."""
        return clip_frames(self.video, {"t": self.t, "wall": self.wall})

    def table_frames(self) -> Iterator[Frame]:
        """The frames the table pipeline saw live: cut to the table view when the app cut one."""
        return table_frames(self.frames(), self.view())


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
