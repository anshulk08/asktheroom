"""Replay a recorded video through the live perception stack for the P6 / D4 check: detector ->
HandTracker -> World, wired as main.py wires them. Reports the processing fps, how often each object
was detected, and every disappearance-type event for an object no hand ever touched. Only a hand can
take, cover or carry something, so on an untouched object such an event is a false "disappeared".

    python -m eval.replay_video --video trials/12/video.mp4                   # config detect.model
    python -m eval.replay_video --video clip.mp4 --model models/askroom-det.engine --fps 12 --json p6.json
    python -m eval.replay_video --video clip.mp4 --untouched keys wallet glasses

Passes with no false disappearances at --min-fps (10) or more. By default "untouched" means detected
at least once and never overlapped by a hand box (contact_overlap of its last box); --untouched names
them instead, for a clip where you know what you left alone. --fps N processes at most N frames per
second of video, dropping the rest as the live loop does when the detector is slower than the camera;
the world's timers run on video time either way. Frame times come from frames.json beside the video
(a guided clip, eval/clip.py) when there is one, else timestamps.json (eval/record.py), else the fps.
For a guided clip with truth.json, eval/score_clip.py replays the whole production pipeline and scores it.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # allow `python eval/<name>.py` as well as `python -m eval.<name>`
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from collections import Counter, defaultdict
from statistics import median
from typing import Callable, Iterable, Optional, Union

from core.embed import make_embedder
from core.geom import overlap_frac
from core.hands import HandTracker
from core.types import Detections, EventType, Frame
from core.world import World

DISAPPEAR = {EventType.PICKED_UP, EventType.COVERED, EventType.PUT_INSIDE, EventType.EXITED_VIEW,
             EventType.LOST_TRACK}


def _every(frames: Iterable[Frame], max_fps: Optional[float]) -> Iterable[Frame]:
    last = None
    for f in frames:
        if max_fps is None or last is None or f.t - last >= 1.0 / max_fps - 1e-6:
            last = f.t
            yield f


def replay(frames: Iterable[Frame], detector: Union[Callable[[Frame], Detections], object], cfg: dict,
           untouched: Optional[list[str]] = None, max_fps: Optional[float] = None,
           min_fps: float = 10.0) -> dict:
    """Run every frame through detector, hand ids and the world; return the report dict."""
    detect = getattr(detector, "detect", detector)
    world = World(cfg, embed=make_embedder(cfg))          # as main.build: None unless reid.enabled
    hands = HandTracker(frame_size=tuple(cfg.get("frame_size_px") or (1280, 720)))
    contact = float(cfg.get("contact_overlap", 0.3))
    objects = list(cfg.get("objects") or {})
    seen, last_box, touched = Counter(), {}, set()
    events, ms = [], []
    n, t0, t_first, t_last = 0, time.perf_counter(), None, None
    for f in _every(frames, max_fps):
        d = detect(f)
        if hasattr(detector, "last_ms"):
            ms.append(detector.last_ms)
        d.hands = hands.update(d.hands, f.t)
        for it in d.items:
            seen[it.cls] += 1
            last_box[it.cls] = it.box_px
        for obj, box in last_box.items():
            if any(overlap_frac(h.box_px, box) >= contact for h in d.hands):
                touched.add(obj)
        events += world.update(d, f)
        n += 1
        t_first = f.t if t_first is None else t_first
        t_last = f.t
    wall = time.perf_counter() - t0
    fps = n / wall if wall > 0 else 0.0
    quiet = sorted(untouched) if untouched is not None else sorted(o for o in objects if seen[o] and o not in touched)
    by_obj: dict[str, Counter] = defaultdict(Counter)
    false = []
    for ev in events:
        typ = EventType(ev.type)
        by_obj[ev.obj][typ.value] += 1
        if ev.obj in quiet and typ in DISAPPEAR:
            false.append({"t": round(ev.t - (t_first or 0.0), 2), "obj": ev.obj, "type": typ.value, "parent": ev.parent})
    return {
        "frames": n, "video_s": round((t_last or 0.0) - (t_first or 0.0), 2), "fps": round(fps, 1),
        "detect_ms_median": round(median(ms), 1) if ms else None,
        "detected": {o: round(seen[o] / max(n, 1), 3) for o in objects},
        "touched": sorted(touched), "untouched": quiet,
        "events": {o: dict(c) for o, c in by_obj.items()},
        "false_disappearances": false,
        "pass": not false and fps >= min_fps and n > 0,
    }


def print_report(r: dict, min_fps: float = 10.0) -> None:
    print(f"{r['frames']} frames over {r['video_s']} s of video: {r['fps']} fps end to end"
          + (f", detector median {r['detect_ms_median']} ms" if r["detect_ms_median"] is not None else ""))
    for o, rate in r["detected"].items():
        tag = "untouched" if o in r["untouched"] else ("touched" if o in r["touched"] else "never seen")
        evs = ", ".join(f"{k} x{v}" for k, v in sorted(r["events"].get(o, {}).items())) or "-"
        print(f"  {o:12s} detected {100 * rate:5.1f}%  {tag:10s}  events: {evs}")
    for e in r["false_disappearances"]:
        print(f"  FALSE {e['type']} of {e['obj']} at {e['t']} s (parent {e['parent']})")
    print(f"P6: {'PASS' if r['pass'] else 'FAIL'} ({len(r['false_disappearances'])} false disappearances, "
          f"{r['fps']} fps, need 0 and >= {min_fps})")


def main(argv=None) -> int:
    from core.capture import VideoFileSource
    from core.config import load_config
    from core.detect import Detector, UltralyticsBackend, _FlatTable
    from core.table import Table
    from eval.clip import clip_frames, load_times
    from eval.record import _fit
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", help="detector weights (default: config detect.model)")
    ap.add_argument("--untouched", nargs="*", help="objects left alone in this clip (default: never overlapped by a hand)")
    ap.add_argument("--fps", type=float, help="process at most this many frames per second of video")
    ap.add_argument("--min-fps", type=float, default=10.0)
    ap.add_argument("--json", help="also write the report here")
    a = ap.parse_args(argv)
    cfg = load_config()
    if load_times(a.video) is not None:         # a guided clip: the camera's own frame times (frames.json)
        src = clip_frames(a.video)
    else:
        src = VideoFileSource(a.video, start=False).frames()
    frames = (Frame(t=f.t, wall=f.wall, img=_fit(f.img), idx=f.idx) for f in src)
    first = next(frames, None)
    if first is None:
        print(f"no frames in {a.video}")
        return 1
    table = Table(cfg)
    if not table.ok and not table.calibrate(first.img):
        print("table not calibrated and markers 0-3 not all in the first frame: using a flat frame->table mapping")
        table = _FlatTable(cfg)
    det = Detector(cfg, table, UltralyticsBackend(cfg, a.model))

    def all_frames():
        yield first
        yield from frames
    r = replay(all_frames(), det, cfg, a.untouched, a.fps, a.min_fps)
    print_report(r, a.min_fps)
    if a.json:
        _Path(a.json).write_text(json.dumps(r, indent=2))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
