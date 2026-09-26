"""Record one evaluation trial (spec 3.13). Owner: eval.

    python -m eval.record --trial-id 17 --category covered --object keys --truth notebook
    python -m eval.record --trial-id 18 --category inside --object keys --truth box --from-file clip.mov
    python -m eval.record --trial-id 17 --detect-only        # (re)run the detector on a saved trial

Records from a camera (default index 0, or a /dev/v4l/by-id/ path; the Mac webcam is fine for testing) or copies a video
file, at 1280x720, for --seconds or until Enter / q. Writes trials/<id>/video.mp4,
timestamps.json (seconds since the first frame, per frame) and truth.json. Ask the question at the
end of the clip; the truth is what is true at that moment. See eval/trial.py for --truth.

--detect runs core.detect.Detector over the video to write detections.jsonl, if that module
exists yet; otherwise it says so and you can run --detect-only later.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # allow `python eval/<name>.py` as well as `python -m eval.<name>`
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Union

from core.config import load_config
from core.types import Detections, Frame
from eval.trial import Trial, check_category, default_question, parse_truth

W, H = 1280, 720


def _writer(path: Path, fps: float):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not w.isOpened():
        raise RuntimeError(f"can't open video writer for {path}")
    return w


def _fit(img):
    import cv2
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def camera_arg(s: str) -> Union[int, str]:
    """--camera: an index ('0') or a device path; the /dev/v4l/by-id/ path names the Brio for good."""
    s = str(s).strip()
    return int(s) if s.isdigit() else s


def record_camera(index: Union[int, str], out: Path, seconds: Optional[float], preview: bool = True) -> list[float]:
    """Record until seconds elapse, Enter on stdin, or q/Enter in the preview window."""
    import cv2

    from core.capture import open_camera
    cap = open_camera(index, W, H)          # MJPG + 2 buffers: 30 fps on the Jetson camera
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps if 1 <= fps <= 120 else 30.0
    writer = _writer(out, fps)
    stop = threading.Event()

    def _stdin():
        try:
            sys.stdin.readline()
        except Exception:
            return
        stop.set()

    threading.Thread(target=_stdin, daemon=True).start()
    print(f"recording camera {index} -> {out} ({'%.0f s' % seconds if seconds else 'no limit'}); "
          "press Enter (or q in the window) to stop")
    ts, t0 = [], None
    try:
        while not stop.is_set():
            ok, img = cap.read()
            if not ok:
                print("camera read failed; stopping")
                break
            now = time.monotonic()
            t0 = now if t0 is None else t0
            writer.write(_fit(img))
            ts.append(round(now - t0, 4))
            if seconds and now - t0 >= seconds:
                break
            if preview:
                try:
                    cv2.imshow("askroom record (q to stop)", img)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 13, 10):
                        break
                except cv2.error:
                    preview = False   # headless OpenCV: stdin only
    finally:
        cap.release()
        writer.release()
        if preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    return ts


def copy_file(src: str, out: Path, seconds: Optional[float]) -> list[float]:
    import cv2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"can't open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = _writer(out, fps)
    ts, i = [], 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = i / fps
            if seconds and t > seconds:
                break
            writer.write(_fit(img))
            ts.append(round(t, 4))
            i += 1
    finally:
        cap.release()
        writer.release()
    return ts


# ---------------------------------------------------------------- detector hook

def load_detector(cfg: dict):
    """(detector, None) if core.detect.Detector is importable, else (None, reason)."""
    try:
        from core.detect import Detector  # type: ignore
    except ImportError as e:
        return None, f"core.detect.Detector is not importable yet ({e})"
    return Detector(cfg), None


def _call_detector(det, frame: Frame) -> Detections:
    for name in ("detect", "process", "__call__"):
        fn = getattr(det, name, None)
        if callable(fn):
            return fn(frame)
    raise TypeError("Detector has no detect(frame) / process(frame) / __call__(frame)")


def detect_trial(trial: Trial, cfg: dict) -> Optional[int]:
    """Runs the detector over trial/video.mp4 -> detections.jsonl. Returns frames written, or None."""
    det, why = load_detector(cfg)
    if det is None:
        print(f"--detect: {why}; detections.jsonl not written. Run "
              f"`python -m eval.record --trial-id {trial.id} --detect-only` once it exists.")
        return None
    import cv2
    ts_path = trial.path / "timestamps.json"
    ts = json.loads(ts_path.read_text()) if ts_path.exists() else None
    cap = cv2.VideoCapture(str(trial.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wall0 = time.time()

    def stream():
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                return
            t = ts[i] if ts and i < len(ts) else i / fps
            yield _call_detector(det, Frame(t=t, wall=wall0 + t, img=_fit(img), idx=i))
            i += 1

    try:
        n = trial.write_detections(stream())
    finally:
        cap.release()
    print(f"wrote {n} frames of detections to {trial.detections_path}")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trial-id", type=int, required=True)
    ap.add_argument("--category")
    ap.add_argument("--object")
    ap.add_argument("--truth", help="notebook | box | left|right|top|bottom | x,y | visible | held")
    ap.add_argument("--trials", default="trials")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=camera_arg, default=0, help="index or /dev/v4l/by-id/ path")
    src.add_argument("--from-file")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--detect", action="store_true", help="run core.detect.Detector after recording")
    ap.add_argument("--detect-only", action="store_true", help="only run the detector on a saved trial")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    if a.detect_only:
        t = Trial(Path(a.trials) / str(a.trial_id))
        return 0 if detect_trial(t, cfg) is not None else 1

    if not (a.category and a.object and a.truth):
        ap.error("--category, --object and --truth are required when recording")
    if a.object not in cfg["objects"]:
        ap.error(f"unknown object {a.object!r}; one of {', '.join(cfg['objects'])}")
    truth = parse_truth(a.truth, cfg)
    warn = check_category(a.category, truth)
    if warn:
        print(f"warning: {warn}")
    t = Trial.create(a.trials, a.trial_id, a.category, a.object, truth,
                     a.question or default_question(cfg, a.object),
                     "file" if a.from_file else "camera", notes=a.notes, truth_spec=a.truth,
                     overwrite=a.overwrite)
    if a.from_file:
        ts = copy_file(a.from_file, t.video_path, a.seconds)
    else:
        ts = record_camera(a.camera, t.video_path, a.seconds, preview=not a.no_preview)
    (t.path / "timestamps.json").write_text(json.dumps(ts))
    print(f"trial {a.trial_id}: {len(ts)} frames, {ts[-1] if ts else 0:.1f} s -> {t.path}")
    print(f"truth: {truth}  question: {t.meta['question']!r}")
    if a.detect:
        detect_trial(t, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
