"""Replay cached detections through each system and score the answers (spec 3.13). Owner: eval.

    python -m eval.replay --trials trials/ --system full|last_seen|current_frame|nearest_object|all

For every trial with a detections.jsonl, each system is fed the whole Detections stream (dummy
Frames for the World), then asked predict(object) once, as if the question came at the end of the
clip. The result goes to trials/<id>/results/<system>.json:

    {"system", "trial_id", "category", "object", "truth", "prediction", "correct",
     "laser_err_cm", "predict_ms", "update_ms_median", "frames", "skipped", "reason", "error"}

Scoring (score()):
  truth INSIDE/UNDER  correct iff predicted parent == truth parent
  truth GONE          correct iff predicted status GONE and edge == truth edge
  truth VISIBLE       correct iff predicted resolved position is within 5 cm of truth pos_cm
                      (covers visible, moved, uncovered, put_back)
  truth HELD          correct iff predicted status HELD
  laser_err_cm        |predicted resolved_cm - truth pos_cm| whenever both exist, else null

'full' wraps core.world.World. While World raises NotImplementedError it is recorded as skipped
(not wrong) with the reason, and the report shows it as skipped.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # allow `python eval/<name>.py` as well as `python -m eval.<name>`
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import sys
import tempfile
import time
import traceback
from statistics import median
from typing import Optional


from core.config import load_config
from core.types import Detections, Frame
from eval.baselines import BASELINES
from eval.trial import Trial, load_trials

VISIBLE_TOL_CM = 5.0
SYSTEMS = ["full", "nearest_object", "last_seen", "current_frame"]



class WorldAdapter:
    """core.world.World behind the baseline interface: update(dets) / predict(obj)."""
    name = "full"

    def __init__(self, cfg: dict, wall0: Optional[float] = None):
        from core.events import EventLog
        from core.world import World
        self._tmp = tempfile.TemporaryDirectory(prefix="askroom_replay_")
        self.events = EventLog(":memory:", self._tmp.name)
        self.world = World(cfg, self.events)
        self.wall0 = time.time() if wall0 is None else wall0
        self.t0: Optional[float] = None

    def update(self, dets: Detections) -> None:
        if self.t0 is None:
            self.t0 = dets.t
        # img=None: replays carry detections only. The world skips its pixel rules without an
        # image; a fake black frame would make every vanished object "still there".
        frame = Frame(t=dets.t, wall=self.wall0 + (dets.t - self.t0), img=None, idx=dets.frame_idx)
        self.world.update(dets, frame)

    def predict(self, obj: str) -> dict:
        from eval.baselines import prediction
        e = self.world.get(obj)
        pos, _chain = self.world.resolve(obj)
        return prediction(e.status.value if hasattr(e.status, "value") else str(e.status),
                          parent=e.parent, pos_cm=e.pos_cm, resolved_cm=pos, edge=e.edge)

    def close(self) -> None:
        self.events.close()
        self._tmp.cleanup()


def make_system(name: str, cfg: dict):
    if name == "full":
        return WorldAdapter(cfg)
    return BASELINES[name](cfg)


# ---------------------------------------------------------------- scoring

def _dist(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def score(pred: dict, truth: dict, tol_cm: float = VISIBLE_TOL_CM) -> tuple[bool, Optional[float]]:
    """(correct, laser_err_cm) for one prediction against one (resolved) truth."""
    st = truth["status"]
    rc, tp = pred.get("resolved_cm"), truth.get("pos_cm")
    err = _dist(rc, tp) if rc is not None and tp is not None else None
    if st in ("INSIDE", "UNDER"):
        ok = pred.get("parent") is not None and pred.get("parent") == truth.get("parent")
    elif st == "GONE":
        ok = pred.get("status") == "GONE" and pred.get("edge") == truth.get("edge")
    elif st == "VISIBLE":
        ok = err is not None and err <= tol_cm
    elif st == "HELD":
        ok = pred.get("status") == "HELD"
    else:
        ok = pred.get("status") == st
    return ok, (round(err, 2) if err is not None else None)


# ---------------------------------------------------------------- running

def run_system(name: str, cfg: dict, trial: Trial, dets: list[Detections]) -> dict:
    tr = trial.truth_resolved(dets)
    base = {"system": name, "trial_id": trial.id, "category": trial.category,
            "object": trial.obj, "truth": tr, "prediction": None, "correct": False,
            "laser_err_cm": None, "predict_ms": None, "update_ms_median": None,
            "frames": len(dets), "skipped": False, "reason": None, "error": None}
    sys_ = None
    try:
        sys_ = make_system(name, cfg)
        ups = []
        for d in dets:
            t0 = time.perf_counter()
            sys_.update(d)
            ups.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        pred = sys_.predict(trial.obj)
        base["predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        base["update_ms_median"] = round(median(ups), 3) if ups else None
    except NotImplementedError as e:
        base["skipped"] = True
        base["reason"] = f"{name}: not implemented yet (NotImplementedError: {e})"
        return base
    except Exception as e:  # a real bug in a system: count it wrong, keep going
        base["error"] = f"{type(e).__name__}: {e}"
        base["traceback"] = traceback.format_exc(limit=6)
        return base
    finally:
        if sys_ is not None and hasattr(sys_, "close"):
            sys_.close()
    base["prediction"] = pred
    base["correct"], base["laser_err_cm"] = score(pred, tr)
    return base


def replay(trials_dir: str, systems: list[str], cfg: dict, verbose: bool = True) -> dict:
    """Runs every system over every trial with cached detections. Returns {system: [results]}."""
    out: dict[str, list[dict]] = {s: [] for s in systems}
    trials = [t for t in load_trials(trials_dir) if t.has_detections()]
    missing = [t for t in load_trials(trials_dir) if not t.has_detections()]
    if verbose and missing:
        print(f"note: {len(missing)} trial(s) have no detections.jsonl yet (run eval/record.py "
              f"--detect-only): {', '.join(str(t.id) for t in missing)}")
    for t in trials:
        dets = list(t.detections())
        for s in systems:
            r = run_system(s, cfg, t, dets)
            t.write_result(s, r)
            out[s].append(r)
    if verbose:
        print(f"replayed {len(trials)} trial(s) from {trials_dir}")
        for s in systems:
            rs = out[s]
            if rs and all(r["skipped"] for r in rs):
                print(f"  {s:15s} SKIPPED - {rs[0]['reason']}")
                continue
            done = [r for r in rs if not r["skipped"]]
            ok = sum(r["correct"] for r in done)
            errs = [r for r in done if r["error"]]
            pct = 100.0 * ok / len(done) if done else 0.0
            line = f"  {s:15s} {ok}/{len(done)} correct ({pct:.0f}%)"
            if errs:
                line += f", {len(errs)} error(s), first: trial {errs[0]['trial_id']}: {errs[0]['error']}"
            print(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", default="trials", help="folder of trial folders")
    ap.add_argument("--system", default="all", choices=SYSTEMS + ["all"])
    ap.add_argument("--config", default=None, help="config.yaml path")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    systems = SYSTEMS if a.system == "all" else [a.system]
    out = replay(a.trials, systems, cfg, verbose=not a.quiet)
    return 0 if any(out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
