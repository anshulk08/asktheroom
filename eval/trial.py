"""Evaluation trial format (spec 3.13). Owner: eval.

A trial is one short clip that ends with a question about one target object. On disk:

    trials/<id>/
        video.mp4           recorded clip (absent for synthetic trials)
        truth.json          what really happened (below)
        detections.jsonl    cached detector output, one Detections per line (see dets_to_json)
        results/<system>.json   one prediction + score per system, written by eval/replay.py

truth.json:

    {"trial_id": 17, "category": "covered", "object": "keys",
     "truth": {"status": "UNDER", "parent": "notebook", "pos_cm": [41.0, 22.5] | null,
               "edge": null},
     "truth_spec": "notebook",            # what was typed after --truth
     "question": "where are my keys?",    # asked at the END of the clip
     "recorded_at": "2026-09-26T14:03:11+00:00", "source": "camera" | "file" | "synth",
     "notes": ""}

Truth semantics. `--truth` takes one word and is parsed by parse_truth():

    --truth notebook   any config object of kind 'cover'     -> status UNDER,  parent notebook
    --truth box        any config object of kind 'container' -> status INSIDE, parent box
    --truth left       left|right|top|bottom                 -> status GONE,   edge left
    --truth 40,22      table cm (x right, y down from marker 0) -> status VISIBLE at (40, 22)
    --truth visible    VISIBLE, position taken from the final detections (see below)
    --truth held       HELD (object in a hand at question time)

truth.pos_cm is "where a correct laser would point". For VISIBLE it is the object itself. For
INSIDE/UNDER it is the parent's position at the END of the clip (so for inside_box_moved it is
the box's new spot). For GONE/HELD it is null (no laser error is scored). When pos_cm is null
for VISIBLE/INSIDE/UNDER (recorded trials where nobody measured with a ruler), Trial.truth_resolved()
fills it from the median of the last detections of the object (VISIBLE) or of the parent
(INSIDE/UNDER) -- a detector-derived truth that is fine for laser error, and does not affect
correctness for INSIDE/UNDER, which is scored on the parent only.

Scoring rules live in eval/replay.py:score().
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator, Optional

from core.types import Detection, Detections, Status

EDGES = ("left", "right", "top", "bottom")

# category -> (minimum trials, expected truth status). Stretch categories last.
CATEGORIES: dict[str, tuple[int, str]] = {
    "visible": (4, "VISIBLE"),
    "moved": (4, "VISIBLE"),
    "covered": (5, "UNDER"),
    "uncovered": (4, "VISIBLE"),
    "inside": (5, "INSIDE"),
    "inside_box_moved": (6, "INSIDE"),
    "put_back": (4, "VISIBLE"),
    "carried_away": (4, "GONE"),
    "ambiguous": (4, "INSIDE|UNDER"),
    "dropped": (4, "GONE"),          # stretch
    "room_surface": (4, "GONE"),     # stretch (found later by the search camera)
}
STRETCH = {"dropped", "room_surface"}
HIDDEN = ("covered", "inside", "inside_box_moved")   # the categories the pitch rests on

TRUTH_KEYS = ("status", "parent", "pos_cm", "edge")


# ---------------------------------------------------------------- Detections <-> JSON

def _r(v: float, nd: int = 2) -> float:
    return round(float(v), nd)


def det_to_json(d: Detection) -> dict:
    return {"cls": d.cls, "conf": _r(d.conf, 3), "box_px": [int(v) for v in d.box_px],
            "center_cm": [_r(v) for v in d.center_cm], "box_cm": [_r(v) for v in d.box_cm]}


def det_from_json(j: dict) -> Detection:
    return Detection(cls=j["cls"], conf=float(j["conf"]),
                     box_px=tuple(int(v) for v in j["box_px"]),
                     center_cm=tuple(float(v) for v in j["center_cm"]),
                     box_cm=tuple(float(v) for v in j["box_cm"]))


def dets_to_json(d: Detections) -> dict:
    return {"t": _r(d.t, 4), "frame_idx": int(d.frame_idx),
            "items": [det_to_json(x) for x in d.items],
            "hands": [det_to_json(x) for x in d.hands]}


def dets_from_json(j: dict) -> Detections:
    return Detections(t=float(j["t"]), frame_idx=int(j["frame_idx"]),
                      items=[det_from_json(x) for x in j.get("items", [])],
                      hands=[det_from_json(x) for x in j.get("hands", [])])


def write_detections(path: str | Path, stream: Iterable[Detections]) -> int:
    n = 0
    with open(path, "w") as f:
        for d in stream:
            f.write(json.dumps(dets_to_json(d), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_detections(path: str | Path) -> Iterator[Detections]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield dets_from_json(json.loads(line))


# ---------------------------------------------------------------- truth

def make_truth(status: str, parent: Optional[str] = None, pos_cm=None,
               edge: Optional[str] = None) -> dict:
    Status(status)  # validates
    return {"status": status, "parent": parent,
            "pos_cm": [_r(pos_cm[0]), _r(pos_cm[1])] if pos_cm is not None else None,
            "edge": edge}


def parse_truth(spec: str, cfg: dict) -> dict:
    """'notebook' | 'box' | 'left' | '40,22' | 'visible' | 'held' -> truth dict (module docstring)."""
    s = spec.strip().lower()
    kinds = cfg.get("objects", {})
    if s in kinds and kinds[s] == "cover":
        return make_truth("UNDER", parent=s)
    if s in kinds and kinds[s] == "container":
        return make_truth("INSIDE", parent=s)
    if s in EDGES:
        return make_truth("GONE", edge=s)
    if s == "visible":
        return make_truth("VISIBLE")
    if s in ("held", "hand"):
        return make_truth("HELD")
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*", s)
    if m:
        return make_truth("VISIBLE", pos_cm=(float(m.group(1)), float(m.group(2))))
    raise ValueError(f"can't parse --truth {spec!r}: expected a cover/container name, an edge "
                     f"({'|'.join(EDGES)}), 'x,y' in table cm, 'visible' or 'held'")


def check_category(category: str, truth: dict) -> Optional[str]:
    """A warning string if the truth status is unusual for the category, else None."""
    if category not in CATEGORIES:
        return f"unknown category {category!r} (known: {', '.join(CATEGORIES)})"
    expected = CATEGORIES[category][1].split("|")
    if truth["status"] not in expected:
        return f"category {category} usually has truth {'/'.join(expected)}, got {truth['status']}"
    return None


def default_question(cfg: dict, obj: str) -> str:
    from core.config import display_name
    name = display_name(cfg, obj)
    return f"where {'are' if name.endswith('s') else 'is'} my {name}?"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- Trial folder

class Trial:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._meta: Optional[dict] = None

    # -- creation
    @classmethod
    def create(cls, trials_dir: str | Path, trial_id, category: str, obj: str, truth: dict,
               question: str, source: str, notes: str = "", truth_spec: str = "",
               overwrite: bool = False) -> "Trial":
        p = Path(trials_dir) / str(trial_id)
        if (p / "truth.json").exists() and not overwrite:
            raise FileExistsError(f"{p} already has a truth.json (use --overwrite)")
        p.mkdir(parents=True, exist_ok=True)
        t = cls(p)
        t.write_meta({"trial_id": trial_id, "category": category, "object": obj,
                      "truth": {k: truth.get(k) for k in TRUTH_KEYS}, "truth_spec": truth_spec,
                      "question": question, "recorded_at": now_iso(), "source": source,
                      "notes": notes})
        return t

    def write_meta(self, meta: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "truth.json", "w") as f:
            json.dump(meta, f, indent=2)
        self._meta = meta

    # -- fields
    @property
    def meta(self) -> dict:
        if self._meta is None:
            with open(self.path / "truth.json") as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def id(self):
        return self.meta["trial_id"]

    @property
    def category(self) -> str:
        return self.meta["category"]

    @property
    def obj(self) -> str:
        return self.meta["object"]

    @property
    def truth(self) -> dict:
        return self.meta["truth"]

    @property
    def video_path(self) -> Path:
        return self.path / "video.mp4"

    @property
    def detections_path(self) -> Path:
        return self.path / "detections.jsonl"

    def has_detections(self) -> bool:
        return self.detections_path.exists() and self.detections_path.stat().st_size > 0

    def detections(self) -> Iterator[Detections]:
        return read_detections(self.detections_path)

    def write_detections(self, stream: Iterable[Detections]) -> int:
        return write_detections(self.detections_path, stream)

    def truth_resolved(self, dets: Optional[list[Detections]] = None, last_n: int = 10) -> dict:
        """truth with pos_cm filled from the final detections when it was not recorded."""
        tr = dict(self.truth)
        if tr.get("pos_cm") is not None or tr["status"] not in ("VISIBLE", "INSIDE", "UNDER"):
            return tr
        who = self.obj if tr["status"] == "VISIBLE" else tr.get("parent")
        if not who:
            return tr
        dets = dets if dets is not None else list(self.detections())
        xs, ys = [], []
        for d in reversed(dets):
            for it in d.items:
                if it.cls == who:
                    xs.append(it.center_cm[0])
                    ys.append(it.center_cm[1])
                    break
            if len(xs) >= last_n:
                break
        if xs:
            tr["pos_cm"] = [_r(median(xs)), _r(median(ys))]
            tr["pos_source"] = "detections"
        return tr

    # -- results
    @property
    def results_dir(self) -> Path:
        return self.path / "results"

    def write_result(self, system: str, result: dict) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / f"{system}.json", "w") as f:
            json.dump(result, f, indent=2)

    def results(self) -> dict[str, dict]:
        out = {}
        if self.results_dir.exists():
            for p in sorted(self.results_dir.glob("*.json")):
                with open(p) as f:
                    out[p.stem] = json.load(f)
        return out


def _id_key(p: Path):
    return (0, int(p.name), "") if p.name.isdigit() else (1, 0, p.name)


def load_trials(trials_dir: str | Path) -> list[Trial]:
    """Every trials/<id>/ that has a truth.json, in numeric id order."""
    root = Path(trials_dir)
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "truth.json").exists()]
    return [Trial(p) for p in sorted(dirs, key=_id_key)]
