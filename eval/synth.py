"""Synthetic trial generator (spec 3.13). Owner: eval.

    python -m eval.synth --out /tmp/askroom_trials --per-category 5 [--seed 0] [--stretch]

Writes trials/<id>/{truth.json, detections.jsonl} (no video) for each category by scripting a
tabletop scene at 10 fps and rendering it as Detections with realistic noise:

  - ~5% missed detections everywhere; +-1 cm centre jitter; conf 0.55-0.95
  - a tracked hand ('hand:N') that enters from an edge, occludes what it covers, carries objects
    (a carried object is detected ~75% of the time, lower conf)
  - the notebook sliding over an object (object hidden once >=60% of its box is covered)
  - an object dropped into the box (hidden, follows the box), then the box being moved
  - a hand carrying an object off an edge, or pushing it off (dropped)
  - ambiguous: the object is closed in a fist (rarely detected) while the hand visits both the
    box and the notebook; truth alternates box / notebook

Table size and object names come from config.yaml. The camera is assumed to see exactly the
table (1280x720 px == size_cm), which is all box_px needs to be plausible.
Categories: the 9 core ones, plus 'dropped' with --stretch. 'room_surface' needs the search
camera and is not synthesised.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # allow `python eval/<name>.py` as well as `python -m eval.<name>`
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

from core.config import load_config
from core.types import Detection, Detections
from eval.trial import EDGES, Trial, default_question, make_truth

FPS = 10.0
T0 = 1000.0                      # fake time.monotonic() at frame 0
IMG_W, IMG_H = 1280, 720
MISS = 0.05
JITTER_SIGMA, JITTER_MAX = 0.4, 1.0

# footprint (w, h) in cm, seen from above
SIZES = {"keys": (6, 4), "pill_bottle": (4, 4), "wallet": (11, 9), "glasses": (14, 5),
         "phone": (15, 7.5), "remote": (18, 5), "box": (20, 15), "notebook": (25, 18),
         "hand": (10, 16)}

CORE = ["visible", "moved", "covered", "uncovered", "inside", "inside_box_moved", "put_back",
        "carried_away", "ambiguous"]
STRETCH_SYNTH = ["dropped"]


def _box(c, size):
    w, h = size
    return (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)


def _cover_frac(inner, outer) -> float:
    """Fraction of `inner`'s area covered by `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / a if a > 0 else 0.0


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class Scene:
    """A scripted tabletop. Methods advance time and record one snapshot per frame."""

    def __init__(self, cfg: dict, rng: random.Random, layout: dict):
        self.cfg = cfg
        self.rng = rng
        self.W, self.H = cfg["table"]["size_cm"]
        self.kinds = cfg["objects"]
        self.pos: dict[str, Optional[tuple]] = {k: tuple(v) for k, v in layout.items()}
        self.hidden_in: dict[str, str] = {}      # obj -> container/cover it is inside/under
        self.hidden_off: dict[str, tuple] = {}   # obj -> offset from its parent
        self.hand: Optional[tuple] = None
        self.hand_id = 0
        self.carry: dict[str, tuple] = {}        # obj -> offset from hand
        self.fist = False                        # carried object closed in the hand
        self.snaps: list[dict] = []

    # ---------------------------------------------------------- script verbs
    def wait(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds * FPS)))):
            self._snap()

    def _entry(self, to, edge: str) -> tuple:
        x, y = to
        return {"left": (-12, y), "right": (self.W + 12, y), "top": (x, -12),
                "bottom": (x, self.H + 12)}[edge]

    def hand_in(self, to, edge: str = "bottom") -> None:
        self.hand_id += 1
        self.hand = self._entry(to, edge)
        self.hand_to(to)

    def hand_to(self, to, speed_cm_s: float = 30.0) -> None:
        start = self.hand
        n = max(4, int(_dist(start, to) / speed_cm_s * FPS))
        for i in range(1, n + 1):
            f = i / n
            f = f * f * (3 - 2 * f)  # smoothstep
            self.hand = (start[0] + (to[0] - start[0]) * f, start[1] + (to[1] - start[1]) * f)
            self._follow()
            self._snap()

    def hand_out(self, edge: str = "bottom") -> None:
        self.hand_to(self._entry(self.hand, edge))
        self.hand = None
        self._snap()

    def grab(self, obj: str, dwell: float = 0.4) -> None:
        self.carry[obj] = (self.pos[obj][0] - self.hand[0], self.pos[obj][1] - self.hand[1])
        self.wait(dwell)

    def release(self, obj: Optional[str] = None, dwell: float = 0.3) -> None:
        for o in ([obj] if obj else list(self.carry)):
            self.carry.pop(o, None)
        self.fist = False
        self.wait(dwell)

    def put_in(self, obj: str, parent: str, dwell: float = 0.5) -> None:
        """obj disappears into/under parent (at a small random offset) and moves with it."""
        self.carry.pop(obj, None)
        self.fist = False
        off = (self.rng.uniform(-3, 3), self.rng.uniform(-2, 2))
        self.hidden_in[obj] = parent
        self.hidden_off[obj] = off
        self._follow()
        self.wait(dwell)

    def _follow(self) -> None:
        if self.hand is not None:
            for o, off in self.carry.items():
                self.pos[o] = (self.hand[0] + off[0], self.hand[1] + off[1])
        for o, p in self.hidden_in.items():
            pp = self.pos[p]
            self.pos[o] = (pp[0] + self.hidden_off[o][0], pp[1] + self.hidden_off[o][1])

    def _snap(self) -> None:
        self.snaps.append({"pos": dict(self.pos), "hidden": set(self.hidden_in),
                           "hand": self.hand, "hand_id": self.hand_id,
                           "carry": set(self.carry), "fist": self.fist})

    # ---------------------------------------------------------- rendering
    def _on_table(self, c) -> bool:
        return 0 <= c[0] <= self.W and 0 <= c[1] <= self.H

    def _det(self, cls: str, c, size, conf: float) -> Detection:
        rng = self.rng
        jx = max(-JITTER_MAX, min(JITTER_MAX, rng.gauss(0, JITTER_SIGMA)))
        jy = max(-JITTER_MAX, min(JITTER_MAX, rng.gauss(0, JITTER_SIGMA)))
        cc = (c[0] + jx, c[1] + jy)
        b = _box(cc, size)
        b = (max(0.0, b[0]), max(0.0, b[1]), min(self.W, b[2]), min(self.H, b[3]))
        sx, sy = IMG_W / self.W, IMG_H / self.H
        bpx = (int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy))
        return Detection(cls=cls, conf=round(conf, 3), box_px=bpx,
                         center_cm=(round(cc[0], 2), round(cc[1], 2)),
                         box_cm=tuple(round(v, 2) for v in b))

    def render(self) -> list[Detections]:
        rng = self.rng
        out = []
        nb = "notebook" if "notebook" in self.kinds else None
        for i, s in enumerate(self.snaps):
            items, hands = [], []
            hand_box = _box(s["hand"], SIZES["hand"]) if s["hand"] is not None else None
            nb_box = (_box(s["pos"][nb], SIZES[nb])
                      if nb and s["pos"].get(nb) is not None and self._on_table(s["pos"][nb])
                      else None)
            for obj, c in s["pos"].items():
                if c is None or obj in s["hidden"] or not self._on_table(c):
                    continue
                size = SIZES.get(obj, (8, 8))
                ob = _box(c, size)
                p_det, conf = 1 - MISS, rng.uniform(0.55, 0.95)
                if obj in s["carry"]:
                    p_det, conf = (0.08 if s["fist"] else 0.75), rng.uniform(0.4, 0.8)
                elif hand_box is not None and _cover_frac(ob, hand_box) >= 0.5:
                    p_det, conf = 0.4, rng.uniform(0.4, 0.7)
                if nb_box is not None and obj != nb and self.kinds.get(obj) == "target":
                    cf = _cover_frac(ob, nb_box)
                    if cf >= 0.6:
                        continue
                    if cf >= 0.3:
                        p_det *= 0.5
                if rng.random() < p_det:
                    items.append(self._det(obj, c, size, conf))
            if s["hand"] is not None and self._on_table(s["hand"]) and rng.random() < 0.95:
                hands.append(self._det(f"hand:{s['hand_id']}", s["hand"], SIZES["hand"],
                                       rng.uniform(0.6, 0.95)))
            out.append(Detections(t=round(T0 + i / FPS, 4), frame_idx=i, items=items, hands=hands))
        return out


# ------------------------------------------------------------------ layouts & scenarios

def _rand_pt(rng, W, H, margin=12.0):
    return (rng.uniform(margin, W - margin), rng.uniform(margin, H - margin))


def _place(rng, W, H, avoid: list, min_d: float, margin=12.0, tries=400):
    best, best_d = None, -1.0
    for _ in range(tries):
        p = _rand_pt(rng, W, H, margin)
        d = min((_dist(p, a) for a in avoid), default=1e9)
        if d >= min_d:
            return p
        if d > best_d:
            best, best_d = p, d
    return best


def _layout(cfg, rng, obj: str, near_pair: bool = False) -> dict:
    """Random positions: target, box, notebook, and two distractor targets."""
    W, H = cfg["table"]["size_cm"]
    box = _place(rng, W, H, [], 0, margin=14)
    if near_pair:   # ambiguous: box and notebook side by side
        for _ in range(200):
            nb = _place(rng, W, H, [box], 0, margin=14)
            if 24 <= _dist(nb, box) <= 30:
                break
    else:
        nb = _place(rng, W, H, [box], 35, margin=14)
    tgt = _place(rng, W, H, [box, nb], 28, margin=10)
    lay = {obj: tgt, "box": box, "notebook": nb}
    targets = [o for o, k in cfg["objects"].items() if k == "target" and o != obj]
    for o in rng.sample(targets, 2):
        lay[o] = _place(rng, W, H, list(lay.values()), 18, margin=8)
    return lay


def _away(rng, W, H, frm, min_d, avoid=(), margin=10):
    return _place(rng, W, H, [frm, *avoid], min_d, margin=margin)


def _far(rng, W, H, frm, avoid=(), clear=20.0, margin=14, n=80):
    """A spot far across the table from `frm` (farthest of n samples that keep `clear` cm from
    `avoid`), so a moved box really lands somewhere else."""
    pts = [_rand_pt(rng, W, H, margin) for _ in range(n)]
    ok = [p for p in pts if all(_dist(p, a) >= clear for a in avoid)] or pts
    return max(ok, key=lambda p: _dist(p, frm))


def scenario(category: str, cfg: dict, rng: random.Random, obj: str, trial_id: int):
    """Returns (scene, truth dict, notes)."""
    W, H = cfg["table"]["size_cm"]
    lay = _layout(cfg, rng, obj, near_pair=(category == "ambiguous"))
    s = Scene(cfg, rng, lay)
    o0 = lay[obj]
    s.wait(rng.uniform(1.0, 2.0))

    if category == "visible":
        others = [k for k in lay if k not in (obj, "box", "notebook")]
        d = rng.choice(others)
        s.hand_in(lay[d]); s.grab(d)
        s.hand_to(_away(rng, W, H, lay[d], 12, avoid=[o0, lay["box"], lay["notebook"]]))
        s.release(); s.hand_out(); s.wait(1.5)
        return s, make_truth("VISIBLE", pos_cm=s.pos[obj]), f"{d} moved, {obj} untouched"

    if category in ("moved", "put_back"):
        s.hand_in(o0); s.grab(obj)
        if category == "moved":
            s.hand_to(_away(rng, W, H, o0, 20, avoid=[lay["box"], lay["notebook"]]))
        else:
            lift = _away(rng, W, H, o0, 10, avoid=[lay["box"], lay["notebook"]])
            s.hand_to(((o0[0] + lift[0]) / 2, (o0[1] + lift[1]) / 2)); s.wait(0.8)
            s.hand_to((o0[0] + rng.uniform(-1.5, 1.5), o0[1] + rng.uniform(-1.5, 1.5)))
        s.release(); s.hand_out(); s.wait(1.5)
        return s, make_truth("VISIBLE", pos_cm=s.pos[obj]), f"{obj} {category}"

    if category in ("covered", "uncovered"):
        nb0 = lay["notebook"]
        s.hand_in(nb0); s.grab("notebook")
        s.hand_to((o0[0] + rng.uniform(-4, 4), o0[1] + rng.uniform(-3, 3)), speed_cm_s=20)
        s.release(); s.hand_out(); s.wait(rng.uniform(2.0, 4.0))
        if category == "covered":
            return (s, make_truth("UNDER", parent="notebook", pos_cm=s.pos["notebook"]),
                    "notebook slid over the object")
        s.hand_in(s.pos["notebook"]); s.grab("notebook")
        s.hand_to(_away(rng, W, H, o0, 28, margin=14), speed_cm_s=20)
        s.release(); s.hand_out(); s.wait(1.5)
        return s, make_truth("VISIBLE", pos_cm=s.pos[obj]), "covered, then notebook slid away"

    if category in ("inside", "inside_box_moved"):
        s.hand_in(o0); s.grab(obj)
        s.hand_to(lay["box"]); s.put_in(obj, "box"); s.hand_out(); s.wait(rng.uniform(1.5, 3.0))
        note = f"{obj} dropped in the box"
        if category == "inside_box_moved":
            s.hand_in(lay["box"]); s.grab("box")
            s.hand_to(_far(rng, W, H, lay["box"], avoid=[lay["notebook"]]), speed_cm_s=20)
            s.release(); s.hand_out(); s.wait(1.5)
            note += f", box moved {_dist(lay['box'], s.pos['box']):.0f} cm"
        else:
            s.wait(1.0)
        return s, make_truth("INSIDE", parent="box", pos_cm=s.pos["box"]), note

    if category == "carried_away":
        edge = EDGES[trial_id % 4]
        s.hand_in(o0); s.grab(obj); s.hand_out(edge); s.wait(2.0)
        return s, make_truth("GONE", edge=edge), f"carried off the {edge} edge"

    if category == "dropped":
        edge = ("left", "right", "top")[trial_id % 3]
        tgt = {"left": (1.0, o0[1]), "right": (W - 1.0, o0[1]), "top": (o0[0], 1.0)}[edge]
        push_from = {"left": (o0[0] + 9, o0[1]), "right": (o0[0] - 9, o0[1]),
                     "top": (o0[0], o0[1] + 11)}[edge]
        s.hand_in(push_from); s.carry[obj] = (o0[0] - push_from[0], o0[1] - push_from[1])
        off = s.carry[obj]
        s.hand_to((tgt[0] - off[0], tgt[1] - off[1]), speed_cm_s=15)
        s.carry.pop(obj)
        s.pos[obj] = {"left": (-8, o0[1]), "right": (W + 8, o0[1]), "top": (o0[0], -8)}[edge]
        s.wait(0.5); s.hand_out("bottom"); s.wait(2.0)
        return s, make_truth("GONE", edge=edge), f"pushed off the {edge} edge"

    if category == "ambiguous":
        parent = ("box", "notebook")[trial_id % 2]
        first, second = ("notebook", "box") if parent == "box" else ("box", "notebook")
        order = [first, second] if rng.random() < 0.5 else [second, first]
        s.hand_in(o0); s.grab(obj); s.fist = True
        for i, h in enumerate(order):
            s.hand_to(lay[h]); s.wait(0.6)
            if h == "notebook":   # lift the notebook a little and put it back
                s.grab("notebook", dwell=0.2)
                s.hand_to((lay[h][0] + 3, lay[h][1] - 2)); s.wait(0.3)
                s.hand_to(lay[h]); s.release("notebook", dwell=0.1)
                s.fist = obj in s.carry
            if h == parent:
                s.put_in(obj, parent, dwell=0.3)
        s.hand_out(); s.wait(2.0)
        kind = "INSIDE" if parent == "box" else "UNDER"
        return (s, make_truth(kind, parent=parent, pos_cm=s.pos[parent]),
                f"object in a fist; hand visited {' then '.join(order)}; really {parent}")

    raise ValueError(f"no synthetic scenario for category {category!r}")


def generate(out_dir: str, cfg: dict, per_category: int = 5, seed: int = 0,
             categories: Optional[list[str]] = None, overwrite: bool = True) -> list[Trial]:
    cats = categories or CORE
    targets = [o for o, k in cfg["objects"].items() if k == "target"]
    trials = []
    tid = 0
    for c in cats:
        for k in range(per_category):
            tid += 1
            rng = random.Random(seed * 100003 + tid)
            obj = targets[(tid - 1) % len(targets)]
            scene, truth, notes = scenario(c, cfg, rng, obj, tid)
            t = Trial.create(out_dir, tid, c, obj, truth, default_question(cfg, obj), "synth",
                             notes=notes, truth_spec="synth", overwrite=overwrite)
            t.write_detections(scene.render())
            trials.append(t)
    return trials


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="trials_synth")
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stretch", action="store_true", help="also generate 'dropped'")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cats = a.categories or (CORE + STRETCH_SYNTH if a.stretch else CORE)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    trials = generate(a.out, cfg, a.per_category, a.seed, cats)
    print(f"wrote {len(trials)} synthetic trials ({len(cats)} categories x {a.per_category}) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
