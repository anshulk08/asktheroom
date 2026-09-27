"""Laser acceptance test (the demo check): 5 objects in different places, "where is my X?" puts the dot
on the object, within ~5 cm.

For each object: POST /ask "where is my <object>?", poll /state for the laser and the object's full-frame
box (room place box_px), estimate the error in cm from the reported px error and the object's known
width (err_px * width_cm / box width px), and ask the operator for a tape measurement. An object passes
when the dot is on target and the tape (else the estimate, else "the dot is surely inside the box") is
within --tol-cm. Overall PASS with --need passes. The dot stays on at most laser_room.max_on_s (4 s)
and room.room_dwell_s (5 s): measure right away, or raise both on the rig for the test.

    python scripts/laser_acceptance.py --url http://localhost:8080 --objects keys wallet remote glasses pill_bottle
    python scripts/laser_acceptance.py --url ... --objects keys phone --known-width-cm keys=8 phone=15 --json out.json
    python scripts/laser_acceptance.py --sim --no-prompt      # offline: simulated room rig, scene boxes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WIDTH_CM = {"keys": 8, "wallet": 11, "remote": 18, "glasses": 14, "pill_bottle": 4.5, "phone": 15,
            "notebook": 21, "box": 20}


# ---------------------------------------------------------------- scoring (pure)

def est_cm(err_px, box, width_cm) -> Optional[float]:
    """Pixel error scaled by the object's known width over its box width (px per cm at its depth)."""
    if err_px is None or box is None or not width_cm or not math.isfinite(err_px):
        return None
    w = float(box[2]) - float(box[0])
    return None if w <= 0 else float(err_px) * float(width_cm) / w


def dot_in_box(box, target=None, err_px=None, dot=None) -> bool:
    """The dot is inside box: checked directly when its px is known, else the circle of radius err_px
    around the target (box centre if none) fits inside the box."""
    if box is None:
        return False
    x1, y1, x2, y2 = (float(v) for v in box)
    if dot is not None:
        return x1 <= dot[0] <= x2 and y1 <= dot[1] <= y2
    if err_px is None or not math.isfinite(err_px):
        return False
    u, v = target if target is not None else ((x1 + x2) / 2, (y1 + y2) / 2)
    return min(u - x1, x2 - u, v - y1, y2 - v) >= float(err_px)


def score(row: dict, tol_cm: float = 5.0) -> dict:
    """Fill row's est_cm, passed and why from on_target, err_px, err_cm, box, target, dot, width_cm, tape_cm."""
    r = dict(row)
    r["est_cm"] = est_cm(r.get("err_px"), r.get("box"), r.get("width_cm"))
    if r["est_cm"] is None and r.get("err_cm") is not None:
        r["est_cm"] = float(r["err_cm"])                 # table aims report cm themselves
    if not r.get("on_target"):
        r["passed"], r["why"] = False, "laser not on target"
    elif r.get("tape_cm") is not None:
        r["passed"] = r["tape_cm"] <= tol_cm
        r["why"] = f"tape {r['tape_cm']:.1f} cm"
    elif r["est_cm"] is not None:
        r["passed"] = r["est_cm"] <= tol_cm
        r["why"] = f"est {r['est_cm']:.1f} cm"
    elif r.get("box") is None:
        r["passed"], r["why"] = False, "no box"
    else:
        r["passed"] = dot_in_box(r["box"], r.get("target"), r.get("err_px"), r.get("dot"))
        r["why"] = "dot in box" if r["passed"] else "dot not surely in box"
    return r


def verdict(rows: list[dict], need: int = 4) -> dict:
    n = sum(bool(r.get("passed")) for r in rows)
    return {"passed": n, "of": len(rows), "need": need, "ok": n >= need}


def parse_widths(items) -> dict:
    out = dict(WIDTH_CM)
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = float(v)
    return out


# ---------------------------------------------------------------- the running app

def http_json(method: str, url: str, body: Optional[dict] = None, timeout: float = 30.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def room_action(action) -> tuple[Optional[list], Optional[list]]:
    """'room:u,v[,x1,y1,x2,y2]' -> (target, box)."""
    if not isinstance(action, str) or not action.startswith("room:"):
        return None, None
    v = [float(x) for x in action.split(":", 1)[1].split(",")]
    return v[:2], (v[2:6] if len(v) >= 6 else None)


def find_box(state: dict, obj: str) -> Optional[list]:
    """The object's full-frame box: its room place, a thing labelled with its name, or an entity box_px."""
    room = state.get("room") or {}
    names = [obj] + [e.get("name") for e in state.get("entities") or []
                     if e.get("label") in (obj, obj.replace("_", " ")) or obj in (e.get("aliases") or [])]
    for n in names:
        if isinstance(room.get(n), dict) and room[n].get("box_px"):
            return list(room[n]["box_px"])
    for e in state.get("entities") or []:
        if e.get("name") in names and e.get("box_px"):
            return list(e["box_px"])
    return None


def run_live(args, http: Optional[Callable] = None, ask_tape: Callable = input,
             sleep: Callable = time.sleep, clock: Callable = time.monotonic) -> list[dict]:
    url, widths, rows = args.url.rstrip("/"), parse_widths(args.known_width_cm), []
    http = http or http_json

    def laser():
        return (http("GET", f"{url}/state").get("state") or {}).get("laser") or {}

    for obj in args.objects:
        t0 = clock()
        while laser().get("on") and clock() - t0 < args.clear_s:   # the last dot must go off first
            sleep(args.poll_s)
        name = obj.replace("_", " ")
        ans = http("POST", f"{url}/ask", {"text": f"where is my {name}?"})
        target, box = room_action(ans.get("action"))
        las, t0 = {}, clock()
        mine = (None, obj, name, ans.get("point_at"))      # the laser's target, when it names one
        while ans.get("action") or ans.get("point_at"):
            las = laser()
            if (las.get("on") and las.get("target") in mine) or clock() - t0 >= args.wait_s:
                break
            sleep(args.poll_s)
        las = las if las.get("target") in mine else {}      # still another object's dot: not ours
        state = http("GET", f"{url}/state").get("state") or {}
        box = box or find_box(state, obj)
        row = {"object": obj, "answer": ans.get("text", ""), "action": ans.get("action"),
               "laser_on": bool(las.get("on")), "on_target": bool(las.get("on") and las.get("on_target", True)),
               "err_px": las.get("err_px"), "err_cm": las.get("err_cm"), "box": box, "target": target,
               "width_cm": widths.get(obj), "tape_cm": None}
        print(f"{obj}: {row['answer']}")
        if row["laser_on"] and not args.no_prompt:
            s = ask_tape("  tape: dot to object centre in cm (Enter to skip) ").strip()
            try:
                row["tape_cm"] = float(s) if s else None
            except ValueError:
                print(f"  not a number: {s!r}; skipped")
        rows.append(score(row, args.tol_cm))
    return rows


# ---------------------------------------------------------------- offline (simulated room)

SIM_OBJECTS = ("bottle", "backpack", "shelf", "table", "coffee_table")


def run_sim(args) -> list[dict]:
    """Sweep the simulated room (act/sim.RoomRig), then aim at 5 scene boxes' centres; the 'tape' is the
    true 3D distance from the dot to the surface point at the target pixel."""
    import numpy as np
    from act.room_map import sweep
    from act.sim import RoomRig
    rig = RoomRig(b_cm=3.0, seed=args.seed)
    laser = rig.make_laser()
    rm = sweep(laser, grid=tuple(args.grid), n_pairs=1)
    rows = []
    for obj in SIM_OBJECTS:
        lo, hi = rig.scene.boxes[obj][:2]
        box = rig.box_px(obj)
        row = {"object": obj, "answer": "", "action": None, "box": list(box) if box else None,
               "width_cm": float(hi[0] - lo[0]), "tape_cm": None, "err_cm": None}
        if box is not None:
            t = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            r = laser.aim_px(t, box, room_map=rm)
            s, _ = rig.scene.cast(rig.scene.cam, rig.scene.ray(t))
            want = rig.scene.cam + s * rig.scene.ray(t)
            pan, tilt, _ = rig.act.state_at(rig.clock.now())
            got, _ = rig.geom.hit3d(pan, tilt)
            row.update(target=t, laser_on=True, on_target=r.on_target, dot=r.dot_px, answer=r.reason,
                       err_px=None if math.isinf(r.err_px) else r.err_px,
                       tape_cm=None if got is None else float(np.linalg.norm(got - want)))
            laser.off()
        rows.append(score(row, args.tol_cm))
    return rows


# ---------------------------------------------------------------- report

def table(rows: list[dict]) -> str:
    def f(v):
        return "-" if v is None else f"{v:.1f}"
    out = [f"{'object':<13}{'on':<4}{'err_px':>7}{'est_cm':>7}{'tape':>6}  {'box':<4}{'result':<6} why"]
    for r in rows:
        out.append(f"{r['object']:<13}{'y' if r.get('on_target') else 'n':<4}{f(r.get('err_px')):>7}"
                   f"{f(r.get('est_cm')):>7}{f(r.get('tape_cm')):>6}  {'y' if r.get('box') else 'n':<4}"
                   f"{'PASS' if r['passed'] else 'FAIL':<6} {r['why']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--objects", nargs="+", default=["keys", "wallet", "remote", "glasses", "pill_bottle"])
    ap.add_argument("--known-width-cm", nargs="*", default=[], metavar="OBJ=CM")
    ap.add_argument("--tol-cm", type=float, default=5.0)
    ap.add_argument("--need", type=int, default=4, help="objects that must pass")
    ap.add_argument("--no-prompt", action="store_true", help="don't ask for tape measurements")
    ap.add_argument("--wait-s", type=float, default=12.0, help="how long to wait for the dot after asking")
    ap.add_argument("--clear-s", type=float, default=12.0, help="how long to wait for the last dot to go off")
    ap.add_argument("--poll-s", type=float, default=0.3)
    ap.add_argument("--json", help="write the rows and verdict here")
    ap.add_argument("--sim", action="store_true", help="offline against the simulated room rig")
    ap.add_argument("--grid", type=int, nargs=2, default=[12, 9], metavar=("NX", "NY"), help="--sim sweep grid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = run_sim(args) if args.sim else run_live(args)
    v = verdict(rows, args.need)
    print(table(rows))
    print(f"{'PASS' if v['ok'] else 'FAIL'}: {v['passed']}/{v['of']} objects (need {v['need']}, tol {args.tol_cm} cm)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "verdict": v, "tol_cm": args.tol_cm, "sim": args.sim,
                       "t": time.time()}, fh, indent=1, default=str)
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
