"""Replay a rig hour through the visual archive's save policy and count what it would keep.

    python -m eval.archive_replay --frames DIR --rows rows.json [--outline-px "x,y x,y ..."] [--set settle_checks=2 ...]

rows.json: {"frames": [[t, path, score, hands, reason], ...], "events": [[wall, obj, type], ...]}, the
visual_frames rows and world events of the hour (read-only from the rig's events.db); DIR holds that hour's
archive JPEGs (data/snapshots/archive/YYYYMMDD-HH). Each saved frame is fed to VisualArchive._check in order
as one check (hands as recorded; dirty when a world event fell since the previous check), with the JPEG
writes stubbed out. The rig saved nearly every check, so its saved frames stand in for the check stream.
--outline-px is a tabletop outline in table-view px, as `python -m core.table --outline-px` would save it;
it defines the truth below, and with --gate-outline the archive gets it too (as a calibrated table whose
cm are px).
Prints saves per hour by reason, and the truth: tabletop states, i.e. runs of checks >= 5 s long where the
outlined tabletop holds still (< 1% of its thumbnail pixels change from check to check) and differs from the
previous state; a state is kept when some saved frame falls inside it. Without an outline the whole view is
the tabletop.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


class PxTable:
    """A calibrated table whose cm are frame px, so table_area.polygon_cm is an outline in px."""
    ok = True

    @staticmethod
    def cm_to_px(pts):
        return pts


def states(times: list, thumbs: list, mask, still_pct: float = 1.0, min_s: float = 5.0, new_pct: float = 0.3) -> list:
    """(t0, t1) of each tabletop state (see the module docstring)."""
    from core.visual_memory import change_pct
    runs, start = [], 0
    for i in range(1, len(thumbs) + 1):
        if i == len(thumbs) or change_pct(thumbs[i], thumbs[i - 1], mask=mask) >= still_pct:
            if times[i - 1] - times[start] >= min_s:
                runs.append((start, i - 1))
            start = i
    out, ref = [], None
    for a, b in runs:
        if ref is None or change_pct(thumbs[a], ref, mask=mask) >= new_pct:
            out.append((times[a], times[b]))
            ref = thumbs[a]
    return out


def replay(frames_dir: str, rows: dict, overrides: dict, outline_px=None, gate_outline: bool = False) -> dict:
    import cv2
    from core.config import load_config
    from core.events import EventLog
    from core.types import Frame
    from core.visual_memory import VisualArchive, VisualConfig
    cfg = load_config()
    table = None
    if outline_px:
        cfg["table_area"] = {**(cfg.get("table_area") or {}), "polygon_cm": outline_px}
    if outline_px and gate_outline:
        table = PxTable()
    c = VisualConfig.from_dict({**cfg["visual_memory"], "embed": {"embed": "none"}})
    for k, v in overrides.items():
        setattr(c, k, v)
    from core.visual_memory import _thumb, table_mask
    evs = sorted(e[0] for e in rows["events"])
    saved: list = []
    times, thumbs, mask = [], [], None
    with tempfile.TemporaryDirectory() as tmp:
        log = EventLog(":memory:", tmp)
        a = VisualArchive(cfg, log, None, embedder=None, start=False, c=c, clock=lambda: 0.0, table=table)
        a._write = lambda img, wall, px, suffix="": (img, "", 0)          # count, don't write
        prev = None
        for t, path, _score, hands, _reason in rows["frames"]:
            img = cv2.imread(os.path.join(frames_dir, os.path.basename(path)))
            if img is None:
                continue
            times.append(t)
            thumbs.append(_thumb(img))
            if outline_px and mask is None:
                mask = table_mask(PxTable(), cfg, (img.shape[1], img.shape[0]))
            dirty = prev is not None and bisect.bisect_right(evs, t) > bisect.bisect_right(evs, prev)
            if a._check(Frame(t=t, wall=t, img=img, idx=0), bool(hands), dirty) is not None:
                saved.append(t)
            prev = t
        rows_out = a.store.window(0, 1e12)
        log.close()
    span_h = (times[-1] - times[0]) / 3600 or 1.0
    truth = states(times, thumbs, mask)
    delays = []
    for t0, t1 in truth:
        i = bisect.bisect_left(saved, t0)
        if i < len(saved) and saved[i] <= t1:
            delays.append(saved[i] - t0)
    return {"checks": len(times), "saves": len(saved), "per_hour": round(len(saved) / span_h),
            "reasons": dict(collections.Counter(r.reason for r in rows_out)),
            "states": len(truth), "states_kept": len(delays),
            "median_delay_s": round(sorted(delays)[len(delays) // 2], 1) if delays else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--set", action="append", default=[], help="VisualConfig field=value")
    ap.add_argument("--outline-px", help='tabletop outline in table-view px: "x,y x,y x,y ..." (the truth)')
    ap.add_argument("--gate-outline", action="store_true", help="the archive measures change inside it too")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    os.environ["ASKROOM_NO_LOCAL_CONFIG"] = "1"
    over = {k: _value(v) for k, v in (s.split("=", 1) for s in a.set)}
    outline = [[float(v) for v in p.split(",")] for p in a.outline_px.split()] if a.outline_px else None
    print(json.dumps({"set": over, "gate_outline": a.gate_outline,
                      **replay(a.frames, json.loads(Path(a.rows).read_text()), over, outline, a.gate_outline)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
