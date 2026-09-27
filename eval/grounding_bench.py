"""Moondream grounding on real room frames: does /detect find each named thing, and where?

    python -m eval.grounding_bench --frames DIR [--gt DIR/gt.json] [--env-file ../.secrets/moondream.env]
        [--zones room_zones.json] [--table-rect x1,y1,x2,y2] [--names keys,wallet] [--no-crops]
        [--grok] [--out results.json]
    python -m eval.grounding_bench --frames DIR --template      # writes DIR/gt.json to fill in

DIR holds raw full camera frames (2560x1440 .jpg/.png, not in git: people are in them) and gt.json:

    {"f1.jpg": {"keys": [x1, y1, x2, y2], "wallet": null}, ...}     # full-frame px; null = not in the frame

For every name x frame it runs the rig's search (core.grounding.find_anywhere: /detect on the whole frame,
then, after a miss, on close-ups of the zones) and scores:
  - full:     the first box /detect returns on the whole frame, unverified;
  - anywhere: what find_anywhere returns, i.e. what the rig would say: boxes confirmed by a closed /query
              (grounding.verify; --no-verify skips it), whole frame then zone close-ups (--no-crops: whole
              frame only).
A hit is IoU >= 0.3 with the true box or the predicted centre inside it; place is the same zone / table /
elsewhere as the true box (core.grounding.place_of). On a name marked null, any box is a false positive.
Latency p50/p90 per HTTP call and per search. --grok also asks Grok's whole-room look (VisualQA.look_room,
what the rig says today, needs XAI_API_KEY) "Where is my <name>?" and scores its words by place: a hit names
the true box's zone (or "table") without saying it doesn't see it; on a null name, any claimed place is a
false positive. Grok gives no box, so it is compared on place only.

Keys: MOONDREAM_API_KEY (and XAI_API_KEY for --grok) from the environment or --env-file (KEY=VALUE lines;
never printed). Calls are spaced 0.5 s apart (Moondream's 2 req/s); 3 names x 10 frames is about 30-120 calls.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG = (".jpg", ".jpeg", ".png")
NEG = re.compile(r"\b(?:don'?t|do not|can'?t|cannot|couldn'?t|not) (?:see|find|spot|tell|make out)\b|\bno sign\b"
                 r"|\bisn'?t (?:in view|there|visible)\b")


def load_env(path: str) -> list[str]:
    """KEY=VALUE lines (optionally 'export KEY=...') into os.environ, unless already set. Returns the key
    names loaded (never the values)."""
    out = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if not m or line.lstrip().startswith("#"):
            continue
        k, v = m.group(1), m.group(2).strip().strip("'\"")
        if v and not os.environ.get(k):
            os.environ[k] = v
            out.append(k)
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def is_hit(pred, gt, thr: float = 0.3) -> bool:
    """IoU >= thr, or pred's centre inside gt."""
    if pred is None or gt is None:
        return False
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return iou(pred, gt) >= thr or (gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3])


def pct(xs: list, q: float):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


class Recorder:
    """A grounder wrapper that records each detect call's boxes and latency."""

    def __init__(self, g):
        self.g, self.c, self.calls = g, g.c, []

    def detect(self, img, phrase, deadline=None):
        t0 = time.perf_counter()
        boxes = self.g.detect(img, phrase, deadline=deadline)
        self.calls.append({"task": "detect", "wh": [img.shape[1], img.shape[0]],
                           "ms": int((time.perf_counter() - t0) * 1000), "boxes": [list(b[:4]) for b in boxes],
                           "status": (self.g.last or {}).get("status")})
        return boxes

    def confirm(self, img, box, phrase, deadline=None):
        t0 = time.perf_counter()
        ok = self.g.confirm(img, box, phrase, deadline=deadline)
        self.calls.append({"task": "query", "box": list(box[:4]), "ok": ok,
                           "ms": int((time.perf_counter() - t0) * 1000), "status": (self.g.last or {}).get("status")})
        return ok

    def point(self, img, phrase, deadline=None):
        return self.g.point(img, phrase, deadline=deadline)


def template(frames: Path, names: list[str]) -> Path:
    files = sorted(p.name for p in frames.iterdir() if p.suffix.lower() in IMG)
    gt = {f: {n: None for n in names} for f in files}
    out = frames / "gt.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text(json.dumps(gt, indent=1))
    return out


def grok_asker(cfg: dict, zones: list, rect, tmp: str):
    """(img, name) -> (answer, s): VisualQA.look_room on img as the live full frame (eval.room_look's
    harness), so the answer is what the rig says today."""
    from core.events import EventLog
    from eval.room_look import Frames, _cfg, _qa
    zd = {n: {"say": say, "poly": [list(p) for p in poly]} for n, say, poly in zones}

    def ask(img, name):
        ev = EventLog(":memory:", os.path.join(tmp, "snaps"))
        try:
            h, w = img.shape[:2]
            r = rect or (0, 0, w, h)
            q = _qa(_cfg(zd, tmp, (w, h)), zd, Frames(img, r, time.time()), ev)
            t0 = time.perf_counter()
            a = q.look_room(f"Where is my {name}?")
            return a.text, time.perf_counter() - t0
        finally:
            ev.close()
    return ask


def grok_score(answer: str, gt, gt_place: str, zones: list) -> tuple[bool, bool]:
    """(hit, false positive) for a Grok answer, by place words."""
    low = (answer or "").lower()
    neg = bool(NEG.search(low)) or "i can't tell" in low
    if gt is None:
        return False, not neg
    if neg:
        return False, False
    words = {"table": ["table"]}
    for n, say, _ in zones:
        words[n] = [re.sub(r"^(?:the|a|an) ", "", say.lower()), n.replace("_", " ")]
    if gt_place == "table":
        ok = re.search(r"\btable\b", low) is not None and not any(
            w in low for n, ws in words.items() if n != "table" for w in ws if "table" in w)
    else:
        ok = any(w in low for w in words.get(gt_place, []))
    return ok, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder of raw full frames (and gt.json)")
    ap.add_argument("--gt", help="default: FRAMES/gt.json")
    ap.add_argument("--env-file", help="KEY=VALUE file for MOONDREAM_API_KEY / XAI_API_KEY (never printed)")
    ap.add_argument("--zones", help="room_zones.json (default: room_memory.zones_path, then FRAMES/room_zones.json)")
    ap.add_argument("--table-rect", help="x1,y1,x2,y2 full-frame px of the table view (default: room_memory.table_view_rect)")
    ap.add_argument("--names", help="comma-separated names (default: every name in gt.json)")
    ap.add_argument("--no-crops", action="store_true", help="whole frame only (no zone close-ups)")
    ap.add_argument("--no-verify", action="store_true", help="say /detect's box without the /query check")
    ap.add_argument("--grok", action="store_true", help="also ask Grok's whole-room look (needs XAI_API_KEY)")
    ap.add_argument("--template", action="store_true", help="write FRAMES/gt.json with every name null, and stop")
    ap.add_argument("--out", help="write every result as JSON")
    a = ap.parse_args(argv)
    for k in ("frames", "gt", "env_file", "zones", "out"):          # the user's paths, before the chdir
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).expanduser().resolve()))
    sys.path.insert(0, str(HERE.parent))
    os.chdir(HERE.parent)
    import cv2

    from core.config import load_config
    from core.grounding import GroundingConfig, MoondreamGrounder, find_anywhere, load_zones, place_of
    frames = Path(a.frames)
    cfg = load_config()
    if a.template:
        names = a.names.split(",") if a.names else [(cfg.get("display_names") or {}).get(n, n.replace("_", " "))
                                                    for n, k in (cfg.get("objects") or {}).items() if k == "target"]
        print(f"wrote {template(frames, names)}: set each name's full-frame box [x1, y1, x2, y2], or leave null "
              f"where it isn't in the frame")
        return 0
    if a.env_file:
        print(f"loaded from {a.env_file}: {', '.join(load_env(a.env_file)) or 'nothing new'}")
    gt = json.loads(Path(a.gt or frames / "gt.json").read_text())
    names = a.names.split(",") if a.names else sorted({n for v in gt.values() for n in v})
    c = GroundingConfig.from_dict({**(cfg.get("grounding") or {}), "enabled": True, "max_per_minute": 10 ** 6,
                                   "max_per_hour": 10 ** 6, "max_wait_s": 5.0})
    if a.no_crops:
        c.zone_crops = 0
    if a.no_verify:
        c.verify = False
    g = MoondreamGrounder(c)
    if not g.available():
        print(f"${c.api_key_env} is not set (use --env-file): nothing to run.")
        return 2
    room = cfg.get("room_memory") or {}
    zpath = a.zones or room.get("zones_path") or "room_zones.json"
    if not Path(zpath).exists() and (frames / "room_zones.json").exists():
        zpath = str(frames / "room_zones.json")
    rect = tuple(int(v) for v in a.table_rect.split(",")) if a.table_rect else room.get("table_view_rect")
    ask_grok = None
    tmp = tempfile.mkdtemp(prefix="grounding_bench_")
    rows, http_ms, query_ms, search_ms, grok_s = [], [], [], [], []
    for fname in sorted(gt):
        img = cv2.imread(str(frames / fname))
        if img is None:
            print(f"!! can't read {fname}; skipped")
            continue
        h, w = img.shape[:2]
        zones = load_zones(zpath, (w, h))
        if a.grok and ask_grok is None:
            ask_grok = grok_asker(cfg, zones, rect, tmp)
        for name in names:
            if name not in gt[fname]:
                continue
            truth = gt[fname][name]
            truth = tuple(float(v) for v in truth) if truth else None
            true_place = place_of(truth, zones, rect, (w, h), c.near_px)[0] if truth else None
            rec = Recorder(g)
            t0 = time.perf_counter()
            found = find_anywhere(rec, img, name, zones, rect, None, c)
            total = int((time.perf_counter() - t0) * 1000)
            full = rec.calls[0] if rec.calls else None
            # the whole frame's first box as /detect gave it (no verify): what a detect-only rig would say
            first = tuple(full["boxes"][0]) if full and full.get("boxes") else None
            r = {"frame": fname, "name": name, "truth": truth, "true_place": true_place,
                 "full_box": first, "full_hit": is_hit(first, truth), "full_fp": truth is None and first is not None,
                 "box": found.box if found else None, "where": found.where if found else None,
                 "place": found.place if found else None, "source": found.source if found else None,
                 "hit": is_hit(found.box if found else None, truth),
                 "place_hit": bool(found and truth and found.place == true_place),
                 "fp": truth is None and found is not None, "calls": rec.calls, "ms": total,
                 "full_status": full and full["status"]}
            http_ms += [x["ms"] for x in rec.calls if x["task"] == "detect"]
            query_ms += [x["ms"] for x in rec.calls if x["task"] == "query"]
            search_ms.append(total)
            line = (f"{fname} {name!r}: truth {true_place or 'absent'}; moondream "
                    f"{(found.where + ' via ' + found.source) if found else 'nothing'} "
                    f"[{'HIT' if r['hit'] else 'FP' if r['fp'] else 'ok' if truth is None else 'miss'}] "
                    f"{total} ms, {len(rec.calls)} calls")
            if ask_grok is not None:
                ans, s = ask_grok(img, name)
                gh, gfp = grok_score(ans, truth, true_place, zones)
                r.update({"grok": ans, "grok_hit": gh, "grok_fp": gfp, "grok_s": round(s, 2)})
                grok_s.append(s)
                line += f" | grok [{'HIT' if gh else 'FP' if gfp else 'ok' if truth is None else 'miss'}] {s:.1f} s: {ans}"
            print(line, flush=True)
            rows.append(r)
    present = [r for r in rows if r["truth"] is not None]
    absent = [r for r in rows if r["truth"] is None]

    def rate(k, rs):
        return f"{sum(1 for r in rs if r.get(k))}/{len(rs)}"

    print("\n== summary ==")
    print(f"present {len(present)}, absent {len(absent)}")
    print(f"moondream full frame : hit {rate('full_hit', present)}, false positives {rate('full_fp', absent)}")
    print(f"moondream + close-ups: hit {rate('hit', present)} (place right {rate('place_hit', present)}), "
          f"false positives {rate('fp', absent)}")
    print(f"latency per /detect p50 {pct(http_ms, 0.5)} ms, p90 {pct(http_ms, 0.9)} ms; per /query p50 "
          f"{pct(query_ms, 0.5)} ms, p90 {pct(query_ms, 0.9)} ms; "
          f"per search p50 {pct(search_ms, 0.5)} ms, p90 {pct(search_ms, 0.9)} ms")
    if grok_s:
        print(f"grok room look (place): hit {rate('grok_hit', present)}, false positives {rate('grok_fp', absent)}; "
              f"p50 {pct(grok_s, 0.5):.2f} s, p90 {pct(grok_s, 0.9):.2f} s")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=list))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
