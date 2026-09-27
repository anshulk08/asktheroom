"""Naming accuracy of core/auto_name (Grok's guess for an unnamed thing) on hand-labelled rig crops.

    set -a && . ./.env && set +a
    .venv/bin/python -m eval.naming --root <dir> [--variants baseline,new] [--json out.json]
    .venv/bin/python -m eval.naming --list          # the rig paths the labels need, one per line

eval/naming_labels.json holds 121 items from the room rig (Sat 26 Sep, ~/askroom_room), labelled by
looking at each crop (WS3, Sun 27 Sep): 64 table things (the APPEARED snapshot, a 1280x720 table view, and
the thing's box in it, recovered from the event's position and the YOLOE proposal there) and 57 room
tracks (the marked crop the room namer saved, ASKROOM_ROOM_CROPS). `label` is what the object really
is, or null when it is no object: a hand, an arm, a watch or sock being worn, a person, jeans, a table
leg. `accept` lists the names that count as right for each label. The images show people, so they are
not in the repo: copy them read-only into --root, keeping the rig's relative paths:

    .venv/bin/python -m eval.naming --list | ssh guru@<rig> 'cd ~/askroom_room && tar cf - $(cat)' | tar xf - -C <root>

Variants (each item goes to Grok once per variant; replies are cached by prompt and image bytes in
--cache, so a rerun costs nothing):
  baseline    the Sat 26 Sep namer: a 128 px crop of the table view (the crop store's size), enlarged to
              384 px at JPEG q85, the overhead-camera prompt, min_confidence 0.5, no object filter
  old-native  the baseline prompt and rules on the native-resolution close-up (the crop change alone)
  new-128     the new prompt, rules and context view on the 128 px crop (the prompt change alone)
  new         core.auto_name as it is: the native close-up (the table view shrunk back to its native
              ~817 px width, as TableView.full_at gives it at 1440p) plus the marked context view
  new-noctx   new without the context view
  new-retry   an experiment, not the app: new, and a reply that is an object but no usable name
              (core.auto_name.unsure) asked once more with a view of RETRY_GROW x the box (at least
              RETRY_MIN_PX); table items only (a saved room crop is all the context there is). Sun 27 Sep,
              3 runs: +0.3 right names, +5 wrong, +2.7 no-objects named, so AutoNamer does not do it
A room item's close-up is the inside of its red box; its context view is the saved crop itself.

Scoring: a real object is right when the kept name fits a label's accepted name (core.auto_name
match_score >= 2 either way: the same head noun, one's words inside the other's); a no-object item is
right when no name is kept. 'wrong' is a kept name that fits nothing: the harm (a hand called a watch
is found when someone asks for their watch). The baseline variants keep names at 0.5 (Sat 26 Sep), the
new ones at config.yaml's auto_name.min_confidence; --sweep prints them at other thresholds. Grok is not
deterministic: runs differ by a few items (--trial N asks again; WS3 reports the mean of 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auto_name import (GENERIC, MAX_WORDS, NAME_SCHEMA, NAME_SYSTEM, AutoNameConfig,  # noqa: E402
                            _jpeg, judge, match_score, unsure)
from core.crops import close_up, marked_view  # noqa: E402
from core.narration import Reply, _parse_json  # noqa: E402
from core.things import norm_name  # noqa: E402

LABELS = Path(__file__).with_name("naming_labels.json")
VIEW_W = 1280                     # table-view snapshot width
STORE_PX = 128                    # the crop store's size_px (config.yaml proposals.crops)
RETRY_GROW, RETRY_MIN_PX = 6.0, 480
VARIANTS = ("baseline", "old-native", "new-128", "new", "new-noctx", "new-retry")

# The Sat 26 Sep prompt, schema and rules (core/auto_name.py before WS3), kept here as the baseline.
OLD_SYSTEM = """You name one object from an overhead close-up of a tabletop (the camera looks straight down).
Reply with the everyday name a person would use when asking where it is.

Rules:
- name: a common noun of 1 to 3 words, lowercase, no brand, no colour, e.g. "deodorant stick", "coffee mug", "phone charger".
- also: up to 3 other short names people might say for it, e.g. "deodorant"; an empty list if there are none.
- confidence: 0 to 1. If you can't tell what it is, set it below 0.5.
- If it is a medicine or pill bottle, just name it plainly ("pill bottle"). Never say anything about medication being taken, its contents or its use.
- Name the main object in the middle of the image only; ignore hands and the table.
Reply with the JSON object only."""
OLD_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["name", "also", "confidence"],
    "properties": {"name": {"type": "string"}, "also": {"type": "array", "items": {"type": "string"}},
                   "confidence": {"type": "number"}},
}


def old_judge(d: dict, min_confidence: float = 0.5) -> Optional[dict]:
    """The Sat 26 Sep AutoNamer._ask rules: clean name (generic words only are none), no med claim,
    confidence >= min_confidence."""
    from core.narration_store import med_claim
    try:
        conf = min(1.0, max(0.0, float(d.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    words = [w for w in norm_name(str(d.get("name") or "")).split() if not w.isdigit()][:MAX_WORDS]
    name = " ".join(words)
    if not name or name in GENERIC or all(w in GENERIC for w in words) or med_claim(name) or conf < min_confidence:
        return None
    return {"name": name, "also": [], "confidence": round(conf, 3)}


class CachedProvider:
    """provider.narrate with replies cached on disk by (system, schema, parts); records the raw reply
    text of the last call per thread in .last."""

    def __init__(self, provider, cache: Path, salt: str = ""):
        self.p, self.cache, self.salt = provider, cache, salt
        self.name, self.model = getattr(provider, "name", "grok"), getattr(provider, "model", None)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls, self.latency = 0, []

    def narrate(self, system, parts, schema):
        h = hashlib.sha256(self.salt.encode() + system.encode() + json.dumps(schema, sort_keys=True).encode())
        for k, v in parts:
            h.update(k.encode() + (v if isinstance(v, bytes) else str(v).encode()))
        f = self.cache / f"{h.hexdigest()}.json"
        if f.exists():
            return Reply(json.loads(f.read_text())["text"], {}, 0)
        t0 = time.perf_counter()
        r = self.p.narrate(system, parts, schema)
        self.latency.append(time.perf_counter() - t0)
        self.calls += 1
        f.write_text(json.dumps({"text": r.text}))
        return r


def _red_box(img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """The inside of the red box core.crops.marked_view drew on a saved room crop, or None."""
    b, g, r = (img[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero((r > 150) & (r - g > 100) & (r - b > 100))
    if len(xs) < 8:
        return None
    t = max(2, round(max(img.shape[:2]) / 120))
    x1, y1, x2, y2 = xs.min() + 2 * t, ys.min() + 2 * t, xs.max() - 2 * t + 1, ys.max() - 2 * t + 1
    return (x1, y1, x2, y2) if x2 - x1 >= 4 and y2 - y1 >= 4 else None


def views(item: dict, root: Path, native: bool, context: bool, margin: float = 0.15, min_side: int = 240,
          grow: float = 3.0):
    """(close-up, context view or None) for an item, as the given namer would have cut it."""
    img = cv2.imread(str(root / item["src"]))
    if img is None:
        raise FileNotFoundError(root / item["src"])
    if item["kind"] == "room":
        box = _red_box(img)
        crop = img if box is None else img[box[1]:box[3], box[0]:box[2]].copy()
        if not native:
            crop = _shrink(crop, STORE_PX)
        return crop, (img if context and box is not None else None)
    box = item["box"]
    if native:                    # the table view is an enlargement of native_w px of the full frame
        s = item["native_w"] / VIEW_W
        img = cv2.resize(img, (item["native_w"], round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        box = [v * s for v in box]
        return close_up(img, box, margin), (marked_view(img, box, min_side, grow) if context else None)
    crop = _shrink(close_up(img, box, margin), STORE_PX)          # core.crops.CropStore._cut
    return crop, (marked_view(img, box, min_side) if context else None)


def _shrink(img: np.ndarray, px: int) -> np.ndarray:
    s = px / max(img.shape[:2])
    return cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA) if s < 1 else img


def fits(name: Optional[str], accept: list) -> bool:
    return bool(name) and any(norm_name(a) == name or match_score(a, {"name": name}) >= 2
                              or match_score(name, {"name": a}) >= 2 for a in accept)


def ask(variant: str, item: dict, root: Path, provider: CachedProvider, system: str = NAME_SYSTEM,
        min_conf: float = 0.65) -> dict:
    """Grok's raw reply for one item under one variant: {"raw": parsed reply} (new-retry: the reply that
    counts, and "first" when it asked twice)."""
    if variant == "new-retry":
        first = ask("new", item, root, provider, system)
        if item["kind"] != "table" or not unsure(first["raw"], min_conf):
            return first
        img, ctx = views(item, root, True, True, min_side=RETRY_MIN_PX, grow=RETRY_GROW)
        return {"raw": _ask_new(img, ctx, provider, system), "first": first["raw"]}
    native = variant in ("old-native", "new", "new-noctx")
    context = variant in ("new", "new-128")
    img, ctx = views(item, root, native, context)
    if variant in ("baseline", "old-native"):
        parts = [("text", "Close-up of one object on the table:"), ("image", _jpeg(img, 384, 85)),
                 ("text", "What is it called?")]
        d = _parse_json(provider.narrate(OLD_SYSTEM, parts, OLD_SCHEMA).text)
    else:
        d = _ask_new(img, ctx, provider, system)
    return {"raw": d}


def _ask_new(img, ctx, provider, system: str) -> dict:
    """The parsed reply to AutoNamer's request for (img, ctx), with system as its prompt."""
    c = AutoNameConfig()
    parts = [("text", "Close-up of the object:"), ("image", _jpeg(img, c.crop_px, c.jpeg_quality))]
    if ctx is not None:
        parts += [("text", "The same spot, wider; the object is in the red box:"),
                  ("image", _jpeg(ctx, c.context_px, c.jpeg_quality))]
    return _parse_json(provider.narrate(system, parts + [("text", "What is it called?")], NAME_SCHEMA).text)


def kept(variant: str, raw: dict, min_conf: float) -> Optional[dict]:
    return old_judge(raw, min_conf) if variant in ("baseline", "old-native") else judge(raw, min_conf)


def score(items: list, accept: dict, replies: dict, variant: str, min_conf: float) -> dict:
    real = [it for it in items if it["label"]]
    none = [it for it in items if not it["label"]]
    res = {"right": 0, "real": len(real), "real_right": 0, "real_wrong": 0, "real_abstain": 0,
           "none": len(none), "none_rejected": 0, "none_named": 0, "wrong": []}
    for it in items:
        g = kept(variant, replies[it["id"]]["raw"], min_conf)
        name = g["name"] if g else None
        if it["label"]:
            if name is None:
                res["real_abstain"] += 1
            elif fits(name, accept[it["label"]]):
                res["real_right"] += 1
            else:
                res["real_wrong"] += 1
                res["wrong"].append((it["id"], it["label"], name))
        elif name is None:
            res["none_rejected"] += 1
        else:
            res["none_named"] += 1
            res["wrong"].append((it["id"], it["note"], name))
    res["right"] = res["real_right"] + res["none_rejected"]
    named = res["real_right"] + res["real_wrong"] + res["none_named"]
    res["precision"] = round(res["real_right"] / named, 3) if named else None
    res["accuracy"] = round(res["right"] / len(items), 3)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="dir holding the rig's data/ files (see --list)")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--variants", default="baseline,new")
    ap.add_argument("--cache", default=str(Path.home() / ".cache/askroom/naming_eval"))
    ap.add_argument("--workers", type=int, default=3, help="concurrent Grok calls (grok-4.3 allows ~3/s)")
    ap.add_argument("--sweep", action="store_true", help="the new variants at min_confidence 0.3 to 0.9")
    ap.add_argument("--json", help="write every reply and score here")
    ap.add_argument("--prompt", help="a file with a system prompt to try instead of core.auto_name.NAME_SYSTEM")
    ap.add_argument("--trial", type=int, default=0, help="a repeat number: >0 asks Grok again (run-to-run spread)")
    ap.add_argument("--list", action="store_true", help="print the rig paths of the labelled images")
    a = ap.parse_args(argv)
    lab = json.loads(Path(a.labels).read_text())
    items, accept = lab["items"], lab["accept"]
    if a.list:
        print("\n".join(sorted({it["src"] for it in items})))
        return 0
    if not a.root:
        ap.error("--root is required")
    if not os.environ.get("XAI_API_KEY"):
        print("XAI_API_KEY not set (set -a && . ./.env && set +a); cached replies only", file=sys.stderr)
    from core.config import load_config
    from core.narration import NarrationConfig, make_provider
    from core.visual_memory import VisualConfig
    cfg = load_config()
    v = VisualConfig.from_dict(cfg.get("visual_memory"))
    min_conf = AutoNameConfig.from_dict(cfg.get("auto_name")).min_confidence
    provider = CachedProvider(make_provider(NarrationConfig.from_dict({
        "provider": v.provider, "model": v.model, "base_url": v.base_url, "api_key_env": v.api_key_env,
        "reasoning_effort": v.reasoning_effort, "timeout_s": 20, "max_tokens": 200})), Path(a.cache),
        salt=f"trial{a.trial}" if a.trial else "")
    system = Path(a.prompt).read_text().strip() if a.prompt else NAME_SYSTEM
    root = Path(a.root)
    variants = [x for x in a.variants.split(",") if x]
    out = {"variants": {}}
    for var in variants:
        if var not in VARIANTS:
            ap.error(f"unknown variant {var}; one of {', '.join(VARIANTS)}")

        def one(it, var=var):
            for attempt in range(3):
                try:
                    return it["id"], ask(var, it, root, provider, system, min_conf)
                except FileNotFoundError:
                    raise
                except Exception as e:                    # the key is never printed, only the error class
                    err = f"{type(e).__name__}"
                    time.sleep(2 * (attempt + 1))
            return it["id"], {"raw": {}, "error": err}

        with ThreadPoolExecutor(max(1, a.workers)) as ex:
            replies = dict(ex.map(one, items))
        mc = 0.5 if var in ("baseline", "old-native") else min_conf
        res = score(items, accept, replies, var, mc)
        out["variants"][var] = {"min_confidence": mc, "score": res, "replies": replies}
        errs = sum(1 for r in replies.values() if r.get("error"))
        print(f"{var:>10} @{mc:.2f}: right {res['right']}/{len(items)} ({res['accuracy']:.0%}) | real {res['real_right']}/"
              f"{res['real']} right, {res['real_wrong']} wrong, {res['real_abstain']} no name | no-object "
              f"{res['none_rejected']}/{res['none']} rejected, {res['none_named']} named | precision "
              f"{res['precision']}" + (f" | {errs} errors" if errs else ""))
        if a.sweep and var not in ("baseline", "old-native"):
            for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                r = score(items, accept, replies, var, t)
                print(f"{'':>10}  min_conf {t:.1f}: right {r['right']}/{len(items)} real {r['real_right']} right "
                      f"{r['real_wrong']} wrong {r['real_abstain']} none | no-object named {r['none_named']} | "
                      f"precision {r['precision']}")
    lat = sorted(provider.latency)
    if lat:
        print(f"{provider.calls} Grok calls, median {lat[len(lat) // 2]:.2f} s, max {lat[-1]:.2f} s")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
