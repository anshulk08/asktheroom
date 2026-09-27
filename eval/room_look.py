"""Room questions on real rig frames, answered by real Grok through VisualQA.route (what the rig would say).

    XAI_API_KEY=... python -m eval.room_look --frames DIR [--repo PATH] [--clip MODELS_DIR] [--only id,id]
        [--out answers.json]

eval/room_questions.json lists the frames (not in git: people are in them; see its _about), the zones drawn
on each, the table view rect, and the questions with what a person sees. Each "look" question gets a fresh
VisualQA over that frame: latest() is the table view cut from it (core.room_view.cut), latest_full() the frame,
an empty world. "recall" questions go to an archive holding the recall frames at their times (--clip DIR with
mobileclip2_s0_{image,text}.onnx: text search as on the rig; without it, time windows only).

--repo runs another checkout's voice/ and core/ (e.g. the base branch, for answers before a change); the
harness only uses APIs both sides have. Prints one line per question and writes --out as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _wall(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def _zones(spec: dict, name: str) -> dict:
    z = spec["frames"][name]["zones"]
    return _zones(spec, z) if isinstance(z, str) else z


class Frames:
    """TableView's API over one still: latest() the table view, latest_full()/full_at() the whole frame."""

    def __init__(self, full, rect, wall: float):
        from core.room_view import cut
        from core.types import Frame
        self.full = Frame(t=0.0, wall=wall, img=full, idx=1)
        self.view = Frame(t=0.0, wall=wall, img=cut(full, rect), idx=1)
        self.rect = tuple(rect)

    def latest(self):
        return self.view

    def at(self, t):
        return self.view

    def latest_full(self):
        return self.full

    def full_at(self, t):
        return self.full


def _cfg(zones: dict, tmp: str, size=(0, 0)) -> dict:
    from core.config import load_config
    cfg = load_config()
    path = os.path.join(tmp, f"zones_{abs(hash(json.dumps(zones, sort_keys=True)))}.json")
    with open(path, "w") as f:
        json.dump({"view": "eval", "size_px": list(size), "zones": zones}, f)
    cfg.setdefault("room_memory", {})["zones_path"] = path
    cfg["visual_memory"] = {**(cfg.get("visual_memory") or {}), "enabled": True}
    return cfg


def _no_embed(cfg: dict) -> dict:
    vm = cfg["visual_memory"]
    return {**vm, "embed": {**(vm.get("embed") or {}), "embed": "none"}}


def _qa(cfg, zones, frames, events, archive=None, clock=time.time):
    from core.fakeworld import FakeWorld
    from core.types import Entity
    from core.visual_memory import VisualConfig
    from voice.visual import VisualQA
    c = VisualConfig.from_dict(_no_embed(cfg))
    world = FakeWorld([Entity(n, k) for n, k in (cfg.get("objects") or {}).items()], events)   # as at start: never seen
    q = VisualQA(cfg, world, events, frames=frames, table=None, archive=archive, clock=clock, c=c)
    q.room_zones = [(n, z["say"]) for n, z in zones.items()]
    return q


def _ask(q, text: str) -> dict:
    from voice.intents import parse
    calls = []
    narrate = q.provider.narrate

    def spy(system, parts, schema):
        calls.append({"system": system.split(".")[0][:60], "images": sum(1 for p in parts if p[0] == "image")})
        return narrate(system, parts, schema)

    q.provider.narrate = spy
    t0 = time.perf_counter()
    try:
        a = q.route(parse(text, q.cfg), text, online=True)
        ans = "(not routed)" if a is None else a.text
    except Exception as ex:                               # a crash is an answer too, for this report
        ans = f"(error: {ex!r})"
    return {"answer": ans, "s": round(time.perf_counter() - t0, 2), "calls": calls,
            "reply": (q.last or {}).get("reply")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--repo", help="checkout whose code answers (default: this one)")
    ap.add_argument("--clip", help="dir with mobileclip2_s0_{image,text}.onnx for recall text search")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve() if a.repo else HERE.parent
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ["ASKROOM_NO_LOCAL_CONFIG"] = "1"
    import cv2
    from core.events import EventLog
    spec = json.loads((HERE / "room_questions.json").read_text())
    only = set(a.only.split(",")) if a.only else None
    imgs = {n: cv2.imread(os.path.join(a.frames, f["file"])) for n, f in spec["frames"].items()}
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in spec["look"]:
            if only and case["id"] not in only:
                continue
            fr = spec["frames"][case["frame"]]
            zones = _zones(spec, case["frame"])
            ev = EventLog(":memory:", os.path.join(tmp, "snaps"))
            size = imgs[case["frame"]].shape[1::-1]
            q = _qa(_cfg(zones, tmp, size), zones, Frames(imgs[case["frame"]], fr["table_rect"], _wall(fr["wall"])), ev)
            r = {"id": case["id"], "q": case["q"], "expect": case["expect"], **_ask(q, case["q"])}
            print(f"[{r['id']}] {r['q']}  ->  {r['answer']}  ({r['s']} s, {r['calls']})", flush=True)
            out.append(r)
            ev.close()
        rc = spec["recall"]
        rq = [c for c in rc["questions"] if not only or c["id"] in only]
        if rq:
            out += _recall(spec, rc, rq, imgs, tmp, a.clip)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def _recall(spec, rc, questions, imgs, tmp, clip_dir) -> list:
    from core.events import EventLog
    from core.types import Frame
    from core.visual_memory import OnnxClipEmbedder, VisualArchive, VisualConfig
    last = rc["frames"][-1]
    zones = _zones(spec, last)
    cfg = _cfg(zones, tmp, imgs[last].shape[1::-1])
    ev = EventLog(os.path.join(tmp, "recall.db"), os.path.join(tmp, "snaps"))
    emb = None
    if clip_dir:
        emb = OnnxClipEmbedder(os.path.join(clip_dir, "mobileclip2_s0_image.onnx"),
                               os.path.join(clip_dir, "mobileclip2_s0_text.onnx"),
                               str(Path("assets/bpe_simple_vocab_16e6.txt.gz").resolve()), providers=("cpu",))
    now = _wall(rc["now"])
    c = VisualConfig.from_dict(_no_embed(cfg))
    arch = VisualArchive(cfg, ev, None, embedder=emb, start=False, c=c, clock=lambda: now)
    for i, n in enumerate(rc["frames"]):
        fr = spec["frames"][n]
        wall = _wall((rc.get("walls") or {}).get(n) or fr["wall"])
        src = Frames(imgs[n], fr["table_rect"], wall)
        arch.frames = src                                  # the full-frame source, where the archive takes one
        arch._check(Frame(t=float(i), wall=wall, img=src.view.img, idx=i + 1), False, True)
    arch.drain()
    rows = arch.store.window(0, now + 1)
    print(f"archive: {len(rows)} rows {[getattr(r, 'view', 'table') for r in rows]}", flush=True)
    if emb is not None:                                   # text search scores, for tuning min_sim on room frames
        for phrase in ("a laptop", "a tv", "an orange cup", "an umbrella", "a cat", "a person standing"):
            print(f"  sim {phrase!r}: " + ", ".join(f"{datetime.fromtimestamp(r.t):%H:%M} {getattr(r, 'view', 'table')} "
                                                     f"{s_:.3f}" for r, s_ in arch.search(phrase, 0, now + 1, 8, 0)))
    fr = spec["frames"][last]
    q = _qa(cfg, zones, Frames(imgs[last], fr["table_rect"], _wall(fr["wall"])), ev, archive=arch, clock=lambda: now)
    out = []
    for case in questions:
        r = {"id": case["id"], "q": case["q"], "expect": case["expect"], **_ask(q, case["q"])}
        print(f"[{r['id']}] {r['q']}  ->  {r['answer']}  ({r['s']} s, {r['calls']})", flush=True)
        out.append(r)
    ev.close()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
