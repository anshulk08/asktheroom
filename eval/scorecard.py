"""Identity scorecard for guided clips (the room demo, spec 0010): does one real object stay one entity?

    python -m eval.scorecard data/clips/room_still_1 data/clips/room_couch_1       # the recorded models
    python -m eval.scorecard data/clips/room_* --hands-off --yoloe-model models/yoloe-26s-seg-pf.pt
    python -m eval.scorecard data/clips/room_* --json card.json --save-track         # keep each replay
    python -m eval.scorecard data/clips/room_* --track track-askroom.jsonl           # rescore, no replay
    python -m eval.scorecard data/clips/room_* --track track-registry.jsonl          # an external tracker

Each clip is replayed as eval.score_clip does (same pipeline, same options: the table-view cut of a
1440p clip, room memory, --hands-off for a Mac with only the YOLOE .pt, --names for names), then
scored per truth prop. Scoring always goes through the tracker output contract (askroom-track/1,
eval/track.py, docs/track-format.md): the replay is written as a track and read back, so an external
tracker's track (--track) is scored by exactly the same code. The props' identities come from score_clip's mapping (each prop is put down with
a cue in the room_* clips, so its entity is the one that arrived for that cue). A table per clip, then one
row per clip; --json writes every number.

Metrics (targets in CardBars):
  entities per real object   1 + the other live (non-merged, not GONE) entities within dup_cm of a prop's
                             entity while it rests on the table, worst over the still segments and at the
                             end. A prop the mapping never located is listed. Also: thing identities created
                             in the clip, and live things at the end against the props on the table plus
                             truth.scene_objects (the fallback when positions can't be compared)
  phantom births / min       APPEARED thing:N that are not the initial scene and not the entity a put-down
                             cue (place / putdown / move / uncover) bound: one birth per cue at most. Split
                             by segment: still (nobody near; target 0) and people (someone moving, sitting,
                             reaching; target <= 1/min). A step's `seg` sets its segment (default: still for
                             hands_out, people otherwise); a still segment starts settle_s after its cue
  body-part / clothing       entities whose label, taught aliases or Grok guess (name or alternatives) is a
                             body part or clothing word (BODY_WORDS). n/a when no entity carries a name
                             (offline replay: use --names)
  position error while still the prop's entity's farthest position from its median (or truth.positions[prop],
                             cm) over each still segment, visible samples only
  room handoffs              each carry_to step: the prop's entity (before it was carried) is in the step's
                             zone within handoff_s (or before the prop's next step); into no drawn zone (the
                             floor by the doorway): it is put in no zone at all
  naming (hook)              the naming block: per prop its entity, the name it goes by and whether it fits
                             (core.auto_name.match_score >= name_min), right / wrong / none, junk_named.
                             n/a without names (replay with --names grok or --names module:factory)

The JSON per clip also has "bindings" (truth prop -> world entity over time: [{t, entity}], from the
scorer's mapping; "mapping" adds why) and "identity" (the headline identity numbers).
  room placements            each place_room step (a prop put straight into a zone, never on the table):
                             an entity turns up in that zone within handoff_s. n/a for a zone not drawn
                             (the floor)
  false handoffs             an entity in a room zone that no cue explains: a prop resting on the table per
                             truth, a phantom (a foot handed off as 'sock'), anything but the carried prop or
                             the first new entity in a place_room zone. Target 0
  identity through occlusion each block step (a person hides a resting prop, then unblock): hidden, not gone
                             (its entity is never GONE or missing from settle_s after the block to unblock),
                             and the same entity, visible, after unblock as before the block
  under the right cover      each cover step (keys under the notebook, under a cup): from hide_s after the cue
                             to the uncover cue the keys' identity is UNDER with its parent the entity bound to
                             the covering prop then (a shuffled cup is followed), in >= under_rate of the frames,
                             and the entity that arrives at the uncover is that identity
  identity after return      each putdown with expect_same (brought back after carry_to / remove): the
                             entity that arrives is the one the prop had before it was taken
  removed, not ghosted       each remove step: from ghost_s after it, no visible entity at the prop's spot

Config overrides (--set dotted.key=value, --mode M for permanence.mode=M) apply to the replay config, so
two trackers score the same clips through the same code: the card header lists them.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Optional

from eval.clip import Clip, load_clip
from eval.score_clip import THING, Bars, Sample, Trace, _dist, _Scorer, _survivor

PUT_DOWN = ("place", "putdown", "move", "uncover")
STILL, PEOPLE = "still", "people"
BODY_WORDS = frozenset("""
foot feet toe toes sock socks shoe shoes sneaker sneakers slipper slippers sandal sandals boot boots heel
leg legs knee knees thigh thighs lap ankle hand hands finger fingers thumb palm fist wrist arm arms elbow
forearm sleeve shoulder head face hair skin body person people man woman
jeans pants trousers shorts shirt tshirt t-shirt sweater hoodie jacket coat sweatshirt clothing clothes
cloth fabric sweatpants leggings skirt dress
""".split())


@dataclass
class CardBars:
    """Scorecard windows and targets. Starting values: tune them from real clips like the thresholds."""
    dup_cm: float = 6.0                 # another entity this close to a resting prop's entity is a duplicate
    settle_s: float = 3.0               # a still segment starts this long after its cue
    pos_cm: float = 5.0                 # a resting prop's entity stays this close to its spot
    handoff_s: float = 30.0             # a carried prop is in its zone within this (or before its next step)
    ghost_s: float = 8.0                # a removed prop's spot is empty from this long after the cue
    hide_s: float = 4.0                 # a covered prop is scored UNDER its cover from this long after the cue
    under_rate: float = 0.8             # ... in at least this share of the frames until it is uncovered
    name_min: float = 2.0               # match_score a name needs (room_memory.name_match_min)
    entities_per_object: int = 1
    still_births_per_min: float = 0.0
    people_births_per_min: float = 1.0
    body_things: int = 0


# ----- segments -----------------------------------------------------------------------------------------

def seg_kind(step: dict) -> str:
    return str(step.get("seg") or (STILL if step.get("event") == "hands_out" else PEOPLE))


def segments(steps: list, t0: float, t1: float, settle_s: float) -> list[tuple[float, float, str]]:
    """(start, end, kind) covering [t0, t1]: each step's segment runs to the next step; a still one
    starts settle_s after its cue (the time before is people: someone is still pulling back)."""
    out: list[tuple[float, float, str]] = []
    steps = sorted(steps, key=lambda s: float(s["t"]))
    cur, kind = t0, PEOPLE
    for i, s in enumerate(steps):
        ts = min(max(float(s["t"]), t0), t1)
        if ts > cur:
            out.append((cur, ts, kind))
            cur = ts
        kind = seg_kind(s)
        if kind == STILL:
            nxt = float(steps[i + 1]["t"]) if i + 1 < len(steps) else t1
            a = min(ts + settle_s, nxt, t1)
            if a > cur:
                out.append((cur, a, PEOPLE))
                cur = a
    if t1 > cur:
        out.append((cur, t1, kind))
    merged: list[tuple[float, float, str]] = []
    for a, b, k in out:                 # neighbours of one kind are one segment
        if merged and merged[-1][2] == k and abs(merged[-1][1] - a) < 1e-9:
            merged[-1] = (merged[-1][0], b, k)
        elif b > a:
            merged.append((a, b, k))
    return merged


def _kind_at(segs: list, t: float) -> str:
    for a, b, k in segs:
        if a <= t < b:
            return k
    return segs[-1][2] if segs and t >= segs[-1][1] else PEOPLE


# ----- names --------------------------------------------------------------------------------------------

def _words(text) -> set:
    return set(re.findall(r"[a-z][a-z-]*", str(text or "").lower()))


def name_phrases(names: dict) -> list[str]:
    """Every name an entity carries: its label, taught aliases, Grok's guess and its alternatives."""
    out = [names.get("label")] + list(names.get("aliases") or [])
    g = names.get("guess")
    if isinstance(g, dict):
        out += [g.get("name")] + list(g.get("also") or [])
    return [str(x) for x in out if x]


def body_part(names: dict) -> Optional[str]:
    """The first name that is a body part or clothing ('sock', 'blue jeans'), else None."""
    for phrase in name_phrases(names):
        if _words(phrase) & BODY_WORDS:
            return phrase
    return None


def primary_name(names: dict) -> Optional[str]:
    """The name an entity goes by: Grok's guess, else its label or first alias."""
    g = (names or {}).get("guess")
    if isinstance(g, dict) and g.get("name"):
        return str(g["name"])
    phrases = name_phrases(names or {})
    return phrases[0] if phrases else None


def name_score(truth: str, names: dict) -> float:
    """How well the entity's names fit the prop's description (core.auto_name.match_score, 0-3): its
    guess as Grok gave it, and each alias or label as a guess of its own."""
    from core.auto_name import match_score
    best = 0.0
    g = names.get("guess")
    if isinstance(g, dict):
        best = match_score(truth, g)
    for phrase in [names.get("label")] + list(names.get("aliases") or []):
        if phrase:
            best = max(best, match_score(truth, {"name": str(phrase)}))
    return best


# ----- the scorecard ------------------------------------------------------------------------------------

class _Card:
    def __init__(self, trace: Trace, clip: Clip, bars: CardBars, sbars: Bars):
        self.tr, self.clip, self.b = trace, clip, bars
        self.sc = _Scorer(trace, clip.truth, sbars)
        self.sc.run_mapping()
        self.samples = trace.samples
        self.t0 = self.samples[0].t if self.samples else 0.0
        self.t1 = self.samples[-1].t if self.samples else 0.0
        self.segs = segments(self.sc.steps, self.t0, self.t1, bars.settle_s)
        self.names = self._latest_names()

    def _latest_names(self) -> dict:
        out: dict = {}
        for s in self.samples:
            out.update(s.names or {})
        return out

    def _others(self, p: str, t: float) -> set:
        """Entities standing for the other props at t (never a duplicate of p)."""
        return {self.sc.mapped(q, t) for q in self.sc.props if q != p} - {None}

    def _near(self, s: Sample, pos, skip: set, visible_only: bool = False) -> list[str]:
        return [n for n, (st, _, q) in s.ents.items()
                if n not in skip and q is not None and st != "GONE" and (st == "VISIBLE" or not visible_only)
                and _dist(q, pos) <= self.b.dup_cm]

    def _resting(self, p: str, t: float) -> bool:
        e = self.sc.expected(p, t)
        return bool(e) and e[0] == "on_table" and not self.sc.involved(p, t)

    # -- entities per real object

    def entities_per_object(self) -> dict:
        per: dict = {}
        last = self.samples[-1] if self.samples else None
        for p in self.sc.props:
            row = {"max": None, "end": None, "worst_t": None, "duplicates": [], "located": False}
            for s in self.samples:
                is_end = s is last
                if not (is_end or _kind_at(self.segs, s.t) == STILL) or not self._resting(p, s.t):
                    continue
                e = self.sc.mapped(p, s.t)
                v = s.ents.get(e) if e else None
                if v is None or v[0] != "VISIBLE" or v[2] is None:
                    continue
                row["located"] = True
                dups = self._near(s, v[2], {e} | self._others(p, s.t))
                n = 1 + len(dups)
                if row["max"] is None or n > row["max"]:
                    row.update(max=n, worst_t=round(s.t, 2), duplicates=sorted(dups), entity=e)
                if is_end:
                    row["end"] = n
            per[p] = row
        located = [r["max"] for r in per.values() if r["max"] is not None]
        return {"per_prop": per, "worst": max(located) if located else None,
                "unlocated": sorted(p for p, r in per.items() if not r["located"]
                                    and any(self._resting(p, s.t) for s in self.samples))}

    def things(self) -> dict:
        created = sorted({ev["obj"] for ev in self.tr.events if ev["type"] == "APPEARED"
                          and ev["obj"].startswith(THING)})
        if not self.samples:
            return {"identities_created": len(created), "live_end": 0, "visible_end": 0, "expected_end": 0}
        s = self.samples[-1]
        live = [n for n, (st, _, q) in s.ents.items() if n.startswith(THING) and st != "GONE" and q is not None]
        vis = [n for n in live if s.ents[n][0] == "VISIBLE"]
        on = [p for p in self.sc.props if (self.sc.expected(p, s.t) or ("",))[0] == "on_table"]
        scene = list(self.clip.truth.get("scene_objects") or [])
        return {"identities_created": len(created), "live_end": len(live), "visible_end": len(vis),
                "expected_end": len(on) + len(scene), "props_on_table_end": len(on), "scene_objects": scene}

    # -- phantom births

    def births(self) -> dict:
        bound = {r["entity"] for r in self.sc.placements
                 if not r["missed"] and r["event"] in PUT_DOWN and r["entity"]}
        minutes = {STILL: 0.0, PEOPLE: 0.0}
        for a, b, k in self.segs:
            minutes[k] += (b - a) / 60.0
        rows = {STILL: [], PEOPLE: []}
        for n, (t, pos) in sorted(self.sc.births.items(), key=lambda x: x[1][0]):
            if self.sc.birth_kind.get(n) == "initial" or n in bound:
                continue
            k = _kind_at(self.segs, t)
            rows[k].append({"entity": n, "t": round(t, 2), "pos_cm": [round(float(v), 1) for v in pos] if pos else None,
                            "kind": self.sc.birth_kind.get(n), "of": self.sc.birth_of.get(n),
                            "merged_into": _survivor(n, self.tr.merged) if n in self.tr.merged else None,
                            "name": (name_phrases(self.names.get(n, {})) or [None])[0]})
        return {k: {"n": len(rows[k]), "minutes": round(minutes[k], 3),
                    "per_min": round(len(rows[k]) / minutes[k], 3) if minutes[k] > 0 else None, "births": rows[k]}
                for k in (STILL, PEOPLE)}

    # -- names

    def body_things(self) -> dict:
        hits = [{"entity": n, "name": body_part(v)} for n, v in sorted(self.names.items()) if body_part(v)]
        return {"names_seen": bool(self.names), "n": len(hits), "things": hits,
                "named": len(self.names)}

    def naming(self) -> dict:
        """The naming block (WS3): per prop the entity (the latest one bound to it that carries a name, else
        the latest bound), its name (Grok's guess, else a taught alias or label) and whether that fits the
        prop's description (core.auto_name.match_score >= name_min; null without a name). right / wrong /
        none count the props; junk_named the named entities that never stood for a prop (clutter, feet)."""
        merged = self.tr.merged
        props, bound = {}, set()
        for p, what in self.sc.types.items():
            ents = [m["entity"] for m in self.sc.mapping.get(p, [])]
            bound |= set(ents) | {_survivor(e, merged) for e in ents}
            pick, nm = (ents[-1] if ents else None), None
            for e in reversed(ents):
                got = self.names.get(e) or self.names.get(_survivor(e, merged) or "")
                if got:
                    pick, nm = e, got
                    break
            props[p] = {"entity": pick, "guess": primary_name(nm) if nm else None,
                        "fits": (name_score(str(what), nm) >= self.b.name_min) if nm else None}
        junk = [{"entity": n, "guess": primary_name(v)} for n, v in sorted(self.names.items())
                if n not in bound and primary_name(v)]
        fits = [v["fits"] for v in props.values()]
        return {"props": props, "right": fits.count(True), "wrong": fits.count(False), "none": fits.count(None),
                "junk_named": len(junk), "junk": junk}

    # -- positions, handoffs, removals

    def positions(self) -> dict:
        truth_pos = self.clip.truth.get("positions") or {}
        rows = []
        for a, b, k in self.segs:
            if k != STILL:
                continue
            for p in self.sc.props:
                pts = []
                for s in self.sc.between(a, b):
                    if not self._resting(p, s.t):
                        continue
                    e = self.sc.mapped(p, s.t)
                    v = s.ents.get(e) if e else None
                    if v is not None and v[0] == "VISIBLE" and v[2] is not None:
                        pts.append(v[2])
                n = len(self.sc.between(a, b))
                if not pts:
                    continue
                ref = tuple(truth_pos[p]) if p in truth_pos else (median(q[0] for q in pts), median(q[1] for q in pts))
                err = max(_dist(q, ref) for q in pts)
                rows.append({"prop": p, "from": round(a, 2), "to": round(b, 2), "max_cm": round(err, 2),
                             "visible": round(len(pts) / n, 3) if n else None,
                             "ref": "truth" if p in truth_pos else "median"})
        return {"rows": rows, "max_cm": max((r["max_cm"] for r in rows), default=None)}

    def _next_step(self, p: str, t: float) -> float:
        return min((float(x["t"]) for x in self.sc.steps if float(x["t"]) > t + 1e-9
                    and p in (x.get("obj"), x.get("parent"))), default=math.inf)

    def _drawn(self, zone: Optional[str]) -> bool:
        drawn = self._zones_drawn()
        return (zone in drawn) if drawn is not None else (zone not in (None, "floor"))

    def handoffs(self) -> dict:
        """carry_to steps. Into a drawn zone: the prop's entity must show up there. Into no drawn zone (the
        floor by the doorway): it must not be put in any zone (nothing to see it there)."""
        rows = []
        for s in self.sc.steps:
            if s["event"] != "carry_to" or not s.get("obj"):
                continue
            p, t, want = s["obj"], float(s["t"]), s.get("zone")
            drawn = self._drawn(want)
            ent = self.sc.mapped(p, t - 1e-6)
            end = min(t + self.b.handoff_s, self._next_step(p, t))
            row = {"t": round(t, 2), "prop": p, "zone": want, "drawn": drawn, "entity": ent, "got": None,
                   "delay_s": None, "ok": False}
            if ent is not None:
                names = {ent, _survivor(ent, self.tr.merged)}
                for smp in self.sc.between(t, end):
                    z = next((smp.zones[n] for n in names if n in (smp.zones or {})), None)
                    if z is None:
                        continue
                    if row["got"] is None or z == want:
                        row.update(got=z, delay_s=round(smp.t - t, 2))
                    if z == want or not drawn:
                        break
                row["ok"] = (row["got"] == want) if drawn else row["got"] is None
            rows.append(row)
        return {"rows": rows, "n": len(rows), "ok": sum(r["ok"] for r in rows)}

    def returns(self) -> dict:
        """putdown steps with expect_same: a prop carried off (or out of view) and brought back has the
        identity it had before it was taken."""
        rows = []
        for s in self.sc.steps:
            if s["event"] != "putdown" or not s.get("expect_same") or not s.get("obj"):
                continue
            p, t = s["obj"], float(s["t"])
            took = [x for x in self.sc.steps if x.get("obj") == p and float(x["t"]) < t
                    and x["event"] in ("carry_to", "remove", "pickup", "exit_edge")]
            t_took = float(took[-1]["t"]) if took else t
            before = _survivor(self.sc.mapped(p, t_took - 1e-6), self.tr.merged)
            row_p = next((r for r in self.sc.placements if r["prop"] == p and abs(r["t"] - t) < 1e-6), None)
            after = _survivor(row_p["entity"], self.tr.merged) if row_p and row_p["entity"] else None
            rows.append({"t": round(t, 2), "prop": p, "taken_t": round(t_took, 2), "before": before, "after": after,
                         "ok": before is not None and after == before})
        return {"rows": rows, "n": len(rows), "ok": sum(r["ok"] for r in rows)}

    def _zones_drawn(self) -> Optional[set]:
        rz = self.clip.meta.get("room_zones") if self.clip.meta else None
        return set((rz or {}).get("zones") or {}) if rz else None

    def room_placements(self) -> dict:
        drawn = self._zones_drawn()
        rows = []
        for s in self.sc.steps:
            if s["event"] != "place_room" or not s.get("obj"):
                continue
            p, t, want = s["obj"], float(s["t"]), s.get("zone")
            is_drawn = (want in drawn) if drawn is not None else (want not in (None, "floor"))
            row = {"t": round(t, 2), "prop": p, "zone": want, "drawn": is_drawn, "entity": None, "delay_s": None,
                   "ok": None}
            before = {n for smp in self.sc.between(-math.inf, t - self.sc.bars.pre_s) for n, z in smp.zones.items()
                      if z == want}
            for smp in self.sc.between(t - self.sc.bars.pre_s, min(self.t1, t + self.b.handoff_s)):
                taken = self._others(p, smp.t)
                new = sorted(n for n, z in (smp.zones or {}).items() if z == want and n not in before and n not in taken)
                if new:
                    row.update(entity=new[0], delay_s=round(smp.t - t, 2))
                    break
            if is_drawn:
                row["ok"] = row["entity"] is not None
            rows.append(row)
        return {"rows": rows, "n": sum(r["drawn"] for r in rows), "ok": sum(bool(r["ok"]) for r in rows)}

    def false_handoffs(self, placements: dict) -> dict:
        claimed = {r["entity"] for r in placements["rows"] if r["entity"]}
        seen: dict = {}
        for smp in self.samples:
            for n, z in (smp.zones or {}).items():
                if n in claimed or (n, z) in seen:
                    continue
                props = [p for p in self.sc.props if self.sc.mapped(p, smp.t) in (n, _survivor(n, self.tr.merged))]
                if any((self.sc.expected(p, smp.t) or ("",))[0] == "room" for p in props):
                    continue                    # the carried prop, where the truth has it off the table
                seen[(n, z)] = {"entity": n, "zone": z, "t": round(smp.t, 2), "prop": props[0] if props else None,
                                "name": (name_phrases(self.names.get(n, {})) or [None])[0]}
        rows = sorted(seen.values(), key=lambda r: r["t"])
        return {"rows": rows, "n": len(rows)}

    def occlusions(self) -> dict:
        rows = []
        for s in self.sc.steps:
            if s["event"] != "block" or not s.get("obj"):
                continue
            p, t = s["obj"], float(s["t"])
            un = next((float(x["t"]) for x in self.sc.steps if x["event"] == "unblock" and x.get("obj") == p
                       and float(x["t"]) > t), t + 15.0)
            t_after = min(un + self.sc.bars.resolve_s, self._next_step(p, un), self.t1)
            before = _survivor(self.sc.mapped(p, t - 1e-6), self.tr.merged)
            after = _survivor(self.sc.mapped(p, t_after), self.tr.merged)
            smp = self.sc.at(t_after)
            v = smp.ents.get(after) if (smp is not None and after) else None
            during = self.sc.between(t + self.b.settle_s, un)
            hidden = [x.ents.get(before, ("missing",))[0] for x in during] if before else []
            gone = sum(h in ("GONE", "missing") for h in hidden)
            rows.append({"t": round(t, 2), "prop": p, "unblock_t": round(un, 2), "before": before, "after": after,
                         "visible_after": bool(v and v[0] == "VISIBLE"),
                         "while_hidden": max(set(hidden), key=hidden.count) if hidden else None,
                         "gone_frames": gone, "frames": len(hidden),
                         "ok": before is not None and before == after and bool(v and v[0] == "VISIBLE") and gone == 0})
        return {"rows": rows, "n": len(rows), "ok": sum(r["ok"] for r in rows)}

    def covers(self) -> dict:
        """cover steps (obj goes under parent): from hide_s after the cue until the uncover cue, the prop's
        identity from before the cover is hidden (UNDER, or a tracker's plain hidden) with its parent the
        entity bound to the covering prop at that moment (so a slid or shuffled cup is followed); after the
        uncover, the entity that arrives is that same identity."""
        rows = []
        merged = self.tr.merged
        for s in self.sc.steps:
            if s["event"] != "cover" or not s.get("obj") or not s.get("parent"):
                continue
            p, c, t = s["obj"], s["parent"], float(s["t"])
            unc = next((x for x in self.sc.steps if x["event"] == "uncover" and x.get("obj") == p
                        and float(x["t"]) > t), None)
            t_un = float(unc["t"]) if unc else self.t1
            ent = _survivor(self.sc.mapped(p, t - 1e-6), merged)
            frames = under = right = 0
            wrong: dict = {}
            for smp in self.sc.between(t + self.b.hide_s, t_un):
                frames += 1
                v = smp.ents.get(ent) if ent else None
                if v is None or v[0] not in ("UNDER", "HIDDEN"):
                    continue
                under += 1
                want = _survivor(self.sc.mapped(c, smp.t), merged)
                got = _survivor(v[1], merged) if v[1] else None
                if want is not None and got == want:
                    right += 1
                else:
                    k = self.sc.as_prop(got, smp.t) if got else None
                    wrong[str(k)] = wrong.get(str(k), 0) + 1
            after = None
            if unc is not None:
                row_p = next((r for r in self.sc.placements if r["prop"] == p and abs(r["t"] - t_un) < 1e-6), None)
                after = _survivor(row_p["entity"], merged) if row_p and row_p["entity"] else None
            rate = right / frames if frames else None
            rows.append({"t": round(t, 2), "prop": p, "cover": c, "uncover_t": round(t_un, 2) if unc else None,
                         "entity": ent, "frames": frames, "under": under, "right_parent": right,
                         "rate": round(rate, 3) if rate is not None else None, "wrong_parents": wrong,
                         "after": after, "same_after": (after == ent) if unc is not None else None,
                         "ok": bool(ent and rate is not None and rate >= self.b.under_rate
                                    and (unc is None or after == ent))})
        return {"rows": rows, "n": len(rows), "ok": sum(r["ok"] for r in rows)}

    def removals(self) -> dict:
        rows = []
        for s in self.sc.steps:
            if s["event"] != "remove" or not s.get("obj"):
                continue
            p, t = s["obj"], float(s["t"])
            ent, spot = self.sc.mapped(p, t - 1e-6), None
            for smp in reversed(self.sc.between(-math.inf, t)):
                v = smp.ents.get(ent) if ent else None
                if v is not None and v[0] == "VISIBLE" and v[2] is not None:
                    spot = v[2]
                    break
            row = {"t": round(t, 2), "prop": p, "entity": ent, "spot_cm": list(spot) if spot else None,
                   "seen_there": None, "ghost": False}
            if spot is not None:
                smp = self.sc.between(t + self.b.ghost_s, min(self.t1, self._next_step(p, t)))
                there = [x for x in smp if self._near(x, spot, self._others(p, x.t), visible_only=True)]
                if smp:
                    row["seen_there"] = round(len(there) / len(smp), 3)
                    row["ghost"] = row["seen_there"] >= 0.5
            rows.append(row)
        return {"rows": rows, "n": len(rows), "ghosts": sum(r["ghost"] for r in rows)}

    # -- all of it

    def report(self) -> dict:
        b = self.b
        epo, things, births = self.entities_per_object(), self.things(), self.births()
        body, naming, pos = self.body_things(), self.naming(), self.positions()
        hand, rem = self.handoffs(), self.removals()
        placed = self.room_placements()
        false_h, occl, back = self.false_handoffs(placed), self.occlusions(), self.returns()
        cov = self.covers()
        room_ran = bool(self.tr.room) or any(smp.zones for smp in self.samples)
        crit = [
            _row("entities per real object", _epo(epo), f"{b.entities_per_object}",
                 None if epo["worst"] is None else epo["worst"] <= b.entities_per_object),
            _row("thing identities created", f"{things['identities_created']} (live at end {things['live_end']}, "
                 f"visible {things['visible_end']}, expected {things['expected_end']})", "", None),
            _row("phantom births/min still", _rate(births[STILL]), f"<= {b.still_births_per_min:g}",
                 None if births[STILL]["per_min"] is None else births[STILL]["per_min"] <= b.still_births_per_min),
            _row("phantom births/min people", _rate(births[PEOPLE]), f"<= {b.people_births_per_min:g}",
                 None if births[PEOPLE]["per_min"] is None else births[PEOPLE]["per_min"] <= b.people_births_per_min),
            _row("body-part / clothing things", (f"{body['n']}" + _few([f"{x['entity']} '{x['name']}'" for x in body["things"]]))
                 if body["names_seen"] else "n/a (no names: --names)", f"{b.body_things}",
                 body["n"] <= b.body_things if body["names_seen"] else None),
            _row("position error while still", f"max {pos['max_cm']:.1f} cm" if pos["max_cm"] is not None else "n/a",
                 f"<= {b.pos_cm:g} cm", None if pos["max_cm"] is None else pos["max_cm"] <= b.pos_cm),
            _row("room handoffs", f"{hand['ok']}/{hand['n']}" + _few([f"{r['prop']}->{r['zone']}: {r['got'] or 'never'}"
                                                                     for r in hand["rows"] if not r["ok"]]),
                 "all", hand["ok"] == hand["n"] if hand["n"] else None),
            _row("room placements", f"{placed['ok']}/{placed['n']}" + _few(
                [f"{r['prop']}->{r['zone']}: never seen" for r in placed["rows"] if r["ok"] is False]), "all",
                 placed["ok"] == placed["n"] if placed["n"] else None),
            _row("false handoffs", f"{false_h['n']}" + _few([f"{r['entity']}->{r['zone']}"
                                                             + (f" ({r['prop']} on the table)" if r["prop"] else "")
                                                             + (f" '{r['name']}'" if r["name"] else "")
                                                             for r in false_h["rows"]]),
                 "0", false_h["n"] == 0 if room_ran else None),
            _row("under the right cover", f"{cov['ok']}/{cov['n']}" + _few(
                [f"{r['prop']} under {r['cover']}: {r['rate'] if r['rate'] is not None else '-'} of frames"
                 + ("" if r["same_after"] is not False else f", back as {r['after'] or 'none'}")
                 for r in cov["rows"] if not r["ok"]]),
                 f">= {b.under_rate:.0%}, same after", cov["ok"] == cov["n"] if cov["n"] else None),
            _row("identity through occlusion", f"{occl['ok']}/{occl['n']}" + _few(
                [f"{r['prop']}: {r['before']} -> {r['after'] or 'none'}" for r in occl["rows"] if not r["ok"]]),
                 "all", occl["ok"] == occl["n"] if occl["n"] else None),
            _row("identity after return", f"{back['ok']}/{back['n']}" + _few(
                [f"{r['prop']}: {r['before']} -> {r['after'] or 'none'}" for r in back["rows"] if not r["ok"]]),
                 "all", back["ok"] == back["n"] if back["n"] else None),
            _row("removed, not ghosted", f"{rem['n'] - rem['ghosts']}/{rem['n']}", "all",
                 rem["ghosts"] == 0 if rem["n"] else None),
            _row("naming (hook)", (f"{naming['right']}/{naming['right'] + naming['wrong']} named props fit, "
                                   f"{naming['none']} unnamed, {naming['junk_named']} junk named")
                 if naming["right"] + naming["wrong"] else "n/a (no names: --names)", f"match >= {b.name_min:g}",
                 naming["wrong"] == 0 if naming["right"] + naming["wrong"] else None),
        ]
        rp = self.tr
        return {"clip": self.clip.name, "video_s": round(self.clip.duration_s or (self.t1 - self.t0), 2),
                "frames_processed": len(self.samples), "replay_wall_s": round(rp.wall_s, 1), "detector": rp.detector, "view": rp.view, "room_memory": rp.room,
                "props": dict(self.sc.types), "mapping": self.sc.mapping,
                "bindings": {p: [{"t": m["t"], "entity": m["entity"]} for m in ms] for p, ms in self.sc.mapping.items()},
                "identity": {"worst_entities_per_object": epo["worst"],
                             "identities_created": things["identities_created"],
                             "phantom_still_per_min": births[STILL]["per_min"],
                             "phantom_people_per_min": births[PEOPLE]["per_min"],
                             "body_things": body["n"] if body["names_seen"] else None,
                             "false_handoffs": false_h["n"], "handoffs_ok": [hand["ok"], hand["n"]],
                             "occlusions_ok": [occl["ok"], occl["n"]], "returns_ok": [back["ok"], back["n"]],
                             "covers_ok": [cov["ok"], cov["n"]]},
                "segments": [{"from": round(a, 2), "to": round(b_, 2), "kind": k} for a, b_, k in self.segs],
                "entities_per_object": epo, "things": things, "phantom_births": births, "body_things": body,
                "position_still": pos, "handoffs": hand, "removals": rem, "naming": naming,
                "room_placements": placed, "false_handoffs": false_h, "occlusions": occl, "returns": back,
                "covers": cov,
                "overrides": list(getattr(rp, "overrides", None) or []),
                "bars": asdict(b), "criteria": crit,
                "pass": all(c["result"] != "FAIL" for c in crit)}


def _row(name: str, value: str, target: str, ok: Optional[bool]) -> dict:
    return {"name": name, "value": value, "target": target, "result": "n/a" if ok is None else ("PASS" if ok else "FAIL")}


def _epo(epo: dict) -> str:
    if epo["worst"] is None:
        return "n/a (no prop located)"
    bad = [f"{p}: {r['max']}" for p, r in epo["per_prop"].items() if r["max"] and r["max"] > 1]
    return f"worst {epo['worst']}" + _few(bad) + (f"; unlocated {', '.join(epo['unlocated'])}" if epo["unlocated"] else "")


def _rate(b: dict) -> str:
    if b["per_min"] is None:
        return "n/a (no such segment)"
    return f"{b['per_min']:.2f} ({b['n']} in {b['minutes']:.1f} min)"


def _few(items: list, n: int = 3) -> str:
    if not items:
        return ""
    return " (" + "; ".join(items[:n]) + (f", +{len(items) - n}" if len(items) > n else "") + ")"


def scorecard(trace: Trace, clip: Clip, bars: Optional[CardBars] = None, score_bars: Optional[Bars] = None) -> dict:
    """The identity scorecard of one replayed clip (see the module docstring)."""
    return _Card(trace, clip, bars or CardBars(), score_bars or Bars()).report()


def score_track(lines: list, clip: Clip, bars: Optional[CardBars] = None,
                score_bars: Optional[Bars] = None) -> dict:
    """The scorecard of a tracker's output in the askroom-track/1 contract (eval/track.py,
    docs/track-format.md): our replay and an external tracker are scored by this same path."""
    from eval.track import track_to_trace
    return scorecard(track_to_trace(lines, clip.meta.get("room_zones")), clip, bars, score_bars)


# ----- command line -------------------------------------------------------------------------------------

def print_card(r: dict) -> None:
    view = f", table view {r['view'][0]} -> {r['view'][1][0]}x{r['view'][1][1]}" if r.get("view") else ""
    print(f"\n== {r['clip']}: {r['video_s']} s, {r['frames_processed']} frames processed in "
          f"{r.get('replay_wall_s', 0):g} s{view}, room memory "
          f"{'on' if r.get('room_memory') else 'off'}")
    print(f"   detector: {r['detector']}")
    if r.get("overrides"):
        print(f"   config overrides: {', '.join(r['overrides'])}")
    for p, what in r["props"].items():
        chain = ", ".join(f"{m['entity']}@{m['t']:g}" for m in r["mapping"].get(p, [])) or "never found"
        print(f"   {p:4s} {what:12s} {chain}")
    w = max(len(c["name"]) for c in r["criteria"])
    print(f"   {'metric':{w}s}  {'result':6s} {'target':14s} value")
    for c in r["criteria"]:
        print(f"   {c['name']:{w}s}  {c['result']:6s} {c['target']:14s} {c['value']}")


def print_summary(cards: list) -> None:
    if len(cards) < 2:
        return
    print("\nclip                         ent/obj  things  still/min  people/min  body  pos cm  handoffs  false h/o  PASS")
    for r in cards:
        b = r["phantom_births"]
        f = lambda v, fmt: ("-" if v is None else format(v, fmt))       # noqa: E731
        print(f"{r['clip'][:28]:28s} {f(r['entities_per_object']['worst'], 'd'):>7s} "
              f"{r['things']['identities_created']:7d} {f(b['still']['per_min'], '.2f'):>10s} "
              f"{f(b['people']['per_min'], '.2f'):>11s} {f(r['body_things']['n'] if r['body_things']['names_seen'] else None, 'd'):>5s} "
              f"{f(r['position_still']['max_cm'], '.1f'):>7s} {r['handoffs']['ok']}/{r['handoffs']['n']:<7d} "
              f"{r['false_handoffs']['n']:9d}  {'yes' if r['pass'] else 'no'}")


def main(argv=None) -> int:
    import logging

    from eval.score_clip import add_replay_args, replay_from_args
    from eval.track import read_lines, trace_to_track, write_track
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="clip directories (video.mp4, frames.json, meta.json, truth.json)")
    ap.add_argument("--json", help="write every clip's scorecard here (a list)")
    ap.add_argument("--track", help="score this tracker output (askroom-track/1 JSONL) instead of replaying: a "
                                    "file name inside each clip directory, or a path when scoring one clip")
    ap.add_argument("--save-track", nargs="?", const="track-askroom.jsonl", metavar="NAME",
                    help="write each replay's track into the clip directory (default name track-askroom.jsonl)")
    ap.add_argument("--tracker", default="askroom", help="the tracker name written in a saved track's header")
    add_replay_args(ap)
    ap.add_argument("--dup-cm", type=float, default=CardBars.dup_cm)
    ap.add_argument("--settle-s", type=float, default=CardBars.settle_s)
    ap.add_argument("--initial-s", type=float, default=Bars.initial_s,
                    help="identities admitted this soon after the first frame are the initial scene")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cards = []
    for d in a.clips:
        clip = load_clip(d)
        if a.track:
            tp = Path(a.track) if (len(a.clips) == 1 and Path(a.track).is_file()) else Path(d) / a.track
            if not tp.is_file():
                print(f"{d}: no track {tp}")
                continue
            lines = read_lines(tp)
        else:
            if not clip.video.exists():
                print(f"{d}: no video.mp4")
                continue
            lines = trace_to_track(replay_from_args(clip, a), clip.name, a.tracker, clip.meta.get("room_zones"))
            if a.save_track:
                write_track(Path(d) / a.save_track, lines)
        r = score_track(lines, clip, CardBars(dup_cm=a.dup_cm, settle_s=a.settle_s), Bars(initial_s=a.initial_s))
        print_card(r)
        cards.append(r)
    print_summary(cards)
    if a.json:
        Path(a.json).write_text(json.dumps(cards, indent=2, default=str))
    return 0 if cards and all(r["pass"] for r in cards) else 1


if __name__ == "__main__":
    raise SystemExit(main())
