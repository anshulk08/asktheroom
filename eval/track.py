"""The tracker output contract the scorecard reads (askroom-track/1): one JSONL file per clip, so any
tracker (ours replayed by eval.score_clip, or an external one) is scored by the same code.
docs/track-format.md is the spec; this module writes and reads it.

Lines (JSON objects, one per line; "type" says which):

  header  {"type": "header", "format": "askroom-track/1", "clip": <clip id>, "tracker": <name>,
           "detector": <free text>, "overrides": [<key=value>], "view": [[x1, y1, x2, y2], [w, h]] | null,
           "room_memory": bool, "replay_wall_s": <s>}             first line, exactly once (only
                                                                  format is required)
  entity  {"type": "entity", "id": <str>, "kind": <str>, "born_t": <s> | null, "merged_into": <id> | null,
           "names": [<str>], "guess": {"name", "also", "confidence"} | null}
                                                                  the identity list; a later line for the same
                                                                  id replaces the earlier one
  frame   {"type": "frame", "t": <s>, "entities": [{"id", "state", "zone", "place", "table_cm", "parent",
           "hidden_in"}], "hands_cm": [[x1, y1, x2, y2]]}          the complete belief after one processed
                                                                  frame, t ascending
  event   {"type": "event", "t": <s>, "id": <str>, "event": <EVENTS key>, "table_cm": [x, y] | null,
           "from_cm": [x, y] | null, "parent": <id> | null}       optional

Times are seconds since the clip's first frame (truth.json's clock). state is one of STATES: visible (seen
now, on the table or in a zone), hidden (under or inside something, or blocked from view: optional
hidden_in 'under' / 'inside' / 'occluded' and parent), carried (in a hand), last_seen (not seen now, last
position known), unknown (no idea), gone (left the view by an edge). zone is 'table', a room zone key
(room_zones.json) or null; without it, place (the spoken place, 'the couch') is matched to a zone by the
clip's recorded zone names. table_cm (table centimetres) only while on the table. An entity missing from a
frame is not believed in at that time (merged away or forgotten). Only header and frame lines are required;
births default to an id's first visible frame.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

from eval.score_clip import THING, Sample, Trace

FORMAT = "askroom-track/1"
STATES = ("visible", "hidden", "carried", "last_seen", "unknown", "gone")
# contract event -> world event type the scorer knows (born = APPEARED; the others are arrivals / departures)
EVENTS = {"born": "APPEARED", "moved": "MOVED", "put_down": "PUT_BACK", "found": "FOUND", "picked_up": "PICKED_UP",
          "hidden": "COVERED", "lost": "LOST_TRACK", "gone": "EXITED_VIEW"}
_TO_EVENT = {"APPEARED": "born", "MOVED": "moved", "PUT_BACK": "put_down", "FOUND": "found", "CORRECTED": "found",
             "UNCOVERED": "found", "TAKEN_OUT": "found", "PICKED_UP": "picked_up", "COVERED": "hidden",
             "PUT_INSIDE": "hidden", "LOST_TRACK": "lost", "EXITED_VIEW": "gone"}
_FROM_STATUS = {"VISIBLE": "visible", "HELD": "carried", "UNDER": "hidden", "INSIDE": "hidden", "GONE": "gone"}
_TO_STATUS = {"visible": "VISIBLE", "carried": "HELD", "last_seen": "UNKNOWN", "unknown": "UNKNOWN", "gone": "GONE"}


# ----- our replay -> the contract -----------------------------------------------------------------------

def _zone_say(zones: Optional[dict], zone: str) -> str:
    if zone == "table":
        return "the table"
    z = ((zones or {}).get("zones") or {}).get(zone) or {}
    return str(z.get("say") or f"the {zone.replace('_', ' ')}")


def trace_to_track(trace: Trace, clip_name: str, tracker: str = "askroom", zones: Optional[dict] = None) -> list:
    """Our replay (eval.score_clip.Trace) as contract lines. zones: the clip's room_zones (spoken places)."""
    lines: list = [{"type": "header", "format": FORMAT, "clip": clip_name, "tracker": tracker,
                    "detector": trace.detector, "overrides": list(trace.overrides or []), "view": trace.view,
                    "room_memory": bool(trace.room), "replay_wall_s": round(float(trace.wall_s), 2)}]
    born = {}
    for ev in trace.events:
        if ev["type"] == "APPEARED":
            born.setdefault(ev["obj"], ev["t"])
    names: dict = {}
    for s in trace.samples:
        names.update(s.names or {})
    # identities the world ever had: born, or seen (a configured object the detector never saw is none)
    ever = set(born) | {n for s in trace.samples for n, v in s.ents.items() if v[0] != "UNKNOWN" or v[2] is not None}
    ids = [n for n in dict.fromkeys(list(trace.kinds) + [n for s in trace.samples for n in s.ents]) if n in ever]
    for n in ids:
        nm = names.get(n) or {}
        plain = [x for x in [nm.get("label")] + list(nm.get("aliases") or []) if x]
        g = nm.get("guess") if isinstance(nm.get("guess"), dict) else None
        lines.append({"type": "entity", "id": n, "kind": trace.kinds.get(n, "thing" if n.startswith(THING) else "object"),
                      "born_t": born.get(n), "merged_into": trace.merged.get(n),
                      "names": list(dict.fromkeys(plain)), "guess": g})
    for s in trace.samples:
        ents = []
        for n, (st, parent, pos) in s.ents.items():
            if n not in ever:
                continue
            zone = (s.zones or {}).get(n) or "table"
            state = _FROM_STATUS.get(st) or ("last_seen" if pos is not None else "unknown")
            row = {"id": n, "state": state, "zone": zone if st != "GONE" else None,
                   "place": _zone_say(zones, zone) if st != "GONE" else None,
                   "table_cm": [round(float(v), 2) for v in pos] if (pos is not None and zone == "table") else None}
            if state == "hidden":
                row.update(hidden_in="inside" if st == "INSIDE" else "under", parent=parent)
            elif state == "carried":
                row["parent"] = parent
            ents.append(row)
        lines.append({"type": "frame", "t": round(float(s.t), 4), "entities": ents,
                      "hands_cm": [[round(float(v), 2) for v in h] for h in s.hands]})
    for ev in trace.events:
        kind = _TO_EVENT.get(ev["type"])
        if kind:
            lines.append({"type": "event", "t": ev["t"], "id": ev["obj"], "event": kind, "table_cm": ev.get("to_cm"),
                          "from_cm": ev.get("from_cm"), "parent": ev.get("parent")})
    return lines


def write_track(path: Union[str, Path], lines: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line, default=str) + "\n")


def read_lines(path: Union[str, Path]) -> list:
    out = []
    with open(path) as f:
        for i, raw in enumerate(f, 1):
            raw = raw.strip()
            if raw:
                try:
                    out.append(json.loads(raw))
                except ValueError as e:
                    raise ValueError(f"{path}:{i}: not JSON: {e}") from e
    return out


# ----- the contract -> what the scorer reads ------------------------------------------------------------

def _zone_of(row: dict, zones: Optional[dict]) -> Optional[str]:
    """zone key from 'zone', else from the spoken 'place' matched to the clip's zone names."""
    z = row.get("zone")
    if z:
        return str(z)
    place = str(row.get("place") or "").strip().lower()
    if not place:
        return None
    if "table" in place.split() and "side" not in place.split():
        return "table"
    for key, zd in ((zones or {}).get("zones") or {}).items():
        say = str((zd or {}).get("say") or "").strip().lower()
        if place in (key.lower(), key.lower().replace("_", " "), say, say.removeprefix("the ")) or \
                (say and say in place):
            return key
    words = [w for w in place.replace("_", " ").split() if w not in ("on", "in", "at", "the", "a")]
    return "_".join(words) or None           # no zone names known: 'on the couch' -> 'couch'


def _sid(i) -> str:
    """Scorer id: every tracker id is an open-world identity to the scorer ('thing:<id>')."""
    i = str(i)
    return i if i.startswith(THING) else THING + i


def track_to_trace(lines: list, zones: Optional[dict] = None) -> Trace:
    """Contract lines -> eval.score_clip.Trace (what eval.scorecard scores). Raises ValueError on a
    malformed track."""
    if not lines or lines[0].get("type") != "header":
        raise ValueError("a track starts with its header line")
    head = lines[0]
    if head.get("format") != FORMAT:
        raise ValueError(f"track format {head.get('format')!r}, this scorer reads {FORMAT}")
    ents: dict = {}
    frames, events = [], []
    for ln in lines[1:]:
        kind = ln.get("type")
        if kind == "entity":
            ents[_sid(ln["id"])] = ln
        elif kind == "frame":
            frames.append(ln)
        elif kind == "event":
            events.append(ln)
    frames.sort(key=lambda f: float(f["t"]))
    samples, first_vis = [], {}
    kinds = {i: str(e.get("kind") or "thing") for i, e in ents.items()}
    for f in frames:
        t = float(f["t"])
        e_s, zones_s = {}, {}
        for row in f.get("entities") or []:
            i = _sid(row["id"])
            state = str(row.get("state") or "unknown")
            if state not in STATES:
                raise ValueError(f"frame t={t}: {row['id']} has state {state!r}, not one of {STATES}")
            status = _TO_STATUS.get(state)
            if status is None:              # hidden
                status = {"under": "UNDER", "inside": "INSIDE"}.get(str(row.get("hidden_in") or ""), "HIDDEN")
            zone = _zone_of(row, zones)
            pos = row.get("table_cm")
            pos = (float(pos[0]), float(pos[1])) if pos is not None and zone in (None, "table") else None
            parent = row.get("parent")
            e_s[i] = (status, _sid(parent) if parent and not str(parent).startswith("hand") else parent, pos)
            if zone not in (None, "table"):
                zones_s[i] = zone
            kinds.setdefault(i, "thing")
            if status == "VISIBLE" and i not in first_vis:
                first_vis[i] = (t, pos)
        samples.append(Sample(t=t, ents=e_s, hands=[tuple(float(v) for v in h) for h in f.get("hands_cm") or []],
                              seen={}, zones=zones_s, names={}))
    names = {}
    for i, e in ents.items():
        nm = {}
        if e.get("names"):
            nm["aliases"] = [str(x) for x in e["names"]]
        if isinstance(e.get("guess"), dict):
            nm["guess"] = e["guess"]
        if nm:
            names[i] = nm
    if samples and names:
        samples[-1].names = names           # the scorecard reads the latest names
    evs = []
    for ev in events:
        typ = EVENTS.get(str(ev.get("event")))
        if typ is None:
            continue
        evs.append({"t": float(ev["t"]), "obj": _sid(ev["id"]), "type": typ,
                    "parent": _sid(ev["parent"]) if ev.get("parent") and not str(ev["parent"]).startswith("hand")
                    else ev.get("parent"), "to_cm": ev.get("table_cm"), "from_cm": ev.get("from_cm")})
    born = {e["obj"] for e in evs if e["type"] == "APPEARED"}
    for i, e in ents.items():               # births: the entity line, else the first visible frame
        if i in born:
            continue
        t = e.get("born_t")
        if t is not None:
            pos = first_vis.get(i, (None, None))[1]
            evs.append({"t": float(t), "obj": i, "type": "APPEARED", "parent": None,
                        "to_cm": list(pos) if pos else None, "from_cm": None})
            born.add(i)
    for i, (t, pos) in first_vis.items():
        if i not in born:
            evs.append({"t": t, "obj": i, "type": "APPEARED", "parent": None, "to_cm": list(pos) if pos else None,
                        "from_cm": None})
    evs.sort(key=lambda e: e["t"])
    merged = {i: _sid(e["merged_into"]) for i, e in ents.items() if e.get("merged_into")}
    return Trace(samples=samples, events=evs, kinds=kinds, merged=merged, objects={},
                 detector=f"{head.get('tracker', '?')}: {head.get('detector') or ''}".rstrip(": "),
                 view=head.get("view"), room=bool(head.get("room_memory")),
                 overrides=list(head.get("overrides") or []), wall_s=float(head.get("replay_wall_s") or 0.0))


def read_track(path: Union[str, Path], zones: Optional[dict] = None) -> tuple[dict, Trace]:
    lines = read_lines(path)
    return (lines[0] if lines else {}), track_to_trace(lines, zones)
