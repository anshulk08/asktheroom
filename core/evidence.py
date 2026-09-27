"""Answer evidence: the proof picture behind an answer ("your keys are on the couch" + the moment they got
there), for the demo page and the phone. Answer.evidence is a list of items, most relevant first:

    {"kind": "event" | "look" | "recall" | "change",
     "snapshot_url": "/snapshots/...jpg",   # room evidence is the whole camera view (<= 1280 px wide)
     "closeup_url": "/snapshots/...jpg" | None,   # a room arrival: the sharp zone crop too
     "t": wall time the picture was taken, "caption": "Your keys, on the couch at 1:42 PM",
     "box": [x1, y1, x2, y2] | None,       # the object, in the snapshot image's own pixels
     "size": [w, h] | None,                # that image's size, to scale the box to a displayed <img>
     "box_px": [x1, y1, x2, y2] | None,    # the same box in the camera's full-frame px (room places)
     "clock": "1:42 PM",                   # t as the rig's own clock says it
     "obj": entity | None, "type": event type | None}

Sources: a logged event's snapshot (EventLog: the table view for table events, the zone crop for room ones,
plus <stem>_room.jpg, the whole view, for a room arrival), an archive frame (core/visual_memory.py, served
under /snapshots/archive/...) or the frame a room look sent (saved as <ms>_look.jpg). Only files inside the
snapshot dir with plain names get a url; anything else is left out rather than exposing a path.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Iterable, Optional

log = logging.getLogger(__name__)

MAX_ITEMS = 3
# what /snapshots serves: a plain file in the snapshot dir, or an archive frame
SNAP_REL_RE = re.compile(r"^(?:archive/\d{8}-\d{2}/)?[A-Za-z0-9][A-Za-z0-9._:-]*\.(?:jpg|jpeg|png)$")   # thing:N

# The event that put an object where it is, by its status (WHERE); any logged event as a last resort.
PLACED = ["PUT_BACK", "MOVED", "APPEARED", "FOUND", "CORRECTED", "TAKEN_OUT", "UNCOVERED"]
BY_STATUS = {"VISIBLE": PLACED, "INSIDE": ["PUT_INSIDE"], "UNDER": ["COVERED"], "GONE": ["EXITED_VIEW"],
             "HELD": ["PICKED_UP"]}


def snapshot_url(path: Optional[str], snap_dir: Optional[str]) -> Optional[str]:
    """'/snapshots/<path relative to snap_dir>' for a file inside it with a plain name, else None."""
    if not path or not snap_dir:
        return None
    try:
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(snap_dir)).replace(os.sep, "/")
    except ValueError:
        return None
    if rel.startswith("..") or not SNAP_REL_RE.match(rel):
        return None
    return f"/snapshots/{rel}"


def image_size(path: str) -> Optional[tuple[int, int]]:
    """(w, h) of a JPEG/PNG, or None (unreadable, or not written yet)."""
    try:
        import cv2
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None if im is None else (im.shape[1], im.shape[0])
    except Exception:
        return None


def context_size(frame_wh: tuple) -> tuple[int, int]:
    """The size EventLog saves a whole-view context at (core/events.CONTEXT_PX long side)."""
    from core.events import CONTEXT_PX
    w, h = frame_wh
    s = min(1.0, CONTEXT_PX / max(w, h))
    return round(w * s), round(h * s)


def clock(wall: float) -> str:
    lt = time.localtime(wall)
    return f"{lt.tm_hour % 12 or 12}:{lt.tm_min:02d} {'AM' if lt.tm_hour < 12 else 'PM'}"


def item(kind: str, path: Optional[str], t: float, caption: str, snap_dir: Optional[str], *, box=None,
         box_space: Optional[tuple] = None, closeup: Optional[str] = None, obj: Optional[str] = None,
         type_: Optional[str] = None) -> Optional[dict]:
    """One evidence item, or None when the picture has no servable url. box is in box_space px ((w, h) of
    the frame it was measured in; None: the snapshot's own px) and is scaled to the snapshot's pixels."""
    url = snapshot_url(path, snap_dir)
    if url is None:
        return None
    size = None
    if box is not None:           # a just-logged snapshot may not be on disk yet: a context's size is known
        size = image_size(path) or (context_size(box_space) if box_space and path.endswith("_room.jpg") else None)
    out_box = None
    if box is not None and size is not None:
        sx, sy = ((size[0] / box_space[0], size[1] / box_space[1]) if box_space and box_space[0] and box_space[1]
                  else (1.0, 1.0))
        x1, y1, x2, y2 = (float(v) for v in box)
        out_box = [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]
    return {"kind": kind, "snapshot_url": url, "closeup_url": snapshot_url(closeup, snap_dir), "t": round(t, 3),
            "clock": clock(t), "caption": caption, "box": out_box,
            "size": list(size) if out_box is not None else None,
            "box_px": [round(float(v)) for v in box] if box is not None and box_space else None,
            "obj": obj, "type": type_}


def room_context(snapshot: Optional[str]) -> Optional[str]:
    """The whole-view file EventLog saved beside a room arrival's snapshot, if it is there."""
    if not snapshot or not snapshot.endswith(".jpg"):
        return None
    p = snapshot[:-4] + "_room.jpg"
    return p if os.path.exists(p) else None


def from_event(ev, snap_dir: Optional[str], caption: str, kind: str = "event", box=None,
               box_space: Optional[tuple] = None) -> Optional[dict]:
    """An event's snapshot as evidence: for a room arrival its whole view (box in box_space, the camera
    frame's px) with the zone crop as the close-up; otherwise the snapshot itself (box in its px)."""
    snap = getattr(ev, "snapshot", None)
    if not snap:
        return None
    ctx = room_context(snap)
    if ctx is not None:
        return item(kind, ctx, ev.wall, caption, snap_dir, box=box, box_space=box_space, closeup=snap,
                    obj=ev.obj, type_=str(ev.type))
    return item(kind, snap, ev.wall, caption, snap_dir, box=None if box_space else box, obj=ev.obj,
                type_=str(ev.type))


def with_snapshot(events, obj: str, types: Optional[Iterable[str]] = None, n: int = 20):
    """The newest logged event of obj (of these types) that has a snapshot, or None."""
    try:
        evs = events.last(obj, n)
    except Exception:
        return None
    want = set(types) if types else None
    return next((e for e in evs if e.snapshot and (want is None or str(e.type) in want)), None)


def nearest(events, obj: str, types: Iterable[str], wall: Optional[float], n: int = 50):
    """The logged event of obj (of these types, with a snapshot) closest in time to wall (None: the newest)."""
    try:
        evs = [e for e in events.last(obj, n) if e.snapshot and str(e.type) in set(types)]
    except Exception:
        return None
    if not evs:
        return None
    return evs[0] if wall is None else min(evs, key=lambda e: abs(e.wall - wall))


def trim(items: Iterable[Optional[dict]]) -> list[dict]:
    """Drop Nones and repeats of a picture; at most MAX_ITEMS."""
    out, seen = [], set()
    for it in items:
        if it and it["snapshot_url"] not in seen:
            seen.add(it["snapshot_url"])
            out.append(it)
    return out[:MAX_ITEMS]
