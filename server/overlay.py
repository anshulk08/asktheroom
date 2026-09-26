"""Draw the world model on top of a camera frame for the dashboard's /video stream.

draw(frame_img, state, table=None, dets=None) -> BGR image (downscaled to ~960 px wide).

- `state` is WorldState JSON (core/types.py).
- `table` (optional) maps table cm -> frame px via `cm_to_px(np.ndarray) -> np.ndarray`. Without it
  nothing is drawn at a table position; entity states go in a legend panel instead.
- `dets` (optional) is a core.types.Detections; its boxes are drawn when given.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

OUT_WIDTH = 960
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR, matched to the dashboard palette
STATUS_BGR = {
    "VISIBLE": (190, 230, 170),
    "HELD": (75, 184, 242),
    "INSIDE": (255, 197, 142),
    "UNDER": (255, 197, 142),
    "GONE": (133, 140, 122),
    "UNKNOWN": (160, 160, 160),
}
LASER_BGR = (68, 90, 255)
PANEL_BGR = (45, 51, 23)
INK_BGR = (232, 238, 230)
DIM_BGR = (170, 181, 157)


def _label(e: dict) -> str:
    st = e.get("status", "UNKNOWN")
    parent = e.get("parent")
    if st in ("INSIDE", "UNDER") and parent:
        where = f"{st.lower()} {parent}"
    elif st == "HELD":
        where = "held"
    elif st == "GONE":
        where = f"gone {e['edge']}" if e.get("edge") else "gone"
    elif st == "VISIBLE":
        where = "on table"
    else:
        where = "unknown"
    return f"{e['name'].replace('_', ' ')}: {where}"


def _text(img, s, org, scale=0.5, color=INK_BGR, thick=1):
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _cm_to_px(table, pts_cm) -> Optional[np.ndarray]:
    """Call table.cm_to_px, tolerating (2,) or (N,2) conventions. Returns (N,2) float or None."""
    if table is None:
        return None
    try:
        arr = np.asarray(pts_cm, dtype=np.float32).reshape(-1, 2)
        out = np.asarray(table.cm_to_px(arr), dtype=np.float32).reshape(-1, 2)
        return out
    except Exception:
        return None


def _panel(img, x, y, w, h, alpha=0.62):
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    tint = np.empty_like(roi)
    tint[:] = PANEL_BGR
    cv2.addWeighted(tint, alpha, roi, 1 - alpha, 0, dst=roi)


def _crosshair(img, c, r=16):
    x, y = int(c[0]), int(c[1])
    cv2.circle(img, (x, y), r, LASER_BGR, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, LASER_BGR, -1, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (x + dx * (r - 6), y + dy * (r - 6)), (x + dx * (r + 8), y + dy * (r + 8)),
                 LASER_BGR, 2, cv2.LINE_AA)


def draw(frame_img: np.ndarray, state: Optional[dict], table=None, dets=None,
         out_width: int = OUT_WIDTH) -> np.ndarray:
    h0, w0 = frame_img.shape[:2]
    s = out_width / float(w0) if w0 > out_width else 1.0
    if s != 1.0:
        img = cv2.resize(frame_img, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA)
    else:
        img = frame_img.copy()
    state = state or {}
    ents = state.get("entities") or []
    by_name = {e.get("name"): e for e in ents}
    H, W = img.shape[:2]

    # detection boxes (px in source-frame coordinates)
    if dets is not None:
        for d in list(getattr(dets, "items", []) or []) + list(getattr(dets, "hands", []) or []):
            try:
                x1, y1, x2, y2 = [int(v * s) for v in d.box_px]
            except Exception:
                continue
            name = d.cls.split(":")[0]
            color = STATUS_BGR.get((by_name.get(name) or {}).get("status", ""), DIM_BGR)
            if name == "hand":
                color = STATUS_BGR["HELD"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            _text(img, f"{d.cls} {d.conf:.2f}", (x1 + 3, max(12, y1 - 5)), 0.45, color)

    # per-entity markers at table positions (only when the table mapping exists)
    if table is not None:
        for e in ents:
            pos = e.get("resolved_cm") or e.get("pos_cm")
            if not pos or e.get("status") == "GONE":
                continue
            px = _cm_to_px(table, pos)
            if px is None:
                continue
            x, y = px[0] * s
            if not (0 <= x < W and 0 <= y < H):
                continue
            color = STATUS_BGR.get(e.get("status"), DIM_BGR)
            if e.get("status") == "VISIBLE" and e.get("kind") == "target":
                cv2.circle(img, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(x), int(y)), 7, color, 2, cv2.LINE_AA)
    # laser target marker
    laser = state.get("laser") or {}
    target = laser.get("target")
    if laser.get("on") and target and table is not None and target in by_name:
        pos = by_name[target].get("resolved_cm") or by_name[target].get("pos_cm")
        px = _cm_to_px(table, pos) if pos else None
        if px is not None:
            _crosshair(img, px[0] * s)

    # labels at positions: stack labels for entities sharing a resolved spot
    if table is not None:
        slots: dict[tuple[int, int], int] = {}
        for e in ents:
            pos = e.get("resolved_cm") or e.get("pos_cm")
            if not pos or e.get("status") == "GONE":
                continue
            px = _cm_to_px(table, pos)
            if px is None:
                continue
            x, y = px[0] * s
            if not (0 <= x < W and 0 <= y < H):
                continue
            key = (int(x) // 24, int(y) // 24)
            k = slots.get(key, 0)
            slots[key] = k + 1
            color = STATUS_BGR.get(e.get("status"), DIM_BGR)
            _text(img, _label(e), (x + 12, y + 5 + 17 * k), 0.45, color)

    # legend panel: every entity's state (always; it is the only state display without a table)
    if ents:
        rows = [(e, _label(e), e.get("confidence")) for e in ents]
        pw = 12 + max(cv2.getTextSize(r[1], FONT, 0.45, 1)[0][0] for r in rows) + 76
        ph = 10 + 19 * len(rows)
        _panel(img, 10, 10, pw, ph)
        for i, (e, lab, conf) in enumerate(rows):
            y = 10 + 18 + 19 * i
            color = STATUS_BGR.get(e.get("status"), DIM_BGR)
            cv2.circle(img, (22, y - 5), 4, color, -1, cv2.LINE_AA)
            _text(img, lab, (32, y), 0.45, INK_BGR)
            if conf is not None:
                _text(img, f"{int(round(conf * 100))}%", (10 + pw - 42, y), 0.42, DIM_BGR)

    # laser badge
    if laser.get("on"):
        msg = f"laser: {str(target).replace('_', ' ')}" if target else "laser on"
        if laser.get("err_cm") is not None:
            msg += f"  ({laser['err_cm']:.1f} cm)"
        tw = cv2.getTextSize(msg, FONT, 0.5, 1)[0][0]
        _panel(img, W - tw - 34, 10, tw + 24, 28)
        cv2.circle(img, (W - tw - 22, 24), 4, LASER_BGR, -1, cv2.LINE_AA)
        _text(img, msg, (W - tw - 14, 29), 0.5, INK_BGR)
    return img


def placeholder(width: int = 1280, height: int = 720, msg: str = "No camera") -> np.ndarray:
    """A 'no camera' frame, drawn in the dashboard palette."""
    img = np.empty((height, width, 3), np.uint8)
    img[:] = (38, 44, 21)
    step = 40
    for x in range(0, width, step):
        cv2.line(img, (x, 0), (x, height), (52, 60, 31), 1)
    for y in range(0, height, step):
        cv2.line(img, (0, y), (width, y), (52, 60, 31), 1)
    (tw, th), _ = cv2.getTextSize(msg, FONT, 1.4, 2)
    cv2.putText(img, msg, ((width - tw) // 2, (height + th) // 2), FONT, 1.4, DIM_BGR, 2, cv2.LINE_AA)
    return img
