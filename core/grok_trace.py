"""What Grok was shown and what it said: a ring of the last N calls, for the demo's "Grok's eyes" panel
(server GET /grok/trace, /grok/img/<id>; the /demo page).

core/xai.Client._create records every chat call here: its time, purpose (from the system prompt: naming,
verify, pick, look, look_room, recall, recall_room, check, refind, confirm, answer, understand,
narration, other), model, latency, the text parts of the request (hint lists, questions; trimmed), small
JPEG thumbnails of the images actually sent (at most THUMB_PX on the long side, in memory only) and the
reply (the message content, trimmed; tool call names). Never the API key, headers or the full images.

Memory: KEEP calls x a few thumbnails of ~20-40 KB, about 5 MB at most. Recording costs one JPEG decode at
a reduced size plus an encode per image, on the calling thread after the reply came back.
"""
from __future__ import annotations

import base64
import itertools
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

log = logging.getLogger(__name__)

KEEP = 30                    # calls kept
THUMB_PX = 480               # thumbnail long side
TEXT_MAX = 600               # request text kept per call
REPLY_MAX = 1500             # reply text kept per call

# System prompt openings -> purpose, most specific first (core/auto_name, core/room, voice/visual, ...).
PURPOSES = [
    ("You name one object", "naming"),
    ("You look at part of a room seen by a ceiling camera. One object is marked with a red box", "verify"),
    ("You check what an object tracker", "check"),
    ("You answer spoken questions about what the room looked like earlier", "recall_room"),
    ("You answer spoken questions about what was on a tabletop earlier", "recall"),
    ("You answer spoken questions about a room", "look_room"),
    ("You answer spoken questions about a tabletop", "look"),
    ("You find one object", "pick"),
    ("You describe short episodes", "narration"),
    ("You sort questions", "understand"),
    ("You are the voice of", "answer"),
]
KEYWORDS = [("same object", "confirm"), ("numbered", "refind"), ("marks", "refind")]

_lock = threading.Lock()
_calls: deque = deque(maxlen=KEEP)
_imgs: dict[str, bytes] = {}
_ids = itertools.count(1)


def purpose(system: str) -> str:
    s = (system or "").strip()
    for start, name in PURPOSES:
        if s.startswith(start):
            return name
    low = s[:400].lower()
    return next((name for k, name in KEYWORDS if k in low), "other")


def _thumb(url: str) -> Optional[bytes]:
    """A data:image/...;base64 URL -> a JPEG at most THUMB_PX on its long side, or None."""
    try:
        import cv2
        import numpy as np
        b64 = url.split(",", 1)[1] if url.startswith("data:") else None
        if not b64:
            return None
        raw = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        s = THUMB_PX / max(h, w)
        if s < 1:
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return buf.tobytes() if ok else None
    except Exception:
        log.debug("grok trace thumbnail failed", exc_info=True)
        return None


def record(body: dict, reply: Optional[dict], ms: float, error: Optional[str] = None) -> None:
    """One chat call (the request body as sent, the parsed reply or None). Never raises."""
    try:
        msgs = body.get("messages") or []
        system = next((m.get("content") for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str)), "")
        texts, thumbs = [], []
        for m in msgs:
            if m.get("role") == "system":
                continue
            c = m.get("content")
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                for part in c:
                    if part.get("type") == "text":
                        texts.append(str(part.get("text", "")))
                    elif part.get("type") == "image_url":
                        t = _thumb(str((part.get("image_url") or {}).get("url", "")))
                        if t is not None:
                            thumbs.append(t)
        content, tools = "", []
        if reply:
            msg = ((reply.get("choices") or [{}])[0] or {}).get("message") or {}
            content = msg.get("content") or ""
            tools = [((c or {}).get("function") or {}).get("name") for c in msg.get("tool_calls") or []]
        n = next(_ids)
        rec = {"id": n, "t": time.time(), "purpose": purpose(system), "model": body.get("model"),
               "ms": int(ms), "ok": error is None, "error": error, "request": " / ".join(t for t in texts if t)[:TEXT_MAX],
               "reply": str(content)[:REPLY_MAX], "tools": [t for t in tools if t],
               "images": [f"{n}-{i}" for i in range(len(thumbs))]}
        with _lock:
            if len(_calls) == _calls.maxlen:                   # the oldest call's thumbnails go with it
                for k in _calls[0]["images"]:
                    _imgs.pop(k, None)
            _calls.append(rec)
            for k, t in zip(rec["images"], thumbs):
                _imgs[k] = t
    except Exception:
        log.debug("grok trace failed", exc_info=True)


def calls(limit: int = KEEP) -> list[dict]:
    """The newest calls first."""
    with _lock:
        return [dict(c) for c in list(_calls)[::-1][:max(0, int(limit))]]


def image(key: str) -> Optional[bytes]:
    with _lock:
        return _imgs.get(key)


def clear() -> None:
    with _lock:
        _calls.clear()
        _imgs.clear()
