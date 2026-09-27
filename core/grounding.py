"""Open-vocabulary grounding: find a NAMED thing anywhere in the room by text, with Moondream 3.1's cloud
/detect and /point (https://api.moondream.ai/v1, model moondream3.1-9B-A2B). Off by default
(config.yaml `grounding.enabled: false`); the key is MOONDREAM_API_KEY in .env, never logged.

    g = MoondreamGrounder(GroundingConfig(enabled=True))
    g.detect(img_bgr, "red coffee mug") -> [(x0, y0, x1, y1, conf)]    # the image's own px
    g.point(img_bgr, "red coffee mug")  -> [(x, y)]
    find_anywhere(g, frames, "keys", zones, table_rect) -> Found | None   # full frame, then zone close-ups

Why: the tracker only knows the props it was trained on and things it saw arrive, and Grok is bad at boxes
(voice/visual.py: 0/5 boxes, mean IoU 0.14). Moondream is a grounding model: any phrase, a box back, zero
shot. Detection is not identity (it finds A mug, not YOUR mug), so every answer from it is hedged ("Your
keys look like they're on the couch") and the laser never aims from it.

Contract (docs.moondream.ai, checked 27 Sep; the PyPI client moondream 2.5.0 agrees): POST /detect and
/point with X-Moondream-Auth, body {model, image_url: base64 JPEG data URL, object, settings}. The model
field is always sent. Coordinates come back normalized 0-1: they are multiplied by the size of the image
the caller passed, so the downscale before upload (max_px long side; the model tiles to ~1.8 MP anyway)
changes nothing. /detect returns no score, so every box gets hit_conf; boxes over max_area_frac of the
image are furniture, not objects, and are dropped. /detect also boxes something for a thing that isn't there
(first real call, rig frame f1440, 27 Sep: 'umbrella' boxed the orange case on the table), so with verify on
(the default) a box is only used after a closed /query on its close-up ("Is the object in the middle of this
picture an umbrella?") says yes: that said no to the umbrella and yes to the pill bottle and tissue box, but
also no to the keys (0.3-0.8 s more per box). Measure before trusting it: eval/grounding_bench.py.
Python's default urllib User-Agent is blocked (HTTP 403, "error code: 1010"): this goes through requests
with its own User-Agent, like core/xai.py.

Failure is never an exception: no key, offline, a timeout, any non-2xx, or a cap reached gives [] with a
log line, and the callers fall through to what they did before. One retry with backoff after a timeout,
connection error or 5xx; a 429 blocks that provider for backoff_429_s (or Retry-After), a 401/403 for
auth_backoff_s. Calls are spaced min_interval_s apart (Moondream allows 2 req/s on 3.1) and capped per
minute and per hour; all of it is thread-safe. With cf_fallback and CF_ACCOUNT_ID + CF_API_TOKEN set,
a failed Moondream call is retried once on Cloudflare Workers AI (@cf/moondream/moondream3.1-9B-A2B,
'target' instead of 'object'); provider: cloudflare uses only that.

refind() is WS8's pluggable re-find backend (core/permanence.py on ws/permanence: refind(name, refs, view
img, view box, suspect boxes) -> [(box full px, confidence, source)], needs_suspects False): select it with
permanence.refind: moondream. It lives here so it works whether or not that branch is merged.
Python 3.10 (JetPack 6).
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.moondream.ai/v1"
MODEL = "moondream3.1-9B-A2B"
CF_MODEL = "@cf/moondream/moondream3.1-9B-A2B"
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
USER_AGENT = "askroom-grounding/1 (+requests)"   # never urllib's default: Cloudflare in front of the API blocks it

Box = tuple  # (x0, y0, x1, y1, conf), image px


@dataclass
class GroundingConfig:
    """config.yaml `grounding:` (a new section at the end). Every key optional; these are the defaults."""
    enabled: bool = False
    provider: str = "moondream"             # moondream | cloudflare
    model: str = MODEL
    base_url: str = BASE_URL
    api_key_env: str = "MOONDREAM_API_KEY"
    cf_fallback: bool = True                # Workers AI when a Moondream call fails, if the CF creds are set
    cf_account_env: str = "CF_ACCOUNT_ID"
    cf_token_env: str = "CF_API_TOKEN"
    timeout_s: float = 3.0                  # per HTTP request
    retries: int = 1                        # after a timeout, connection error or 5xx
    backoff_s: float = 0.3                  # before retry i: backoff_s * 2**i
    backoff_429_s: float = 10.0             # a 429 stops every call this long (or Retry-After, if longer)
    auth_backoff_s: float = 60.0            # a 401/403 stops every call this long
    min_interval_s: float = 0.5             # between call starts (2 req/s: Moondream's default limit on 3.1)
    max_wait_s: float = 1.0                 # a call that would wait longer than this for its slot is skipped
    max_per_minute: int = 30
    max_per_hour: int = 300
    max_px: int = 1792                      # long side sent (1792x1008 of a 2560x1440 frame: ~1.8 MP, what the
                                            # model's 12 tiles of ~378 px plus a global view see anyway)
    jpeg_quality: int = 90
    max_objects: int = 5
    verify: bool = True                     # a box is said only after a closed /query on its close-up says yes
    verify_boxes: int = 2                   # boxes per search checked that way, best first
    hit_conf: float = 0.8                   # /detect has no score: the confidence every box gets
    min_conf: float = 0.5                   # boxes below this are dropped (a scored provider, or a tuned hit_conf)
    max_area_frac: float = 0.05             # a box over this share of the image is furniture, not the object
    where_deadline_s: float = 2.5           # voice/visual.py: longest a WHERE question waits for find_anywhere
    zone_crops: int = 3                     # zone close-ups searched after a miss on the full frame, smallest first
    zone_pad: float = 0.15                  # each zone's box grown by this share of its size on every side ...
    zone_rise: float = 0.5                  # ... and upwards by this share of its height (things stand above it)
    min_crop_px: int = 384                  # a close-up is widened to at least this (about one model tile)
    near_px: float = 250.0                  # outside every zone: 'near <zone>' within this many px of it (2560 wide)

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "GroundingConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in known})


class Grounder(Protocol):
    """Anything that finds a phrase in an image: boxes (x0, y0, x1, y1, conf) and points (x, y) in the
    image's own px. Never raises; [] when nothing is found or the service can't be reached."""

    def detect(self, img_bgr: np.ndarray, phrase: str, deadline: Optional[float] = None) -> list: ...

    def point(self, img_bgr: np.ndarray, phrase: str, deadline: Optional[float] = None) -> list: ...


def _a(phrase: str) -> str:
    """'a red mug', 'an apple', 'keys' (a plural is left bare)."""
    p = phrase.strip()
    if p.lower().startswith(("a ", "an ", "the ", "my ")) or (p.endswith("s") and not p.endswith("ss")):
        return p
    return f"{'an' if p[:1].lower() in 'aeiou' else 'a'} {p}"


def encode(img_bgr: np.ndarray, max_px: int = 1792, quality: int = 90) -> str:
    """A base64 JPEG data URL of img (shrunk to max_px on the long side; never enlarged)."""
    import cv2
    h, w = img_bgr.shape[:2]
    s = max_px / max(h, w) if max_px else 1.0
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _norm(v: Any, n: float) -> Optional[float]:
    """A coordinate from the API (0-1, the contract; a little out of range is clamped) as px of an image n px
    along that axis. None for anything else (Workers AI's schema doesn't say its coordinates are normalized:
    a px answer is dropped, never misplaced)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return min(max(f, 0.0), 1.0) * n if f <= 1.5 else None


def boxes_px(objects: Any, wh: tuple, conf: float) -> list:
    """/detect's objects [{x_min, y_min, x_max, y_max}] (0-1) -> [(x0, y0, x1, y1, conf)] in px of a w x h
    image. Malformed or empty boxes are skipped."""
    w, h = wh
    out = []
    for o in objects or []:
        if not isinstance(o, dict):
            continue
        x0, y0 = _norm(o.get("x_min"), w), _norm(o.get("y_min"), h)
        x1, y1 = _norm(o.get("x_max"), w), _norm(o.get("y_max"), h)
        if None in (x0, y0, x1, y1) or x1 <= x0 or y1 <= y0:
            continue
        score = o.get("confidence")
        out.append((x0, y0, x1, y1, float(score) if isinstance(score, (int, float)) else conf))
    return out


def points_px(points: Any, wh: tuple) -> list:
    """/point's points [{x, y}] (0-1) -> [(x, y)] in px of a w x h image."""
    w, h = wh
    out = []
    for p in points or []:
        if not isinstance(p, dict):
            continue
        x, y = _norm(p.get("x"), w), _norm(p.get("y"), h)
        if x is not None and y is not None:
            out.append((x, y))
    return out


class MoondreamGrounder:
    """Moondream 3.1 cloud /detect and /point (Cloudflare Workers AI as the fallback). Thread-safe: one
    requests.Session, and the slot, cap and backoff state behind one lock."""

    def __init__(self, c: Optional[GroundingConfig] = None, session: Optional[requests.Session] = None,
                 clock=time.monotonic, sleep=time.sleep, env=None):
        self.c = c or GroundingConfig()
        self._session = session
        self.clock, self.sleep = clock, sleep
        self.env = os.environ if env is None else env
        self._lock = threading.Lock()
        self._calls: deque = deque()            # monotonic start times, the last hour
        self._next = 0.0                        # earliest start of the next call
        self._blocked: dict = {}                # provider -> no calls to it before this (429 / auth backoff)
        self._said: dict = {}                   # log key -> last time said (log lines at most once a minute)
        self.last: Optional[dict] = None        # the last call: provider, status, ms, n (for logs and the bench)

    # -- keys (read from the environment on every call, never logged)

    def key(self) -> str:
        return str(self.env.get(self.c.api_key_env, "") or "").strip()

    def cf_creds(self) -> Optional[tuple]:
        a = str(self.env.get(self.c.cf_account_env, "") or "").strip()
        t = str(self.env.get(self.c.cf_token_env, "") or "").strip()
        return (a, t) if a and t else None

    def available(self) -> bool:
        """Configured with a key for its provider (says nothing about the network)."""
        if self.c.provider == "cloudflare":
            return self.cf_creds() is not None
        return bool(self.key()) or (self.c.cf_fallback and self.cf_creds() is not None)

    # -- the public calls

    def detect(self, img_bgr: np.ndarray, phrase: str, deadline: Optional[float] = None) -> list:
        """[(x0, y0, x1, y1, conf)] in img's px, best first as the API orders them; boxes over max_area_frac
        of img or under min_conf are dropped. [] on no key, offline, timeout, any error, or a cap."""
        r = self._ask("detect", img_bgr, phrase, deadline)
        if not r:
            return []
        h, w = img_bgr.shape[:2]
        out = []
        for b in boxes_px(r.get("objects"), (w, h), self.c.hit_conf):
            if (b[2] - b[0]) * (b[3] - b[1]) > self.c.max_area_frac * w * h or b[4] < self.c.min_conf:
                continue
            out.append(b)
        return out

    def point(self, img_bgr: np.ndarray, phrase: str, deadline: Optional[float] = None) -> list:
        """[(x, y)] centres in img's px. [] on no key, offline, timeout, any error, or a cap."""
        r = self._ask("point", img_bgr, phrase, deadline)
        if not r:
            return []
        h, w = img_bgr.shape[:2]
        return points_px(r.get("points"), (w, h))

    def confirm(self, img_bgr: np.ndarray, box: Sequence, phrase: str, deadline: Optional[float] = None) -> bool:
        """A closed /query on a close-up of box (padded by its own size): is the object in the middle this
        phrase? /detect answers for things that aren't there (on a rig frame 'umbrella' boxed an orange
        case); this 'no' drops it. False on 'no', no answer, or any failure: an unconfirmed box is never said."""
        h, w = img_bgr.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in box[:4])
        p = max(x1 - x0, y1 - y0)
        crop = img_bgr[max(0, int(y0 - p)):min(h, int(y1 + p)), max(0, int(x0 - p)):min(w, int(x1 + p))]
        if crop.size == 0:
            return False
        r = self._ask("query", crop, f"Is the object in the middle of this picture {_a(phrase)}? Answer yes or no.",
                      deadline)
        return bool(r) and str(r.get("answer") or "").strip().lower().startswith("yes")

    # -- plumbing

    def _say(self, key: str, level: int, msg: str, *args) -> None:
        now = self.clock()
        if now - self._said.get(key, -1e9) >= 60.0:
            self._said[key] = now
            log.log(level, msg, *args)
        else:
            log.debug(msg, *args)

    def session(self) -> requests.Session:
        with self._lock:
            if self._session is None:
                self._session = requests.Session()
            return self._session

    def _admit(self, end: Optional[float], provider: str = "moondream") -> Optional[float]:
        """Reserve a call slot: seconds to wait before starting it, or None (backing off, a cap reached, or
        the slot too far off / past the deadline). Counts the call."""
        with self._lock:
            now = self.clock()
            blocked = self._blocked.get(provider, 0.0)
            if now < blocked:
                self._say(f"{provider}:blocked", logging.INFO, "grounding: backing off %s for %.0f s more", provider,
                          blocked - now)
                return None
            while self._calls and self._calls[0] <= now - 3600:
                self._calls.popleft()
            minute = sum(1 for t in self._calls if t > now - 60)
            if minute >= self.c.max_per_minute or len(self._calls) >= self.c.max_per_hour:
                self._say("cap", logging.WARNING, "grounding: call cap reached (%d a minute, %d an hour)",
                          self.c.max_per_minute, self.c.max_per_hour)
                return None
            start = max(now, self._next)
            wait = start - now
            if wait > self.c.max_wait_s or (end is not None and start + 0.2 > end):
                return None
            self._next = start + float(self.c.min_interval_s)
            self._calls.append(start)
            return wait

    def _ask(self, task: str, img_bgr: np.ndarray, phrase: str, deadline: Optional[float]) -> Optional[dict]:
        """The provider's JSON reply (Moondream's shape: objects / points), or None on any failure."""
        phrase = (phrase or "").strip()
        if img_bgr is None or not phrase or not self.c.enabled:
            return None
        md, cf = self.key(), self.cf_creds()
        use_md = self.c.provider != "cloudflare" and bool(md)
        use_cf = cf is not None and (self.c.provider == "cloudflare" or self.c.cf_fallback)
        if not use_md and not use_cf:
            self._say("nokey", logging.INFO, "grounding: no %s set; not looking",
                      self.c.api_key_env if self.c.provider != "cloudflare" else self.c.cf_token_env)
            return None
        try:
            url = encode(img_bgr, self.c.max_px, self.c.jpeg_quality)
        except Exception:
            log.exception("grounding: image encode failed")
            return None
        if use_md:
            body = ({"model": self.c.model, "image_url": url, "question": phrase} if task == "query" else
                    {"model": self.c.model, "image_url": url, "object": phrase,
                     "settings": {"max_objects": int(self.c.max_objects)}})
            r = self._post("moondream", f"{self.c.base_url.rstrip('/')}/{task}",
                           {"X-Moondream-Auth": md, "Content-Type": "application/json", "User-Agent": USER_AGENT},
                           body, deadline)
            if r is not None or not use_cf or self.c.provider == "cloudflare":
                return r
        body = ({"task": task, "image": url, "question": phrase, "reasoning": False} if task == "query" else
                {"task": task, "image": url, "target": phrase, "max_objects": int(self.c.max_objects)})
        r = self._post("cloudflare", CF_URL.format(account=cf[0], model=CF_MODEL),
                       {"Authorization": f"Bearer {cf[1]}", "Content-Type": "application/json",
                        "User-Agent": USER_AGENT}, body, deadline)
        if r is None:
            return None
        res = r.get("result") if isinstance(r.get("result"), dict) else r      # Cloudflare's envelope
        return {"objects": res.get("objects") or [], "points": res.get("points") or [], "answer": res.get("answer")}

    def _post(self, provider: str, url: str, headers: dict, body: dict, deadline: Optional[float]) -> Optional[dict]:
        """POST with the slot, caps, retry and backoff rules; the JSON body on 2xx, else None."""
        for attempt in range(1 + max(0, int(self.c.retries))):
            wait = self._admit(deadline, provider)
            if wait is None:
                return None
            if wait > 0:
                self.sleep(wait)
            timeout = float(self.c.timeout_s)
            if deadline is not None:
                timeout = min(timeout, deadline - self.clock())
                if timeout <= 0.1:
                    return None
            t0 = time.perf_counter()
            status, retry = None, False
            try:
                r = self.session().post(url, headers=headers, json=body, timeout=timeout)
                status = r.status_code
            except requests.Timeout:
                self._say(f"{provider}:timeout", logging.INFO, "grounding: %s timed out after %.1f s",
                          provider, timeout)
                retry = True
            except requests.RequestException as ex:
                self._say(f"{provider}:net", logging.INFO, "grounding: %s unreachable (%s)", provider,
                          type(ex).__name__)
                retry = True
            ms = int((time.perf_counter() - t0) * 1000)
            self.last = {"provider": provider, "status": status, "ms": ms, "attempt": attempt}
            if status is not None and 200 <= status < 300:
                try:
                    d = r.json()
                except ValueError:
                    log.warning("grounding: %s sent a non-JSON reply", provider)
                    return None
                return d if isinstance(d, dict) else None
            if status == 429:
                back = float(self.c.backoff_429_s)
                try:
                    back = max(back, float(r.headers.get("Retry-After") or 0))
                except (TypeError, ValueError, AttributeError):
                    pass
                with self._lock:
                    self._blocked[provider] = max(self._blocked.get(provider, 0.0), self.clock() + back)
                log.warning("grounding: %s rate-limited (429); no calls for %.0f s", provider, back)
                return None
            if status in (401, 403):
                with self._lock:
                    self._blocked[provider] = max(self._blocked.get(provider, 0.0),
                                                  self.clock() + float(self.c.auth_backoff_s))
                self._say(f"{provider}:auth", logging.WARNING, "grounding: %s refused the request (HTTP %s); "
                          "check the key", provider, status)
                return None
            if status is not None:
                retry = status >= 500
                self._say(f"{provider}:{status}", logging.WARNING, "grounding: %s HTTP %s", provider, status)
            if not retry or attempt >= int(self.c.retries):
                return None
            pause = float(self.c.backoff_s) * (2 ** attempt)
            if deadline is not None and self.clock() + pause + 0.3 > deadline:
                return None
            self.sleep(pause)
        return None


# ================================================================================ places

@dataclass
class Found:
    """Where find_anywhere found the phrase: its full-frame px box and the place it stands in."""
    box: tuple                  # (x0, y0, x1, y1) full-frame px
    conf: float
    place: str                  # a zone name, 'table', or 'room' (in no zone)
    where: str                  # spoken: 'on the couch', 'on the table', 'near the doorway', 'somewhere on the left ...'
    source: str                 # 'full' or 'zone:<name>'
    ms: int = 0


def load_zones(path: str, frame_wh: tuple) -> list:
    """[(name, say, poly in this frame's px)] from a room_zones.json drawn at any size. [] if unreadable."""
    try:
        from core.room_zones import Zones
        zs = Zones.load(path)
    except Exception:
        log.info("grounding: no zones at %s", path)
        return []
    (dw, dh), (fw, fh) = zs.size_px, frame_wh
    sx, sy = (fw / dw if dw else 1.0), (fh / dh if dh else 1.0)
    return [(z.name, z.say, [(x * sx, y * sy) for x, y in z.poly]) for z in zs.zones.values()]


def crop_box(poly, frame_wh: tuple, pad: float = 0.15, rise: float = 0.5, min_px: int = 384) -> Optional[tuple]:
    """The frame px box a zone's close-up is cut from: the polygon's box grown by pad of its size on every
    side and upwards by rise of its height (what stands on a surface rises above it in the image), widened to
    at least min_px (slid back inside the frame at its edges). None if empty."""
    if len(poly) < 3:
        return None
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    b = [x1 - pad * w, y1 - pad * h - rise * h, x2 + pad * w, y2 + pad * h]
    fw, fh = frame_wh
    for lo, hi, n in ((0, 2, fw), (1, 3, fh)):
        if b[hi] - b[lo] < min_px:
            c = (b[lo] + b[hi]) / 2
            b[lo], b[hi] = c - min_px / 2, c + min_px / 2
            shift = max(0.0, -b[lo]) - max(0.0, b[hi] - n)       # slid back inside the frame, not cut
            b[lo], b[hi] = b[lo] + shift, b[hi] + shift
    x1, y1, x2, y2 = max(0, round(b[0])), max(0, round(b[1])), min(fw, round(b[2])), min(fh, round(b[3]))
    return (x1, y1, x2, y2) if x2 - x1 > 1 and y2 - y1 > 1 else None


def _inside(poly, pt) -> bool:
    import cv2
    if len(poly) < 3:
        return False
    return cv2.pointPolygonTest(np.asarray(poly, np.float32).reshape(-1, 1, 2), (float(pt[0]), float(pt[1])),
                                False) >= 0


def _dist(poly, pt) -> float:
    import cv2
    if len(poly) < 3:
        return float("inf")
    return max(0.0, -cv2.pointPolygonTest(np.asarray(poly, np.float32).reshape(-1, 1, 2),
                                          (float(pt[0]), float(pt[1])), True))


def _on(name: str, say: str) -> str:
    """The preposition for being at a zone: 'by the doorway', 'on the couch'."""
    low = f"{name} {say}".lower()
    return "by" if "door" in low or "window" in low else "on"


def place_of(box, zones: Sequence, table_rect: Optional[Sequence] = None, frame_wh: tuple = (2560, 1440),
             near_px: float = 250.0, table_say: str = "the table") -> tuple:
    """(place, spoken where) for a full-frame px box, from its footprint (the bottom-centre: where it
    stands): the first zone whose polygon holds it ('couch', 'on the couch'), else the table view's rect
    ('table', 'on the table'), else 'room' with the nearest zone within near_px (scaled from a 2560-wide
    frame: 'near the doorway') or a rough region ('somewhere on the left side of the room')."""
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    foot = ((x0 + x1) / 2, y1)
    for name, say, poly in zones or []:
        if _inside(poly, foot):
            return name, f"{_on(name, say)} {say}"
    if table_rect is not None:
        tx0, ty0, tx1, ty1 = (float(v) for v in table_rect)
        if tx0 <= foot[0] <= tx1 and ty0 <= foot[1] <= ty1:
            return "table", f"on {table_say}"
    fw, fh = frame_wh
    lim = near_px * fw / 2560.0
    near = min(((_dist(poly, foot), name, say) for name, say, poly in zones or []), default=None,
               key=lambda q: q[0])
    if near is not None and near[0] <= lim:
        return "room", f"near {near[2]}"
    cx, cy = foot[0] / fw, foot[1] / fh
    if cx < 1 / 3:
        side = "on the left side of the room"
    elif cx > 2 / 3:
        side = "on the right side of the room"
    elif cy < 0.45:
        side = "on the far side of the room"
    else:
        side = "in the middle of the room"
    return "room", f"somewhere {side}"


def _full_img(frames) -> Optional[np.ndarray]:
    """The camera's whole view from a TableView-like source (latest_full), a Frame, or an image."""
    if frames is None:
        return None
    if isinstance(frames, np.ndarray):
        return frames
    full = getattr(frames, "latest_full", None)
    try:
        f = full() if callable(full) else (frames.latest() if hasattr(frames, "latest") else frames)
    except Exception:
        log.debug("grounding: no full frame", exc_info=True)
        return None
    img = getattr(f, "img", None)
    return img if isinstance(img, np.ndarray) else None


def find_anywhere(grounder, frames, phrase: str, zones: Optional[Sequence] = None,
                  table_rect: Optional[Sequence] = None, deadline_s: Optional[float] = None,
                  c: Optional[GroundingConfig] = None, table_say: str = "the table") -> Optional[Found]:
    """Look for phrase in the camera's whole raw view (frames: TableView.latest_full, a Frame or an image),
    and if nothing is there, in native-resolution close-ups of the zones ([(name, say, poly)] in that
    frame's px), smallest first, at most c.zone_crops. The first box found, mapped to full-frame px and to a
    place (place_of). None when nothing is found, the view is missing, or deadline_s runs out (no new call
    starts then; one in flight is bounded by the same deadline)."""
    c = c or getattr(grounder, "c", None) or GroundingConfig()
    img = _full_img(frames)
    if img is None or not (phrase or "").strip():
        return None
    t0 = time.monotonic()
    end = t0 + float(deadline_s) if deadline_s is not None else None
    fh, fw = img.shape[:2]
    zones = list(zones or [])

    confirm = getattr(grounder, "confirm", None) if c.verify else None

    def hit(boxes, dx, dy, source) -> Optional[Found]:
        checked = 0
        for b in boxes:
            box = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
            if (box[2] - box[0]) * (box[3] - box[1]) > c.max_area_frac * fw * fh:
                continue                                  # furniture-sized in the whole view
            if confirm is not None:
                if checked >= c.verify_boxes or (end is not None and time.monotonic() + 0.3 > end):
                    return None
                checked += 1
                if not confirm(img, box, phrase, deadline=end):
                    log.info("grounding: %r at %s not confirmed", phrase, tuple(round(v) for v in box))
                    continue
            place, where = place_of(box, zones, table_rect, (fw, fh), c.near_px, table_say)
            return Found(box, float(b[4]), place, where, source, int((time.monotonic() - t0) * 1000))
        return None

    got = hit(grounder.detect(img, phrase, deadline=end), 0, 0, "full")
    if got is not None:
        return got
    crops = []
    for name, _, poly in zones:
        b = crop_box(poly, (fw, fh), c.zone_pad, c.zone_rise, c.min_crop_px)
        if b is not None and (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * fw * fh:
            crops.append(((b[2] - b[0]) * (b[3] - b[1]), name, b))
    for _, name, (x1, y1, x2, y2) in sorted(crops)[: max(0, int(c.zone_crops))]:
        if end is not None and time.monotonic() + 0.3 > end:
            break
        got = hit(grounder.detect(img[y1:y2, x1:x2], phrase, deadline=end), x1, y1, f"zone:{name}")
        if got is not None:
            return got
    return None


# ================================================================================ app wiring

_default: Optional[MoondreamGrounder] = None
_default_lock = threading.Lock()


def from_config(cfg: Optional[dict], force: bool = False) -> Optional[MoondreamGrounder]:
    """The app's grounder, or None when grounding.enabled is false (the default) and not force. A missing
    key is not an error here: every call then returns [] with a log line."""
    c = GroundingConfig.from_dict((cfg or {}).get("grounding"))
    if not (c.enabled or force):
        return None
    if force:
        c.enabled = True
    if c.provider not in ("moondream", "cloudflare"):
        log.warning("grounding: unknown provider %r; off", c.provider)
        return None
    g = MoondreamGrounder(c)
    log.info("grounding on: %s %s%s", c.provider, c.model,
             "" if g.available() else f" (no {c.api_key_env} yet: every look returns nothing)")
    return g


def default_grounder() -> Optional[MoondreamGrounder]:
    """A process-wide grounder for refind(): configure(cfg)'s, else one from load_config()'s grounding
    section with enabled forced on (choosing permanence.refind: moondream is the opt-in)."""
    global _default
    with _default_lock:
        if _default is None:
            try:
                from core.config import load_config
                _default = from_config(load_config(), force=True)
            except Exception:
                log.exception("grounding: no config; re-find off")
                return None
        return _default


def configure(cfg: Optional[dict], grounder: Optional[MoondreamGrounder] = None) -> Optional[MoondreamGrounder]:
    """Set the grounder refind() uses (given, or from cfg with enabled forced on)."""
    global _default
    with _default_lock:
        _default = grounder if grounder is not None else from_config(cfg, force=True)
        return _default


def refind(name: str, refs: list, view_img: np.ndarray, view_box: Sequence, suspect_boxes: list) -> list:
    """WS8's re-find backend (permanence.refind: moondream): /detect of name on the view, boxes mapped from
    the view to full-frame px. refs and suspect_boxes are unused (text grounding needs neither).
    [(box full px, confidence, 'moondream')]; [] when nothing is found or the service is unavailable."""
    g = default_grounder()
    if g is None or view_img is None:
        return []
    x0, y0 = float(view_box[0]), float(view_box[1])
    boxes = g.detect(view_img, name)
    if g.c.verify:
        boxes = [b for b in boxes[: max(0, int(g.c.verify_boxes))] if g.confirm(view_img, b, name)]
    return [((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), float(b[4]), "moondream") for b in boxes]


refind.needs_suspects = False
