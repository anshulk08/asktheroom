"""Visual memory: an archive of what the table (and, with room memory on, the whole room) looked like,
searchable by text, for past-tense questions ('was there a red mug here this morning?', 'what was on the
couch earlier?').

Archive policy. The perception thread hands a frame reference to a worker at most every check_every_s
(O(1) there; if the worker is behind, the check is skipped, never queued). The worker compares a 128x72
grey thumbnail with the last archived one and saves a JPEG (<= frame_px, quality jpeg_quality) when:
  - it is the first frame, or archive_every_s have passed since the last save (a floor: something is
    kept at least every 30 s even if nothing moved), or
  - the scene changed (share of thumbnail pixels changed >= change_thr %, or the world model logged an event since
    the last save) and no hand is over the table: the settled view after a change, not the hand
    in the middle of it.
  The change gate (config.yaml turns it on; the code defaults keep the policy above): with a tabletop outline
  (core/table_area.py, via proposals.table_roi) only pixels on the tabletop count (change_on_table: the
  corner rig's table view also shows legs, laps and the couch); a world event counts only when the view
  changed too (events_need_pixels: the tracker's COVERED/UNCOVERED churn); change frames are at least
  change_min_gap_s apart; settle_checks waits for a still view. eval/archive_replay.py measures it on a rig hour.
Each row records the time, the path, the change score, whether hands were in view, why it was saved,
and a digest of the world model then (which entities were visible and roughly where), so text search
works on the world model's names as well as on pixels.

Room frames. With room memory on the frame source is a core.room_view.TableView: the table pipeline sees a
cut of the camera frame, and full_at(t) / latest_full() give the whole view. The worker then also keeps the
whole view (<= room_frame_px, rows with view = 'room') every room_every_s, and after a world event once
room_min_gap_s has passed: people on a couch move all the time, so pixel change is no trigger there. Room
rows are embedded with the finer room_tiles grid (a cup on the far counter is ~30 px of a 1280 px view) and
have their own disk cap (room_max_mb), so a busy table does not push the room's history out.

Search. A pluggable embedder (MobileCLIP2-S0 in ONNX by default: image and text towers from
scripts/export_mobileclip.py, tokenized by core/clip_tokenizer.py) embeds each saved frame on a third
thread: the whole frame letterboxed, plus a grid of overlapping tiles, since a mug is only ~20 px in a
256 px view of the whole table. Embeddings are float16 blobs in SQLite. A text query is embedded with
'a photo of {query}' and each frame scores its best tile. Frames the embedder could not keep up with
are skipped then and backfilled when it is idle.

Storage lives in the EventLog's SQLite file (table visual_frames) and snapshots dir (archive/), pruned
after keep_h hours and capped at max_mb, oldest first.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS visual_frames (id INTEGER PRIMARY KEY, t REAL, path TEXT, bytes INTEGER,
  score REAL, hands INTEGER, reason TEXT, digest TEXT, emb BLOB, emb_n INTEGER, emb_model TEXT, view TEXT);
CREATE INDEX IF NOT EXISTS visual_frames_t ON visual_frames(t);
"""
VIEWS = ("table", "room")


@dataclass
class VisualConfig:
    """The config.yaml visual_memory: section (its archive: and embed: sub-sections are merged in)."""
    enabled: bool = False
    # the VLM for looking and recalling (same providers as core/narration.py)
    provider: str = "grok"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    reasoning_effort: Optional[str] = "none"   # on the voice path: speed first
    timeout_s: float = 8.0
    max_tokens: int = 400
    max_per_hour: int = 60
    look_px: int = 1280            # current frame long side sent to the VLM
    crop_px: int = 384             # close-ups of named entities are upscaled to this
    max_crops: int = 3
    abstain_below: float = 0.5     # VLM confidence under this: "I can't tell from here"
    recall_frames: int = 6         # archived frames per past-tense question
    recall_px: int = 768
    # archive
    archive_every_s: float = 30.0
    check_every_s: float = 1.0
    change_thr: float = 0.3        # % of 128x72 grey thumbnail pixels changed by > 20 levels (keys ~0.4%)
    change_on_table: bool = True   # with a calibrated table, change is measured on the tabletop outline only
    settle_checks: int = 0         # a change is kept once the view has held still this many checks in a row
    change_min_gap_s: float = 0.0  # ... and at least this long after the last kept frame
    events_need_pixels: bool = False   # a world event marks a change only if the view changed too
    frame_px: int = 1280
    jpeg_quality: int = 80
    max_mb: float = 500.0
    keep_h: float = 24.0
    # whole-room frames (room memory on: the frame source has latest_full)
    room_every_s: float = 30.0     # a room frame at least this often (0: none)
    room_min_gap_s: float = 10.0   # ... and after a world event, once this long has passed
    room_frame_px: int = 1280
    room_max_mb: float = 300.0     # the room frames' own disk cap
    room_tiles: tuple = (4, 3)
    # the whole-room look (voice/visual.py look_room): native-resolution close-ups of drawn zones
    room_crops: int = 3            # at most this many zone close-ups per room question
    room_crop_px: int = 640        # their long side (upscaled from the native crop)
    room_crop_max_frac: float = 0.1   # unnamed in the question, only zones under this share of the frame
    # embedder
    embed: str = "onnx"            # onnx | none | fake
    image_model: str = "models/mobileclip2_s0_image.onnx"
    text_model: str = "models/mobileclip2_s0_text.onnx"
    vocab: str = "assets/bpe_simple_vocab_16e6.txt.gz"
    providers: tuple = ("tensorrt", "cuda", "cpu")   # onnxruntime execution providers, tried in order
    cache_dir: str = "models/cache/clip"   # TensorRT engine cache (first build is slow; later starts load it)
    tiles: tuple = (3, 2)          # grid (cols, rows) of tiles embedded besides the whole frame
    min_sim: float = 0.21          # best text match below this: nothing matched (one stock photo: present 0.22-0.28, absent 0.11-0.19)
    embed_backlog: int = 4

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VisualConfig":
        d = dict(d or {})
        flat = {**{k: v for k, v in d.items() if not isinstance(v, dict)}, **(d.get("archive") or {}),
                **(d.get("embed") or {})}
        if "kind" in flat and "embed" not in flat:
            flat["embed"] = flat.pop("kind")
        if isinstance(d.get("embed"), str):
            flat["embed"] = d["embed"]
        known = {f.name for f in fields(cls)}
        c = cls(**{k: v for k, v in flat.items() if k in known})
        c.tiles = tuple(c.tiles) if c.tiles else (0, 0)
        c.room_tiles = tuple(c.room_tiles) if c.room_tiles else (0, 0)
        return c


# ---------------------------------------------------------------- embedders

def letterbox(img: np.ndarray) -> np.ndarray:
    """Pad to a square (grey) so a 16:9 frame keeps its edges when the model wants a square input."""
    h, w = img.shape[:2]
    n = max(h, w)
    out = np.full((n, n, 3), 114, np.uint8)
    out[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = img
    return out


def tiles(img: np.ndarray, grid=(3, 2), overlap: float = 0.15) -> list[np.ndarray]:
    """The whole image plus a cols x rows grid of tiles, each grown by `overlap` so an object on a
    tile border is whole in at least one."""
    out = [img]
    cols, rows = grid
    if cols <= 0 or rows <= 0:
        return out
    h, w = img.shape[:2]
    tw, th = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            x1, y1 = max(0, int(c * tw - overlap * tw)), max(0, int(r * th - overlap * th))
            x2, y2 = min(w, int((c + 1) * tw + overlap * tw)), min(h, int((r + 1) * th + overlap * th))
            out.append(img[y1:y2, x1:x2])
    return out


class Embedder:
    """image(list of BGR arrays) -> (n, d) unit rows; text(list of str) -> (n, d) unit rows."""
    name = "none"
    can_text = False

    def image(self, imgs: list) -> np.ndarray:
        raise NotImplementedError

    def text(self, texts: list) -> np.ndarray:
        raise NotImplementedError


_EP = {"tensorrt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}


def ort_providers(names, available, cache_dir: str, profile: Optional[tuple] = None) -> list:
    """Config names (tensorrt, cuda, cpu) -> onnxruntime providers this build has, in order; TensorRT
    runs FP16 with its engines cached (as core/embed.py does for re-ID) and, given profile = (input,
    min_shape, max_shape, opt_shape), one engine for that range instead of a rebuild per new shape."""
    out = []
    for n in names:
        ep = _EP.get(str(n).lower())
        if ep not in available:
            continue
        if ep == "TensorrtExecutionProvider":
            os.makedirs(cache_dir, exist_ok=True)
            opts = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": cache_dir}
            if profile:
                name, lo, hi, opt = profile
                opts.update(trt_profile_min_shapes=f"{name}:{lo}", trt_profile_max_shapes=f"{name}:{hi}",
                            trt_profile_opt_shapes=f"{name}:{opt}")
            out.append((ep, opts))
        else:
            out.append(ep)
    return out or ["CPUExecutionProvider"]


class OnnxClipEmbedder(Embedder):
    """MobileCLIP2-S0 towers in onnxruntime. Sessions are built on first use, which the archive's embed
    thread does at start (warm()), never the perception or voice thread: the first TensorRT build takes
    ~4 minutes on the Orin (then it loads from cache_dir). The image tower may use TensorRT (FP16,
    measured 5.2 ms / image, 26 ms for a frame's 7 tiles on the Orin Nano); the text tower stays on
    CUDA / CPU (7.2 ms a query on the Orin in FP32), so a question never waits on an engine build. The
    text tower (254 MB) loads on first use. Preprocessing matches open_clip's for this model: RGB,
    resize to 256 (bilinear), centre crop, 0-1, no mean/std (inputs here are letterboxed squares, so
    the crop cuts nothing)."""
    name = "mobileclip2_s0"

    def __init__(self, image_path: str, text_path: Optional[str] = None, vocab: Optional[str] = None,
                 providers=("tensorrt", "cuda", "cpu"), size: int = 256, cache_dir: str = "models/cache/clip",
                 max_batch: int = 8):
        import onnxruntime as ort
        self._ort, self.size = ort, size
        avail = ort.get_available_providers()
        self.providers = ort_providers(providers, avail, cache_dir,
                                       ("images", f"1x3x{size}x{size}", f"{max_batch}x3x{size}x{size}",
                                        f"7x3x{size}x{size}"))
        self.text_providers = ort_providers([p for p in providers if str(p).lower() != "tensorrt"], avail, cache_dir)
        self.image_path, self.text_path, self.vocab, self.max_batch = image_path, text_path, vocab, max_batch
        self._img = self._txt = self._tok = None
        self._lock = threading.Lock()
        self.can_text = bool(text_path and vocab and os.path.exists(text_path) and os.path.exists(vocab))

    def warm(self) -> None:
        """Build both sessions (call from a background thread)."""
        with self._lock:
            if self._img is None:
                self._img = self._ort.InferenceSession(self.image_path, providers=self.providers)
            if self.can_text and self._txt is None:
                from core.clip_tokenizer import ClipTokenizer
                self._tok = ClipTokenizer(self.vocab)
                self._txt = self._ort.InferenceSession(self.text_path, providers=self.text_providers)

    def _prep(self, img: np.ndarray) -> np.ndarray:
        import cv2
        sq = letterbox(img)
        x = cv2.resize(sq, (self.size, self.size), interpolation=cv2.INTER_LINEAR if sq.shape[0] < self.size
                       else cv2.INTER_AREA)
        return x[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

    def image(self, imgs: list) -> np.ndarray:
        if self._img is None:
            self.warm()
        x = np.stack([self._prep(i) for i in imgs])
        return np.concatenate([self._img.run(None, {"images": x[i:i + self.max_batch]})[0]
                               for i in range(0, len(x), self.max_batch)]).astype(np.float32)

    def text(self, texts: list) -> np.ndarray:
        if not self.can_text:
            raise RuntimeError("no text tower")
        if self._txt is None:
            self.warm()
        return self._txt.run(None, {"tokens": self._tok(texts)})[0].astype(np.float32)


class FakeEmbedder(Embedder):
    """For tests: an image is its mean colour (plus a constant, so no row is zero); a text is the colour
    words in it. 'a photo of a red mug' is close to red frames and far from blue ones."""
    name = "fake"
    can_text = True
    COLORS = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255),
              "black": (0, 0, 0), "yellow": (0, 255, 255)}

    def __init__(self):
        self.calls = 0

    @staticmethod
    def _unit(v) -> np.ndarray:
        v = np.asarray(v, np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def image(self, imgs: list) -> np.ndarray:
        self.calls += 1
        return np.stack([self._unit(list(i.reshape(-1, 3).mean(0) / 255.0 - 0.5) + [0.05]) for i in imgs])

    def text(self, texts: list) -> np.ndarray:
        out = []
        for t in texts:
            cs = [np.array(c) / 255.0 - 0.5 for w, c in self.COLORS.items() if w in t.lower()]
            v = list(np.mean(cs, 0)) + [0.05] if cs else [0.0, 0.0, 0.0, 1.0]
            out.append(self._unit(v))
        return np.stack(out)


def make_embedder(c: VisualConfig) -> Optional[Embedder]:
    """The configured embedder, or None (then recall works by time window only)."""
    if c.embed == "fake":
        return FakeEmbedder()
    if c.embed != "onnx":
        return None
    from core.config import ROOT

    def at_root(p: str) -> str:
        return p if os.path.isabs(p) else str(ROOT / p)
    image, text, vocab, cache = (at_root(p) for p in (c.image_model, c.text_model, c.vocab, c.cache_dir))
    if not os.path.exists(image):
        log.warning("visual memory: %s missing (python scripts/export_mobileclip.py); no text search", image)
        return None
    try:
        return OnnxClipEmbedder(image, text, vocab, providers=c.providers, cache_dir=cache)
    except Exception:
        log.exception("visual memory: embedder failed to load; no text search")
        return None


# ---------------------------------------------------------------- storage

@dataclass
class FrameRow:
    id: int
    t: float
    path: Optional[str]
    bytes: int
    score: float
    hands: bool
    reason: str
    digest: dict
    emb: Optional[np.ndarray]
    view: str = "table"


def _frame_row(r, dim: Optional[int] = None) -> FrameRow:
    i, t, path, nbytes, score, hands, reason, digest, blob, n, view = r
    emb = None
    if blob is not None and n:
        emb = np.frombuffer(blob, np.float16).astype(np.float32).reshape(n, -1)
    try:
        dg = json.loads(digest) if digest else {}
    except ValueError:
        dg = {}
    return FrameRow(i, t, path, nbytes or 0, score or 0.0, bool(hands), reason or "", dg, emb, view or "table")


class ArchiveStore:
    """visual_frames in the EventLog's database (same connection rules as core/narration_store.py)."""

    def __init__(self, events):
        self.events = events
        with events._locked():
            c = events._conn()
            c.executescript(SCHEMA)
            if "view" not in {r[1] for r in c.execute("PRAGMA table_info(visual_frames)")}:
                c.execute("ALTER TABLE visual_frames ADD COLUMN view TEXT")     # archives from before room frames
                c.commit()

    def _exec(self, sql: str, args=()) -> int:
        with self.events._locked():
            c = self.events._conn()
            cur = c.execute(sql, args)
            c.commit()
            return cur.lastrowid

    def _q(self, sql: str, args=()) -> list:
        with self.events._locked():
            return self.events._conn().execute(sql, args).fetchall()

    def add(self, t: float, path: str, nbytes: int, score: float, hands: bool, reason: str, digest: dict,
            view: str = "table") -> int:
        return self._exec("INSERT INTO visual_frames (t, path, bytes, score, hands, reason, digest, view) "
                          "VALUES (?,?,?,?,?,?,?,?)",
                          (t, path, nbytes, score, int(hands), reason, json.dumps(digest), view))

    def set_embedding(self, id_: int, emb: np.ndarray, model: str) -> None:
        e = np.asarray(emb, np.float16)
        self._exec("UPDATE visual_frames SET emb = ?, emb_n = ?, emb_model = ? WHERE id = ?",
                   (e.tobytes(), int(e.shape[0]), model, id_))

    _COLS = "id, t, path, bytes, score, hands, reason, digest, emb, emb_n, view"
    _VIEW = "COALESCE(view, 'table') = ?"

    def window(self, t0: float, t1: float, view: Optional[str] = None) -> list[FrameRow]:
        """Rows in [t0, t1] by time; only one view's ('table' or 'room') when given."""
        w, args = ("", (t0, t1)) if view is None else (f" AND {self._VIEW}", (t0, t1, view))
        return [_frame_row(r) for r in self._q(f"SELECT {self._COLS} FROM visual_frames WHERE t >= ? AND t <= ?{w} "
                                               "ORDER BY t", args)]

    def get(self, id_: int) -> Optional[FrameRow]:
        r = self._q(f"SELECT {self._COLS} FROM visual_frames WHERE id = ?", (id_,))
        return _frame_row(r[0]) if r else None

    def unembedded(self, limit: int = 4) -> list[FrameRow]:
        return [_frame_row(r) for r in self._q(
            f"SELECT {self._COLS} FROM visual_frames WHERE emb IS NULL AND path IS NOT NULL ORDER BY t DESC LIMIT ?",
            (limit,))]

    def stats(self) -> dict:
        n, nb, ne, nr = self._q("SELECT COUNT(*), COALESCE(SUM(bytes), 0), COUNT(emb), "
                                "COALESCE(SUM(view = 'room'), 0) FROM visual_frames")[0]
        return {"frames": n, "mb": round(nb / 1e6, 1), "embedded": ne, "room_frames": nr}

    def prune(self, keep_h: float, max_mb: float, now: Optional[float] = None, view: Optional[str] = None) -> int:
        """Drop frames older than keep_h, then the oldest until the JPEGs fit in max_mb (of one view's
        frames when given: the table and the room have separate caps)."""
        now = time.time() if now is None else now
        w, args = ("", ()) if view is None else (f" AND {self._VIEW}", (view,))
        gone = self._q(f"SELECT id, path FROM visual_frames WHERE t < ?{w}", (now - keep_h * 3600, *args))
        rows = self._q(f"SELECT id, path, bytes FROM visual_frames WHERE t >= ?{w} ORDER BY t",
                       (now - keep_h * 3600, *args))
        total = sum(b or 0 for _, _, b in rows)
        cap = max_mb * 1e6
        for i, p, b in rows:
            if total <= cap:
                break
            gone.append((i, p))
            total -= b or 0
        for i, p in gone:
            if p:
                try:
                    os.remove(p)
                    os.rmdir(os.path.dirname(p))          # the hour's folder, once it is empty
                except OSError:
                    pass
            self._exec("DELETE FROM visual_frames WHERE id = ?", (i,))
        return len(gone)


# ---------------------------------------------------------------- the archive

def change_pct(a: np.ndarray, b: Optional[np.ndarray], level: float = 20.0, mask: Optional[np.ndarray] = None) -> float:
    """Percent of thumbnail pixels (those in mask, when given) whose grey level moved by more than `level`
    (100 with no reference). A share of changed pixels, not a mean difference: a phone is ~2% of the table
    and would vanish in a mean, while sensor noise averaged down to 128x72 stays far under 20 levels."""
    if b is None:
        return 100.0
    moved = np.abs(a - b) > level
    return float((moved[mask] if mask is not None else moved).mean() * 100.0)


def table_mask(table, cfg: dict, frame_wh: tuple) -> Optional[np.ndarray]:
    """The 128x72 thumbnail pixels on the tabletop: core/proposals.table_roi (the operator's outline, else
    the calibrated area) scaled from frame px. None without a calibrated table, or when the outline covers
    under 2% of the view (then the whole view counts, as before)."""
    import cv2
    from core.proposals import table_roi
    roi = table_roi(table, cfg)
    if not roi:
        return None
    fw, fh = frame_wh
    pts = np.array([[x * 128 / fw, y * 72 / fh] for x, y in roi], np.float32)
    m = np.zeros((72, 128), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool) if m.mean() >= 0.02 else None


def _thumb(img: np.ndarray) -> np.ndarray:
    import cv2
    small = cv2.resize(img, (128, 72), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) if small.ndim == 3 else small.astype(np.float32)


def digest(world, cfg: dict) -> dict:
    """What the world model believed when the frame was saved: visible entities by spoken name and
    rough area, and the hidden ones with where they are."""
    from core.config import display_name
    from core.labels import thing_labels
    from core.viewframe import View
    try:
        st = world.state_json()
    except Exception:
        return {}
    labels = st.get("aliases") or {}
    shown = thing_labels(st)                     # a thing's taught name, else 'mug?' or 'something new
    view = View.from_cfg(cfg)
    vis, hid = [], []
    for e in st.get("entities", []):
        n = e.get("name", "")
        name = e.get("label") or (labels.get(n) if n.startswith("thing:") else None) or \
            (shown.get(n) or "something new" if n.startswith("thing:") else display_name(cfg, n))
        pos = e.get("resolved_cm") or e.get("pos_cm")
        area = view.area_word(pos)
        if e.get("status") == "VISIBLE":
            vis.append({"name": name, "area": area})
        elif e.get("status") in ("INSIDE", "UNDER") and e.get("parent"):
            hid.append({"name": name, "status": e["status"].lower(), "parent": display_name(cfg, e["parent"])})
    return {"visible": vis, "hidden": hid}


def digest_text(d: dict) -> str:
    vis = ", ".join(f"{v['name']}" + (f" ({v['area']})" if v.get("area") else "") for v in d.get("visible", []))
    hid = ", ".join(f"{h['name']} {h['status']} the {h['parent']}" for h in d.get("hidden", []))
    return "; ".join(p for p in (f"tracked on the table: {vis}" if vis else "", hid) if p) or "no tracked objects"


class VisualArchive:
    """Saves, embeds and searches frames (see the module docstring). start=False: no threads (tests
    call drain()). frames: the app's frame source; one with latest_full (TableView) adds room frames."""

    def __init__(self, cfg: dict, events, world=None, embedder: Optional[Embedder] = None,
                 clock: Callable[[], float] = time.time, start: bool = True, c: Optional[VisualConfig] = None,
                 frames=None, table=None):
        self.cfg = cfg
        self.frames = frames
        self.table = table              # with a tabletop outline, only change on the tabletop counts
        self.c = c or VisualConfig.from_dict(cfg.get("visual_memory"))
        self.store = ArchiveStore(events)
        self.root = os.path.join(events.snap_dir, "archive")
        self.world = world
        self.embedder = embedder
        self.clock = clock
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._eq: queue.Queue = queue.Queue(maxsize=max(1, self.c.embed_backlog))
        self._last_check: Optional[float] = None
        self._dirty = False
        self._last_thumb: Optional[np.ndarray] = None
        self._last_save: Optional[float] = None
        self._last_room: Optional[float] = None
        self._prev_thumb: Optional[np.ndarray] = None     # the previous check's (for settle_checks)
        self._still = 0
        self._pending_change = False
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._pruned = 0.0
        self.skipped = 0
        if start:
            for target, name in ((self._loop, "archive"), (self._embed_loop, "archive-embed")):
                th = threading.Thread(target=target, name=name, daemon=True)
                th.start()
                self._threads.append(th)

    # -- perception thread

    def feed(self, frame, dets, new_events=()) -> None:
        """O(1); never blocks, never raises."""
        try:
            if new_events:
                self._dirty = True
            if frame is None or getattr(frame, "img", None) is None:
                return
            if self._last_check is not None and frame.t - self._last_check < self.c.check_every_s:
                return
            self._last_check = frame.t
            item = ("check", frame, bool(getattr(dets, "hands", None)), self._dirty)
            try:
                self._q.put_nowait(item)
                self._dirty = False
            except queue.Full:
                self.skipped += 1
        except Exception:
            log.exception("archive feed failed")

    def attach(self, world) -> "VisualArchive":
        update = world.update

        def fed_update(dets, frame):
            out = update(dets, frame)
            if frame is not None:
                self.feed(frame, dets, out)
            return out

        world.update = fed_update
        self.world = world
        return self

    # -- archive worker

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._check(*item[1:])
            except Exception:
                log.exception("archive check failed")

    def drain(self) -> None:
        """Process queued checks and embeddings now (start=False)."""
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            self._check(*item[1:])
        while True:
            try:
                item = self._eq.get_nowait()
            except queue.Empty:
                break
            self._embed(*item)
        if self.embedder is not None:
            self._backfill(1000)

    def _backfill(self, n: int) -> None:
        """Embed saved frames the embed queue had no room for (from their JPEGs)."""
        import cv2
        for r in self.store.unembedded(n):
            im = cv2.imread(r.path) if r.path else None
            if im is not None:
                try:
                    self._embed(r.id, im, r.view)
                except Exception:
                    log.exception("archive backfill failed")

    def _check(self, frame, hands: bool, dirty: bool) -> Optional[int]:
        """Decide whether this frame is archived (see the module docstring); returns the table row id if so.
        A room frame is kept on its own schedule."""
        if self._room_due(frame.wall, dirty):
            try:
                self._save_room(frame)
            except Exception:
                log.exception("archive room frame failed")
        th = _thumb(frame.img)
        mask = self._mask(frame.img) if self.c.change_on_table else None
        score = change_pct(th, self._last_thumb, mask=mask)
        still = self._prev_thumb is not None and change_pct(th, self._prev_thumb, mask=mask) < self.c.change_thr
        self._still, self._prev_thumb = (self._still + 1 if still else 0), th
        # A world event marks a change (with events_need_pixels, only if the view changed too); a pixel
        # difference only counts without hands in view (a hand passing over an unchanged table is not a change).
        changed = score >= self.c.change_thr
        if (dirty and (changed or not self.c.events_need_pixels)) or (not hands and changed):
            self._pending_change = True
        if self._last_save is None:
            reason = "first"
        elif frame.wall - self._last_save >= self.c.archive_every_s:
            reason = "interval"
        elif (self._pending_change and not hands and self._still >= self.c.settle_checks
              and frame.wall - self._last_save >= self.c.change_min_gap_s):
            reason = "change"
        else:
            return None
        return self._save(frame, th, score, hands, reason)

    def _mask(self, img: np.ndarray) -> Optional[np.ndarray]:
        """table_mask for this frame (recomputed each check: a recalibration moves the outline)."""
        if self.table is None:
            return None
        try:
            return table_mask(self.table, self.cfg, (img.shape[1], img.shape[0]))
        except Exception:
            log.debug("archive: no tabletop mask", exc_info=True)
            return None

    def _write(self, img: np.ndarray, wall: float, long_side: int, suffix: str = "") -> tuple[np.ndarray, str, int]:
        """img shrunk to long_side and saved as the hour folder's JPEG: (saved img, path, bytes)."""
        import cv2
        h, w = img.shape[:2]
        s = long_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        d = os.path.join(self.root, datetime.fromtimestamp(wall).strftime("%Y%m%d-%H"))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{int(wall * 1000)}{suffix}.jpg")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.c.jpeg_quality)])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return img, path, len(buf)

    def _queue_embed(self, id_: int, img: np.ndarray, view: str) -> None:
        if self.embedder is not None:
            try:
                self._eq.put_nowait((id_, img, view))
            except queue.Full:
                self.skipped += 1           # backfilled when the embedder is idle

    def _save(self, frame, th, score: float, hands: bool, reason: str) -> int:
        img, path, nbytes = self._write(frame.img, frame.wall, self.c.frame_px)
        dg = digest(self.world, self.cfg) if self.world is not None else {}
        id_ = self.store.add(frame.wall, path, nbytes, round(score, 2), hands, reason, dg)
        self._last_thumb, self._last_save, self._pending_change = th, frame.wall, False
        self._queue_embed(id_, img, "table")
        now = self.clock()
        if now - self._pruned > 60:
            self._pruned = now
            if self._room_source() is None:
                self.store.prune(self.c.keep_h, self.c.max_mb, now=now)
            else:
                self.store.prune(self.c.keep_h, self.c.max_mb, now=now, view="table")
                self.store.prune(self.c.keep_h, self.c.room_max_mb, now=now, view="room")
        return id_

    # -- room frames

    def _room_source(self):
        """frames.full_at (else latest_full) when the source has the whole camera view, else None."""
        if self.c.room_every_s <= 0 or self.frames is None:
            return None
        return getattr(self.frames, "full_at", None) or getattr(self.frames, "latest_full", None)

    def _room_due(self, wall: float, dirty: bool) -> bool:
        if self._room_source() is None:
            return False
        if self._last_room is None:
            return True
        gap = wall - self._last_room
        return gap >= self.c.room_every_s or (dirty and gap >= self.c.room_min_gap_s)

    def _save_room(self, frame) -> Optional[int]:
        """The whole camera view at the table frame's time (full_at; the newest one if the ring has moved on)."""
        src = getattr(self.frames, "full_at", None)
        full = src(frame.t) if src is not None else None
        if full is None or getattr(full, "img", None) is None:
            latest = getattr(self.frames, "latest_full", None)
            full = latest() if latest is not None else None
        if full is None or getattr(full, "img", None) is None:
            return None
        img, path, nbytes = self._write(full.img, frame.wall, self.c.room_frame_px, "_room")
        dg = digest(self.world, self.cfg) if self.world is not None else {}
        id_ = self.store.add(frame.wall, path, nbytes, 0.0, False, "room", dg, view="room")
        self._last_room = frame.wall
        self._queue_embed(id_, img, "room")
        return id_

    # -- embed worker

    def _embed(self, id_: int, img: np.ndarray, view: str = "table") -> None:
        emb = self.embedder.image(tiles(img, self.c.room_tiles if view == "room" else self.c.tiles))
        self.store.set_embedding(id_, emb, self.embedder.name)

    def _embed_loop(self) -> None:
        if self.embedder is not None and hasattr(self.embedder, "warm"):
            try:
                self.embedder.warm()                      # engine builds happen here, off every hot path
            except Exception:
                log.exception("visual memory: embedder failed to load; no text search")
                self.embedder = None
        while not self._stop.is_set():
            try:
                item = self._eq.get(timeout=5.0)
            except queue.Empty:
                if self.embedder is not None:
                    self._backfill(2)                     # idle: catch up on what was skipped
                continue
            try:
                self._embed(*item)
            except Exception:
                log.exception("archive embedding failed")

    # -- search

    def search(self, query: str, t0: float, t1: float, k: int = 6, min_gap_s: float = 60.0,
               view: Optional[str] = None) -> list[tuple]:
        """(FrameRow, similarity) best first: 'a photo of {query}' against each frame's best tile, in
        [t0, t1] (one view's frames when given), at most one frame per min_gap_s unless nothing else is
        left. [] without a text embedder."""
        if self.embedder is None or not getattr(self.embedder, "can_text", False):
            return []
        q = self.embedder.text([f"a photo of {query}"])[0]
        rows = [r for r in self.store.window(t0, t1, view) if r.emb is not None and r.path]
        scored = sorted(((r, float((r.emb @ q).max())) for r in rows), key=lambda x: -x[1])
        out: list[tuple] = []
        for r, s in scored:
            if all(abs(r.t - o.t) >= min_gap_s for o, _ in out):
                out.append((r, s))
            if len(out) >= k:
                break
        return out

    def sample(self, t0: float, t1: float, k: int = 6, view: Optional[str] = None) -> list[FrameRow]:
        """k frames spread evenly over [t0, t1], preferring change frames (for time-only questions); one
        view's frames when given."""
        rows = [r for r in self.store.window(t0, t1, view) if r.path]
        if len(rows) <= k:
            return rows
        changes = [r for r in rows if r.reason in ("change", "first")]
        pool = changes if len(changes) >= k else rows
        idx = np.linspace(0, len(pool) - 1, k).round().astype(int)
        return [pool[i] for i in dict.fromkeys(idx.tolist())]

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
