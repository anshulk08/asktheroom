"""Public 'hand' training images from EgoHands (Bambach et al., ICCV 2015), in our YOLO layout.

    python scripts/finetune/public_hands.py                     # ~700 frames (~200 MB fetched) + qa.jpg
    python scripts/finetune/public_hands.py --qa-only --audit-model models/yoloe-26s-askroom.pt

EgoHands: 48 Google Glass videos of two people playing cards / chess / Jenga / a puzzle across a table
(conference room, patio, living room), 100 frames per video with a pixel polygon for every hand. The
polygons stop at the wrist, which is our 'hand' convention (bglabel.label_hands keeps only an arm's
last ~260 px), so a polygon's bounding box is our box. No other class is labelled here, so videos
with one of our objects on the table are skipped (EXCLUDE); --audit-model double-checks the rest.

The official download is gone (vision.soic.indiana.edu now redirects), so this reads the Internet
Archive copy of egohands_data.zip (1.3 GB) with HTTP range requests and fetches only the chosen frames
and the 48 polygons.mat files. Output under --out (default data/hands_public):
    images/pubhand_<video*10000 + frame>.jpg    one trial group, 'pubhand', for train.py
    labels/pubhand_*.txt                        '<hand index> cx cy w h' per hand (hand = 8 today)
    manifest.csv                                stem -> source video / frame, hands, size
    qa.jpg                                      30 random frames with their boxes
Train on them only: copy images/ and labels/ into the training data dir and pass --val-trials (so the
'pubhand' group can't be drawn as validation). No scipy: polygons.mat is read by a small MAT v5 reader.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # run as a script or imported by tests
from bglabel import Box, contact_sheet, read_yolo, yolo_line  # noqa: E402
from common import ROOT, class_names, dhash, hamming  # noqa: E402

URL = ("http://web.archive.org/web/20200713164330id_/"
       "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip")
PREFIX = "pubhand"
HAND_FIELDS = ("myleft", "myright", "yourleft", "yourright")
DEFAULT_OUT = ROOT / "data" / "hands_public"
# Videos with one of OUR objects lying on the table (found by eye on 4 frames per video, 2026-09-26).
# Frames are labelled 'hand' only, so these would teach the detector that the object is background.
EXCLUDE = {
    "CARDS_COURTYARD_B_T": "eyeglasses on the table", "CARDS_COURTYARD_H_S": "eyeglasses on the table",
    "CARDS_OFFICE_B_S": "eyeglasses and a small box on the table",
    "CHESS_COURTYARD_B_T": "eyeglasses on the table",
    "CHESS_OFFICE_B_S": "a phone on the table", "CHESS_OFFICE_H_T": "eyeglasses on the table",
    "CHESS_OFFICE_S_B": "a phone on the table", "JENGA_COURTYARD_S_T": "eyeglasses on the table",
}


# ---------------------------------------------------------------- polygon / mask -> box -> YOLO line

def polygon_box(poly, w: int, h: int, min_side: int = 4) -> Optional[Box]:
    """Tight (x1, y1, x2, y2) box, exclusive ends, around an (N, 2) array of (x, y) pixel points (a
    vertex at x covers pixel column floor(x), as for a filled mask), clipped to the w x h image. None
    for an empty/degenerate polygon or one under min_side px."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2) if np.size(poly) else np.zeros((0, 2))
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        return None
    x1, y1 = np.floor(p.min(axis=0)).astype(int)
    x2, y2 = np.floor(p.max(axis=0)).astype(int) + 1
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def mask_box(mask: np.ndarray, min_side: int = 4) -> Optional[Box]:
    """Tight box (exclusive ends) of a boolean mask's pixels, None if it is empty or tiny."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return (x1, y1, x2, y2) if x2 - x1 >= min_side and y2 - y1 >= min_side else None


def hand_lines(polys: Sequence, w: int, h: int, hand_id: int, scale: float = 1.0) -> list[str]:
    """YOLO label lines for one frame's hand polygons (source px, scaled by `scale` to the saved
    image of size w x h)."""
    out = []
    for poly in polys:
        if poly is None or not np.size(poly):
            continue
        b = polygon_box(np.asarray(poly, dtype=np.float64).reshape(-1, 2) * scale, w, h)
        if b is not None:
            out.append(yolo_line(hand_id, b, w, h))
    return out


def fit_long_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


# ---------------------------------------------------------------- minimal MAT v5 reader

_NUM = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}
_MI_MATRIX, _MI_COMPRESSED = 14, 15
_MX_CELL, _MX_STRUCT, _MX_CHAR = 1, 2, 4


def _elements(buf: bytes, pos: int = 0, end: Optional[int] = None):
    """(type, payload) data elements of a little-endian MAT v5 stream, small-element form included."""
    end = len(buf) if end is None else end
    while pos + 8 <= end:
        t, n = struct.unpack_from("<II", buf, pos)
        if t >> 16:                                      # small data element: 2-byte size, 4-byte data
            yield t & 0xFFFF, buf[pos + 4:pos + 4 + (t >> 16)]
            pos += 8
            continue
        yield t, buf[pos + 8:pos + 8 + n]
        pos += 8 + n
        if t != _MI_COMPRESSED:
            pos += (-n) % 8


def _matrix(payload: bytes):
    """One miMATRIX -> (name, value). Numeric/char -> ndarray/str; cell -> list; struct -> list of
    dicts (column-major element order). Enough for EgoHands' polygons.mat, not a general reader."""
    if not payload:
        return "", np.zeros((0, 0))
    it = _elements(payload)
    _, flags = next(it)
    cls, complex_ = flags[0], bool(flags[1] & 0x08)
    _, dims = next(it)
    dims = tuple(int(d) for d in np.frombuffer(dims, "<i4"))
    _, name = next(it)
    name = name.decode("latin-1")
    count = int(np.prod(dims))
    if cls == _MX_CELL:
        return name, [_matrix(p)[1] for _, p in (next(it) for _ in range(count))]
    if cls == _MX_STRUCT:
        _, flen = next(it)
        flen = int(np.frombuffer(flen, "<i4")[0])
        _, raw = next(it)
        fields = [raw[i:i + flen].split(b"\0")[0].decode("latin-1") for i in range(0, len(raw), flen)]
        return name, [{f: _matrix(next(it)[1])[1] for f in fields} for _ in range(count)]
    t, data = next(it)
    if cls == _MX_CHAR:
        enc = "utf-8" if t in (2, 16) else "utf-16-le"
        return name, data.decode(enc, "replace")
    if t not in _NUM:
        raise ValueError(f"unsupported MAT data type {t} (class {cls})")
    arr = np.frombuffer(data, _NUM[t]).astype(np.float64)
    if complex_:
        ti, im = next(it)
        arr = arr + 1j * np.frombuffer(im, _NUM.get(ti, "<f8")).reshape(dims, order="F")
    return name, arr.reshape(dims, order="F")


def read_mat(data: bytes) -> dict:
    """Variables of a MAT v5 file (not v7.3/HDF5)."""
    if len(data) < 128 or data[126:128] != b"IM":
        raise ValueError("not a little-endian MAT v5 file")
    out = {}
    for t, payload in _elements(data, 128):
        if t == _MI_COMPRESSED:
            payload = zlib.decompress(payload)
            t, payload = next(_elements(payload))
        if t == _MI_MATRIX:
            name, value = _matrix(payload)
            out[name] = value
    return out


def frame_polygons(mat: dict) -> list[list[np.ndarray]]:
    """polygons.mat -> per labelled frame (in sorted frame-file order), the (N, 2) x,y hand polygons."""
    return [[np.asarray(fr.get(f, np.zeros((0, 2)))) for f in HAND_FIELDS] for fr in mat["polygons"]]


# ---------------------------------------------------------------- remote zip (HTTP range requests)

class RemoteZip:
    """Reads single members of a zip over HTTP without downloading the whole archive."""

    def __init__(self, url: str, retries: int = 5):
        import requests
        self.url, self.retries, self._requests, self._local = url, retries, requests, threading.local()
        self.bytes = 0
        self._lock = threading.Lock()
        r = self._session().head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        tail = self._get(max(0, self.size - 65536), self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError("no zip end-of-central-directory record")
        eocd = tail[i:i + 22]
        cd_size, cd_off = struct.unpack_from("<II", eocd, 12)
        cd = self._get(cd_off, cd_off + cd_size - 1)
        fake = cd + eocd[:16] + struct.pack("<I", 0) + eocd[20:]      # central directory at offset 0
        self.infos = {zi.filename: zi for zi in zipfile.ZipFile(io.BytesIO(fake)).infolist()}

    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = self._requests.Session()
        return s

    def _get(self, a: int, b: int) -> bytes:
        for k in range(self.retries):
            try:
                r = self._session().get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code == 206:
                    with self._lock:
                        self.bytes += len(r.content)
                    return r.content
                err = f"HTTP {r.status_code}"
            except Exception as e:                        # network hiccup: back off and retry
                err = repr(e)
            time.sleep(2 ** k)
        raise IOError(f"range {a}-{b} of {self.url}: {err}")

    def read(self, name: str) -> bytes:
        zi = self.infos[name]
        head = 30 + len(zi.filename.encode()) + len(zi.extra) + 256      # local extra may differ
        b = self._get(zi.header_offset, zi.header_offset + head + zi.compress_size - 1)
        n, e = struct.unpack_from("<HH", b, 26)
        d = b[30 + n + e:30 + n + e + zi.compress_size]
        if zi.compress_type == zipfile.ZIP_DEFLATED:
            d = zlib.decompress(d, -15)
        elif zi.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{name}: compression {zi.compress_type}")
        if zlib.crc32(d) != zi.CRC:
            raise IOError(f"{name}: CRC mismatch")
        return d


# ---------------------------------------------------------------- selection and build

def videos_of(names: Sequence[str]) -> dict[str, list[str]]:
    """'_LABELLED_SAMPLES/<VIDEO>/frame_0123.jpg' names -> {video: sorted frame names}."""
    out: dict[str, list[str]] = {}
    for n in names:
        parts = n.split("/")
        if len(parts) == 3 and parts[0] == "_LABELLED_SAMPLES" and parts[2].endswith(".jpg"):
            out.setdefault(parts[1], []).append(n)
    return {v: sorted(fs) for v, fs in sorted(out.items())}


def spread(n: int, k: int, offset: float = 0.5) -> list[int]:
    """k indices spread evenly over range(n)."""
    if k >= n:
        return list(range(n))
    return sorted({int((i + offset) * n / k) for i in range(k)})


def frame_order(n: int, seed: int, base: int = 15) -> list[int]:
    """Which of a video's n labelled frames to take first: `base` evenly spaced ones, then the rest in
    seeded random order, so a larger --per-video only adds frames (reruns reuse what is on disk)."""
    first = spread(n, base)
    rest = [i for i in range(n) if i not in set(first)]
    random.Random(seed).shuffle(rest)
    return first + rest


def stem_for(video_idx: int, frame_name: str) -> str:
    frame = int(Path(frame_name).stem.split("_")[-1])
    return f"{PREFIX}_{video_idx * 10000 + frame:06d}"


MANIFEST = ["stem", "video", "frame", "hands", "w", "h"]


def write_manifest(out: Path, rows: Optional[list[dict]] = None) -> Path:
    """manifest.csv of the frames on disk (rows=None: keep the existing rows whose image is still there)."""
    path = Path(out) / "manifest.csv"
    if rows is None:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if (Path(out) / "images" / f"{r['stem']}.jpg").exists()]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, MANIFEST, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["stem"]))
    return path


def build(out: Path, url: str = URL, per_video: int = 18, max_side: int = 1280, workers: int = 6,
          min_hamming: int = 6, hand_id: Optional[int] = None) -> list[dict]:
    """Fetch per_video frames of each video not in EXCLUDE (minus rejected/ ones and near-duplicates),
    write images/, labels/ and manifest.csv, and remove pubhand files of earlier runs not chosen now."""
    from core.config import load_config
    names = class_names(load_config())
    hand_id = names.index("hand") if hand_id is None else hand_id
    out = Path(out)
    for sub in ("images", "labels", "cache"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rz = RemoteZip(url)
    vids = videos_of(list(rz.infos))
    print(f"{len(vids)} videos, {sum(map(len, vids.values()))} labelled frames in {url} ({rz.size / 1e9:.2f} GB)")

    def polys(v: str) -> list:
        cp = out / "cache" / f"{v}_polygons.mat"
        if not cp.exists():
            tmp = cp.with_suffix(".part")
            tmp.write_bytes(rz.read(f"_LABELLED_SAMPLES/{v}/polygons.mat"))
            tmp.replace(cp)
        return frame_polygons(read_mat(cp.read_bytes()))

    def one(job) -> Optional[dict]:
        vi, v, fname, fpolys = job
        stem = stem_for(vi, fname)
        ip, lp = out / "images" / f"{stem}.jpg", out / "labels" / f"{stem}.txt"
        if ip.exists() and lp.exists():                  # fetched by an earlier run
            small = cv2.imread(str(ip))
            n = sum(1 for x in lp.read_text().splitlines() if x.strip())
        else:
            img = cv2.imdecode(np.frombuffer(rz.read(fname), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            small, s = fit_long_side(img, max_side)
            lines = hand_lines(fpolys, small.shape[1], small.shape[0], hand_id, s)
            cv2.imwrite(str(ip), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lp.write_text("".join(x + "\n" for x in lines))
            n = len(lines)
        h, w = small.shape[:2]
        return {"stem": stem, "video": v, "frame": Path(fname).name, "hands": n, "w": w, "h": h,
                "hash": dhash(small)}

    rejected = {p.stem for p in (out / "rejected" / "images").glob("*.jpg")}
    with ThreadPoolExecutor(workers) as ex:
        all_polys = dict(zip(vids, ex.map(polys, vids)))
        jobs = []
        for vi, (v, frames) in enumerate(vids.items()):
            if v in EXCLUDE:
                continue
            ps = all_polys[v]
            if len(ps) != len(frames):
                raise ValueError(f"{v}: {len(ps)} polygon sets for {len(frames)} frames")
            order = [i for i in frame_order(len(frames), vi) if stem_for(vi, frames[i]) not in rejected]
            jobs += [(vi, v, frames[i], ps[i]) for i in order[:per_video]]
        rows = [r for r in ex.map(one, jobs) if r is not None]

    kept = []                                # near-duplicates within a video (a still stretch) go
    for r in rows:
        if not any(k["video"] == r["video"] and hamming(k["hash"], r["hash"]) < min_hamming for k in kept):
            kept.append(r)
    keep = {r["stem"] for r in kept}
    stale = [p for p in (out / "images").glob(f"{PREFIX}_*.jpg") if p.stem not in keep]
    for p in stale:
        p.unlink()
        (out / "labels" / f"{p.stem}.txt").unlink(missing_ok=True)
    write_manifest(out, kept)
    print(f"kept {len(kept)} frames ({sum(r['hands'] for r in kept)} hands); removed {len(stale)} "
          f"near-duplicate/stale; {len(rejected)} rejected; fetched {rz.bytes / 1e6:.0f} MB this run")
    return kept


def write_qa(out: Path, n: int = 30, seed: int = 0, name: str = "qa.jpg") -> Path:
    out = Path(out)
    stems = sorted(p.stem for p in (out / "images").glob(f"{PREFIX}_*.jpg"))
    entries = []
    for stem in random.Random(seed).sample(stems, min(n, len(stems))):
        img = cv2.imread(str(out / "images" / f"{stem}.jpg"))
        h, w = img.shape[:2]
        entries.append((img, [("hand", b) for _, b in read_yolo(out / "labels" / f"{stem}.txt", w, h)], stem))
    path = out / name
    cv2.imwrite(str(path), contact_sheet(entries, cols=6, thumb_w=320), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


WORN = ("glasses", "eyeglasses")      # EgoHands partners wear Google Glass: a face, not a tabletop object


def worn_glasses(name: str, box: Sequence[float], h: int, top: float = 0.3) -> bool:
    """A glasses hit centred in the top part of the frame is the partner's Google Glass, on a face."""
    return name in WORN and (box[1] + box[3]) / 2 < top * h


def audit(out: Path, model: str, conf: float = 0.3, drop: bool = False) -> list[dict]:
    """Frames where an open-vocabulary detector (e.g. models/yoloe-26s-askroom.pt, prompted with our
    object names) sees one of OUR objects. They are labelled 'hand' only, so such an object would be
    trained as background: list them in flagged.csv and a qa_flagged.jpg sheet, and with drop=True
    move them to rejected/ (build() never fetches them again). Worn Google Glass is not flagged."""
    from ultralytics import YOLO
    out = Path(out)
    m = YOLO(model)
    rows, entries = [], []
    for p in sorted((out / "images").glob(f"{PREFIX}_*.jpg")):
        img = cv2.imread(str(p))
        r = m.predict(img, conf=conf, verbose=False)[0]
        hits = [(r.names[int(c)], float(s), tuple(int(v) for v in b))
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        hits = [x for x in hits if x[0] != "hand" and not worn_glasses(x[0], x[2], img.shape[0])]
        if hits:
            rows.append({"stem": p.stem, "hits": "; ".join(f"{n} {s:.2f} {list(b)}" for n, s, b in hits)})
            entries.append((img, [(f"{n} {s:.2f}", b) for n, s, b in hits], p.stem))
    with open(out / "flagged.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, ["stem", "hits"])
        wr.writeheader()
        wr.writerows(rows)
    cv2.imwrite(str(out / "qa_flagged.jpg"), contact_sheet(entries[:60], cols=6, thumb_w=320),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    if drop:
        for sub in ("images", "labels"):
            (out / "rejected" / sub).mkdir(parents=True, exist_ok=True)
        for r in rows:
            for sub, ext in (("images", "jpg"), ("labels", "txt")):
                src = out / sub / f"{r['stem']}.{ext}"
                if src.exists():
                    src.replace(out / "rejected" / sub / src.name)
        write_manifest(out)
    print(f"{len(rows)} frames show one of our objects (conf >= {conf}) -> {out / 'flagged.csv'}"
          + (", moved to rejected/" if drop else ""))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--url", default=URL)
    ap.add_argument("--per-video", type=int, default=18, help="frames per video (48 videos, 100 labelled each)")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--qa-only", action="store_true", help="just redraw qa.jpg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-model", help="flag frames where this detector sees one of our objects")
    ap.add_argument("--audit-conf", type=float, default=0.3)
    ap.add_argument("--drop-flagged", action="store_true", help="move flagged frames to rejected/")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not a.qa_only:
        build(out, a.url, a.per_video, a.max_side, a.workers)
    if a.audit_model:
        audit(out, a.audit_model, a.audit_conf, a.drop_flagged)
    print(f"contact sheet: {write_qa(out, seed=a.seed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
