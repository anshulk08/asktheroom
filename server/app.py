"""Dashboard + phone server (spec V7, V8, V11).

create_app(cfg, world, events, frames=None, ask_fn=None, table=None, care=None) -> FastAPI

  GET  /                 dashboard (server/web/index.html; everything served locally, works offline)
  GET  /video            MJPEG of the latest frame with the world drawn on it (server/overlay.py)
  GET  /frame.jpg, /full.jpg   one frame drawn on like /video (the table view; room memory: the whole frame);
                         ?raw=1 the camera frame as captured, nothing drawn (full.jpg at capture size)
  WS   /ws             WorldState JSON at server.push_hz plus new events since the last push
  GET  /events?since=t   events with wall >= t (oldest first), each with a snapshot_url
  GET  /snapshots/{name} one event snapshot jpg (snapshot dir only)
  POST /ask              {"text", "source"?: "dashboard" | "phone"} -> {"text", "point_at", "action", "latency_ms"}
  POST /voice            {"engine"?: "grok" | "rig" | "builtin", "grok_voice"?, "speed"?} the phone app's
                         voice for the rig's speaker (BLE bridge) -> the stored {"engine", "grok_voice", "speed"}
  POST /orientation      {"front": "bottom" | "right" | "top" | "left" | null} the user's seat, the camera-frame side of
                         the table they sit at (the phone's "I sit here", BLE bridge; core/viewframe.py): used by
                         the next answer, saved in data/viewer.json (null: back to config viewer.front) -> the new
                         view {"front", "table", "m", "outline"}
  POST /sms              Twilio webhook (signature checked, whitelist only)

Demo view (WS9, the filmed page; everything is drawn in the browser, the rig only serves JSON and JPEGs):
  GET  /demo             the page (server/web/demo.html, demo.js, demo.css)
  GET  /demo/meta        what the page needs to draw on the camera frame: the image it polls and its size,
                         "cm_to_img" (3x3 homography table cm -> that image's px), the table view rect, the drawn
                         zones (full-frame px) and the viewer frame
  GET  /demo/boxes       {"t", "boxes": {entity: [x1, y1, x2, y2]}} each entity's last observed box, in the
                         image's px (a table box_cm through cm_to_img)
  GET  /demo/evidence?obj=NAME   the newest event of obj with a snapshot: {obj, type, t, snapshot_url} or {}
  GET  /room_layout      the room map (config room_layout:, camera table axes) turned to the user's seat:
                         {"v": 1, "size", "front", "table": {"rect", "origin"}, "zones": [{id, say, rect, kind}],
                         "you"}; a table point's map position is View.m . pos_cm + table.origin (the phone too)
  GET  /grok/trace?limit=N   the last Grok calls, newest first (core/grok_trace.py): {id, t, purpose, model,
                         ms, ok, request, reply, tools, images: [key]}; GET /grok/img/{key} one thumbnail sent
  /full.jpg?raw=1&w=N    the full frame shrunk to N px wide before encoding (less CPU and Wi-Fi than 2560 px)
/state and /ws also carry "listening": true while the rig listens for a question after the wake word (main.py).

/state and /ws also carry "view": the user's frame (core/viewframe.py View.to_json(): {"front", "table": [w, h]
cm from the seat, "m": 2x3 affine camera table cm -> viewer cm, "outline"}) and, when config viewer.sides is
set, "sides": {camera side: label}. Positions in "state" stay camera-frame; the BLE bridge turns them.

With care=voice.care.Care (reminders, reports; see voice/care.py), additively:
  /state and /ws         also carry "notices": [{id, t, kind, text, point_at, acknowledged, ...}] and
                         "answers": the last 10 answers from every source [{seq, t, src, q, text, point_at, action}]
  POST /notices/{id}/ack acknowledge a notice (the phone app's button)
  GET  /report?date=YYYY-MM-DD&format=markdown|text|json   the caregiver summary for a day

Room handoff scoreboard (server/scoreboard.py: real scripts/room_trials.py results only):
  GET  /scoreboard?date=today|yesterday|all|YYYY-MM-DD      handoffs / returns passed of tried, per object and zone
  POST /scoreboard/trials?object=NAME   body: a room_trials.py results list; stored in scoreboard.trials_dir

Run the dev version with fake data:  python -m server.app --fake
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional
from xml.sax.saxutils import escape

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.types import Answer, Event, Status
from core.viewframe import View, apply_saved, set_front
from server import overlay, scoreboard

log = logging.getLogger("askroom.server")

WEB_DIR = Path(__file__).resolve().parent / "web"
BOUNDARY = "askroomframe"
SNAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(jpg|jpeg|png)$")
ASK_TIMEOUT_S = 10.0          # dashboard: ask_fn has its own 4 s LLM timeout; this is a backstop (main.ANSWER_LATE_S)
ASK_SOURCES = {"dashboard", "phone"}       # /ask sources a client may name; both are spoken and aimed
SMS_TIMEOUT_S = 10.0          # Twilio gives a webhook 15 s
INITIAL_EVENTS = 200          # events sent on a fresh WS connection
EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
SCORE_MAX_BYTES = 256 * 1024  # a room_trials.py results file is a few KB
AskFn = Callable[[str, str], Answer]


# ---------------------------------------------------------------- helpers

def _json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Status,)):
        return o.value
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


def _event_key(ev: Event) -> tuple:
    return (ev.wall, ev.obj, ev.type)


class EventCursor:
    """Tracks what one client has already seen, so each push carries only new events."""

    def __init__(self, since: float = 0.0):
        self.since = since
        self.seen_at_edge: set[tuple] = set()

    def poll(self, events, limit: Optional[int] = None) -> list[Event]:
        evs = events.since(self.since)
        out = [e for e in evs if _event_key(e) not in self.seen_at_edge]
        if limit is not None and len(out) > limit:
            out = out[-limit:]
        if evs:
            top = max(e.wall for e in evs)
            if top > self.since:
                self.since = top
                self.seen_at_edge = set()
            self.seen_at_edge |= {_event_key(e) for e in evs if e.wall == top}
        return out


def canned_ask(cfg: dict, world) -> AskFn:
    """A stand-in for voice/answers: finds an object name in the question and reads the world."""
    syn = {k.lower(): v for k, v in (cfg.get("synonyms") or {}).items()}
    names = list((cfg.get("objects") or {}).keys())

    def spoken(n: Optional[str]) -> str:
        if not n:
            return "something"
        if n.startswith("hand"):
            return "someone's hand"
        return (cfg.get("display_names") or {}).get(n, n.replace("_", " "))

    def ask(text: str, source: str) -> Answer:
        q = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
        obj = None
        for phrase in sorted(syn, key=len, reverse=True):
            if f" {phrase} " in q:
                obj = syn[phrase]
                break
        if obj is None:
            for n in names:
                if f" {n.replace('_', ' ')} " in q or f" {n} " in q:
                    obj = n
                    break
        if obj is None:
            return Answer("Ask me where something is, like: where are my keys?")
        try:
            e = world.get(obj)
        except KeyError:
            return Answer(f"I'm not tracking the {spoken(obj)}.")
        st = e.status.value if hasattr(e.status, "value") else str(e.status)
        name = spoken(obj)
        is_ = "are" if name.endswith("s") else "is"
        if st == "VISIBLE":
            return Answer(f"The {name} {is_} on the table. I'm pointing at it.", point_at=obj, action="point")
        if st == "INSIDE":
            return Answer(f"The {name} {is_} inside the {spoken(e.parent)}.", point_at=obj, action="point")
        if st == "UNDER":
            return Answer(f"The {name} {is_} under the {spoken(e.parent)}.", point_at=obj, action="point")
        if st == "HELD":
            return Answer(f"Someone is holding the {name} right now.")
        if st == "GONE":                      # spoken from the user's seat; the sweep is the laser's (camera frame)
            side = View.from_cfg(cfg).off_table(e.edge)
            return Answer(f"The {name} {'were' if is_ == 'are' else 'was'} carried off {side}.",
                          action=f"sweep:{e.edge}" if e.edge else None)
        return Answer(f"I lost track of the {name}. I last saw {'them' if is_ == 'are' else 'it'} here.", point_at=obj, action="circle")

    return ask


# ---------------------------------------------------------------- app

def create_app(cfg: dict, world, events, frames=None, ask_fn: Optional[AskFn] = None,
               table=None, care=None, voice_fn: Optional[Callable[..., Any]] = None,
               listening_fn: Optional[Callable[[], bool]] = None) -> FastAPI:
    apply_saved(cfg)                     # the seat the phone chose last time (data/viewer.json), into cfg
    scfg = cfg.get("server") or {}
    push_period = 1.0 / float(scfg.get("push_hz", 5) or 5)
    mjpeg_period = 1.0 / float(scfg.get("mjpeg_fps", 10) or 10)
    whitelist = {str(n).strip() for n in ((cfg.get("sms") or {}).get("whitelist") or [])}
    snap_dir = Path(getattr(events, "snap_dir", None) or (cfg.get("paths") or {}).get("snapshots", "data/snapshots"))
    snap_root = snap_dir.resolve()
    ask = ask_fn or canned_ask(cfg, world)

    app = FastAPI(title="Ask the Room", docs_url=None, redoc_url=None)
    app.state.last_answer = None
    app.state.answers = collections.deque(maxlen=10)   # recent answers from every source, for the phone
    app.state.answer_seq = 0
    answers_lock = threading.Lock()

    def record_answer(question: str, ans: Answer, source: str) -> None:
        """Log an answer for /state 'answers' (main.py calls this for voice questions too)."""
        with answers_lock:
            app.state.answer_seq += 1
            app.state.answers.append({"seq": app.state.answer_seq, "t": time.time(), "src": source,
                                      "q": question, "text": ans.text, "point_at": ans.point_at,
                                      "action": ans.action})

    app.state.record_answer = record_answer
    placeholder_img = overlay.placeholder()

    # -- snapshot urls
    def snapshot_url(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        try:
            if p.resolve().parent != snap_root:
                return None
        except OSError:
            return None
        if not SNAP_NAME_RE.match(p.name):
            return None
        return f"/snapshots/{p.name}"

    def event_json(ev: Event) -> dict:
        d = asdict(ev)
        d["snapshot_url"] = snapshot_url(ev.snapshot)
        d.pop("snapshot", None)          # don't leak filesystem paths to the browser
        d["id"] = f"{ev.wall:.4f}:{ev.obj}:{ev.type}"
        return d

    def meta() -> dict:
        out = {"last_answer": app.state.last_answer, "server_t": time.time(), "answers": list(app.state.answers)}
        try:                                                   # the user's frame, for the phone (BLE bridge)
            out["view"] = View.from_cfg(cfg).to_json()
        except Exception:
            log.exception("viewer frame failed")
        sides = (cfg.get("viewer") or {}).get("sides")
        if isinstance(sides, dict) and sides:
            out["sides"] = {str(k): str(v) for k, v in sides.items()}
        if care is not None:                                   # care layer: additive key
            try:
                out["notices"] = care.notices_json()
            except Exception:
                log.exception("care notices failed")
        if listening_fn is not None:                           # the listening light (voice/cues.ListenIndicator)
            try:
                out["listening"] = bool(listening_fn())
            except Exception:
                out["listening"] = False
        return out

    # -- pages
    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(WEB_DIR / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # -- video
    def render_jpeg() -> bytes:
        f = None
        if frames is not None:
            try:
                f = frames.latest()
            except Exception:
                log.exception("frames.latest() failed")
        img = f.img if f is not None and getattr(f, "img", None) is not None else placeholder_img
        try:
            state = world.state_json()
        except Exception:
            state = None
        dets = None
        get_dets = getattr(frames, "latest_dets", None)
        if callable(get_dets):
            try:
                dets = get_dets()
            except Exception:
                dets = None
        try:
            out = overlay.draw(img, state, table=table, dets=dets)
        except Exception:
            log.exception("overlay.draw failed")
            out = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else b""

    async def mjpeg(request: Request, max_frames: Optional[int] = None):
        # Starlette cancels this generator when the client goes away; the disconnect check is a
        # second guard for servers that don't.
        n = 0
        while max_frames is None or n < max_frames:
            t0 = time.monotonic()
            if await request.is_disconnected():
                break
            jpg = await asyncio.to_thread(render_jpeg)
            yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            n += 1
            await asyncio.sleep(max(0.0, mjpeg_period - (time.monotonic() - t0)))

    @app.get("/video")
    async def video(request: Request, frames_n: Optional[int] = None):
        return StreamingResponse(
            mjpeg(request, frames_n),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache",
                     "X-Accel-Buffering": "no"},
        )

    def render_raw(full: bool, width: Optional[int] = None) -> Optional[bytes]:
        """The camera frame as captured, nothing drawn (?raw=1): the table view, or the full frame at capture
        size. For reading tabletop corners (python -m core.table --outline-full) and demo_check's view check.
        width: shrink to this many px wide first (the demo page polls a 1600 px frame, not 2560)."""
        get = getattr(frames, "latest_full" if full else "latest", None)
        f = get() if callable(get) else None
        if f is None or getattr(f, "img", None) is None:
            return None
        img = f.img
        if width and 64 <= width < img.shape[1]:
            img = cv2.resize(img, (int(width), round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92 if img is f.img else 85])
        return buf.tobytes() if ok else None

    @app.get("/frame.jpg")
    async def frame_jpg(raw: bool = False):
        jpg = await asyncio.to_thread(render_raw, False) if raw else await asyncio.to_thread(render_jpeg)
        if jpg is None:
            raise HTTPException(404, "no camera frame yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # -- room memory (spec 0009): the whole camera frame with the zones, the table view and room places
    room_zones: dict = {}

    def render_full() -> Optional[bytes]:
        latest_full = getattr(frames, "latest_full", None)
        if not callable(latest_full):
            return None
        f = latest_full()
        if f is None or getattr(f, "img", None) is None:
            return None
        img = f.img.copy()
        if "zones" not in room_zones:
            try:
                from core.room_zones import Zones
                path = (cfg.get("room_memory") or {}).get("zones_path", "room_zones.json")
                room_zones["zones"] = list(Zones.load(path).zones.values())
            except Exception:
                room_zones["zones"] = []
        for z in room_zones["zones"]:
            pts = np.array(z.poly, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 220, 0), 3)
            cv2.putText(img, z.say, (int(z.poly[0][0]) + 6, int(z.poly[0][1]) + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 220, 0), 2)
        rect = getattr(frames, "rect", None)
        if rect is not None:
            cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 160, 0), 3)
        try:
            places = (world.state_json() or {}).get("room") or {}
        except Exception:
            places = {}
        for name, st in places.items():
            if isinstance(st, dict) and st.get("box_px"):
                x1, y1, x2, y2 = (int(v) for v in st["box_px"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        h, w = img.shape[:2]
        if w > 1280:
            img = cv2.resize(img, (1280, round(h * 1280 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    @app.get("/full.jpg")
    async def full_jpg(raw: bool = False, w: Optional[int] = None):
        jpg = await asyncio.to_thread(render_raw, True, w) if raw else await asyncio.to_thread(render_full)
        if jpg is None:
            raise HTTPException(404, "no full camera frame (room memory is off)")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # -- live state
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            since = float(websocket.query_params.get("since", 0) or 0)
        except ValueError:
            since = 0.0
        cursor = EventCursor(since)
        first = True
        try:
            while True:
                t0 = time.monotonic()
                state = await asyncio.to_thread(world.state_json)
                new = await asyncio.to_thread(cursor.poll, events, INITIAL_EVENTS if first else None)
                msg = {"type": "state", "state": state, "events": [event_json(e) for e in new],
                       "initial": first, **meta()}
                await websocket.send_text(dumps(msg))
                first = False
                await asyncio.sleep(max(0.0, push_period - (time.monotonic() - t0)))
        except Exception:
            # client went away (WebSocketDisconnect / closed transport) or the app is shutting down
            pass

    @app.get("/state")
    async def state_route():
        st = await asyncio.to_thread(world.state_json)
        return Response(dumps({"state": st, **meta()}), media_type="application/json")

    # -- events + snapshots
    @app.get("/events")
    async def events_route(since: float = 0.0, limit: int = 500):
        evs = await asyncio.to_thread(events.since, since)
        if limit and len(evs) > limit:
            evs = evs[-limit:]
        return Response(dumps([event_json(e) for e in evs]), media_type="application/json")

    @app.get("/snapshots/{name:path}")
    def snapshot(name: str):
        if not SNAP_NAME_RE.match(name) or ".." in name:
            raise HTTPException(400, "bad snapshot name")
        p = (snap_root / name).resolve()
        if p.parent != snap_root or not p.is_file():
            raise HTTPException(404, "no such snapshot")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    # -- ask
    async def run_ask(text: str, source: str, timeout: float) -> tuple[Answer, int]:
        t0 = time.perf_counter()
        try:
            ans = await asyncio.wait_for(asyncio.to_thread(ask, text, source), timeout)
        except asyncio.TimeoutError:
            ans = Answer("Sorry, that took too long. Please ask again.")
        except Exception:
            log.exception("ask_fn failed")
            ans = Answer("Sorry, something went wrong answering that.")
        ms = int((time.perf_counter() - t0) * 1000)
        app.state.last_answer = {"question": text, "text": ans.text, "point_at": ans.point_at,
                                 "action": ans.action, "latency_ms": ms, "source": source,
                                 "t": time.time()}
        record_answer(text, ans, source)
        return ans, ms

    @app.post("/ask")
    async def ask_route(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "expected JSON {\"text\": ...}")
        text = str((body or {}).get("text", "")).strip() if isinstance(body, dict) else ""
        if not text:
            raise HTTPException(400, "text is required")
        source = body.get("source") if body.get("source") in ASK_SOURCES else "dashboard"
        ans, ms = await run_ask(text[:500], source, ASK_TIMEOUT_S)
        return JSONResponse({"text": ans.text, "point_at": ans.point_at, "action": ans.action,
                             "latency_ms": ms})

    # -- sms (Twilio)
    def _public_urls(request: Request) -> list[str]:
        urls = [str(request.url)]
        base = os.environ.get("ASKROOM_PUBLIC_URL")
        if base:
            q = f"?{request.url.query}" if request.url.query else ""
            urls.insert(0, base.rstrip("/") + request.url.path + q)
        proto = request.headers.get("x-forwarded-proto")
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        if proto and host:
            q = f"?{request.url.query}" if request.url.query else ""
            urls.append(f"{proto.split(',')[0].strip()}://{host}{request.url.path}{q}")
        return urls

    def twiml(text: str) -> Response:
        body = ('<?xml version="1.0" encoding="UTF-8"?><Response><Message>'
                + escape(text[:1500]) + "</Message></Response>")
        return Response(body, media_type="application/xml")

    @app.post("/voice")
    async def voice_route(request: Request):
        if voice_fn is None:
            raise HTTPException(503, "no speaker on this rig")
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "expected JSON {\"engine\", \"grok_voice\", \"speed\"}")
        if not isinstance(body, dict):
            raise HTTPException(400, "expected a JSON object")
        v = voice_fn(body.get("engine"), body.get("grok_voice"), body.get("speed"))
        return JSONResponse(dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v)

    @app.post("/orientation")
    async def orientation_route(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "expected JSON {\"front\": \"bottom\" | \"right\" | \"top\" | \"left\"}")
        if not isinstance(body, dict) or "front" not in body:
            raise HTTPException(400, "expected a JSON object with \"front\"")
        try:
            front = await asyncio.to_thread(set_front, cfg, body["front"])
        except ValueError as ex:
            raise HTTPException(400, str(ex))
        log.info("viewer: the user sits at the camera's %s side", front)
        return JSONResponse(View.from_cfg(cfg).to_json())

    @app.post("/sms")
    async def sms(request: Request):
        form = await request.form()
        params = {k: str(v) for k, v in form.items()}
        if os.environ.get("ASKROOM_SMS_INSECURE") != "1":
            token = os.environ.get("TWILIO_AUTH_TOKEN")
            sig = request.headers.get("x-twilio-signature", "")
            if not token or not sig:
                raise HTTPException(403, "missing Twilio signature")
            from twilio.request_validator import RequestValidator
            v = RequestValidator(token)
            if not any(v.validate(u, params, sig) for u in _public_urls(request)):
                raise HTTPException(403, "bad Twilio signature")
        sender = params.get("From", "").strip()
        if sender not in whitelist:
            log.info("sms from non-whitelisted number ignored")
            return Response(EMPTY_TWIML, media_type="application/xml")
        text = params.get("Body", "").strip()
        if not text:
            return twiml("Text me a question, like: where are my keys?")
        ans, _ = await run_ask(text[:500], "sms", SMS_TIMEOUT_S)
        return twiml(ans.text)

    # -- care layer (voice/care.py): acknowledge a notice, the caregiver summary
    if care is not None:
        @app.post("/notices/{notice_id}/ack")
        async def ack_notice(notice_id: int):
            if not await asyncio.to_thread(care.ack, notice_id):
                raise HTTPException(404, "no such notice")
            return {"ok": True}

        @app.get("/report")
        async def report(date: Optional[str] = None, format: str = "markdown"):
            fmt = format if format in ("markdown", "text", "json") else "markdown"
            try:
                out = await asyncio.to_thread(care.report, date, fmt)
            except ValueError:
                raise HTTPException(400, "date must be YYYY-MM-DD, today or yesterday")
            if fmt == "json":
                return JSONResponse(out)
            return Response(out, media_type="text/markdown; charset=utf-8" if fmt == "markdown"
                            else "text/plain; charset=utf-8")

    # -- room handoff scoreboard (server/scoreboard.py): real trial results, uploaded from the laptop
    score_dir = scoreboard.trials_dir(cfg)

    @app.get("/scoreboard")
    async def scoreboard_route(date: Optional[str] = None):
        try:
            runs = await asyncio.to_thread(scoreboard.load_runs, score_dir)
            return scoreboard.summarize(runs, date)
        except ValueError:
            raise HTTPException(400, "date must be today, yesterday, all or YYYY-MM-DD")

    @app.post("/scoreboard/trials")
    async def scoreboard_upload(request: Request, object: str = ""):
        body = await request.body()
        if len(body) > SCORE_MAX_BYTES:
            raise HTTPException(413, "results file too large")
        try:
            records = json.loads(body or b"null")
            path = await asyncio.to_thread(scoreboard.save_upload, score_dir, object, records)
        except ValueError as e:
            raise HTTPException(400, str(e))
        log.info("scoreboard: %d room trial runs for %r saved to %s", len(records), object, path)
        runs = await asyncio.to_thread(scoreboard.load_runs, score_dir)
        return {"ok": True, "saved": path.name, "runs": len(records), "today": scoreboard.summarize(runs)}

    # -- demo view (WS9): the page draws everything; these routes only hand it geometry and boxes
    @app.get("/demo", response_class=HTMLResponse)
    def demo_page():
        return FileResponse(WEB_DIR / "demo.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/demo/meta")
    async def demo_meta():
        return Response(dumps(await asyncio.to_thread(demo_geometry, cfg, frames, table)),
                        media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get("/demo/boxes")
    async def demo_boxes():
        def boxes() -> dict:
            m = demo_geometry(cfg, frames, table, zones=False).get("cm_to_img")
            out = {}
            if m is not None:
                M = np.asarray(m, float)
                for e in (world.state_json() or {}).get("entities") or []:
                    try:
                        ent = world.get(e["name"])
                    except Exception:
                        continue
                    b = getattr(ent, "box_cm", None)
                    if b is None or getattr(ent, "zone", "table") != "table":
                        continue
                    out[e["name"]] = box_through(M, b)
            return {"t": time.time(), "boxes": out}
        return Response(dumps(await asyncio.to_thread(boxes)), media_type="application/json",
                        headers={"Cache-Control": "no-store"})

    @app.get("/demo/evidence")
    async def demo_evidence(obj: str = ""):
        def find() -> dict:
            for ev in events.last(obj, 8) if obj else []:
                url = snapshot_url(ev.snapshot)
                if url:
                    return {"obj": ev.obj, "type": str(getattr(ev.type, "value", ev.type)), "t": ev.wall,
                            "snapshot_url": url}
            return {}
        return JSONResponse(await asyncio.to_thread(find), headers={"Cache-Control": "no-store"})

    @app.get("/grok/trace")
    def grok_trace_route(limit: int = 10):
        from core import grok_trace
        return JSONResponse(grok_trace.calls(min(max(1, limit), grok_trace.KEEP)), headers={"Cache-Control": "no-store"})

    @app.get("/grok/img/{key}")
    def grok_img(key: str):
        from core import grok_trace
        jpg = grok_trace.image(key) if re.fullmatch(r"\d+-\d+", key) else None
        if jpg is None:
            raise HTTPException(404, "no such image")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.get("/room_layout")
    async def room_layout_route():
        lay = await asyncio.to_thread(room_layout, cfg)
        if lay is None:
            raise HTTPException(404, "no room map (room_memory off or no room_layout: section)")
        return JSONResponse(lay, headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    return app


# ---------------------------------------------------------------- demo view geometry

def box_through(M: np.ndarray, box) -> list:
    """A cm box's four corners through the homography M, as the px box around them (rounded)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    p = cv2.perspectiveTransform(np.float64([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]]), M).reshape(-1, 2)
    return [round(float(p[:, 0].min()), 1), round(float(p[:, 1].min()), 1),
            round(float(p[:, 0].max()), 1), round(float(p[:, 1].max()), 1)]


def _zones(cfg: dict) -> list:
    try:
        from core.room_zones import Zones
        path = (cfg.get("room_memory") or {}).get("zones_path", "room_zones.json")
        return [{"name": z.name, "say": z.say, "poly": [[float(x), float(y)] for x, y in z.poly]}
                for z in Zones.load(path).zones.values()]
    except Exception:
        return []


def demo_geometry(cfg: dict, frames, table, zones: bool = True) -> dict:
    """/demo/meta: the image the demo polls (the full frame when room memory runs, else the table view) and
    the homography from table cm to its px: the table's cm -> table-view px (Hinv), then the table view's
    place in the full frame (TableView.rect / out_size)."""
    rect = getattr(frames, "rect", None)
    out_size = getattr(frames, "out_size", None) or tuple(cfg.get("frame_size_px") or (1280, 720))
    full = rect is not None and callable(getattr(frames, "latest_full", None))
    cap = (cfg.get("room_memory") or {}).get("capture_size") or [2560, 1440]
    img_size = [int(cap[0]), int(cap[1])] if full else [int(out_size[0]), int(out_size[1])]
    M = None
    Hinv = getattr(table, "Hinv", None) if table is not None else None
    if Hinv is not None:
        M = np.asarray(Hinv, float)
        if full:
            sx = (rect[2] - rect[0]) / float(out_size[0])
            sy = (rect[3] - rect[1]) / float(out_size[1])
            M = np.array([[sx, 0.0, rect[0]], [0.0, sy, rect[1]], [0.0, 0.0, 1.0]]) @ M
    out = {"image": "/full.jpg?raw=1" if full else "/frame.jpg?raw=1", "image_size": img_size, "full": full,
           "cm_to_img": M.tolist() if M is not None else None,
           "table_view_rect": [float(v) for v in rect] if rect is not None else None,
           "table_size_cm": list((cfg.get("table") or {}).get("size_cm") or []),
           "zones": _zones(cfg) if full and zones else []}
    try:
        out["view"] = View.from_cfg(cfg).to_json()
    except Exception:
        out["view"] = None
    return out


def room_layout(cfg: dict) -> Optional[dict]:
    """GET /room_layout: config room_layout: (camera table axes, cm) turned to the user's seat with the viewer
    frame's affine (core/viewframe.py), each rect re-boxed after turning, all shifted to start at 0. None
    (a 404: the phone's bridge then sends no map) when room memory is off or there is no layout."""
    rl = cfg.get("room_layout") or {}
    if not (cfg.get("room_memory") or {}).get("enabled") or not rl.get("zones"):
        return None
    view = View.from_cfg(cfg).to_json()
    m = np.asarray(view["m"], float)
    size = (cfg.get("table") or {}).get("size_cm") or [100, 70]

    def turn(x, y):
        return m @ np.array([x, y, 1.0])

    def rbox(r):
        x, y, w, h = (float(v) for v in r)
        pts = np.array([turn(x, y), turn(x + w, y), turn(x + w, y + h), turn(x, y + h)])
        return pts.min(axis=0), pts.max(axis=0)

    says = {z["name"]: z["say"] for z in _zones(cfg)}
    raw = []
    for zid, z in (rl.get("zones") or {}).items():
        lo, hi = rbox(z["rect"])
        raw.append((zid, z, lo, hi))
    tlo, thi = rbox([0, 0, size[0], size[1]])
    tw, th = view["table"]
    you = np.array([tw / 2.0, th + float(rl.get("seat_cm", 45))])       # viewer frame: y grows toward the seat
    los = [tlo, you] + [lo for _, _, lo, _ in raw]
    his = [thi, you] + [hi for _, _, _, hi in raw]
    pad = float(rl.get("pad_cm", 20))
    o = np.min(los, axis=0) - pad
    W, H = (np.max(his, axis=0) - o + pad).tolist()

    def rect(lo, hi):
        return [round(float(lo[0] - o[0]), 1), round(float(lo[1] - o[1]), 1),
                round(float(hi[0] - lo[0]), 1), round(float(hi[1] - lo[1]), 1)]

    zones = [{"id": zid, "say": says.get(zid) or z.get("say") or zid.replace("_", " "), "rect": rect(lo, hi),
              "kind": z.get("kind", "surface")} for zid, z, lo, hi in raw]
    sides = (cfg.get("viewer") or {}).get("sides")
    return {"v": 1, "size": [round(W, 1), round(H, 1)], "front": view["front"],
            "sides": {str(k): str(v) for k, v in sides.items()} if isinstance(sides, dict) else {},
            "table": {"rect": rect(tlo, thi), "origin": [round(float(tlo[0] - o[0]), 1), round(float(tlo[1] - o[1]), 1)]},
            "zones": zones, "you": [round(float(you[0] - o[0]), 1), round(float(you[1] - o[1]), 1)]}


# ---------------------------------------------------------------- --fake dev run

class FakeTable:
    """Linear table cm -> frame px for the synthetic frame (90x60 cm table inset in 1280x720)."""

    def __init__(self, cfg: dict, w: int = 1280, h: int = 720):
        tw, th = (cfg.get("table") or {}).get("size_cm", [90, 60])
        self.tw, self.th = float(tw), float(th)
        self.sx = (w - 220) / self.tw
        self.sy = (h - 120) / self.th
        self.s = min(self.sx, self.sy)
        self.ox = (w - self.tw * self.s) / 2
        self.oy = (h - self.th * self.s) / 2

    def cm_to_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        return np.stack([self.ox + pts[:, 0] * self.s, self.oy + pts[:, 1] * self.s], axis=1)


class TestPatternFrames:
    """Synthetic camera: a tabletop with a cm grid, objects drawn where the fake world has them,
    and a hand drifting across. Rendered on demand at most ~15 fps."""

    __test__ = False   # not a pytest class

    OBJ_BGR = {"keys": (60, 200, 230), "pill_bottle": (70, 110, 240), "wallet": (60, 80, 120),
               "glasses": (200, 200, 90), "phone": (40, 40, 40), "remote": (90, 90, 90),
               "box": (80, 140, 190), "notebook": (230, 225, 215)}
    SIZE = {"box": (15, 11), "notebook": (13, 10), "remote": (4, 12), "phone": (5, 9),
            "wallet": (7, 5), "glasses": (9, 3), "pill_bottle": (3, 3), "keys": (3, 3)}

    def __init__(self, world, table: FakeTable):
        from core.types import Frame
        self._Frame = Frame
        self.world = world
        self.table = table
        self.idx = 0
        self._last: Any = None
        self._lock = threading.Lock()
        self._bg = self._background()

    def _background(self) -> np.ndarray:
        img = np.empty((720, 1280, 3), np.uint8)
        img[:] = (58, 62, 66)
        t = self.table
        (x1, y1), (x2, y2) = t.cm_to_px(np.array([[0, 0], [t.tw, t.th]]))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (70, 98, 44), -1)
        for cx in range(0, int(t.tw) + 1, 5):
            (px, _), = t.cm_to_px(np.array([[cx, 0]]))
            cv2.line(img, (int(px), int(y1)), (int(px), int(y2)), (84, 116, 58) if cx % 10 else (98, 132, 70), 1)
        for cy in range(0, int(t.th) + 1, 5):
            (_, py), = t.cm_to_px(np.array([[0, cy]]))
            cv2.line(img, (int(x1), int(py)), (int(x2), int(py)), (84, 116, 58) if cy % 10 else (98, 132, 70), 1)
        for (mx, my) in ((0, 0), (t.tw, 0), (t.tw, t.th), (0, t.th)):   # ArUco-ish corner markers
            (px, py), = t.cm_to_px(np.array([[mx, my]]))
            cv2.rectangle(img, (int(px) - 18, int(py) - 18), (int(px) + 18, int(py) + 18), (20, 20, 20), -1)
            cv2.rectangle(img, (int(px) - 10, int(py) - 10), (int(px) + 2, int(py) + 2), (235, 235, 235), -1)
        return img

    def _render(self):
        img = self._bg.copy()
        t = time.time()
        s = self.table.s
        order = sorted(self.world.entities.values(), key=lambda e: {"container": 0, "target": 1, "cover": 2}[e.kind])
        for e in order:
            if e.status.value not in ("VISIBLE",) or not e.pos_cm:
                continue
            (px, py), = self.table.cm_to_px(np.array([e.pos_cm]))
            w, h = self.SIZE.get(e.name, (5, 5))
            x1, y1 = int(px - w * s / 2), int(py - h * s / 2)
            x2, y2 = int(px + w * s / 2), int(py + h * s / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), self.OBJ_BGR.get(e.name, (200, 200, 200)), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 25), 1)
        # a hand drifting in a slow figure eight, plus whatever it holds
        hx = 640 + 380 * math.sin(t * 0.5)
        hy = 380 + 150 * math.sin(t * 1.0)
        cv2.ellipse(img, (int(hx), int(hy)), (46, 60), math.degrees(math.sin(t)) * 0.3, 0, 360, (150, 175, 215), -1)
        for e in self.world.entities.values():
            if e.status.value == "HELD":
                cv2.rectangle(img, (int(hx) - 10, int(hy) - 30), (int(hx) + 10, int(hy) + 20),
                              self.OBJ_BGR.get(e.name, (200, 200, 200)), -1)
        cv2.putText(img, time.strftime("%H:%M:%S") + "  test pattern", (18, 704),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        self.idx += 1
        return self._Frame(t=time.monotonic(), wall=t, img=img, idx=self.idx)

    def latest(self):
        with self._lock:
            if self._last is None or time.monotonic() - self._last.t > 1 / 15:
                self._last = self._render()
            return self._last


def _fake_animator(world, events, frames: TestPatternFrames, stop: threading.Event, period: float = 4.0):
    """Cycle a short scene forever so the dashboard's graph and timeline visibly change."""
    import random

    def ev(obj, typ, **kw):
        now = time.time()
        e = Event(t=time.monotonic(), wall=now, obj=obj, type=typ, **kw)
        events.add(e, frames.latest())

    steps = [
        lambda: (world.set("remote", status=Status.VISIBLE, parent=None, pos_cm=(52.0, 20.0),
                           last_seen=time.time(), confidence=1.0),
                 ev("remote", "MOVED", from_cm=(50.0, 45.0), to_cm=(52.0, 20.0), confidence=1.0)),
        lambda: (world.set("wallet", status=Status.HELD, parent="hand:1", confidence=0.9),
                 ev("wallet", "PICKED_UP", from_cm=(60.0, 15.0), parent="hand:1", confidence=0.9)),
        lambda: (world.set("wallet", status=Status.INSIDE, parent="box", pos_cm=(70.4, 38.1), confidence=0.85),
                 ev("wallet", "PUT_INSIDE", to_cm=(70.4, 38.1), parent="box", confidence=0.85)),
        lambda: (world.set("glasses", status=Status.VISIBLE, parent=None, pos_cm=(80.0, 50.0),
                           last_seen=time.time(), confidence=1.0),
                 ev("glasses", "CORRECTED", to_cm=(80.0, 50.0), confidence=1.0)),
        lambda: (world.set("wallet", status=Status.VISIBLE, parent=None, pos_cm=(60.0, 15.0),
                           last_seen=time.time(), confidence=1.0),
                 ev("wallet", "TAKEN_OUT", from_cm=(70.4, 38.1), to_cm=(60.0, 15.0), confidence=1.0)),
        lambda: (world.set("glasses", status=Status.UNKNOWN, parent="unknown", confidence=0.4),
                 ev("glasses", "LOST_TRACK", from_cm=(80.0, 50.0), confidence=0.4)),
        lambda: (world.set("remote", status=Status.HELD, parent="hand:2", pos_cm=(52.0, 20.0), confidence=0.9),
                 ev("remote", "PICKED_UP", from_cm=(52.0, 20.0), parent="hand:2", confidence=0.9)),
        lambda: (world.set("remote", status=Status.VISIBLE, parent=None, pos_cm=(50.0, 45.0),
                           last_seen=time.time(), confidence=1.0),
                 ev("remote", "PUT_BACK", to_cm=(50.0, 45.0), confidence=1.0)),
        lambda: (world.set("remote", status=Status.HELD, parent="hand:2", confidence=0.9),
                 ev("remote", "PICKED_UP", from_cm=(50.0, 45.0), parent="hand:2", confidence=0.9)),
    ]
    i = 0
    next_step = time.monotonic() + period
    while not stop.is_set():
        world.online = True
        world.fps = round(12.0 + random.uniform(-0.8, 0.8), 1)
        # decay confidence of things we can't see, like the real engine would
        for e in world.entities.values():
            if e.status.value in ("INSIDE", "UNDER", "UNKNOWN"):
                e.confidence = max(0.45, e.confidence * 0.9998)
        las = getattr(world, "_laser_until", 0)
        if world.laser.get("on") and time.time() > las:
            world.laser = {"on": False, "target": None, "err_cm": None}
        if time.monotonic() >= next_step:
            try:
                steps[i % len(steps)]()
            except Exception:
                log.exception("fake step failed")
            i += 1
            next_step = time.monotonic() + period
        stop.wait(0.2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask the Room dashboard server")
    ap.add_argument("--fake", action="store_true", help="demo world + synthetic camera + canned answers")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import uvicorn
    from core.config import load_config
    cfg = load_config(args.config)
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]

    if not args.fake:
        raise SystemExit("Only --fake is runnable standalone; the main loop builds the app with "
                         "create_app(cfg, world, events, frames, ask_fn).")

    import tempfile
    from core.events import EventLog
    from core.fakeworld import demo_world
    snap = tempfile.mkdtemp(prefix="askroom_fake_snaps_")
    events = EventLog(":memory:", snap)
    world = demo_world(events)
    world.online, world.fps = True, 12.0
    table = FakeTable(cfg)
    frames = TestPatternFrames(world, table)
    # the real router: intents + answer templates, Grok for OTHER when online
    from net import NetMonitor
    from voice.pipeline import make_ask
    net = NetMonitor(cfg)
    net.start()
    base_ask = make_ask(cfg, world, events, net=net)

    def ask_fn(text: str, source: str) -> Answer:
        ans = base_ask(text, source)
        if ans.point_at:
            world.laser = {"on": True, "target": ans.point_at, "err_cm": 0.8}
            world._laser_until = time.time() + cfg.get("laser_timeout_s", 10)
        return ans

    stop = threading.Event()
    threading.Thread(target=_fake_animator, args=(world, events, frames, stop), daemon=True).start()
    app = create_app(cfg, world, events, frames=frames, ask_fn=ask_fn, table=table)
    log.info("fake dashboard on http://%s:%s  (snapshots in %s)", host, port, snap)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info", timeout_graceful_shutdown=2)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
