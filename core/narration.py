"""Episode narration memory: what the person DID, in words, grounded in the world model's facts.

The world model (core/world.py) knows each object's state and emits events ('keys PUT_INSIDE box').
Narration is the complementary layer: it cuts the camera stream into activity episodes, keeps a few
keyframes of each, and after the episode ends asks a VLM to describe it, with the episode's world events
in the prompt as ground truth and the tracked objects' names as the only vocabulary. Answers then use
it for 'what was I doing before lunch?', 'what happened while I was away?', 'did I use the stove?'.

(The idea of captioning recorded room activity with a VLM into a searchable store comes from Project
Memoria, MIT licensed; this is a separate implementation: episodes come from hands and world events
rather than fixed clips, the prompt is grounded in the tracker's events, and it all lives in the
EventLog's SQLite file instead of MongoDB/Chroma.)

Threads. feed() runs on the perception thread, once per frame, and is O(1): the Segmenter decides, and
image work (downscale, JPEG encode, delete) goes to a 'narration-frames' worker by queue (keyframes are
skipped, never queued without bound, if it falls behind). A 'narration' worker sends finished episodes
to the provider. An episode is written to SQLite as 'pending' the moment it ends, so being offline, an
API failure or a restart only delays it; retries back off exponentially, and the queue and the calls
per hour are capped from config.

Privacy: off by default (config narration.enabled). When on, keyframes of the table (hands and objects;
the camera sees nothing else) go to the configured provider, and status()['disclosure'] says so on the
dashboard.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Callable, Optional, Union

from core.config import display_name
from core.narration_store import NarrationStore, med_claim, redact_meds
from core.types import Event, EventType

log = logging.getLogger(__name__)

MAX_IMG_BACKLOG = 16        # keyframes waiting for the frames worker; more are skipped
DEFAULT_MODELS = {"grok": "grok-4.3", "claude": "claude-haiku-4-5", "fake": "fake"}


@dataclass
class NarrationConfig:
    """The config.yaml narration: section. Every key is optional."""
    enabled: bool = False
    provider: str = "grok"                 # grok (xAI, or any OpenAI-compatible API) | claude | fake
    model: Optional[str] = None            # default per provider (DEFAULT_MODELS): grok-4.3
    base_url: Optional[str] = None         # grok only; default https://api.x.ai/v1
    api_key_env: Optional[str] = None      # env var holding the key; default XAI_API_KEY / ANTHROPIC_API_KEY
    reasoning_effort: Optional[str] = "low"  # grok only: none | low | medium | high | xhigh
    quiet_s: float = 4.0                   # no hands and no events this long ends an episode
    max_episode_s: float = 120.0           # longer episodes are split
    min_episode_s: float = 2.0             # shorter ones with no world event are not narrated
    min_hand_frames: int = 2               # consecutive hand frames to open an episode (flicker guard)
    keyframe_every_s: float = 1.0
    keyframes: int = 24                    # kept per episode; beyond this they are thinned evenly
    frame_px: int = 640                    # keyframe long side
    jpeg_quality: int = 80
    max_images: int = 8                    # per VLM call
    max_per_hour: int = 30                 # VLM calls per hour (cost cap)
    max_queued: int = 20                   # pending episodes kept while offline; the oldest go first
    timeout_s: float = 30.0
    max_tokens: int = 1024
    max_attempts: int = 4                  # unusable replies before an episode is marked failed
    backoff_s: float = 5.0                 # first retry delay; doubles per attempt ...
    backoff_max_s: float = 300.0           # ... up to this
    keep_h: float = 24.0                   # keyframe JPEG retention (the EventLog's snapshot policy)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "NarrationConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    @property
    def model_name(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")


# ---------------------------------------------------------------- segmentation

@dataclass
class Keyframe:
    t: float                    # monotonic
    wall: float
    name: str                   # file name inside the episode's folder
    event: Optional[str] = None  # 'keys PUT_INSIDE box' when taken at a world event
    ep: str = ""                # episode id (folder)


@dataclass
class Episode:
    id: str
    t_start: float
    wall_start: float
    t_active: float             # last frame with a hand or an event
    wall_active: float
    min_s: float = 2.0
    t_end: float = 0.0
    wall_end: float = 0.0
    keyframes: list[Keyframe] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    @property
    def worth_narrating(self) -> bool:
        """A world event makes any episode worth describing; otherwise only real activity does (a hand
        passing over the table for a moment is not worth a VLM call)."""
        return bool(self.events) or (self.t_active - self.t_start) >= self.min_s


@dataclass
class Step:
    keep: Optional[Keyframe] = None         # save this frame as a keyframe
    dropped: list[Keyframe] = field(default_factory=list)   # delete these keyframes' files
    ended: Optional[Episode] = None


def _ev_dict(e: Event) -> dict:
    return {"wall": e.wall, "obj": e.obj, "type": EventType(e.type).value, "parent": e.parent, "edge": e.edge}


class Segmenter:
    """Frames -> activity episodes. Pure logic (no images, no threads, no clock of its own), so it is
    cheap on the perception thread and fully testable.

    An episode opens on min_hand_frames consecutive hand frames or on any world event, and closes after
    quiet_s with neither, or at max_episode_s (the next one opens on the same frame if activity goes
    on). A keyframe is taken every keyframe_every_s and at every world event; past `keyframes` they are
    thinned by closing the smallest gap, so what is kept stays spread over the episode, and event
    frames (plus the first and the newest) are never thinned while others remain. Of the frames taken
    in the quiet tail, only the last (the settled 'after' view) is kept."""

    def __init__(self, c: NarrationConfig):
        self.c = c
        self.ep: Optional[Episode] = None
        self._hand_run = 0
        self._seq = 0

    def update(self, t: float, wall: float, hands: bool, events=()) -> Step:
        step = Step()
        self._hand_run = self._hand_run + 1 if hands else 0
        active = bool(events) or (hands and (self.ep is not None or self._hand_run >= self.c.min_hand_frames))
        ep = self.ep
        if ep is not None and not active and t - ep.t_active >= self.c.quiet_s - 1e-9:
            step.ended = self._close(step)
            ep = None
        elif ep is not None and t - ep.t_start >= self.c.max_episode_s - 1e-9:
            step.ended = self._close(step)
            ep = None
        if ep is None:
            if not active:
                return step
            ep = self.ep = Episode(id=f"{int(wall * 1000)}", t_start=t, wall_start=wall, t_active=t,
                                   wall_active=wall, min_s=self.c.min_episode_s)
        if active:
            ep.t_active, ep.wall_active = t, wall
        evs = [_ev_dict(e) for e in events]
        ep.events += evs
        last = ep.keyframes[-1].t if ep.keyframes else None
        if evs or last is None or t - last >= self.c.keyframe_every_s - 1e-6:
            self._seq += 1
            kf = Keyframe(t, wall, f"{int(wall * 1000)}_{self._seq}.jpg",
                          "; ".join(event_desc(e) for e in evs) or None, ep.id)
            ep.keyframes.append(kf)
            step.keep = kf
            while len(ep.keyframes) > max(3, self.c.keyframes):
                step.dropped.append(self._thin(ep.keyframes))
        return step

    def close_now(self) -> Optional[Episode]:
        """End the running episode (shutdown); None if there is none."""
        if self.ep is None:
            return None
        step = Step()
        ep = self._close(step)
        ep.dropped = step.dropped           # the caller deletes the trimmed tail
        return ep

    def _close(self, step: Step) -> Episode:
        ep, self.ep = self.ep, None
        tail = [k for k in ep.keyframes if k.t > ep.t_active]
        for k in tail[:-1]:
            ep.keyframes.remove(k)
            step.dropped.append(k)
        last = ep.keyframes[-1] if ep.keyframes else None
        ep.t_end = max(ep.t_active, last.t if last else ep.t_active)
        ep.wall_end = max(ep.wall_active, last.wall if last else ep.wall_active)
        return ep

    @staticmethod
    def _thin(kfs: list[Keyframe]) -> Keyframe:
        """Remove the keyframe whose neighbours are closest together (the smallest merged gap; ties go
        to the newer one), never the first or the newest, and an event frame only when nothing else is
        left. Applied once per new frame, this keeps the survivors evenly spread (measured: 10 kept
        over a 60 s episode sit 4-9 s apart)."""
        n = len(kfs)
        for allow_events in (False, True):
            cands = [i for i in range(1, n - 1) if allow_events or kfs[i].event is None]
            if cands:
                return kfs.pop(min(cands, key=lambda i: (kfs[i + 1].t - kfs[i - 1].t, -i)))
        return kfs.pop(1)


# ---------------------------------------------------------------- job assembly

def _hms(wall: float) -> str:
    return datetime.fromtimestamp(wall).strftime("%H:%M:%S")


def _name(obj: Optional[str], names: Optional[dict], cfg: Optional[dict] = None) -> str:
    """How an entity is written for the VLM: its display name or taught alias; an unnamed thing is
    'unnamed object' (never a thing:N id)."""
    if obj is None:
        return "something"
    if names and obj in names:
        return names[obj] or "unnamed object"
    if obj.startswith("thing:"):
        return "unnamed object"
    return display_name(cfg or {}, obj)


def event_desc(e: dict, names: Optional[dict] = None, cfg: Optional[dict] = None) -> str:
    """'keys PICKED_UP by hand', 'keys PUT_INSIDE box', 'phone EXITED_VIEW off the left edge'."""
    typ, parent = e.get("type"), e.get("parent")
    s = f"{_name(e.get('obj'), names, cfg)} {typ}"
    if parent and parent.startswith("hand"):
        s += " by hand"
    elif parent and parent != "unknown":
        p = _name(parent, names, cfg)
        s += {"COVERED": f" by {p}", "TAKEN_OUT": f" of {p}"}.get(typ, f" {p}")
    elif parent == "unknown":
        s += " by something" if typ == "COVERED" else ""
    if e.get("edge"):
        s += f" off the {e['edge']} edge"
    return s


def event_line(e: dict, names: Optional[dict] = None, cfg: Optional[dict] = None) -> str:
    return f"{_hms(e['wall'])} {event_desc(e, names, cfg)}"


def select_frames(kfs: list[Keyframe], k: int) -> list[Keyframe]:
    """At most k keyframes spanning the episode: the first, the last and the event frames first, then
    farthest-point fill in time. With more event frames than room, events are spread evenly."""
    if len(kfs) <= k:
        return list(kfs)
    k = max(2, k)
    first, last = kfs[0], kfs[-1]
    evs = [f for f in kfs[1:-1] if f.event]
    if len(evs) > k - 2:
        step = len(evs) / (k - 2)
        chosen = [first, last] + [evs[int(i * step + step / 2)] for i in range(k - 2)]
    else:
        chosen = [first, last] + evs
        rest = [f for f in kfs if f not in chosen]
        while len(chosen) < k and rest:
            best = max(rest, key=lambda f: min(abs(f.t - c.t) for c in chosen))
            chosen.append(best)
            rest.remove(best)
    return sorted(chosen, key=lambda f: f.t)


SYSTEM = """You describe short episodes recorded by an overhead camera that looks straight down at a tabletop. It sees the table surface, the objects on it and people's hands and forearms; it never sees faces, bodies or the rest of the room. You get keyframes from one episode (oldest first, each with its clock time) and the events that a rule-based object tracker logged during it. Those events are ground truth about the tracked objects: never contradict them, and don't just repeat them; describe what the person was doing around them.

Write for the person who owns the table, in second person, e.g. "You sorted some papers, then put your keys in the box."

Rules:
- Describe only what is visible in the frames or stated in the events. If something is not clear, say "unclear" rather than guess.
- For tracked objects use exactly the names in the object list. Anything else goes in "objects" as "unknown object"; you may describe it briefly in "detail" and in the summary (e.g. "a mug").
- Never state or imply that medication was taken, swallowed, skipped or missed. You may say the pill bottle was picked up, opened, moved or put down, and nothing more about pills.
- Don't try to identify anyone; say "you".
- summary: one or two plain sentences, no lists, no clock times.
- actions: in time order; t is the frame time (HH:MM:SS) when it happened, or null.
- activity_tags: a few short lowercase words (e.g. "tidying", "reading", "cooking").
- confidence: 0 to 1, how sure you are of the summary; below 0.5 when the frames are ambiguous.
- Reply with the JSON object only."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "actions", "objects_involved", "activity_tags", "confidence", "notes"],
    "properties": {
        "summary": {"type": "string"},
        "actions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["t", "verb", "objects", "detail"],
            "properties": {"t": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "verb": {"type": "string"},
                           "objects": {"type": "array", "items": {"type": "string"}},
                           "detail": {"type": "string"}}}},
        "objects_involved": {"type": "array", "items": {"type": "string"}},
        "activity_tags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
}

REPLY_SHAPE = ('{"summary": str, "actions": [{"t": "HH:MM:SS" or null, "verb": str, "objects": [str], '
               '"detail": str}], "objects_involved": [str], "activity_tags": [str], "confidence": 0-1, '
               '"notes": str}')


@dataclass
class Job:
    system: str
    parts: list                 # ('text', str) | ('image', jpeg bytes), in order
    names: list[str]            # the vocabulary given to the model
    events_text: str
    frames: list[str]           # image paths sent
    t_start: float = 0.0
    t_end: float = 0.0


def _f(row, key, default=None):
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def entity_names(cfg: dict, labels: Optional[dict] = None) -> dict:
    """entity -> spoken name (None for an unnamed thing)."""
    out = {o: display_name(cfg, o) for o in (cfg.get("objects") or {})}
    out.update(labels or {})
    return out


def build_job(row, cfg: dict, labels: Optional[dict] = None, max_images: int = 8) -> Job:
    """One pending row (a Row or a dict with t_start, t_end, frames_dir, episode) -> the VLM request."""
    ep = _f(row, "episode") or {}
    fdir = _f(row, "frames_dir")
    names = entity_names(cfg, labels)
    vocab = list(dict.fromkeys(n for n in names.values() if n))
    evs = sorted(ep.get("events") or [], key=lambda e: e.get("wall", 0))
    events_text = "; ".join(event_line(e, names, cfg) for e in evs)
    kfs = [Keyframe(k["t"], k["wall"], k["name"], k.get("event")) for k in ep.get("keyframes") or []]
    kfs = [k for k in kfs if fdir and os.path.exists(os.path.join(fdir, k.name))]
    chosen = select_frames(kfs, max(1, int(max_images)))
    t0, t1 = _f(row, "t_start", 0.0), _f(row, "t_end", 0.0)
    parts: list = [("text", f"Episode from {_hms(t0)} to {_hms(t1)} ({max(0, round(t1 - t0))} s).\n"
                            f"Tracked objects (use exactly these names): {', '.join(vocab)}.\n"
                            f"Tracker events (ground truth): {events_text or 'none'}.\n"
                            f"{len(chosen)} keyframes follow, oldest first.")]
    paths = []
    for i, k in enumerate(chosen, 1):
        at = [event_desc(e, names, cfg) for e in evs if abs(e.get("wall", 0) - k.wall) < 0.05]
        what = "; ".join(at) or k.event
        parts.append(("text", f"Frame {i} at {_hms(k.wall)}" + (f", when: {what}" if what else "")))
        p = os.path.join(fdir, k.name)
        with open(p, "rb") as fh:
            parts.append(("image", fh.read()))
        paths.append(p)
    parts.append(("text", f"Describe this episode. Reply with JSON only: {REPLY_SHAPE}"))
    return Job(SYSTEM, parts, vocab, events_text, paths, t0, t1)


# ---------------------------------------------------------------- output validation

class NarrationError(ValueError):
    """The provider's reply could not be used."""


def _parse_json(text: str) -> dict:
    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < i:
        raise NarrationError("no JSON object in the reply")
    s = s[i:j + 1]
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            v = json.loads(cand)
        except ValueError:
            continue
        if isinstance(v, dict):
            return v
    raise NarrationError("reply is not a JSON object")


def _vocab_map(names: list[str]) -> dict:
    m = {}
    for n in names:
        k = n.lower().strip()
        for v in (k, k.replace(" ", "_"), k.rstrip("s"), k + "s"):
            m.setdefault(v, n)
    return m


def _texts(v) -> list[str]:
    return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()] \
        if isinstance(v, list) else []


def validate(text: str, names: list[str]) -> dict:
    """The provider's reply -> the stored narration dict, or NarrationError. Repairs what is cheap to
    repair (code fences, prose around the object, trailing commas, a string confidence), maps object
    names onto the vocabulary ('unknown object' otherwise), and enforces the medication rule on every
    text field: offending sentences, actions and tags are dropped and counted in 'redacted'."""
    d = _parse_json(text)
    summary = d.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise NarrationError("no summary")
    vocab = _vocab_map(names)

    def canon(x: str) -> Optional[str]:
        return vocab.get(x.lower().strip())

    summary = " ".join(" ".join(summary.split()).split(" ")[:80])
    summary = " ".join(re.split(r"(?<=[.!?])\s+", summary)[:2])
    actions = []
    for a in d.get("actions") or []:
        if not isinstance(a, dict):
            continue
        t = a.get("t")
        t = t.strip() if isinstance(t, str) and re.fullmatch(r"\d{1,2}:\d\d(?::\d\d)?", t.strip()) else None
        objs = [canon(o) or "unknown object" for o in _texts(a.get("objects"))]
        actions.append({"t": t, "verb": str(a.get("verb") or "").strip()[:40],
                        "objects": list(dict.fromkeys(objs)), "detail": str(a.get("detail") or "").strip()[:200]})
    involved = [c for c in (canon(o) for o in _texts(d.get("objects_involved"))) if c]
    tags = list(dict.fromkeys(t.lower()[:40] for t in _texts(d.get("activity_tags"))))[:8]
    try:
        conf = float(d.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = min(1.0, max(0.0, conf)) if conf == conf else 0.5
    notes = str(d.get("notes") or "").strip()[:300]

    redacted = 0
    summary, n = redact_meds(summary)
    redacted += n
    kept = []
    for a in actions:
        if med_claim(f"{a['verb']} {' '.join(a['objects'])} {a['detail']}."):
            redacted += 1
        else:
            kept.append(a)
    tags2 = [t for t in tags if not med_claim(t)]
    redacted += len(tags) - len(tags2)
    notes, n = redact_meds(notes)
    redacted += n
    if not summary:
        pill = next((n for n in involved if "pill" in n.lower()), None)
        summary = f"You handled the {pill}." if pill else "Unclear."
    return {"summary": summary, "actions": kept[:12], "objects_involved": list(dict.fromkeys(involved)),
            "activity_tags": tags2, "confidence": round(conf, 3), "notes": notes, "redacted": redacted}


# ---------------------------------------------------------------- providers

class ProviderError(RuntimeError):
    """A provider call failed. retryable: worth trying again later (network, rate limit, server, auth
    or configuration that an operator can fix); otherwise the reply itself was the problem."""

    def __init__(self, msg: str, retryable: bool = True):
        super().__init__(msg)
        self.retryable = retryable


@dataclass
class Reply:
    text: str
    usage: dict
    latency_ms: int


def _classify(ex: Exception) -> ProviderError:
    status = getattr(ex, "status_code", None)
    retry = status is None or status in (401, 403, 408, 409, 429) or status >= 500
    return ProviderError(f"{type(ex).__name__}: {status or ''} {str(ex)[:200]}".strip(), retryable=retry)


class Provider:
    name = "base"
    model = ""

    def run(self, job: Job) -> Reply:
        return self.narrate(job.system, job.parts, SCHEMA)

    def narrate(self, system: str, parts: list, schema: Optional[dict]) -> Reply:
        raise NotImplementedError


class ClaudeProvider(Provider):
    """Anthropic Messages API with structured outputs (output_config json_schema), kept as an
    alternative provider (the project default is Grok). The key comes from the environment only
    (ANTHROPIC_API_KEY, or cfg api_key_env); it is never logged."""
    name = "claude"

    def __init__(self, c: NarrationConfig):
        self.c, self.model = c, c.model or DEFAULT_MODELS["claude"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get(self.c.api_key_env) if self.c.api_key_env else None
            kw = {"api_key": key} if key else {}
            self._client = anthropic.Anthropic(max_retries=0, timeout=self.c.timeout_s, **kw)
        return self._client

    def narrate(self, system: str, parts: list, schema: Optional[dict]) -> Reply:
        content = []
        for kind, v in parts:
            if kind == "image":
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                            "data": base64.standard_b64encode(v).decode()}})
            else:
                content.append({"type": "text", "text": v})
        kw = dict(model=self.model, max_tokens=self.c.max_tokens, system=system,
                  messages=[{"role": "user", "content": content}])
        if schema:
            kw["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        t0 = time.perf_counter()
        try:
            r = self._get_client().messages.create(**kw)
        except ImportError as ex:
            raise ProviderError(f"anthropic SDK missing: {ex}", retryable=True)
        except Exception as ex:
            raise _classify(ex)
        ms = int((time.perf_counter() - t0) * 1000)
        if getattr(r, "stop_reason", None) == "refusal":
            raise ProviderError("refused", retryable=False)
        text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
        u = getattr(r, "usage", None)
        return Reply(text, {"input_tokens": getattr(u, "input_tokens", None),
                            "output_tokens": getattr(u, "output_tokens", None)}, ms)


class OpenAICompatProvider(Provider):
    """xAI Grok (the default: grok-4.3 at https://api.x.ai/v1) or any OpenAI-compatible chat API with
    image input. Images go as base64 JPEG data URLs in image_url parts; the reply is constrained with
    response_format json_schema (strict) and validated here anyway. reasoning_effort comes from config.
    If the API rejects reasoning_effort or json_schema (HTTP 400), the call is retried once without it
    (json_schema falls back to json_object) and that choice is kept for later calls. The key comes from
    $XAI_API_KEY (or cfg api_key_env), read per call, never logged. The client is core.xai's (plain requests
    on the shared connection, no retries: the narrator's queue does the retrying)."""
    name = "grok"

    def __init__(self, c: NarrationConfig):
        self.c, self.model = c, c.model or DEFAULT_MODELS["grok"]
        self._client = None
        self._drop: set[str] = set()          # request features this API refused

    def narrate(self, system: str, parts: list, schema: Optional[dict]) -> Reply:
        key = os.environ.get(self.c.api_key_env or "XAI_API_KEY", "").strip()
        if not key:
            raise ProviderError(f"no API key in ${self.c.api_key_env or 'XAI_API_KEY'}", retryable=True)
        content = []
        for kind, v in parts:
            if kind == "image":
                url = "data:image/jpeg;base64," + base64.standard_b64encode(v).decode()
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
            else:
                content.append({"type": "text", "text": v})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        t0 = time.perf_counter()
        for _ in range(3):
            kw = dict(model=self.model, max_completion_tokens=self.c.max_tokens, messages=messages)
            if schema and "json_schema" not in self._drop:
                kw["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "reply", "schema": schema, "strict": True}}
            else:
                kw["response_format"] = {"type": "json_object"}
            if self.c.reasoning_effort and "reasoning_effort" not in self._drop:
                kw["reasoning_effort"] = self.c.reasoning_effort
            try:
                if self._client is None:
                    from core.xai import Client
                    self._client = Client(self.c.base_url or "https://api.x.ai/v1", key, timeout=self.c.timeout_s)
                r = self._client.chat.completions.create(**kw)
                break
            except Exception as ex:
                if getattr(ex, "status_code", None) == 400:
                    # Drop the feature the error names; if it names neither, reasoning_effort first.
                    msg = str(ex).lower()
                    used = [f for f in ("reasoning_effort", "json_schema") if f not in self._drop and
                            (f in kw if f == "reasoning_effort" else kw["response_format"]["type"] == f)]
                    named = [f for f in used if f in msg or (f == "json_schema" and "response_format" in msg)]
                    feature = (named or used or [None])[0]
                    if feature is not None:
                        log.warning("%s rejected %s; retrying without it", self.model, feature)
                        self._drop.add(feature)
                        continue
                raise _classify(ex)
        else:
            raise ProviderError("request rejected", retryable=False)
        ms = int((time.perf_counter() - t0) * 1000)
        u = getattr(r, "usage", None)
        return Reply(r.choices[0].message.content or "",
                     {"input_tokens": getattr(u, "prompt_tokens", None),
                      "output_tokens": getattr(u, "completion_tokens", None)}, ms)


def _echo_reply(job: Job) -> str:
    """The fake provider's default: a narration made of the tracker events, for --fake runs and demos."""
    first = job.events_text.split("; ")[0] if job.events_text else ""
    what = f"The tracker logged: {first.split(' ', 1)[1]}." if first else "Something moved on the table."
    return json.dumps({"summary": f"You were busy at the table. {what}", "actions": [], "objects_involved": [],
                       "activity_tags": ["fake"], "confidence": 0.5, "notes": "fake provider"})


class FakeProvider(Provider):
    """For tests and offline demos. reply: a string, a function(job) -> string, or a list of those
    and exceptions, used in order (the last one repeats). Records every job in .calls."""
    name = "fake"
    model = "fake"

    def __init__(self, reply: Union[str, Callable, list, None] = None):
        self.replies = reply if isinstance(reply, list) else [reply if reply is not None else _echo_reply]
        self.calls: list[Job] = []

    def narrate(self, system: str, parts: list, schema: Optional[dict]) -> Reply:
        """Direct calls (voice/visual.py) are recorded as a Job too."""
        return self.run(Job(system, parts, [], "", []))

    def run(self, job: Job) -> Reply:
        self.calls.append(job)
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(r, Exception):
            raise r
        text = r(job) if callable(r) else r
        return Reply(text, {"input_tokens": 0, "output_tokens": 0}, 1)


def make_provider(c: NarrationConfig) -> Provider:
    p = (c.provider or "grok").lower()
    if p in ("grok", "xai", "openai_compat", "openai"):
        return OpenAICompatProvider(c)
    if p in ("claude", "anthropic"):
        return ClaudeProvider(c)
    if p == "fake":
        return FakeProvider()
    raise ValueError(f"unknown narration provider {c.provider!r}")


# ---------------------------------------------------------------- the narrator

class Narrator:
    """Owns the segmenter, the two workers and the narrations queue. See the module docstring.

    feed(frame, dets, new_events) once per perception frame, or attach(world) to have World.update do
    it (that also covers --fake runs, where server/sim.py drives the world without a perception loop).
    start=False runs no threads: tests call drain() and run_pending() instead."""

    def __init__(self, cfg: dict, events, provider: Optional[Provider] = None,
                 online: Optional[Callable[[], bool]] = None, labels: Optional[Callable[[], dict]] = None,
                 clock: Callable[[], float] = time.time, start: bool = True):
        self.cfg = cfg
        self.c = NarrationConfig.from_dict(cfg.get("narration"))
        self.store = NarrationStore(events)
        self.root = os.path.join(events.snap_dir, "narration")
        os.makedirs(self.root, exist_ok=True)
        self.seg = Segmenter(self.c)
        self.provider = provider or make_provider(self.c)
        self.online = online or (lambda: True)
        self.labels = labels or (lambda: {})
        self.clock = clock
        self.last: Optional[dict] = None
        self.last_dir: Optional[str] = None
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._imgs = 0
        self._busy = False
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._calls = deque(self.store.calls_since(clock() - 3600))
        self._pruned_at = 0.0
        self._feed_errors = 0
        self._queued = self.store.pending_count()
        self._threads: list[threading.Thread] = []
        if start:
            self.start()

    # -- perception thread

    def feed(self, frame, dets, new_events=()) -> None:
        """One call per processed frame. O(1); never blocks on disk or network; never raises."""
        try:
            hands = bool(getattr(dets, "hands", None))
            evs = [e for e in (new_events or ()) if isinstance(e, Event)]
            step = self.seg.update(frame.t, frame.wall, hands, evs)
            if step.ended is not None:
                self._q.put(("end", step.ended))
            for k in step.dropped:
                self._q.put(("del", self._path(k)))
            if step.keep is not None and getattr(frame, "img", None) is not None:
                with self._lock:
                    if self._imgs >= MAX_IMG_BACKLOG:
                        return              # the worker is behind: a missing keyframe beats a stall
                    self._imgs += 1
                # The capture thread allocates a new array per frame, so a reference is safe to hand off.
                self._q.put(("img", self._path(step.keep), frame.img))
        except Exception:
            self._feed_errors += 1
            if self._feed_errors in (1, 10, 100) or self._feed_errors % 1000 == 0:
                log.exception("narration feed failed (%d so far)", self._feed_errors)

    def attach(self, world) -> "Narrator":
        """Hook this narrator into a World: after each update() it is fed that update's frame, detections
        and events; state_json() gains a 'narration' entry (status and the privacy disclosure); thing
        aliases come from world.thing_labels(). Instance attributes, so the class is untouched."""
        update, state = world.update, world.state_json

        def fed_update(dets, frame):
            out = update(dets, frame)
            if frame is not None:
                self.feed(frame, dets, out)
            return out

        def state_json(*a, **kw):
            st = state(*a, **kw)
            try:
                st["narration"] = self.status()
            except Exception:
                log.exception("narration status failed")
            return st

        world.update, world.state_json = fed_update, state_json
        if hasattr(world, "thing_labels"):
            self.labels = world.thing_labels
        return self

    # -- workers

    def start(self) -> None:
        for target, name in ((self._frames_loop, "narration-frames"), (self._narrate_loop, "narration")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def _path(self, k: Keyframe) -> str:
        return os.path.join(self.root, k.ep, k.name)

    def _frames_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                self._do(item)
            except Exception:
                log.exception("narration frame work failed")
            finally:
                self._q.task_done()

    def drain(self) -> None:
        """Do all queued image and episode work now, on this thread (start=False)."""
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                if item is not None:
                    self._do(item)
            finally:
                self._q.task_done()

    def _do(self, item) -> None:
        kind = item[0]
        if kind == "img":
            _, path, img = item
            try:
                self._write(path, img)
            finally:
                with self._lock:
                    self._imgs -= 1
        elif kind == "del":
            try:
                os.remove(item[1])
            except FileNotFoundError:
                pass
        elif kind == "end":
            self._persist(item[1])

    def _write(self, path: str, img) -> None:
        import cv2
        h, w = img.shape[:2]
        s = self.c.frame_px / max(h, w)
        if s < 1:
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, int(self.c.jpeg_quality)]):
            log.warning("keyframe write failed: %s", path)

    def _persist(self, ep: Episode) -> None:
        """A finished episode: discard it, or queue it in SQLite (bounded; the oldest pending go)."""
        fdir = os.path.join(self.root, ep.id)
        for k in getattr(ep, "dropped", []):
            try:
                os.remove(self._path(k))
            except FileNotFoundError:
                pass
        if not ep.worth_narrating:
            shutil.rmtree(fdir, ignore_errors=True)
            return
        episode = {"keyframes": [{"t": k.t, "wall": k.wall, "name": k.name, "event": k.event}
                                 for k in ep.keyframes], "events": ep.events}
        self.store.add_pending(ep.wall_start, ep.wall_end, fdir, episode, now=self.clock())
        self.last_dir = fdir
        for r in self.store.drop_oldest_pending(self.c.max_queued):
            log.warning("narration queue full: dropped the episode from %s", _hms(r.t_start))
            if r.frames_dir:
                shutil.rmtree(r.frames_dir, ignore_errors=True)
        self._queued = self.store.pending_count()
        self._wake.set()

    # -- narration

    def _backoff(self, attempts: int) -> float:
        return max(0.5, min(self.c.backoff_max_s, self.c.backoff_s * 2 ** max(0, attempts - 1)))

    def _capped(self, now: float) -> bool:
        while self._calls and self._calls[0] <= now - 3600:
            self._calls.popleft()
        return len(self._calls) >= self.c.max_per_hour

    def _narrate_one(self) -> Optional[str]:
        """Narrate the oldest due episode. Returns 'done' / 'retry' / 'failed', or None when nothing can
        be sent now (none due, offline, or the hourly cap reached)."""
        now = self.clock()
        if now - self._pruned_at > 3600:
            self._pruned_at = now
            try:
                self.store.prune(self.c.keep_h, now=time.time(), root=self.root)
            except Exception:
                log.exception("narration prune failed")
        row = self.store.next_due(now)
        if row is None or not self.online() or self._capped(now):
            return None
        try:
            labels = self.labels() or {}
        except Exception:
            labels = {}
        try:
            job = build_job(row, self.cfg, labels, self.c.max_images)
        except Exception as ex:
            self.store.mark_failed(row.id, f"job: {ex}")
            return "failed"
        if not job.frames and not job.events_text:
            self.store.mark_failed(row.id, "no frames left")
            return "failed"
        self._calls.append(now)
        self.store.mark_called(row.id, now)
        try:
            reply = self.provider.run(job)
            data = validate(reply.text, job.names)
        except (ProviderError, NarrationError) as ex:
            retry = isinstance(ex, ProviderError) and ex.retryable
            if not retry and row.attempts + 1 >= self.c.max_attempts:
                log.warning("narration of %s failed for good: %s", _hms(row.t_start), ex)
                self.store.mark_failed(row.id, str(ex))
                self._queued = self.store.pending_count()
                return "failed"
            log.info("narration of %s will be retried: %s", _hms(row.t_start), ex)
            self.store.mark_retry(row.id, now + self._backoff(row.attempts + 1), str(ex))
            return "retry"
        data.update(usage=reply.usage, frames=[os.path.basename(p) for p in job.frames],
                    events=job.events_text)
        self.store.mark_done(row.id, data["summary"], data, self.provider.name, self.provider.model,
                             reply.latency_ms)
        self._queued = self.store.pending_count()
        self.last = {"summary": data["summary"], "t": row.t_end, "latency_ms": reply.latency_ms}
        log.info("narration %s-%s (%d ms): %s", _hms(row.t_start), _hms(row.t_end), reply.latency_ms,
                 data["summary"])
        return "done"

    def run_pending(self) -> int:
        """Narrate everything that can be sent now (start=False). Returns how many were narrated."""
        n = 0
        while True:
            r = self._narrate_one()
            if r is None:
                return n
            n += r == "done"

    def _wait_s(self) -> float:
        now = self.clock()
        if not self.online():
            return 2.0
        if self._capped(now):
            return max(1.0, self._calls[0] + 3600 - now)
        nxt = self.store.next_try_after()
        return 30.0 if nxt is None else min(30.0, max(0.2, nxt - now))

    def _narrate_loop(self) -> None:
        while not self._stop.is_set():
            self._busy = True
            try:
                r = self._narrate_one()
            except Exception:
                log.exception("narration failed")
                r = None
            if r is not None:
                continue
            self._busy = False
            self._wake.wait(self._wait_s())
            self._wake.clear()

    def idle(self, timeout: float = 5.0) -> bool:
        """Wait until queued frame work is done and nothing sendable is left (threads running)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.unfinished_tasks == 0 and not self._busy and \
                    (self.store.next_due(self.clock()) is None or not self.online() or self._capped(self.clock())):
                return True
            time.sleep(0.02)
        return False

    # -- lifecycle and status

    def close_episode(self) -> None:
        """End the running episode now (so it is kept), e.g. at shutdown."""
        ep = self.seg.close_now()
        if ep is not None:
            self._q.put(("end", ep))

    def stop(self) -> None:
        """Keep the running episode (queued for next time if it can't be sent now) and stop the workers.
        Must run before the EventLog closes."""
        self.close_episode()
        if not self._threads:
            self.drain()
            return
        self._q.put(None)
        self._threads[0].join(timeout=5)
        self._stop.set()
        self._wake.set()
        self._threads[1].join(timeout=2)

    def status(self) -> dict:
        return {"enabled": True, "provider": self.provider.name, "model": self.provider.model,
                "queued": self._queued,
                "last_summary": self.last["summary"] if self.last else None,
                "last_t": self.last["t"] if self.last else None,
                "disclosure": (f"Narration is on: keyframes of the table (hands and objects) are sent to "
                               f"{self.provider.name} ({self.provider.model}) to describe what happened.")}


def from_config(cfg: dict, events, world=None, online: Optional[Callable[[], bool]] = None,
                start: bool = True) -> Optional[Narrator]:
    """The app's Narrator, or None when cfg narration.enabled is false (the default). With a world,
    it is attached (see Narrator.attach)."""
    c = NarrationConfig.from_dict(cfg.get("narration"))
    if not c.enabled:
        return None
    n = Narrator(cfg, events, online=online, start=start)
    if world is not None:
        n.attach(world)
    log.info("narration on: %s", n.status()["disclosure"])
    return n


# ---------------------------------------------------------------- selftest (a real provider call)

def _selftest(argv=None) -> int:
    """python -m core.narration --selftest [--photos a.jpg b.jpg ...] [--n 2] [--reasoning none]

    Narrates real episodes with the configured provider (default Grok; needs $XAI_API_KEY) and prints
    latency, token usage and the validated JSON. Without --photos, episodes come from server/sim.py's
    scripted story through the real World (captions painted over, so the model can't read them)."""
    import argparse
    import tempfile

    import cv2

    from core.config import load_config
    from core.events import EventLog
    from core.types import Detections, Frame
    ap = argparse.ArgumentParser(description=_selftest.__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--photos", nargs="*", help="still photos played as one episode, oldest first")
    ap.add_argument("--n", type=int, default=2, help="episodes to narrate")
    ap.add_argument("--reasoning", help="override narration.reasoning_effort (grok: none | low | ...)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    nc = dict(cfg.get("narration") or {}, enabled=True)
    if a.reasoning:
        nc["reasoning_effort"] = a.reasoning
    cfg = {**cfg, "narration": nc}
    c = NarrationConfig.from_dict(nc)
    key_env = c.api_key_env or ("XAI_API_KEY" if c.provider == "grok" else "ANTHROPIC_API_KEY")
    if c.provider != "fake" and not os.environ.get(key_env):
        print(f"${key_env} is not set: the real {c.provider} call is pending a key.")
        return 2
    events = EventLog(":memory:", tempfile.mkdtemp(prefix="narration_selftest_"))
    n = Narrator(cfg, events, start=False)
    if a.photos:
        t = 0.0
        for p in a.photos:
            img = cv2.imread(p)
            if img is None:
                raise SystemExit(f"can't read {p}")
            for _ in range(20):                      # 2 s of each photo at 10 fps, a hand 'present'
                f = Frame(t=t, wall=time.time() - 60 + t, img=img, idx=int(t * 10))
                n.feed(f, Detections(t=t, frame_idx=0, hands=[object()]), [])
                n.drain()
                t = round(t + 0.1, 3)
        n.close_episode()
    else:
        from core.world import World
        from eval.synth import FPS, IMG_H
        from server.sim import Painter, story
        world = World(cfg, events)
        n.attach(world)
        scene, steps = story(cfg, 0)
        painter = Painter(cfg)
        wall0 = time.time() - len(scene.snaps) / FPS
        for i, (snap, d) in enumerate(zip(scene.snaps, scene.render())):
            img = painter.draw(snap, "")
            img[IMG_H - 44:] = painter.bg[IMG_H - 44:]     # no caption bar
            t = 1000.0 + i / FPS
            world.update(Detections(t=t, frame_idx=i, items=d.items, hands=d.hands),
                         Frame(t=t, wall=wall0 + i / FPS, img=img, idx=i))
            n.drain()
        n.close_episode()
    n.drain()
    done = 0
    while done < a.n:
        r = n.store.next_due(n.clock())
        if r is None:
            break
        job = build_job(r, cfg, n.labels(), n.c.max_images)
        print(f"\n--- episode {_hms(r.t_start)}-{_hms(r.t_end)}: {len(job.frames)} frames; events: "
              f"{job.events_text or 'none'}")
        t0 = time.perf_counter()
        try:
            reply = n.provider.run(job)
            data = validate(reply.text, job.names)
        except (ProviderError, NarrationError) as ex:
            print(f"FAILED after {time.perf_counter() - t0:.2f} s: {ex}")
            n.store.mark_failed(r.id, str(ex))
            continue
        n.store.mark_done(r.id, data["summary"], data, n.provider.name, n.provider.model, reply.latency_ms)
        print(f"{n.provider.name} {n.provider.model}: {reply.latency_ms} ms, usage {reply.usage}")
        print(json.dumps(data, indent=1))
        done += 1
    events.close()
    return 0 if done else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftest(sys.argv[1:]))
