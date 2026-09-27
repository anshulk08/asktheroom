"""A Grok provider with its replies cached on disk, for replays that name things (eval/score_clip.py
--names eval.grok_cache:provider, eval/naming.py): the same prompt and the same image bytes are asked once,
so re-scoring a clip or re-running an eval costs no calls and gives the same names. Grok is not
deterministic; salt (a trial label) asks again for a run-to-run spread.

The cache is ~/.cache/askroom/grok (env ASKROOM_GROK_CACHE), one JSON file per reply. The key and the
replies are never printed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from core.narration import Reply

DEFAULT_DIR = Path.home() / ".cache/askroom/grok"


class CachedProvider:
    """provider.narrate(system, parts, schema) with replies cached by (salt, system, schema, parts).
    .calls counts the real calls and .latency their durations (s)."""

    def __init__(self, provider, cache: Path, salt: str = ""):
        self.p, self.cache, self.salt = provider, Path(cache), salt
        self.name, self.model = getattr(provider, "name", "grok"), getattr(provider, "model", None)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls, self.latency = 0, []

    def narrate(self, system, parts, schema):
        h = hashlib.sha256(self.salt.encode() + system.encode() + json.dumps(schema, sort_keys=True).encode())
        for k, v in parts:
            h.update(k.encode() + (v if isinstance(v, bytes) else str(v).encode()))
        f = self.cache / f"{h.hexdigest()}.json"
        if f.exists():
            return Reply(json.loads(f.read_text())["text"], {}, 0)
        t0 = time.perf_counter()
        r = self.p.narrate(system, parts, schema)
        self.latency.append(time.perf_counter() - t0)
        self.calls += 1
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"text": r.text}))
        tmp.replace(f)                             # a reader never sees half a file
        return r


def grok(cfg: dict, timeout_s: float = 20.0):
    """The app's naming provider (visual_memory's Grok settings), with a longer timeout for batch runs."""
    from core.narration import NarrationConfig, make_provider
    from core.visual_memory import VisualConfig
    v = VisualConfig.from_dict((cfg or {}).get("visual_memory"))
    return make_provider(NarrationConfig.from_dict({
        "provider": v.provider, "model": v.model, "base_url": v.base_url, "api_key_env": v.api_key_env,
        "reasoning_effort": v.reasoning_effort, "timeout_s": timeout_s, "max_tokens": 200}))


def provider(cfg: Optional[dict] = None, cache: Optional[str] = None, salt: str = "", inner=None) -> CachedProvider:
    """eval/score_clip.py --names eval.grok_cache:provider: the configured Grok provider (or inner), cached."""
    if cfg is None:
        from core.config import load_config
        cfg = load_config()
    d = Path(cache or os.environ.get("ASKROOM_GROK_CACHE") or DEFAULT_DIR)
    return CachedProvider(inner if inner is not None else grok(cfg), d, salt=salt)
