"""TEACH intent -> Answer: 'this is my charger' names the thing just put in the teach square.

answers.answer() calls this for TEACH, so every entry point that already routes questions through
voice.pipeline (voice, dashboard Ask box, SMS) teaches with no extra wiring. The app may also call
teach_answer directly, e.g. for a name typed into the dashboard.
"""
from __future__ import annotations

from typing import Optional

from core.config import display_name
from core.things import is_thing, norm_name
from core.types import Answer


def teach_answer(name: Optional[str], world, cfg: dict) -> Answer:
    """Bind name to the thing the world's teach rule picks and confirm it (pointing the laser at
    it), or say why not: no name, a configured object's name, or nothing in the teach square."""
    key = norm_name(name)
    if not key:
        return Answer("What should I call it? Say 'this is my' and then its name.")
    if not hasattr(world, "teach"):
        return Answer("I can't learn new names right now.")
    known = world.find(key)
    if known is not None and not is_thing(known):
        whose = "your" if (cfg.get("objects") or {}).get(known) == "target" else "the"
        return Answer(f"I already know {whose} {display_name(cfg, known)}. Give this one a different name.")
    thing = world.teach(key)
    if thing is None:
        where = "in the teach square" if getattr(world.cfg, "teach_zone_cm", None) else "down on the table"
        return Answer(f"Put it {where} first, then say 'this is my {key}'.")
    return Answer(f"Got it, I'll remember this as your {key}.", point_at=thing, action="point")
