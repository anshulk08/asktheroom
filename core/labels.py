"""What to call an entity when showing or saying it. A thing the person named is called by that name
('charger'), else by what it looks like ('pill bottle?'), else 'something new': neither the internal id
thing:7 nor its number ever reaches a person, or Grok (which would say it).
Read from state_json (the world emits label / aliases / maybe_same_as per thing, and merged ids; the
auto-namer adds a guess), so
the overlay, Grok's world state and the dashboard (server/web/app.js mirrors this) say the same thing.
Configured objects keep their names; callers space them out as before.
"""
from __future__ import annotations

from typing import Optional

PREFIX = 'thing:'
NEW = 'something new'       # a thing with no taught name and no guess


def thing_label(name: str, label: Optional[str] = None, guess: Optional[str] = None) -> str:
    """'thing:7' -> the taught label, else the automatic guess of what it is (core/auto_name.py) as
    'deodorant stick?', else NEW. Any other name comes back as is."""
    if label:
        return str(label)
    if isinstance(name, str) and name.startswith(PREFIX):
        return f'{guess}?' if guess else NEW
    return name


def _guess(e: dict) -> Optional[str]:
    g = e.get('guess')
    return str(g['name']) if isinstance(g, dict) and g.get('name') else None


def thing_labels(state: Optional[dict]) -> dict[str, str]:
    """Every thing in a state_json dict -> what to call it, including ids merged into another thing
    (their old events and parents then read as the survivor). Grok picks things by these names, so a
    name two live things share gets ' (2)', ' (3)' after the first."""
    state = state or {}
    out: dict[str, str] = {}
    taken: dict[str, int] = {}
    for e in state.get('entities') or []:
        if str(e.get('name', '')).startswith(PREFIX):
            label = thing_label(e['name'], e.get('label'), _guess(e))
            taken[label.lower()] = k = taken.get(label.lower(), 0) + 1
            out[e['name']] = label if k == 1 else f'{label} ({k})'
    for old, into in (state.get('merged') or {}).items():
        seen = set()
        while into in (state.get('merged') or {}) and into not in seen:     # chains of merges
            seen.add(into)
            into = state['merged'][into]
        out[old] = out.get(into) or thing_label(into)
    return out


def spoken(name: Optional[str], labels: dict[str, str]) -> str:
    """Display form of any entity or parent name: a thing's label, 'pill bottle' for pill_bottle,
    'hand 1' for hand:1; '' for None."""
    if not name:
        return ''
    if name in labels:
        return labels[name]
    if name.startswith(PREFIX):
        return thing_label(name)
    if name.startswith('hand:'):
        return 'hand ' + name.split(':', 1)[1]
    return name.replace('_', ' ')
