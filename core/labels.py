"""What to call an entity when showing or saying it. A thing the person named is called by that name
('charger'), an unnamed one 'unnamed object 7'; the internal id thing:7 never reaches a person.
Read from state_json (the world emits label / aliases / maybe_same_as per thing, and merged ids), so
the overlay, Grok's world state and the dashboard (server/web/app.js mirrors this) say the same thing.
Configured objects keep their names; callers space them out as before.
"""
from __future__ import annotations

from typing import Optional

PREFIX = 'thing:'


def thing_label(name: str, label: Optional[str] = None) -> str:
    """'thing:7' -> the taught label, else 'unnamed object 7'. Any other name comes back as is."""
    if label:
        return str(label)
    if isinstance(name, str) and name.startswith(PREFIX):
        return f'unnamed object {name[len(PREFIX):]}'
    return name


def thing_labels(state: Optional[dict]) -> dict[str, str]:
    """Every thing in a state_json dict -> what to call it, including ids merged into another thing
    (their old events and parents then read as the survivor)."""
    state = state or {}
    out = {e['name']: thing_label(e['name'], e.get('label'))
           for e in state.get('entities') or [] if str(e.get('name', '')).startswith(PREFIX)}
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
