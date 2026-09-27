# Tracker output contract: `askroom-track/1`

`eval/scorecard.py` scores identity on the guided room clips (`docs/runbooks/room-identity-clips.md`). It
never reads a tracker's internals: it reads this file format. Our own replay is written in it and read
back before scoring, so any other tracker that writes it (for example WS8's `permanence.mode: registry`)
goes through exactly the same scoring code. `eval/track.py` writes and reads it.

## File

One JSON Lines file per clip, conventionally `data/clips/<clip>/track-<tracker>.jsonl`. Every line is a JSON
object with a `type`. Times `t` are **seconds since the clip's first frame** (the clock of the clip's
`frames.json` `t` and `truth.json`). Positions are **table centimetres** (origin at the table tag or
marker 0, x right, y down), as everywhere in the repo.

### `header` (required, first line, once)

```json
{"type": "header", "format": "askroom-track/1", "clip": "room_still_1", "tracker": "registry",
 "detector": "free text", "overrides": ["permanence.mode=registry"], "view": [[0, 980, 817, 1440], [1280, 720]],
 "room_memory": true, "replay_wall_s": 212.4}
```

Only `format` is required. The others are shown on the scorecard.

### `frame` (required, one per processed frame, t ascending)

The tracker's complete belief after one processed frame. An entity missing from a frame is not believed
in at that time (merged away, forgotten).

```json
{"type": "frame", "t": 12.4,
 "entities": [
   {"id": "7", "state": "visible", "zone": "table", "table_cm": [41.2, 30.5]},
   {"id": "9", "state": "hidden", "hidden_in": "occluded", "zone": "table", "table_cm": [60.0, 30.0]},
   {"id": "3", "state": "visible", "zone": "couch", "place": "the couch"},
   {"id": "4", "state": "carried", "parent": "hand:1"}],
 "hands_cm": [[30.0, 20.0, 42.0, 31.0]]}
```

| Field | Meaning |
|---|---|
| `id` | the tracker's identity id (any string). The scorer shows it as `thing:<id>` (ids already starting with `thing:` stay) |
| `state` | `visible` (seen now, on the table or in a room zone), `hidden` (under / inside something or blocked from view), `carried` (in a hand), `last_seen` (not seen now, last position known), `unknown`, `gone` (left the view) |
| `zone` | `table`, a room zone key from the clip's `room_zones.json` (`couch`, `side_table`, `counter`), or null |
| `place` | optional spoken place (`the couch`, `on the table`); used to find the zone when `zone` is missing (matched to the clip's recorded zone names) |
| `table_cm` | `[x, y]` while the belief is on the table; null in a room zone |
| `hidden_in` | optional with `hidden`: `under`, `inside` or `occluded` |
| `parent` | optional: the cover / container id, or `hand:N` |
| `hands_cm` | optional per frame: hand boxes `[x1, y1, x2, y2]` in table cm (the scorer uses them to prefer the entity near a hand at a put-down cue) |

### `entity` (optional: the identity list)

```json
{"type": "entity", "id": "7", "kind": "thing", "born_t": 12.3, "merged_into": null,
 "names": ["brown wallet"], "guess": {"name": "brown wallet", "also": ["wallet"], "confidence": 0.8}}
```

A later line for the same id replaces the earlier one (write the final one at the end). `born_t` is when
the identity was created (default: its first `visible` frame). `merged_into` names the identity it was
folded into. `names` are the names it goes by (taught aliases, labels); `guess` a VLM guess. Names feed
the body-part / clothing count and the naming block.

### `event` (optional)

```json
{"type": "event", "t": 12.3, "id": "7", "event": "put_down", "table_cm": [41.2, 30.5], "from_cm": null, "parent": null}
```

`event` is one of `born`, `moved`, `put_down`, `found`, `picked_up`, `hidden`, `lost`, `gone`. An arrival
event (`moved`, `put_down`, `found`) helps bind an entity to a prop put down on a cue when the entity never
stopped being visible; without events, arrivals come from state changes to `visible`.

## Scoring a track

```bash
python -m eval.scorecard data/clips/room_*_1 --track track-registry.jsonl --json card-registry.json
python -m eval.scorecard data/clips/room_still_1 --track /path/to/any.jsonl      # one clip, any path
python -m eval.scorecard data/clips/room_*_1 --hands-off --yoloe-model models/yoloe-26s-seg-pf.pt \
    --save-track                                   # our replay, written as track-askroom.jsonl, then scored
```

The scorer binds each truth prop to the entity that arrived after the prop's put-down cue (so ids need not
match anything of ours) and scores entities per real object, phantom births per minute (still / people),
body-part things, position error, room handoffs and placements, false handoffs, identity through
occlusion and after return, removed-prop ghosts and the naming block (`eval/scorecard.py` docstring).
