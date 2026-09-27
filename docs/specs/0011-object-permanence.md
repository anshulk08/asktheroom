# 0011: Object permanence: a registry of the objects that matter, re-found anywhere in the room

Status: design and build started Sun 27 Sep 1 AM EDT (WS8, branch `ws/permanence`), approved by the user
("start now, behind a flag"). Everything here runs only with `permanence.mode: registry`; with the default
(`off`) the rig runs exactly as before. This spec amends the room identity model of 0009 section 3 and 0010:
with the flag on, there is no table-to-zone handoff.

## 1. Why the current design fails (not its tuning)

Measured on the rig on the night of Sep 26-27 (`/state`, logs, frames):

1. **Only the table is watched continuously.** The room is a few drawn zones visited in turn (`core/room.py`).
   An object put down outside a zone does not exist.
2. **A room place needs a handoff** (`core/room_world.py`): the object must first leave the table as a known
   entity, then one new track in one zone must arrive within 120 s and match by Grok name. Keys put straight on
   the couch, or carried past two zones, never become "your keys".
3. **Every YOLOE proposal can become a lasting thing.** Identity is box overlap plus labels, so one object
   becomes about 100 things (100 of 108 visible things at one spot), and people's feet, knees and sleeves are
   things too. The room assumes one of each prop.
4. **Appearance re-ID exists but is off** (`core/embed.py`, DINOv2-S/14, `reid.enabled: false`).
5. **People are only half handled**: a birth inside a person box is skipped (`core/things.py` `_may_create`),
   but body parts and clothing still become objects, and an object behind a person looks missing.

## 2. The design in one paragraph

Keep a **registry** of the objects people care about: the configured demo props and anything taught ("Room,
this is my charger"). Each has a few reference crops and their appearance embeddings. A **slow whole-frame
loop** (about 1 to 2 full sweeps a second, over tiles of the native 2560x1440 frame, with zoomed views of far
areas) looks only for registered objects: YOLOE proposes candidates, people are masked out, the embedding is
compared with the references, and a candidate at a new spot is confirmed by a closed Grok question (same object
as this reference, yes or no). Each registered object has one clear **state**: visible (where), hidden (a person
covers its last spot), carried (a hand or person touched it and it left), last seen (place, time), found again
(anywhere). Places are said in human terms. The fast table loop stays as it is for hands, hidden-under rules and
the laser.

## 3. Components (`core/permanence.py`)

### 3.1 Registry

`RegObject`: `name` (the world entity it answers for: a prop such as `remote`, or a taught `thing:N`), `bank`
(an `ExemplarBank` of up to `refs_max` embeddings, `dup_sim` 0.97), `refs` (small reference crops, for Grok and
the dashboard), `state`, `place`, `box_px` (full frame), `seen_wall`, `since_wall`, `arrived_wall`,
`contact_wall`.

References come from, in order:
1. **Enrollment files**: `permanence.refs_dir/<name>/*.jpg` (a crop per file). Written by
   `python -m core.permanence enroll --name remote --image full.jpg --box x1 y1 x2 y2` from a raw 1440p still.
2. **Teaching**: "this is my X" binds a thing (unchanged, `voice/teach.py`); the registry then adds X with the
   thing's box cut from the native full frame (`TableView.latest_full`) and registers it under the entity name.
3. **Confident re-finds**: a match at `sim >= learn_sim` adds a view at most once per `learn_every_s`, so the
   bank learns the room's lighting and angles (never from a Grok-only match).

Unnamed proposals never become registry objects. They stay the table world's business (and with
`hide_unnamed`, unnamed things that aren't visible now are left out of `/state`).

### 3.2 Two-speed watching

- **Fast loop, unchanged**: the table view at up to 15 fps (hands, the table's hidden-object rules, the laser).
- **Slow loop, new**: `Permanence.step(full_frame, hands_px)` runs on the perception thread (the models are
  shared and not thread-safe) and processes **one view per call** every `every_n` frames, so each call costs
  one YOLOE pass plus a batched embedding, never a stall of a whole sweep. Views: a `tiles` grid (default 3x2,
  15% overlap) over the native frame, plus `zoom` views (full-frame boxes of far areas, default the drawn
  zones' boxes, cut at native resolution). At 15 fps and `every_n` 1 with 8 views, a full sweep takes about
  0.5 s (2 Hz).
- Per view: YOLOE prompt-free gives boxes with class names. **People** (person/man/woman/child) are blockers;
  a candidate at least `person_inside` (0.6) inside a person box is dropped (a knee, a sock, a sleeve), and so
  is any box of a body or clothing class (`ignore_classes`). The rest are embedded in one batch.

### 3.3 Matching (thresholds measured, not guessed)

Measured on 4,000 room crops from the rig (DINOv2-S/14, background painted out of Grok's marked crops, a
hand-checked set of props): the same prop at the same spot scores a median of 0.85 (p25 0.69) against its
references; the same remote on another surface scored 0.09 to 0.34 against its couch references; clutter at one
spot (a crumpled bag, a tissue, the TV stand) reached 0.77 to 0.94; table-view references never reached 0.75
against room crops of the same prop. Appearance therefore **keeps** an object at its spot but can't **find** it
somewhere new. The rules:

| Situation | Decision |
|---|---|
| A candidate at the object's last spot (centre within `near_frac` of its diagonal) with `sim >= sim_accept` (0.55) | still there (refresh) |
| A candidate anywhere with `sim >= sim_accept_far` (0.88) | found (rare) |
| The object is carried or last seen | its **suspects** are candidates that first appeared (at that spot) after it went, less `arrival_s`: a carried object turns up somewhere new. Grok gets the reference crops and the view with the suspects in numbered red boxes and answers `{mark, confidence}` (0 = none): set-of-marks, the closed question 0010 found works. Yes at `verify_conf` (0.7): found, and that view's embedding joins the references, so the new spot refreshes without Grok from then on |
| The object was never seen, or nothing new arrived for `backstop_s`, or someone asked where it is | its suspects are the candidates most like it (`sim_backstop` 0.35 and up), asked the same way |

Candidates held by another registered object are never suspects; candidates inside a person box are asked about
last. One question in flight per object, one per `ask_every_s` per object, `verify_per_minute` in all. A Grok
find is hedged in answers ("I think") until the object is re-found by appearance alone. Offline, only the
appearance rules run.

### 3.4 States

| State | Enters when | Spoken (WHERE) |
|---|---|---|
| visible | matched on this sweep | "Your keys are on the couch. They appeared there 2 minutes ago." |
| hidden | its view was processed, it wasn't matched, and a person box covers its last spot by `hidden_overlap` | "Someone is in front of your keys; I last saw them on the couch." |
| carried | it wasn't matched on `miss_visits` valid views over `miss_s`, and a person or hand touched it within `contact_s` before (or it was hidden and the person left without it) | "Someone picked up your keys from the table at 3:12. I haven't seen where they went yet." |
| last seen | not matched as above, with no contact | "I last saw your keys on the couch at 3:12. I can't see them there now." |
| found again | matched after hidden, carried or last seen | (visible again; the event says found again) |

A **valid view** for an object is a processed view that contains its last box. Time without valid views
counts for nothing (as 0009's absence rule). Transitions write ordinary events to the EventLog (FOUND,
PICKED_UP, LOST_TRACK, MOVED), so HISTORY, HANDLED and "what changed" work unchanged.

### 3.5 Places in human terms

A full-frame point is said as: the table (`table_view_rect`, `table_say`), else a drawn zone (`room_zones.json`),
else an extra place from `permanence.places` (polygons with a `say`, for "the floor by the doorway", "the
kitchen counter"), else "near <zone>" within `near_px`, else "on the left/right side of the room". Table
centimetres stay for the table map and the laser only (computed by the table world, in WS6's viewer frame).

### 3.6 Backstop

Folded into 3.3: objects never seen, objects with no new arrivals for `backstop_s`, and objects someone just
asked about (a WHERE while it isn't visible calls `request`) get their best-matching candidates asked about.

### 3.7 Answers and /state (adapter, `Permanence.attach(world)`)

Like `auto_name.attach`, it wraps two World methods and leaves `world.py` alone:
- `world.place(name)`: for a registered object, the table world's place wins while the table has it (VISIBLE,
  HELD, UNDER or INSIDE within `table_fresh_s`), so the hidden-object rules and the laser keep working; otherwise
  a `RegPlace` (a `Place` with `state`) that the room templates speak (`voice/answers.py` `_where_room` gains the
  hidden and carried sentences).
- `world.state_json()`: registered entities carry `zone`, `status` and a `registry` dict; `state['room'][name]`
  mirrors the room place (so `/full.jpg` and the phone's zone string keep working); `state['permanence']` has
  the loop's health; with `hide_unnamed`, unnamed things that aren't visible are dropped.
- `world.get` is not wrapped (the World calls it internally).

## 4. Config (`permanence:`, a new section at the end of config.yaml)

`mode` (`off` | `registry`), `every_n`, `tiles`, `tile_overlap`, `zoom` (`zones` or a list of boxes),
`view_px`, `sim_accept`, `sim_accept_far`, `sim_backstop`, `learn_sim`, `learn_every_s`, `refs_max`, `refs_dir`,
`near_frac`, `person_inside`, `hidden_overlap`, `contact_s`, `miss_visits`, `miss_s`, `verify`,
`verify_conf`, `verify_per_minute`, `ask_every_s`, `arrival_s`, `marks`, `backstop_s`, `places`, `table_say`, `near_px`,
`table_fresh_s`, `hide_unnamed`. Thresholds come from offline measurement on rig crops and replays, never from
guessing during a live run (AGENTS.md). The embedder reuses `reid:` (model, input size, providers) with
`enabled` forced on for the registry only: `reid.enabled` stays false, so the table world is unchanged.

## 5. Wiring (`main.py`, additive)

With `mode: registry`: `open_frames` builds the `TableView` (needs `room_memory.capture_size` and
`table_view_rect`, as today) even when `room_memory.enabled` is false; the zone round-robin (`RoomMemory`) is not
built (the registry replaces the handoff); `perceive` calls the registry step after `world.update`, guarded
like `_room_step`; RESET clears states but keeps references.

## 6. Acceptance tests (no hardware)

1. Mode off: `make_permanence` returns None and the World's methods are untouched.
2. Registry: enrollment from a crop embeds and stores references; a matching candidate refreshes, a
   non-matching one never does; one-to-one assignment between two similar objects.
3. People: a candidate inside a person box is never matched; a person over the last spot gives hidden, not
   carried or last seen; the person leaving without the object gives carried.
4. States: visible → carried after contact and misses → found again elsewhere (FOUND, and the place says the new
   spot); visible → last seen without contact; misses on views that don't contain the object count for nothing.
5. Keys straight to the couch: an object first seen on the couch (never on the table) is visible there.
6. Grok: a carried object is found at a new arrival only when Grok picks its mark; a "none" or offline leaves it
   carried; candidates held by another object are never marked.
7. Answers: WHERE for each state through the adapter; the table world wins while it has the object; HISTORY and
   CHANGES read the registry's events.
8. `/state`: the registered entity has its zone and `registry` dict; unnamed things not visible are hidden.

## 7. Evaluation (rig clips, WS2's scorecard)

Replay raw 2560x1440 guided clips (WS7 records them; WS2 owns `eval/guided.py` and `eval/score_clip.py`) through
both trackers and score: one entity per object, no phantoms, the correct place after carries (including straight
to the couch and the floor), hidden vs gone behind a person. The registry mode must beat the current tracker
before it goes on the rig.

## 8. Open

- Things the table world births are still born (WS2 owns `core/things.py`); the registry only hides them. A hook
  to stop births in registry mode is a WS2 decision.
- "Asked about" registration (a WHERE for a name nobody taught, answered by a whole-room Grok look that then
  registers the object) is not in the first cut.
- The DINOv2 TensorRT engine is not built on the rig yet (a WS7 job); ONNX Runtime CPU on the Jetson is the
  fallback, and its speed decides `every_n`.
