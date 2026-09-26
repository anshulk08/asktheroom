# 0009: Room product: room memory and room pointing from one fixed camera

Status: proposed Sat 26 Sep 2026. Design agreed in chat with Anshul (sections 1-5), then revised after three
external critiques; every code claim in them was checked against the repo. Nothing here is built. This spec
supersedes the ordering of spec 0005 (tags first, laser-guided placement) and spec 0006 (dot map first) and
reuses their parts: 0006's `aim_px`, `find_dot_px`, gates and zones, 0005's planner ideas and verification.
The 6 PM freeze no longer applies to this work (Anshul, Sat).

## Problem

Ask the Room only knows one table. Positions are table centimetres from one AprilTag homography
(`core/table.py`), the laser fit maps table cm to pulses (`act/laser.py`), and `detect_filter.table_margin_cm`
drops detections off the table. The Brio sees much more than the table: it is mounted high on an angle, but
`scripts/camera_setup.sh` sets digital zoom 160 (a crop of its 4K sensor), capture is 1280x720, and the detector
shrinks every frame to 640 px (`imgsz: 640`). Most of the room, and most of the sensor, is thrown away.

## Goal

Two capabilities that ship separately:

1. **Room memory (M1-M3).** "Where are my keys?" gets "Your keys are on the bookshelf. They appeared there
   3 minutes ago." or "I last saw your keys on the bookshelf at 3:12. I can't see them there now." No laser
   needed. This is the product; it stands on its own.
2. **Room pointing (M4-M5).** The laser indicates a room object's location, confirmed by the camera. This is an
   experiment gated on measurements and on a documented laser module. It may end as "pointable zones: none".

The demo moment: a judge puts a prop on the shelf or couch, walks off, asks, and hears where it is (and, if
pointing passed its gates, sees the dot on it).

## Non-goals (v1)

- Hidden, covered or hand-carried objects in room zones. Table rules (covers, containers, holds, parent
  chains) stay table-only. The one exception is chain traversal: a container observed in a room zone carries
  its recorded children (see `place()`).
- A moving camera or a second camera.
- Walls and anything above `laser_max_height_cm`.
- Inferring object height, or claiming the dot lands on the object's surface.
- Automatic open-loop room aiming. An unconfirmed aim turns the laser off.
- "You put it there" for room objects. The room pipeline does not attribute placements to a person.

## Setup (facts as of Sat 26 Sep)

- Logitech Brio, fixed, high on an angle; at zoom 100 it sees the table and the room zones. It does not move.
- Pan-tilt laser head mounted high, pivot within about 10 cm of the lens, mechanically able to rotate fully in
  both axes.
- Green laser module, power and class unknown (see Safety: it must be documented before it is powered).
- Jetson Orin Nano 8 GB, JetPack 6.2, Python 3.10; perception runs at `perception_max_fps: 15`.

## Design

### 1. Capture and views (M1)

The camera runs at zoom 100 and 4K MJPG. Two views are cut from each frame:

- **Table view.** Digital zoom 160 is a crop of the sensor, so the table pipeline can keep receiving the image
  it was tuned on: crop the zoom-160 region from the 4K frame and resize it to 1280x720. The crop rectangle is
  not assumed to be the exact centre: M1 measures it by aligning a 4K frame to a zoom-160 reference frame
  (ECC or feature match) and stores it as `room_memory.table_view_rect`. Everything downstream of the table
  view is unchanged: `table_cal.json`, `table_area.json`, the detector, hands, world rules, snapshots. Code
  that assumes 1280x720 (for example `core/detect.py:283`) keeps working because it only ever sees this view.
- **Room tiles.** 640x640 tiles at native 4K resolution, covering only the drawn room zones. One tile (two if
  M1 shows headroom) is processed per perception frame, round-robin, so every room tile is revisited every
  ~1-3 s. Detections are mapped back to 4K frame pixels. A detection whose centre is inside the table polygon
  is dropped from room tiles (the table view owns it). Boxes cut by a tile edge are merged with the
  neighbouring tile's box (tiles overlap by 64 px).

Room tiles run YOLOE prompt-free (proposals with class hypotheses kept as uncertain evidence, plus person boxes).
The known-prop detector runs on a tile only when that tile holds a confirmed but unassociated proposal, to keep
the per-frame budget. `capture.view(name)` returns the table view or a tile with its frame id and capture time.

**Buffers.** `FrameBuffer` keeps the compressed JPEG bytes of recent frames (about 1-2 MB each at 4K) and decodes
lazily, instead of holding decoded 4K frames (about 25 MB each). The ring length stays what the laser latency
lookup (`at(t)`) needs. Decode cost is measured in M1: CPU `cv2.imdecode`, the reduced-size decode flags for
the full-frame person pass, and the Orin's JPEG engine through GStreamer if it is usable from our container.
If the table view cannot hold 12 fps, the room tile rate drops before the table rate does.

**View version.** Every observation carries `view_version`: a hash of (device, resolution, zoom, focus,
`table_view_rect`, calibration file ids). A mismatch invalidates room calibration and backgrounds; the
dashboard says so and room memory pauses rather than using stale geometry.

### 2. Calibration, split by capability

**Room memory needs (M2):** the view version, zone polygons in 4K pixels, each zone's spoken name
(`say: "the bookshelf"`), and an empty-room background still per tile (a median of about 30 frames). No lens
intrinsics, no tags, no planes, no depth. A couch works as a zone.

**Room pointing needs (M5), per pointable zone:**

- Lens intrinsics: ChArUco board shown on an iPad or laptop screen, about 20 views, rational model,
  pass at RMS below 0.5 px (spec 0005 step 0). Only valid at the locked zoom and focus.
- A support plane: one AprilTag per pointable zone (16 cm; bigger prints for far zones), pose by `solvePnP`.
  The dashboard shows where to put each tag. Laser-guided placement (spec 0005) comes back only after the dot
  test (M4) passes.
- The head's offset from the lens, measured with a tape (3 numbers, cm).
- The aim model fit and its held-out validation (section 5).
- Optional: a depth map of the empty room (Depth Pro or Depth Anything V2 Metric-Indoor, run once on the Mac,
  scaled to the tag points). Only for pointable zones without a tag plane; a zone with neither is not pointable.
- Optional: the 0006 dot sweep, only after M4.

### 3. Room sightings and identity (M3)

**Observation record.** Each room detection is a `RoomObservation`: frame id, capture time, view version,
tile id, zone, `box_px` (4K), class hypotheses with scores, crop reference, optional DINOv2 embedding, person
boxes in the tile, source (`yoloe` or `props`). Observations older than the entity's latest applied one are
rejected. All world updates happen under the world lock.

**`core/room.py`: `RoomTracker`.** Keeps short tracks per tile (`r:N`). A track is **confirmed** after two
consecutive valid visits with a matching detection (IoU at least 0.3 or centre within 0.5 box diagonals).
Confirmed tracks go to association, then to `World.observe_room(...)`. This is a new method with its own
semantics; it does not wrap `observe_external()`, which writes `pos_cm`, clears the parent and sets confidence
1.0.

**Births (unnamed things).** A confirmed track becomes a new `thing:N` only if all hold:
- a persistent proposal (confirmed, above), and
- change evidence: at least `birth_change_frac` of the box's pixels differ from the tile background, and
- no person box covers it, and it is inside a room zone.

Change evidence prioritises discovery; it is not the only way in. Objects already present at calibration are
reachable by teaching ("this is my X", open question below) and, once the Grok look covers the full frame
(open question below), at question time.

**Background update.** Per tile, a slow running average (`bg_alpha`) applied only to pixels outside every
tracked box, person box and recent-change region, and only after the tile has been stable for `bg_stable_visits`
valid visits. Tracked objects are never absorbed. A view version change rebuilds backgrounds from scratch.

**Identity: known props.** The eight props are treated as one instance per class. **This is a demo
assumption, not identity evidence**, and the dashboard and this spec say so. For a confirmed track whose props
detector hypothesis for class C is at least `room_prop_conf` on two visits:

| The prop entity C is... | Then |
|---|---|
| VISIBLE on the table, table observation fresh (`table_fresh_s`) | Not C. The track may become an unnamed thing. |
| Believed hidden on the table (UNDER, INSIDE, HELD) | Keep the belief. Record a **conflict** sighting linked to C. No status change. |
| Left the table (EXITED_VIEW, LOST_TRACK, or lost from a hand) within `handoff_s` (120 s) | Associate: the departure is the causal link. FOUND with the zone. |
| UNKNOWN with no recent departure (for example at startup) | Associate with `arrival_observed: false`. |

Repetition does not upgrade a conflict: three shelf sightings can be three observations of the same wrong
object. A conflict resolves only through table evidence (the existing table rules revealing the belief was
wrong) or the user.

**Identity: handoff of unnamed things.** A recent departure only makes a thing a **candidate**. Association is
one-to-one across all departures and new room tracks in the window. Automatic association requires all of:
- exactly one open departure candidate and exactly one new confirmed room track in the window,
- appearance similarity at least `reid_room_thr` (DINOv2; the threshold is measured on this camera's table
  view vs room tile crops; with the embedder off there is no automatic association), and
- no contradicting observation (the candidate seen on the table after the track appeared).

A category match alone (YOLOE or Grok says "mug" for both) only records `maybe_same_as`. Never a silent merge.

**Absence.** A **valid visit** for an entity is a tile visit where: the frame is fresh, the entity's box region
is inside the tile (not cut by the tile edge), no person, hand or other track box overlaps it by more than
30%, and mean luminance in the box is inside `[lum_lo, lum_hi]`. Time without valid visits contributes nothing.
After `absent_visits` (3) valid visits without a matching detection, the entity's status becomes **UNKNOWN**
with its zone and `last_seen` kept, and a LOST_TRACK event is emitted. UNKNOWN in a room zone means "not seen
there on recent valid views", never "gone".

**Freshness.** A room entity is **currently verified** when its last matching observation is within
`fresh_visits` valid visits of its tile (`fresh_visits`: 2) and at most `fresh_s` (10 s) old. Answers use present tense only then.

**`world.place(name) -> Place`** (dataclass in `core/room.py`; `core/types.py` does not change):
`kind` ('table' | 'room' | 'none'), `zone`, `pos_cm` (table only), `box_px` and `point_m` (room; `point_m` only
when the zone has a plane), `chain`, `via` (the entity whose location is used), `observed_directly`, `fresh`,
`arrived_wall`, `last_seen_wall`, `arrival_observed`, `status`, `tentative` (candidate ids from
`maybe_same_as`), `conflicts` (conflict sightings). `place()` walks the parent chain itself: keys INSIDE box,
box seen on the shelf, gives `kind='room'`, `zone='bookshelf'`, `via='box'`, `observed_directly=False`, with the
box's freshness. The child inherits the container's location and its uncertainty.

**Legacy readers stay safe.** When an entity is associated to a room zone, `World` clears its `pos_cm` and
`box_cm` (the last table position moves to the side record for history). `resolve()` returns `(None, chain)`
when the outermost entity of the chain is in a room zone. Both are needed: `voice/answers._where` reads
`e.pos_cm` directly (line 224) as well as `resolve()`, and `main.aim` uses `resolve()`. When an entity is seen on
the table again, `_observe()` already resets `zone` to 'table'.

**Persistence.** A new `room_sightings` table in the EventLog (same pattern as `grok_checks`): entity, zone,
kind (`acquire`, `confirm`, `absent`, `conflict`, `tentative`), capture time, wall time, frame id, view
version, `box_px`, `point_m`, evidence (source, scores, similarity), association confidence, plus a tile crop
snapshot for `acquire` rows. `confirm` rows are rate-limited to one per entity per minute. Events stay as they
are (`Event` has no zone field): FOUND on acquisition and reacquisition only, LOST_TRACK on absence, each
linked to its `room_sightings` row. History answers ("bookshelf at 3:12, couch at 3:18") read this table.

### 4. Answers (M3)

`_where` calls `world.place(obj)` first. `kind='table'` goes to today's templates unchanged. Room templates
(one or two spoken sentences, no markdown):

| Place | Spoken |
|---|---|
| Fresh, arrival observed | "Your keys are on the bookshelf. They appeared there 3 minutes ago." |
| Fresh, arrival not observed | "Your keys are on the bookshelf. I've seen them there since 3:12." |
| Not fresh, not absent | "I last saw your keys on the bookshelf at 3:12." |
| UNKNOWN after valid absent visits | "I last saw your keys on the bookshelf at 3:12. I can't see them there now." |
| Via a container, fresh | "They're in the box. The box is on the bookshelf." |
| Via a container, not fresh | "They're in the box, which I last saw on the bookshelf at 3:12." |
| Conflict | "I think they're under the notebook, but I also see keys on the bookshelf." |
| Tentative (`maybe_same_as`) | "That might be your mug, on the couch." |

The tentative case names the candidate; pointing (if any) targets the candidate's own id and never binds it.
Pill-bottle wording rules still apply. Grok's world state (`voice/llm`) includes room places (zone, freshness,
tentative), so open answers can say "on the couch" with the same hedges. The phone gets the zone string in the
state it already receives; a zone map on the phone is later.

### 5. Room pointing (M4-M5, gated)

**Gate to start.** A documented laser module (see Safety) and a passed M4 dot test. Until then
`room.enabled: false` and room answers highlight the box on the dashboard and phone.

**Target.** The camera pixel at the object's box centre. Because the head sits near the lens, a beam along
nearly the camera's ray lands on whatever the camera sees at that pixel. The zone's plane (or the depth map)
gives a depth estimate Z for parallax compensation only. Success is defined in the image: the dot confirmed
inside the object's box shrunk by 20%. The spoken answer never claims the dot is on the object.

**Parallax.** With a lateral offset b, an angular map calibrated at distance Z0 misses by about
b * |1 - Z/Z0| at distance Z: with b = 10 cm, calibrated at 3 m, a target at 1 m misses by about 6.7 cm.
So the model uses the tape-measured offset and the target's Z, not a pure angle map.

**Aim model (a candidate, validated by measurement).** Pixel to ray (intrinsics), scaled to Z, gives P in
camera coordinates. Head coordinates P_h = R (P - t), with t the measured offset. Pan and tilt angles from P_h;
pulses = centre + gain * angle per axis. Gauge: each axis's zero is fixed at its centre pulse and the zero
offsets are absorbed into R, since a free pan offset and R's yaw (tilt offset and R's pitch) trade off exactly.
Fitted parameters: R (3) and the two gains (2). A quadratic term per axis (2 more) is added only if held-out
residuals show curvature.

**Fit data.** Dashboard jog mode (arrow keys, laser on only while jogging, dwell cap): a person nudges the dot
onto tag centres and marked points by eye. At least 12 fit points spread over near (at most 1.5 m), middle and far
targets and zone edges; at least 4 of them approached from both sides to expose backlash. At least 4 held-out
points, including a near one and a zone edge, never used in the fit.

**Search region.** The dot search area around the predicted pixel has a radius of 1.5 x the p95 open-loop error
measured on held-out points in that depth band (px), not the object's box.

**Confirmation loop.** `aim_px`-style: move with the laser off, then on/off pairs in the search region
(`find_dot_px`, 4K crop), step through the model's Jacobian with the Broyden update, at most `max_tries`. Only
`in_box` or `within_tol` count. Any other outcome turns the laser off and highlights the box. This keeps
`_aim_room`'s existing rule: never leave the dot somewhere it wasn't confirmed.

**Timing.** `respond()` starts speech and aim in parallel (`main.py:214-220`), so:
- Before the answer text is composed: static checks (a pointable zone, servo limits, the height limit, a valid
  aim calibration for this view version). A static failure adds "I can't reach it with the pointer." to the text.
- Just before actuation: revalidate the entity, its association, and freshness. Force a visit of the target's
  tile; if the object is not seen within `aim_fresh_s`, abort.
- During illumination: the dynamic checks below. A dynamic failure turns the laser off and highlights the box;
  after the speech ends, one short follow-up at most ("I couldn't get the pointer on it.").

**Safety (all required for room pointing):**
- **Documented laser.** A module from a reputable supplier with a datasheet (power, wavelength, class, IR
  filtering) before any powered use, including calibration and the M4 test. Unknown modules are not "assumed
  3R": some green pointers exceed their label and leak invisible infrared, so apparent brightness says nothing.
  Class 2 is necessary, not a validation of the automated system.
- **Downward-only servo limits** in config: only angles that point into drawn pointable zones, at least
  `min_depression_deg` below horizontal. Never horizontal or upward, whatever the head can do mechanically.
  Mechanical stops too, if the head has any.
- **Height.** A target is refused when its point is above `laser_max_height_cm` over the floor plane. A straight
  beam is never higher than its higher end, and the head is high, so this limits the target, not the beam.
- **Blockers.** A fresh full-frame person pass (YOLOE on a downscaled frame, about 5 fps) runs whenever the
  laser may be on. The laser is off if the last pass is older than 0.5 s, or any person or hand box overlaps the
  target box grown by `blocker_grow_px` (40 px at 4K). This is a check, not proof of a clear beam path: the head is not on the lens
  centre, so near the rig the beam and the camera's line of sight to the target separate by up to about 10 cm.
- **Mounting.** The rig is mounted out of reach and away from walkways, since the only part of the beam outside
  the camera's view is within about 30 cm of the rig.
- **No-fire zones** for mirrors, TVs, windows and glossy surfaces. Laser off during servo moves. `room_dwell_s`
  cap. Hardware kill switch (demo_check check 8).

### 6. Dashboard and phone (M2-M3, not polish)

Needed to debug room memory, so it comes with M2-M3: a downscaled full-frame view with zone polygons, tile
outlines (the tile being visited highlighted), room tracks and their state (candidate, confirmed, associated,
conflict, tentative, UNKNOWN), and a zone editor (draw polygon, set spoken name, mark pointable, mark no-fire).
The phone shows the zone string.

## Data

- `config.yaml`, new section `room_memory:` at the end: `enabled`, `capture_size`, `table_view_rect`,
  `tile_px`, `tile_overlap_px`, `tiles_per_frame`, `confirm_visits`, `handoff_s`, `table_fresh_s`,
  `room_prop_conf`, `reid_room_thr`, `birth_change_frac`, `bg_alpha`, `bg_stable_visits`, `absent_visits`,
  `lum_lo`, `lum_hi`, `fresh_visits`, `fresh_s`. Pointing keys join the existing `room:` section:
  `min_depression_deg`, `aim_fresh_s`, `person_fresh_s`, `blocker_grow_px`, `laser_documented` (false until a datasheet is on file).
- `room_zones.json` (gitignored): view version, zones (polygon in 4K px, `say`, `pointable`, `no_fire`),
  tile layout.
- `room_bg/` (gitignored): per-tile backgrounds with their view version.
- `camera_intrinsics.json`, `room_map.json` (gitignored; pointing only): tag poses and planes per zone, head
  offset, aim model parameters, fit and held-out residuals, search radii per depth band.
- EventLog: new `room_sightings` table.

## Changes by file

| File | Change | Sign-off |
|---|---|---|
| `core/capture.py` | 4K MJPG, JPEG ring with lazy decode, `view()`, table-view crop, view version | capture owner |
| `core/room.py` (new) | `RoomObservation`, tiles schedule, `RoomTracker`, backgrounds, association, `Place` | us |
| `core/world.py`, `core/relations.py` | `observe_room`, `place()`, room-zone `resolve()` returns None, clear `pos_cm` on room association | W |
| `core/events.py` | `room_sightings` table | us |
| `core/detect.py` | run on a tile image; YOLOE person boxes kept | P |
| `voice/answers.py` | room templates via `place()` | V |
| `voice/llm.py` | room places in the world state | V |
| `main.py` | wire room tiles into perception; aim timing split; room branch in `aim` | us |
| `act/laser.py`, `act/room_aim.py` (new) | aim model, jog mode, confirmation with search radius | H |
| `server/` | room view, zone editor, jog UI | us |
| `demo_check.py` | room memory and room pointing checks | us |
| `core/types.py` | no change | - |

## Milestones and gates

| # | What | Needs | Gate to pass |
|---|---|---|---|
| M1 | 4K capture, table view, tiles schedule, decode, RAM | Jetson, Brio | Table view parity and throughput numbers below |
| M2 | Zones, spoken names, backgrounds, dashboard room view and zone editor | Rig, ~20 min | Zones drawn; view version recorded |
| M3 | `RoomTracker`, `observe_room`, `place()`, `room_sightings`, templates | Synthetic tests, then M1+M2 | Unit tests; rig room-memory numbers below |
| M4 | Dot visibility test (4K search area, on/off pairs, 6 spots, two exposures, 720p comparison) | Documented laser, room empty | Per-spot dot SNR and size reported; decides sweep vs jog-only |
| M5 | Intrinsics, tag planes, aim fit, held-out validation, confirmation, safety gates | M2, M4, documented laser | Pointing numbers below, per zone |

M3's synthetic part can run in parallel with M1 and M2. Room memory (M1-M3) ships whatever happens to M4-M5.

## Acceptance tests (no hardware, `python -m pytest -q`)

Capture and views:
- The table view cut from a synthetic 4K frame matches the zoom-160 reference within 1 px (given the rect).
- Tile scheduling visits every zone tile within the expected number of frames; boxes cut by a tile edge merge.
- Detections inside the table polygon are dropped from room tiles.
- A view version mismatch pauses room memory.

Room memory:
- Keys on the shelf after leaving the table: FOUND with zone, `arrival_observed` true.
- Keys visible on the table plus "keys" on the shelf: the shelf track is not the keys.
- Keys believed UNDER the notebook plus repeated shelf sightings: belief unchanged, conflict recorded, conflict answer.
- Your mug leaves the table, a visibly different mug appears on the shelf within 120 s: separate (embedding
  below threshold). With the embedder off: `maybe_same_as` only.
- Two things leave the table at once, one room track appears: no automatic association.
- One departure, two new room tracks: no automatic association (one-to-one).
- Calibration clutter produces no births; a lighting step on a tile without a persistent proposal produces none.
- A person box over the spot is not a valid visit; three valid empty visits give UNKNOWN with zone kept.
- A stale observation gives the "last saw" wording; the present tense needs freshness.
- Box containing keys: table to shelf to table round trip; losing the box's room observation makes the keys'
  answer "last saw the box", never "I see the keys".
- A room entity never reaches `aim_object` or any table-cm path: `resolve()` gives None, `pos_cm` is None.
- Out-of-order observations are rejected.

Room pointing:
- The aim model recovers a known head pose from noisy synthetic samples at several depths and refuses a
  single-plane or single-depth sample set.
- Parallax: with b = 10 cm, the model with measured t and Z beats the pure angle map at 1 m.
- A missed dot turns the laser off and highlights; a person entering during illumination turns it off; a stale
  person pass (over 0.5 s) turns it off.
- The target moving between answer and actuation triggers revalidation (re-aim or abort, never the old pixel).
- Static failures produce "can't reach" in the text with no servo calls; `laser_documented: false` blocks every
  room aim.
- Downward-only limits reject upward and horizontal targets; no-fire zones and the height limit refuse.

## Rig measurements (pass criteria)

M1:
- Table view: ECC alignment residual at most 1 px against a zoom-160 reference; D17 and the eval clips scored
  from the table view equal or better than from zoom 160.
- Throughput with room tiles on: table view at least 12 fps; room tile revisit p95 at most 3 s; decode p50/p95
  reported. Stale-view handling is separate from the p95: a tile not visited within 3x its target period is
  marked stale on the dashboard and gives no absence evidence.
- RAM: measure today's table stack peak first; the combined peak must stay under 6.5 GB of the Orin's ~7.4 GB
  usable, measured before room tiles are enabled by default.

M3 (report detection, zone, identity and latency separately, never one combined score):
- Smoke: 10 placements of demo props across zones: detection rate, zone accuracy, identity accuracy, time to
  FOUND p50/p95.
- Adversarial, zero wrong automatic merges required: a same-category distractor (two different mugs), two
  simultaneous departures, placement into existing clutter, a lamp switched on and off, a person walking past.
- Idle 10 minutes: at most 1 false birth.
- Removal: UNKNOWN within 3 valid visits (seconds reported); a blocked view never counts.

M5 (per pointable zone):
- Held-out targets near, middle, far and at zone edges: confirmation rate with every failure listed, open-loop
  error distribution, confirmed error median at most 3 cm and p90 at most 6 cm. A zone below 80% confirmation
  is marked not pointable.
- Repeatability: each held-out point revisited 3 times from alternating sides; spread reported.
- Zero illuminations with a person box overlapping, in scripted walk-ins.
- Time to confirm p95 reported.

## Risks

- **The props detector has only seen the table from above.** Shelf views are oblique. Handoff and YOLOE
  class hypotheses cover part of it; the M3 adversarial set measures the rest.
- **4K decode and RAM on the Orin Nano.** The fallback order is fewer room tiles per frame, then a smaller capture
  size, never a slower table view.
- **The Brio's zoom may not be a pure centre crop**, and ISP scaling differs from our resize. M1 measures parity
  before anything else depends on it.
- **Appearance thresholds across views.** The table view and room tiles see objects at different angles; until
  `reid_room_thr` is measured, unnamed handoffs stay `maybe_same_as` and answers hedge.
- **Lighting.** The empty-room background drifts; the persistent-proposal rule and the gated background update
  are the defence. The Brio needs the lamp in dim rooms.
- **Laser.** Unknown module; dot visibility on dark and glossy surfaces; servo slop (spec 0006 Limits).
  Pointing may end with few or no pointable zones, and that is an acceptable outcome.
- **Owned files.** `core/world.py`, `core/detect.py`, `voice/answers.py` and `act/laser.py` need their owners.

## Open questions

- Teaching in room zones: which object does "this is my X" bind to when the teach square is on the table
  (proposal: the most recently confirmed room track, confirmed back by name)?
- Grok look (set-of-marks) on the full frame with room entities marked: in scope for M3 or later?
- Does the table demo use the same green laser? If so, the documented-module rule applies to the table too.
- Floor zones that touch the table edge: where does the table polygon end for handoff purposes?
- Phone zone map: after M3.
