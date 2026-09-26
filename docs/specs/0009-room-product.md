# 0009: Room product: room memory and room pointing from one fixed camera

Status: proposed Sat 26 Sep 2026, revision 2. Design agreed in chat with Anshul (sections 1-5), revised after
three external critiques, then after a code-verified spec review (16 findings; the reviewer's code claims were
re-checked). Four items stay open as milestone blockers, listed in Open blockers below. Nothing here is built. This spec supersedes the ordering of spec 0005 (tags
first, laser-guided placement) and spec 0006 (dot map first) and reuses their parts: 0006's `aim_px`,
`find_dot_px`, gates and zones, 0005's planner ideas and verification. Anshul dropped the Sat 6 PM freeze for
this work; the freeze line in `AGENTS.md` needs updating to match.

## Problem

Ask the Room only knows one table. Positions are table centimetres from one AprilTag homography
(`core/table.py`), the laser fit maps table cm to pulses (`act/laser.py`), and `detect_filter.table_margin_cm`
drops detections off the table. The Brio is mounted high on an angle and sees much more than the table, but
`scripts/camera_setup.sh` sets digital zoom 160 (a crop of its 4K sensor), capture is 1280x720, and the detector
shrinks every frame to 640 px (`imgsz: 640`). Most of the room is thrown away.

## Goal

Two capabilities that ship separately:

1. **Room memory (M0-M3).** "Where are my keys?" gets "Your keys are on the bookshelf. They appeared there
   3 minutes ago." or "I last saw your keys on the bookshelf at 3:12. I can't see them there now." No laser.
   This is the product and stands on its own. M0 is the thin slice for Sunday's judging.
2. **Room pointing (M4-M5).** The laser indicates a room object's location, confirmed by the camera. An
   experiment gated on measurements and on a documented laser module. "Pointable zones: none" is an acceptable
   outcome.

The demo moment: a judge takes a prop from the table, puts it on the shelf or couch, walks off, asks, and hears
where it is (and, only if pointing passed its gates, sees the dot on it).

## Non-goals (v1)

- Hidden, covered or hand-carried objects in room zones. Table rules (covers, containers, holds, parent chains)
  stay table-only. Exception: chain traversal in `place()` (a container seen in a room zone carries its
  recorded children), from M2 on.
- A moving camera or a second camera.
- Walls and anything above `laser_max_height_cm`.
- Inferring object height, or claiming the dot lands on the object's surface.
- Automatic open-loop room aiming. An unconfirmed aim turns the laser off.
- "You put it there" for room objects. Room code does not attribute placements to a person.

## Setup (facts as of Sat 26 Sep)

- Logitech Brio, fixed, high on an angle; at zoom 100 it sees the table and the room zones. It does not move.
  Exposure is locked for the table (16.6 ms, gain 80); far shelves will be darker (see Risks).
- Pan-tilt laser head mounted high, pivot within about 10 cm of the lens, mechanically able to rotate fully.
- Green laser module, power and class unknown (see Safety: documented before it is powered).
- Jetson Orin Nano 8 GB, JetPack 6.2, Python 3.10; perception at `perception_max_fps: 15`; local config runs
  `proposals.kind: change` (no YOLOE loaded).

## M0: thin slice for Sunday (props only)

Everything later milestones add is left out unless the demo moment needs it.

**Capture.** Brio at 1920x1080 MJPG, zoom 100 (`scripts/camera_setup.sh` zoom argument). No 4K, no JPEG ring.
A wrapper in `core/room.py`, `TableView(frames, rect)`, implements the `FrameSource` API over the existing
`FrameBuffer`: `latest()` and `at(t)` return the **table view** (the zoom-160 region cropped from the 1080p frame
and resized to 1280x720), `latest_full()` returns the full frame. `main.py` wraps the `FrameBuffer` in it when
`room_memory.enabled`, so the 16 `latest()` callers (dashboard, visual Q&A, laser, calibrate, demo_check) keep
getting 1280x720 table frames, and `core/capture.py` (owned) doesn't change. `FrameBuffer(camera, ring_s=1.0)`
when room memory is on: the default 10 s ring of decoded 1080p frames would be about 1.8 GB (only `at(t)` for
laser latency reads the ring, and it needs well under a second). `table_view_rect` is measured once by
aligning a 1080p zoom-100 frame to a zoom-160 reference frame (ECC), stored in `config.local.yaml`.

The table view at 1080p is about 1200 px of output width resized to 1280. That is a sampling estimate only: it
says nothing about how the camera's own processing at zoom 160 compares, so the evidence is the ECC alignment
and D17 below, not this number. D17 (teach, hide, move box, ask, at least 4/5) must pass on the table view before room
memory is turned on for the demo; if it doesn't, the camera goes back to zoom 160 and M0 is off.

**Zones.** `python -m core.room --zone bookshelf --say "the bookshelf" --poly x,y x,y x,y ...` writes
`room_zones.json` (gitignored): polygons in full-frame px, spoken names, and the capture settings they were
drawn at. (The existing `act/room_map --zone` needs a laser-sweep map first, so it is not used here.)
`--show` saves the full frame with zones drawn, for checking.

**Detection.** Every `room_every_n` (5) perception frames, one zone is processed, round-robin: its bounding box
is cropped from the full frame at native resolution (resized down only if its long side exceeds 1280) and the
already-loaded known-prop model runs on it, sharing the engine instance (no second load). `Detector.detect`
is not used for this: it requires a table calibration and converts to table cm; the room pass calls the model
backend directly and keeps `box_px` in full-frame pixels. Detections whose centre is outside the zone polygon,
or inside the table-view rectangle, are dropped. Blocker evidence: hand boxes from the same model, plus, when
YOLOE is loaded, its person boxes.

**Association (props only; the transition rules are in section 3).** A zone track of class C confirmed on 2
consecutive visits of its zone **acquires** prop C only if C has an unconsumed table departure (EXITED_VIEW,
LOST_TRACK, or lost from a hand, emitted while `zone == 'table'`) within `handoff_s`, and the track was **first
seen after** that departure; acquisition consumes the departure. Otherwise it is a conflict sighting on the
dashboard and never upgrades. **Confirmed** table presence of C (the table's presence flip, not a single
detection) triggers return and drops the room association.

**M0 is a controlled demo.** One instance per prop class is assumed: a different wallet that first appears in a
zone after your wallet leaves the table inherits your wallet's identity. M0 demos keep one of each prop in the
room, and the decoy tests below show the rules hold, not that the rig recognises instances.

**Absence.** After 3 valid visits of the zone with no matching detection (a visit is invalid if a hand or
person box, or a frame-difference blob larger than the object, covers the spot), C becomes UNKNOWN with zone and
last-seen kept, and LOST_TRACK is emitted. Only for an entity whose `zone` is that zone.

**State and answers.** Room zone, box and room timestamps live in a side dict in `core/room.py`, set through
`World.observe_room()` (section 3). `place()` and the four non-container templates of section 4 (fresh arrival
observed, not fresh, UNKNOWN, conflict). No `room_sightings` table, no unnamed-thing handoff, no backgrounds, no
zone editor, no pointing: those are M2-M5.

**M0 done when:** the M0 unit tests below pass, D17 passes on the table view, and on the rig: 5 of 5 scripted
demo-moment runs (a prop from the table to a zone, walk off, ask) give the right zone, and the table-return and
decoy tests below give no wrong answer.

## Design (M1-M5)

### 1. Capture and views (M1)

M1 decides the capture size by measurement. **Baseline 1920x1080** (M0). **Stretch 3840x2160**, only if full 4K
decode passes: a full 4K JPEG must be decoded every frame for the table view (`cv2.imdecode` cannot decode a
region), which on the Orin's CPU is likely 80-150 ms, below the table's 12 fps before any detector runs. M1
measures CPU decode and the Orin's JPEG engine through GStreamer (probably not in the pip OpenCV of the
Ultralytics container; check). 4K needs decode p95 under 25 ms. If 4K passes: `FrameBuffer` keeps undecoded
JPEG bytes (`CAP_PROP_CONVERT_RGB = False`, a `core/capture.py` change for its owner) with `ring_s` at most 1 s,
and decodes on demand.

**Views.** `latest()` and `at(t)` stay the table view for every existing caller. Room code uses
`view('full')` or `view(zone)`. Only room code and the room-pointing loop switch; the table laser path keeps the
table view. The real 1280x720 assumptions to keep satisfied are `core/capture.py` `W, H`, `config.yaml`
`frame_size_px`, and `table_cal.json` (`core/detect.py:283` is only the report fallback `_FlatTable`).

**Room tiles (M1, replacing M0's per-zone crops once measured).** 640x640 tiles at native capture resolution
covering the zones, overlapping by 64 px, `tiles_per_frame` (1) per perception frame, round-robin. The number of
tiles is capped so every tile is revisited within the target: `max_tiles = room_revisit_s (3) x 15 fps x
tiles_per_frame`. A box cut by a tile edge is merged with the neighbour's box; a merged box's visit is valid
only when both tiles were visited within one revisit period. **Tracks are per zone in full-frame coordinates,
not per tile:** each tile's detections are mapped to full-frame pixels and de-duplicated across overlapping
tiles (class-agnostic NMS at IoU 0.5 plus the edge merge) before they reach `RoomTracker`, so one object seen
by two tiles is one observation and one track.

**View version.** Hash of device, capture size, zoom, focus and `table_view_rect` only. Table calibration is
not in it: a spoken RECAL (`main.py:186`) rewrites `table_cal.json` and must not wipe room memory. A mismatch
pauses room memory and says so on the dashboard.

### 2. Calibration, split by capability

**Room memory (M0/M2):** view version, zone polygons, spoken names; from M2 an empty-room background per tile
(median of about 30 frames). No intrinsics, tags, planes or depth. A couch works as a zone.

**Room pointing (M5), per pointable zone:**
- Lens intrinsics: ChArUco board on an iPad or laptop screen, about 20 views, rational model, RMS below 0.5 px
  (spec 0005 step 0). Valid only at the locked capture size, zoom and focus.
- A floor plane: one AprilTag on the floor, required for any pointing (the height rule needs it).
- A support plane per pointable zone: one AprilTag (16 cm; bigger prints for far zones), pose by `solvePnP`.
  The dashboard shows where to put each tag; laser-guided placement (0005) returns only after M4.
- The head's offset from the lens, tape-measured (3 numbers, cm).
- The aim model fit and held-out validation (section 5).
- Optional: an empty-room depth map (Depth Pro or Depth Anything V2 Metric-Indoor, run once on the Mac, scaled
  to the tag points), only for zones without a tag plane. A zone with neither is not pointable.
- Optional: the 0006 dot sweep, only after M4.

### 3. Room sightings and identity (M0 props, M3 full)

**Observation record.** `RoomObservation`: frame id, capture time (monotonic and wall), view version, zone,
tile id (M1), `box_px` (full frame), class hypotheses with scores, crop, optional DINOv2 embedding, blocker boxes,
source (`props` or `yoloe`). Class hypotheses from YOLOE need a change in `core/proposals.py`: `YOLOEProposer`
drops class names today; they are kept as uncertain evidence on the room path only. An observation older than
the entity's latest applied one is rejected. All world updates happen under the world lock.

**`RoomTracker`** (`core/room.py`) keeps short per-zone (M0) or per-tile (M1) tracks `r:N`. A track is
**confirmed** after `confirm_visits` (2) consecutive valid visits with a match (IoU at least 0.3, or centre within
0.5 box diagonals). Each track records `first_seen` (capture time). Confirmed tracks go to association.

**`World.observe_room(name, obs, kind)`** is a new method; it does not wrap `observe_external()` (which writes
`pos_cm`, clears the parent and sets confidence 1.0). It sets `zone`, status VISIBLE (acquire) or UNKNOWN
(absent), clears `pos_cm` and `box_cm`, records the room side state (zone, `box_px`, `room_seen_t`,
`room_seen_wall`, `arrived_wall`, `arrival_observed`, `assoc_track`), and emits FOUND or LOST_TRACK with the
observation's capture time and its crop as the snapshot, not the last table frame (`_emit` today stamps
`self._now/_wall` and snapshots `self._frame`, world.py:600-604; it gains optional `t`, `wall` and `img`).

**Departures.** A departure is an event that took entity E off the table: EXITED_VIEW, LOST_TRACK, or loss from
a hand (`held_timeout_s`, `hand_lost_s`), **emitted while `E.zone == 'table'`**. Room absence (LOST_TRACK in a
room zone) is not a departure. The table decides a departure up to ~2 s after the object left, so its time is
the last table evidence (the holding hand's last sighting, or the object's own last sighting), at most 5 s
before the event (revised after the M0 code review: a zone next to the table can see the object first).

**Identity: known props.** One instance per class. **This is a demo assumption, not identity evidence**; the
dashboard and this spec say so. For a confirmed track whose class-C score is at least `room_prop_conf` (0.45,
tuned from M0 replays) on both confirming visits:

| Prop C is... | Then |
|---|---|
| VISIBLE on the table, table observation within `table_fresh_s` (2 s) | Not C. Unnamed-thing candidate (M3) or ignored (M0). |
| Believed hidden on the table (UNDER, INSIDE, HELD) | Keep the belief. Conflict sighting. |
| VISIBLE in a room zone, and this is a **different** track from its `assoc_track` | Keep it. Conflict sighting; never re-association. (Its own track only refreshes.) |
| Has an unconsumed table departure within `handoff_s` (120 s) **and** track `first_seen` after it | Acquire: FOUND, `arrival_observed: true`; the departure is consumed. |
| Departed, but the track was first seen before the departure | Conflict sighting (it was already there: a decoy). |
| UNKNOWN in a room zone, same zone, track at its last spot (IoU at least 0.3), first seen after the absence | Reacquire: FOUND, same zone. |
| Anything else (UNKNOWN on the table, startup, departure too old) | Conflict sighting. |

A conflict sighting **never upgrades** by repetition or by a later departure. It resolves only through table
evidence (the existing table rules revealing the belief was wrong, after which a *new* track must form) or the
user.

**Transitions.** Every change to a room entity is one of these operations, with its precondition:

| Operation | Precondition | Effect |
|---|---|---|
| **Acquire** | Entity has an unconsumed table departure within `handoff_s`; track first seen after it | zone = track's zone, VISIBLE, `assoc_track` = track, departure consumed, FOUND |
| **Refresh** | Observation matches the entity's `assoc_track` (same zone, same track) | room timestamps updated; no event (a `confirm` row from M3) |
| **Absence** | `assoc_track` matches; `absent_visits` valid empty visits **and** `absent_min_s` (3 s) since its last match (a zone is visited ~3 times a second) | UNKNOWN, zone kept, LOST_TRACK |
| **Reacquire** | Entity in room zone Z whose own track is missing (valid empty visits) or absent; a track in Z first seen after the own track's last match (anywhere in Z: moved along the shelf; revised after the M0 code review) | VISIBLE, `assoc_track` = new track, FOUND |
| **Return** | Confirmed table presence (`_observe()` after the presence flip, not a single detection) | room side state cleared, track released, zone 'table', room-to-table MOVED |
| **Conflict** | Any other confirmed track of the entity's class, including a *different* track while the entity is VISIBLE in a room zone | conflict sighting only; the entity is unchanged; the track never upgrades |

A departure authorises at most one acquisition. Observations from the entity's own `assoc_track` are always
Refresh, never Conflict.

**Table return detail.** `_observe()` resets `zone` to 'table' (world.py:255-258) but emits no event when the
previous status was VISIBLE (world.py:267), so Return adds the MOVED event and notifies the tracker. A single
table detection without the presence flip only writes `pos_cm`/`last_seen` through `_debounce` and does not
trigger Return (see Freshness).

**Identity: unnamed things (M3).** A departure makes a thing a **candidate**. Association is one-to-one across
all departures and new room tracks in the window. Automatic association requires all of: exactly one open
departure and exactly one new confirmed room track in the window; the track first seen after the departure;
appearance similarity at least `reid_room_thr` (DINOv2, measured on this camera's table view vs room crops; with
the embedder off, never automatic); no contradicting observation. A category match alone only records
`maybe_same_as`.

**Births (M3).** A confirmed track becomes a new `thing:N` only with a persistent proposal and change evidence
(at least `birth_change_frac` (0.3) of the box's pixels differ from the tile background), no blocker box over it,
inside a room zone. **M3 does not discover unnamed objects that were already in the room at calibration.** That
needs room teaching or a full-frame Grok look, both open questions; until one is specced, no milestone claims it.

**Background update (M3).** Per tile, a running average (`bg_alpha` 0.02) on pixels outside every tracked,
blocker and recent-change box, only after `bg_stable_visits` (5) stable visits. Tracked objects are never
absorbed. A view version change rebuilds backgrounds.

**Absence.** A **valid visit** for an entity: the zone (M0) or tile (M1) was processed from a fresh frame, the
entity's box is fully inside it (or, for a merged box, both tiles visited within one period), no blocker box and
no other track overlaps it by more than 30%, and mean luminance in the box is within `[lum_lo, lum_hi]`
([25, 235]). Time without valid visits counts for nothing. After `absent_visits` (3) valid visits without a match,
status UNKNOWN (zone and room timestamps kept), LOST_TRACK. UNKNOWN in a room zone means "not seen there on recent
valid views", never "gone".

**Freshness uses room timestamps only.** `_debounce` writes `pos_cm`, `box_cm` and `last_seen` on every
detection above threshold, before the presence flip (world.py:243-246), so one table flicker would refresh
`ent.last_seen` for an entity whose zone is 'bookshelf'. Room freshness and every room answer read
`room_seen_t` / `room_seen_wall`, never `ent.last_seen`. A room entity is **currently verified** when its last
match is within `fresh_s` (10 s) and it has not had `fresh_visits` (2) valid misses lasting `stale_min_s` (2 s).

**`world.place(name) -> Place`** (dataclass in `core/room.py`; `core/types.py` does not change): `kind`
('table' | 'room' | 'none'), `zone`, `pos_cm` (table only), `box_px` and `point_m` (room; `point_m` only with a
plane), `chain`, `via`, `observed_directly`, `fresh`, `arrived_wall`, `last_seen_wall` (room timestamp for room
places), `arrival_observed`, `status`, `tentative`, `conflicts`. `place()` walks the parent chain (from M2): keys
INSIDE box, box on the shelf gives `kind='room'`, `zone='bookshelf'`, `via='box'`, `observed_directly=False`, with
the box's freshness. `place` joins `get / resolve / history / state_json` in `WorldAPI` (an `AGENTS.md` update);
`FakeWorld` gets a `place()` that returns table places only; `_where` guards with `hasattr(world, "place")`, as
it already does for `open_container_of`.

**Legacy readers stay safe.** `resolve()` returns `(None, chain)` when the outermost entity of the chain has
`zone != 'table'`, keyed on the zone, not on `pos_cm` (which a flicker can repopulate). `_where` calls `place()`
first and returns before any direct `e.pos_cm` read for room places (`area(e.pos_cm)` near line 206, `_clause`,
`_nearest`, and the HELD branch at line 224). `world.py:133` already skips table absence for non-table zones.

**Persistence (M3).** A `room_sightings` table in the EventLog (like `grok_checks`): entity, zone, kind
(`acquire`, `confirm`, `absent`, `conflict`, `tentative`, `return`), capture time, wall time, frame id, view
version, `box_px`, `point_m`, evidence, association confidence, and a crop for `acquire` and `conflict` rows.
`confirm` rows are at most one per entity per minute. `Event` stays unchanged (it has no zone): FOUND on
acquisition and reacquisition only, LOST_TRACK on absence, each linked to its row. History answers read this
table. In M0 the side dict holds only the current state.

### 4. Answers (M0 first four rows, M3 the rest)

`_where` calls `world.place(obj)` first; `kind='table'` goes to today's templates unchanged.

| Place | Spoken |
|---|---|
| Fresh, arrival observed | "Your keys are on the bookshelf. They appeared there 3 minutes ago." |
| Not fresh, not absent | "I last saw your keys on the bookshelf at 3:12." |
| UNKNOWN after valid absent visits | "I last saw your keys on the bookshelf at 3:12. I can't see them there now." |
| Conflict (belief elsewhere, sighting here) | "I think they're under the notebook, but I also see keys on the bookshelf." |
| Fresh, arrival not observed (M3) | "Your keys are on the bookshelf. I've seen them there since 3:12." |
| Via a container, fresh (M3) | "They're in the box. The box is on the bookshelf." |
| Via a container, not fresh (M3) | "They're in the box, which I last saw on the bookshelf at 3:12." |
| Tentative, `maybe_same_as` (M3) | "That might be your mug, on the couch." |

The tentative case names the candidate; pointing targets the candidate's own id and never binds it. Pill-bottle
wording rules apply. Grok's world state (`voice/llm`) includes room places (zone, freshness, tentative). The
phone gets the zone string in the state it already receives.

### 5. Room pointing (M4-M5, gated)

**Dot colour.** The dot finder is red-only today: `dot_score` is the red rise minus half the green/blue rise, and
`dot_px_hsv` looks for red hues (act/laser.py), so a green dot scores negative and is never found. M4 adds
`laser.color: red | green` (config, default red): for green, the score is the green rise minus half the red/blue
rise and the HSV fallback uses green hues (about 40-85 in OpenCV). This also applies to the table laser if it
uses the green module.

**Gate to start.** A documented laser module (Safety) and a passed M4 dot test. Until then `room.enabled: false`
and room answers highlight the box on the dashboard and phone.

**Actions carry the entity, not pixels.** Room answers use `Answer(point_at=<entity>, action='room')`; `aim`
resolves the pixel from `place()` at actuation time. The 0006 form `room:u,v,x1,y1,x2,y2` bakes pixels in at
compose time and stays only for Grok points at untracked spots (off by default, spec 0006).

**Target.** The camera pixel at the object's box centre. With the head near the lens, a beam along nearly the
camera's ray lands on what the camera sees at that pixel. The zone's plane (or depth map) gives a depth Z for
parallax compensation only. Success is defined in the image: the dot confirmed inside the object's box shrunk
by 20%. The spoken answer never claims the dot is on the object.

**Parallax.** With a lateral offset b, an angular map calibrated at distance Z0 misses by about
b * |1 - Z/Z0| at distance Z: b = 10 cm, calibrated at 3 m, a target at 1 m misses by about 6.7 cm. So the model
uses the tape-measured offset and the target's Z.

**Aim model (a candidate, validated by measurement).** Pixel to ray (intrinsics), scaled to Z, gives P in camera
coordinates; head coordinates P_h = R (P - t), t the measured offset. The head's pan axis is outer and its tilt
axis turns with pan, so a direction is R · Rpan(pan) · Rtilt(tilt). A pan zero offset is absorbed exactly by R
(Rpan(d) · Rpan(p) = Rpan(p + d)); a tilt zero offset is not (Rtilt(d) sits between Rpan and Rtilt and does not
commute with Rpan), so it stays a free parameter. Pulses = centre + gain * (angle + offset). Fitted: R (3), the
two gains (2), the tilt offset (1): **6 parameters**. A quadratic term per axis is added only if held-out
residuals show curvature after the tilt offset is fitted.

**Fit data.** Dashboard jog mode (arrow keys; laser on only while jogging; dwell cap) nudges the dot onto tag
centres and marked points by eye. At least 12 fit points over near (at most 1.5 m), middle and far targets and
zone edges; at least 4 approached from both sides. At least 4 held-out points, including a near one and a zone
edge.

**Search region.** Radius 1.5 x the p95 open-loop error on held-out points in the target's depth band (px).

**Confirmation loop.** `aim_px`-style, with one change: **the laser is off during every servo move.** Today
`find_dot_px` ends with the laser on ("Laser ends on") and `aim_px` then calls `move_to` with it still on
(act/laser.py, `aim_px` loop); the room loop calls `act.laser(False)` before every `move_to`. Then on/off pairs
in the search region, a step through the model's Jacobian with the Broyden update, at most `max_tries`. Only
`in_box` or `within_tol` count; anything else turns the laser off and highlights the box.

**Timing.** `respond()` starts speech and aim in parallel (`main.py:214-220`), so:
- Before the answer text is composed: static checks (pointable zone, servo limits, height, a valid aim
  calibration for this view version). A static failure adds "I can't reach it with the pointer." to the text.
- Just before actuation: `place()` again (the entity may have moved or changed zone), then a forced visit of its
  zone or tile; if it is not seen within `aim_fresh_s` (1.0 s), abort.
- During illumination: the dynamic checks below. A dynamic failure turns the laser off and highlights; after the
  speech ends, one short follow-up at most ("I couldn't get the pointer on it.").

**Safety (all required for room pointing):**
- **Documented laser.** A module from a reputable supplier with a datasheet (power, wavelength, class, IR
  filtering) before any powered use, including calibration and M4. Unknown modules are not "assumed 3R": some
  green pointers exceed their label and leak invisible infrared. Class 2 is necessary, not a validation of the
  automated system. `room.laser_documented: false` blocks every room aim and the jog mode.
- **Downward-only servo limits** in config: only angles into drawn pointable zones and at least
  `min_depression_deg` (15) below horizontal; never horizontal or upward. Mechanical stops too, if available.
- **Height.** A target is refused when it is above `laser_max_height_cm` over the floor plane (the floor tag).
  A straight beam is never higher than its higher end; the head is high, so this limits the target, not the beam.
- **Blockers.** While the laser may be on, a full-frame person pass (YOLOE on a downscaled frame, about 5 fps;
  pointing therefore requires YOLOE loaded) runs. The laser is off if the last pass is older than
  `person_fresh_s` (0.5 s), or a person or hand box overlaps the target box grown by `blocker_grow_px` (40 px at
  1080p). This is a check, not proof of a clear beam: near the rig the beam and the camera's line of sight
  separate by up to about 10 cm. The 40 px margin is a starting value; M5 validates it with scripted walk-ins,
  including an arm reaching in near the rig.
- **Mounting.** Out of reach and away from walkways. Assumption to measure in M5, not a fact: the part of the
  beam outside the camera's view should be short (estimated under about 30 cm from the rig, from the head offset
  and the camera's view cone); M5 computes it from the measured head offset and intrinsics.
- **No-fire zones** for mirrors, TVs, windows and glossy surfaces. Laser off during servo moves. `room_dwell_s`
  cap. Hardware kill switch (demo_check check 8).

### 6. Dashboard and phone (M0-M3, not polish)

M0: the saved `--show` image and the room side state in `state_json`. M2-M3: a downscaled full-frame view with
zone polygons, tile outlines (the visited one highlighted), room tracks with their state (candidate, confirmed,
associated, conflict, tentative, UNKNOWN), and a zone editor (polygon, spoken name, pointable, no-fire). The phone
shows the zone string.

## Data

- `config.yaml`, new section `room_memory:` at the end: `enabled` (false), `capture_size` ([1920, 1080]),
  `room_every_n` (5), `confirm_visits` (2), `handoff_s` (120), `table_fresh_s` (2), `room_prop_conf` (0.45),
  `absent_visits` (3), `lum_lo` (25), `lum_hi` (235), `fresh_visits` (2), `fresh_s` (10), `ring_s` (1.0); from M1-M3
  `tile_px` (640), `tile_overlap_px` (64), `tiles_per_frame` (1), `room_revisit_s` (3), `reid_room_thr` (measured),
  `birth_change_frac` (0.3), `bg_alpha` (0.02), `bg_stable_visits` (5). Pointing keys join the existing `room:`
  section: `min_depression_deg` (15), `aim_fresh_s` (1.0), `person_fresh_s` (0.5), `blocker_grow_px` (40),
  `laser_documented` (false). All starting values are tuned from replays, not live.
- `config.local.yaml`: `room_memory.table_view_rect` (measured per rig).
- `room_zones.json` (gitignored): view version, zones (polygon in full-frame px, `say`, `pointable`, `no_fire`).
- `room_bg/` (gitignored, M3): per-tile backgrounds with their view version.
- `camera_intrinsics.json`, `room_map.json` (gitignored; pointing only): tag poses and planes, floor plane, head
  offset, aim model, fit and held-out residuals, search radii per depth band.
- EventLog: `room_sightings` table (M3).

## Changes by file

| File | Change | Sign-off |
|---|---|---|
| `core/room.py` (new) | `TableView`, zone CLI, `RoomObservation`, `RoomTracker`, association, `Place`; M3 tiles, backgrounds | us |
| `core/world.py`, `core/relations.py` | `observe_room`, `place()`, zone-keyed `resolve()` None, table-return drop + MOVED, `_emit` optional `t/wall/img` | world owner |
| `core/fakeworld.py` | `place()` (table places) | us |
| `main.py` | `TableView` wrap, `ring_s`, room pass every N frames, `action='room'`, aim timing split | us |
| `voice/answers.py` | `place()` branch first in `_where`, room templates | voice owner |
| `voice/llm.py` | room places in the world state | voice owner |
| `scripts/camera_setup.sh` | 1080p zoom 100 profile | us |
| `core/capture.py` | M1 only, if 4K: undecoded JPEG ring | capture owner |
| `core/proposals.py` | M3: keep YOLOE class hypotheses for the room path | proposals owner |
| `act/laser.py`, `act/room_aim.py` (new) | M5: aim model, jog mode, laser-off moves, confirmation | laser owner |
| `server/` | M2-M3 room view, zone editor; M5 jog UI | us |
| `demo_check.py` | room memory check (M0), room pointing check (M5) | us |
| `AGENTS.md`, `CONTEXT.md`, `docs/FEATURE_STATUS.md` | `place` in `WorldAPI`, freeze line, status | us |
| `core/types.py`, `core/detect.py` | no change | - |

## Milestones and gates

| # | What | Needs | Gate |
|---|---|---|---|
| M0 | Thin slice: 1080p, `TableView`, zone CLI, props-only association, absence, `place()`, four templates | Laptop tests, then rig | M0 done criteria |
| M1 | Decode measurement, 4K decision, tiles, view version | Jetson | M1 numbers |
| M2 | Backgrounds, dashboard room view, zone editor, container chains in `place()` | Rig | Zones redrawn in the editor; view version recorded |
| M3 | Unnamed handoff and births, `room_sightings`, remaining templates, Grok world state | M1, M2 | M3 numbers |
| M4 | Green-dot detection (`laser.color`), then the dot visibility test (search area, on/off pairs, 6 spots, two exposures) | Documented laser, room empty | Green dot found in the unit test; per-spot dot SNR and size; decides sweep vs jog-only |
| M5 | Intrinsics, floor and zone planes, aim fit, held-out validation, confirmation, safety | M2, M4, documented laser | M5 numbers, per zone |

Room memory ships whatever happens to M4-M5.

## Acceptance tests (no hardware, `.venv/bin/python -m pytest -q`)

M0:
- `TableView.latest()` returns 1280x720 cropped from a synthetic 1080p frame at the given rect; `latest_full()`
  returns the full frame; `at(t)` returns the table view.
- Zone CLI round trip; detections outside the polygon or inside the table rectangle are dropped.
- Demo moment: keys depart the table (EXITED_VIEW while zone 'table'), a new shelf track confirms: FOUND on the
  bookshelf, `pos_cm` None, `resolve()` None, the "appeared there" answer.
- **Table return:** keys associated to the shelf, then seen on the table: room side state dropped, MOVED emitted,
  three later empty shelf visits do not make them UNKNOWN, the table answer is used.
- **Decoy after room absence:** keys on the shelf go UNKNOWN (room LOST_TRACK); a new keys track on the couch within
  120 s is a conflict, not a reassociation.
- **HELD timeout decoy:** a keys-like shelf track already present (conflict) while the keys are HELD; the hold
  times out (LOST_TRACK on the table): the old track stays a conflict and never upgrades.
- Keys VISIBLE on the shelf and a second keys track on the couch: conflict, keys stay on the shelf.
- Keys VISIBLE on the table plus a shelf keys track: not the keys.
- Keys believed UNDER the notebook plus repeated shelf sightings: belief unchanged, conflict answer.
- **Flicker:** a room entity plus one table detection above threshold (no presence flip): `place()` freshness and
  the answer are unchanged, `resolve()` still None.
- Absence: three valid empty visits give UNKNOWN with zone kept; a visit with a hand box or a large change blob over
  the spot is invalid.
- Stale observation: the "last saw" wording; present tense only when fresh.
- A room place never reaches `aim_object` or any table-cm path.
- Events from `observe_room` carry the observation's capture time, not the table frame's.
- Out-of-order observations are rejected. `FakeWorld.place()` exists; `_where` works on a world without `place`.
- **Own track is not a conflict:** the associated shelf track keeps confirming: Refresh only, no conflict rows.
- **Departure consumed:** after one acquisition, a second new keys track elsewhere within `handoff_s` is a conflict.
- **Return needs the presence flip:** one table detection of a room entity does not trigger Return; a confirmed
  table presence does (MOVED emitted, room state cleared).
- RECAL does not change the view version.

M1-M3:
- Tile scheduling visits every tile within `room_revisit_s`; merged boxes need both tiles for a valid visit.
- Your mug departs, a visibly different mug appears on the shelf within 120 s: separate (embedding below
  threshold); embedder off: `maybe_same_as` only. Two departures, one track: no automatic association. One
  departure, two tracks: none.
- Calibration clutter and a lighting step with no persistent proposal produce no births.
- Box with keys: table to shelf to table round trip; losing the box's room observation gives "last saw the box".
- `room_sightings` rows for acquire, confirm (rate-limited), absent, conflict, return.

M4:
- With `laser.color: green`, a synthetic green dot on off/on frames is found within 1 px; a red dot is not found
  as green; a lighting change (all channels rise) scores below threshold. Red behaviour is unchanged.

M5:
- The 6-parameter model recovers a known head pose **with a nonzero tilt offset** from noisy synthetic samples at
  several depths; it refuses a single-plane or single-depth set.
- With b = 10 cm, the model with measured t and Z beats the pure angle map at 1 m.
- **The sim actuator log shows the laser off during every `move_to`** of the room loop.
- A missed dot turns the laser off and highlights; a person entering turns it off; a stale person pass (over
  0.5 s) turns it off.
- The entity moving between answer and actuation: `aim` re-reads `place()` and re-aims or aborts, never the old pixel.
- Static failures give "can't reach" with no servo calls; `laser_documented: false` blocks every room aim and jog.
- Downward-only limits reject upward and horizontal targets; no-fire zones and the height limit refuse; no floor
  plane means nothing is pointable.

## Rig measurements (pass criteria)

M0:
- D17 at least 4/5 on the table view (the existing check, run live).
- 5/5 scripted demo-moment runs with the right zone; table-return and decoy scripts give no wrong answer.
- Time from placement to FOUND at most 10 s; table view at least 12 fps with the room pass on.

M1:
- Table view parity: ECC residual at most 1 px against a zoom-160 reference. Scoring the table view against
  zoom 160 needs **paired recordings**: the same scripted scenes recorded once at zoom 160 and once at zoom 100,
  scored with `eval.score_clip`; the table view must score equal or better.
- Decode p95 under 25 ms at the chosen size; table view at least 12 fps with tiles on; room tile revisit p95 at
  most 3 s. A tile not visited within 3x its target period is marked stale and gives no absence evidence.
- RAM: today's table stack peak measured first; the combined peak stays under 6.5 GB of the Orin's ~7.4 GB usable.

M3 (detection, zone, identity and latency reported separately):
- Smoke: 10 placements across zones: at least 9 detected, every detected one in the right zone, zero wrong
  identities, time to FOUND p95 at most 10 s.
- Adversarial, zero wrong automatic merges: two different mugs, two simultaneous departures, placement into
  existing clutter, a lamp switched on and off, a person walking past.
- Idle 10 minutes: at most 1 false birth.
- Removal: UNKNOWN within 15 s when the view is clear; a blocked view never counts.

M5 (per pointable zone):
- Held-out targets near, middle, far and at zone edges: confirmation rate with every failure listed, confirmed
  error median at most 3 cm and p90 at most 6 cm. A zone below 80% confirmation is not pointable.
- Repeatability: each held-out point revisited 3 times from alternating sides; spread reported.
- Time to confirm p95 at most 4 s.
- Zero illuminations with a person box overlapping, in scripted walk-ins.

## Risks

- **The props detector has only seen the table from above.** Shelf views are oblique; M0's scripted runs and
  the M3 adversarial set measure it. Small props (keys, pills) at 3 m are about 19 px at 1080p; the demo uses
  larger props in room zones unless 4K passes M1.
- **Decode and RAM on the Orin Nano.** Fallback order: fewer room passes, then a smaller capture size, never a
  slower table view.
- **The Brio's zoom may not be a pure centre crop**, and its ISP scaling differs from our resize. The ECC
  alignment and D17 gate cover it.
- **Shared exposure.** Exposure is locked for the table; far shelves may be dark. `lum_lo/lum_hi` make those
  visits invalid instead of counting as absence; a lamp on the room zones helps.
- **Appearance thresholds across views.** Until `reid_room_thr` is measured, unnamed handoffs stay
  `maybe_same_as` and answers hedge.
- **Lighting drift** of backgrounds (M3): the persistent-proposal rule and gated updates are the defence.
- **Laser.** Unknown module; dot visibility on dark and glossy surfaces; servo slop (spec 0006 Limits).
- **Owned files.** `core/world.py`, `core/relations.py`, `voice/answers.py`, `core/proposals.py`,
  `core/capture.py` and `act/laser.py` need their owners.

## Open blockers (from the reviews, not yet resolved)

| Blocker | Resolve before |
|---|---|
| Green-dot detection (`laser.color`, section 5) is specified but not built | M4 |
| The 40 px blocker margin and the short out-of-view beam segment are assumptions | M5 (measured) |
| Discovering unnamed objects present at calibration needs room teaching or a full-frame Grok look | Any claim that M3 discovers them |
| One instance per prop class is a demo assumption, not instance recognition | Any demo beyond M0's controlled setup |

## Open questions

- Teaching in room zones: which object does "this is my X" bind to (proposal: the most recently confirmed room
  track, confirmed back by name)?
- Grok look (set-of-marks) on the full frame with room entities marked: M3 or later?
- Does the table demo use the same green laser? If so, the documented-module rule applies to the table too, and
  the table's closed-loop dot finding is red-only today (section 5, Dot colour): it cannot see a green dot.
- Floor zones that touch the table edge: where does the table rectangle end for handoff?
- Phone zone map: after M3.

## Rig findings (Sat 26 Sep evening, corner camera)

- The fine-tuned prop detector does not recognise props from the corner (the remote read as a wallet): on
  that rig prop labels are off (`conf_threshold` 0.9) and unnamed things named by Grok carry the demo.
- Unnamed things are handed from the table to a zone by departure + Grok name. Room crops go to Grok only
  while a handoff is possible, boxed in red with context, as "is it one of these?", and a yes counts only
  at confidence 0.7 with Grok's own description fitting.
- A hand placing an object makes the table re-birth it several times; departed candidates with the same
  Grok name are one object (the latest is handed over). Like one-per-class for props, this is a demo
  assumption, not identity evidence.
