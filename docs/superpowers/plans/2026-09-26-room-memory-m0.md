# Room memory M0 Implementation Plan

> **For agentic workers:** Tasks A-E run in parallel (subagents) against the shared contract
> `core/room_types.py`; task F (integration) runs after them. Steps use checkbox (`- [ ]`) syntax.

**Goal:** "Where are my keys?" answers "on the bookshelf" when a known prop is carried from the table to a
drawn room zone, from one fixed 1080p camera, with no laser.

**Architecture:** The Brio runs at 1920x1080, zoom 100. `TableView` cuts the zoom-160 region out of each
frame and resizes it to 1280x720, so the whole table pipeline is unchanged. Every `room_every_n` perception
frames `RoomMemory` crops one drawn zone at native resolution, runs the already-loaded prop model on it,
tracks per-zone detections (`RoomTracker`) and hands a `ZoneVisit` to `World.room_update`, which applies the
spec's transition table (Acquire / Refresh / Absence / Reacquire / Return / Conflict). `world.place()` and
room templates in `voice/answers.py` turn that into speech.

**Tech Stack:** Python 3.10, numpy, OpenCV, pytest. Venv: `/Users/anshul/askroom/.venv/bin/python`.

**Spec:** `docs/specs/0009-room-product.md` (M0 section, and section 3 Transitions).

## Global Constraints

- Python 3.10: no `match`, no 3.11+ stdlib. Runs on the Jetson (JetPack 6).
- Room positions are full-frame px; never write room positions into `Entity.pos_cm` (table cm).
- `core/types.py` does not change. `core/capture.py`, `core/detect.py`, `core/relations.py` do not change in M0.
- New config keys only in a new `room_memory:` section at the end of `config.yaml`; `room_memory.enabled: false` by default.
- With `room_memory.enabled: false`, behaviour is byte-for-byte today's (every existing test passes unchanged).
- Spoken answers: 1-2 short sentences, no markdown; room answers never say "you put it there".
- Tests need no hardware. Run with `ASKROOM_NO_LOCAL_CONFIG=1 /Users/anshul/askroom/.venv/bin/python -m pytest -q`.
- One instance per prop class is a demo assumption; code comments must not call it identity evidence.

## Review Focus

- A prop carried back to the table must stop being "on the bookshelf" once the table confirms it (presence flip), and a single table flicker must not.
- A decoy (a second key ring) never inherits the keys: only an unconsumed *table* departure, with the track first seen after it.
- A person walking in front of a zone must not make its objects UNKNOWN (blocked visits count for nothing).
- `resolve()` for a room entity returns `(None, chain)`, so no table-cm path (laser `aim_object`) can use a stale table spot.
- Room memory off or misconfigured (no zones file, view mismatch, no rect) must log why and leave the table demo untouched.

---

### Task A: zones and table view (`core/room_zones.py`, `core/room_view.py`)

**Files:** Create `core/room_zones.py`, `core/room_view.py`, `tests/test_room_zones.py`, `tests/test_room_view.py`.

**Produces:**
- `Zone(name: str, say: str, poly: list[tuple[float, float]])` with `contains(pt: tuple[float, float]) -> bool` (inside or on edge) and `bbox() -> BoxPx` (int, x1,y1,x2,y2 inclusive-exclusive, clipped at 0).
- `Zones(view: str, size_px: tuple[int, int], zones: dict[str, Zone])` with `at(pt) -> Optional[Zone]`, `to_dict()`, `from_dict(d)`, `save(path)`, `load(path)` (JSON; `load` raises FileNotFoundError when missing).
- `view_version(size_px, zoom, table_view_rect) -> str`: first 12 hex chars of sha1 over the canonical JSON of the three values. No table calibration in it.
- `default_rect(size_px, zoom=100, ref_zoom=160, out_size=(1280, 720)) -> BoxPx`: centred rect covering `1/(ref_zoom/zoom)` of the frame, same aspect as out_size.
- `cut(img, rect, out_size=(1280, 720)) -> np.ndarray`: crop + `cv2.resize(..., INTER_AREA)` (INTER_LINEAR when upscaling).
- `TableView(source, rect, out_size=(1280, 720))`: the FrameSource API over a full-frame source (`FrameBuffer` or anything with `latest()`, `at(t)`, `wait_new(after_idx, timeout)`): `latest()`, `at(t)`, `wait_new(after_idx, timeout=1.0)` return table-view `Frame`s (same t, wall, idx; img cut, None stays None; cache the cut per idx, last 4). `latest_full()`, `full_at(t)` return the source's full frames. `fps`, `stop()`, other attributes delegate to the source (`__getattr__`).
- `measure_rect(full_img, ref_img, init_rect) -> tuple[BoxPx, float]`: ECC (`cv2.findTransformECC`, MOTION_AFFINE, grey, both downscaled so the full frame is at most 960 px wide) aligning `ref_img` (the zoom-160 table image) into `full_img` starting from `init_rect`; returns the fitted rect (x0, y0, x0 + sx*ref_w, y0 + sy*ref_h rounded) and the ECC correlation.

- [ ] Tests first: polygon contains/at/bbox; JSON round trip; view_version stable and changes with each input; default_rect for 1920x1080 is 1200x675 centred; cut shape; TableView latest/at/wait_new give 1280x720 with the source idx/t, full_at gives the source frame, attribute delegation; measure_rect recovers a known rect within 1 px on a textured synthetic image (random blobs, `np.random.default_rng(0)`), from an init rect 20 px off.
- [ ] Implement; run the two test files; all pass.

### Task B: tracker and driver (`core/room.py`)

**Files:** Create `core/room.py`, `tests/test_room_tracker.py`, `tests/test_room_driver.py`.

**Consumes:** `core/room_types.py`; Task A's `Zone`, `Zones`, `view_version`, `cut`, `default_rect`, `measure_rect` (import them; write tests with small Zones built in code).

**Produces:**
- `RoomTracker(cfg: RoomConfig)` with `visit(zone: str, say: str, obs: list[RoomObservation], blockers: list[BoxPx], changes: list[BoxPx], t: float, wall: float, frame_idx: int, lum: Optional[Callable[[BoxPx], float]] = None, crop=None) -> ZoneVisit` and `tracks(zone=None) -> list[RoomTrack]`.
  - Match: per class, greedy by IoU; a match needs IoU >= 0.3 or centre distance <= 0.5 x the track box diagonal.
  - Matched track: box, last_seen/last_wall updated, hits += 1, misses = 0; confirmed once hits >= confirm_visits; confirmed tracks matched this visit go in `confirmed`.
  - Unmatched track: the visit is **valid** for it unless a blocker box covers >= blocker_overlap of the track box, or a change box whose area >= change_area_ratio x the track box area overlaps it (same fraction), or `lum(box)` is outside [lum_lo, lum_hi]. Invalid: nothing changes. Valid and unconfirmed: dropped. Valid and confirmed: misses += 1, into `missed`; when misses >= absent_visits it is also dropped (appended to `dropped`).
  - Unmatched observation: new track `r:N` (N global, increasing), hits 1, first_seen = t.
- `RoomMemory(cfg: RoomConfig, zones: Zones, backend, to_obj: dict[str, str], world, table_rect: BoxPx)` with `step(full: Frame) -> list[Event]` (call once per perception frame; every `room_every_n`-th call processes the next zone round-robin; returns `world.room_update(visit)`), and `from_config(cfg: dict, world, backend, table_rect: BoxPx) -> Optional[RoomMemory]` (None, with a log line saying why, when disabled, zones file missing, zero zones, or `zones.view != view_version(capture_size, zoom, table_rect)`).
  - Zone pass: crop `zone.bbox()` from `full.img` (skip when img is None); resize down if the long side > max_crop_px; `backend.infer(crop) -> [(label, conf, (x1,y1,x2,y2))]`; map labels with `to_obj`; `hand` boxes are blockers; a prop needs conf >= room_prop_conf and IoU < hand_iou with every hand; map boxes back to full px; drop if the centre is outside the zone polygon or inside `table_rect`. Changes: grey absdiff (> change_thr) against this zone's previous crop, dilated, external contours' bounding boxes in full px (none on the first visit). `lum`: mean grey in the box. Then `tracker.visit(...)` and `world.room_update(visit)`.
- CLI `python -m core.room`: `--zone NAME --say TEXT --poly x,y x,y x,y ...` (adds/replaces in `zones_path`, creating the file with the current view version from config: capture_size, zoom, table_view_rect or default_rect), `--delete-zone NAME`, `--list`, `--show IMAGE --out PATH` (draws zones), `--measure-rect --full FULL.jpg --ref REF.jpg` (prints `room_memory: {table_view_rect: [...]}` for config.local.yaml and the correlation), `--grab DEVICE --out PATH` (one full frame via `core.capture.open_camera(dev, w, h)`). `main(argv=None) -> int`.

- [ ] Tests first (tracker): confirm after 2 visits; unconfirmed dropped on a valid miss; blocked miss changes nothing; big change blob blocks, small one doesn't; dark box (lum) blocks; confirmed track goes UNKNOWN path: misses 1, 2, 3 then dropped; two classes don't cross-match; ids increase.
- [ ] Tests first (driver, with a fake backend returning fixed raw boxes in crop coordinates and a stub world recording `room_update` calls): round-robin over zones every N steps; crop offset mapping back to full px; hand-overlapping prop dropped; low conf dropped; outside-polygon and inside-table-rect dropped; from_config returns None when disabled / missing file / view mismatch; CLI zone round trip and `--list` in a tmp dir.
- [ ] Implement; run the two test files; all pass.

### Task C: World integration (`core/room_world.py`, hooks in `core/world.py`)

**Files:** Create `core/room_world.py`, `tests/test_room_world.py`. Modify `core/world.py` (minimal hooks only).

**Consumes:** `core/room_types.py` (build `ZoneVisit` / `RoomTrack` objects directly in tests; do not import core/room.py).

**Produces:** `class RoomRules` mixin, `World(ThingRules, RoomRules)`:
- `_reset_room()` (called at the end of `World.reset()`): `self._room: dict[str, RoomState] = {}`, `self._departures: dict[str, tuple[float, float]] = {}` (name -> (t, wall) of the latest unconsumed table departure), `self._conflicts: dict[str, dict[str, Conflict]] = {}` (name -> track id -> Conflict), `self.room_cfg = RoomConfig.from_dict(<the raw dict's 'room_memory'>)` (World.__init__ keeps the raw dict when given one; a Config gives defaults).
- Departure hook in `_emit`: EXITED_VIEW or LOST_TRACK emitted while `entities[name].zone == 'table'` records `_departures[name] = (t, wall)`.
- `_emit(name, etype, t=None, wall=None, img=None, **fields)`: optional capture time and snapshot image for room events (defaults: today's `self._now`, `self._wall`, `self._frame`); a room event with `img` passes a `Frame(t, wall, img, idx)` to `events.add`.
- `room_update(visit: ZoneVisit) -> list[Event]` under `self.lock`, per spec section 3 **Transitions** and the prop table: for each track in `visit.confirmed`, skip unless `track.cls` is an entity; `decide` the operation:
  - `refresh` if `_room[name].track == track.tid` (update box, seen_t/seen_wall, misses 0; if it was absent that can't happen: absent tracks are dropped);
  - otherwise, if `track.role == 'conflict'`: keep/refresh the Conflict, nothing else (never upgrades);
  - entity zone 'table': VISIBLE and `now - _seen_t[name] <= table_fresh_s` -> role 'ignored' (re-decided on later visits); UNDER/INSIDE/HELD -> conflict; unconsumed departure within handoff_s and `track.first_seen > departure t` -> **acquire**; else conflict;
  - entity in a room zone: `_room[name].absent`, same zone, IoU(track box, last box) >= 0.3 and `track.first_seen > absent_t` -> **reacquire**; else conflict.
  - **acquire**: consume the departure; `_room[name] = RoomState(zone, say, box, track=tid, seen_t=track.last_seen, seen_wall=track.last_wall, arrived_wall=track.first_wall, arrival_observed=True, table_pos_cm=ent.pos_cm)`; entity: status VISIBLE, zone = visit.zone, parent None, candidates [], confidence 1.0, edge None, pos_cm None, box_cm None, held_since None, pre_pickup_pos None; clear its table debounce (`_bits[name].clear()`, `_present[name] = False`); track.role 'assoc', track.entity = name; FOUND event with the visit's t/wall/crop.
  - **reacquire**: as acquire but arrival_observed False, arrived_wall = track.first_wall, absent False; FOUND.
  - For `visit.missed`: if `_room[name].track == tid`: misses = track.misses; when misses >= absent_visits and not absent: absent True, absent_t = visit.t, entity status UNKNOWN (zone kept), LOST_TRACK (room; not a departure because zone != 'table').
  - For `visit.dropped`: remove Conflicts with that track id.
- Return: `_observe()` gets one early hook: if `ent.zone != 'table'` before it resets the zone, call `_room_return(name, ent)` after the reset: drop `_room[name]`, drop its conflicts, and emit MOVED (`to_cm=ent.pos_cm`). `_observe` runs only after the table presence flip, so a single flicker never returns.
- `resolve(name)`: unchanged except it returns `(None, chain)` when the outermost chain entity has `zone != 'table'`.
- `place(name, now: Optional[float] = None) -> Place` (now = wall time, default time.time()): chain from `relations.resolve_chain`; `via` = outermost; if `via`'s zone is a room zone and in `_room`: kind 'room', zone, say, box_px, observed_directly (via == name), absent, fresh = not absent and misses < fresh_visits and now - seen_wall <= fresh_s, arrived_wall, last_seen_wall = seen_wall, arrival_observed, status of via; else kind 'table' (pos from resolve_chain; kind 'none' if the name is unknown). `conflicts` = the object's own Conflicts seen within fresh_s of now.
- `state_json()` gains `'room': self.room_json()` (name -> zone, say, box_px, seen_wall, absent, arrival_observed; plus 'conflicts').
- `WorldAPI` gets `place` (docstring updated).

- [ ] Tests first (synthetic, `tests.synth.Scene` for the table side, `ZoneVisit`s built by hand with t on the same clock): the M0 list in the spec's acceptance tests — demo moment (EXITED_VIEW then shelf track first seen after -> FOUND, pos_cm None, resolve None, place kind room fresh); table return via presence flip (MOVED, room state gone, later shelf misses change nothing); single flicker (one table detection) does not return and place() freshness unchanged; decoy after room absence -> conflict; HELD-timeout decoy (track first seen before the LOST_TRACK) stays conflict; departure consumed (second track elsewhere -> conflict); own track keeps refreshing, no conflict; keys VISIBLE on table + shelf keys -> ignored; keys UNDER notebook + shelf keys -> conflict, belief unchanged; absence after 3 valid misses -> UNKNOWN zone kept, LOST_TRACK, not a departure; reacquire at the last spot; room FOUND event carries the visit's t/wall; box with keys INSIDE carried to shelf -> place('keys') via box, observed_directly False; room_update with a class that is not an entity is ignored; RECAL-independent (no table calibration involved).
- [ ] Implement; run `tests/test_room_world.py` plus `tests/test_world_core.py tests/test_world_rules.py tests/test_openworld.py tests/test_thing_identity.py`; all pass.

### Task D: answers (`voice/answers.py`, `core/fakeworld.py`)

**Files:** Modify `voice/answers.py`, `core/fakeworld.py`. Create `tests/test_room_answers.py`.

**Consumes:** `Place`, `Conflict` from `core/room_types.py`; `world.place(name, now)` when present (`hasattr` guard).

**Produces:**
- `_where` becomes: `place = world.place(obj, now) if hasattr(world, "place") else None`; room places go to `_where_room`; everything else to today's body (renamed `_where_table`, unchanged). Then, if `place` has conflicts, append one sentence: `"I also see {name} on {say}."` for the first conflict (name = display name without "your").
- `_where_room` templates (Y = "Your"/"The", n = display name, be = is/are, It = It/They, it = it/them):
  - direct, fresh, arrival observed: `"{Y} {n} {be} on {say}. {It} appeared there {ago(arrived)}."`
  - direct, fresh, arrival not observed: `"{Y} {n} {be} on {say}. I've seen {it} there since {clock(arrived)}."`
  - direct, not fresh, not absent: `"I last saw {y} {n} on {say} at {clock(last_seen)}."`
  - direct, absent: `"I last saw {y} {n} on {say} at {clock(last_seen)}. I can't see {it} there now."`
  - via a container, fresh: `"{Y} {n} {be} in {pn(via)}. {Pn(via)} is on {say}."`
  - via a container, not fresh: `"{Y} {n} {be} in {pn(via)}, which I last saw on {say} at {clock(last_seen)}."`
  - Room answers carry no laser action (`point_at=None`, `action=None`).
- `FakeWorld.place(name, now=None) -> Place`: a table place from its own resolve (kind 'table'), plus a test helper `set_place(name, place)` that makes `place(name)` return the given Place.

- [ ] Tests first: each template row from a FakeWorld with `set_place`; pill bottle wording unaffected; plural (keys) vs singular (wallet); conflict tail on a table answer and on a room answer; a world without `place` works; existing `tests/test_answers.py` unchanged and passing.
- [ ] Implement; run `tests/test_room_answers.py tests/test_answers.py tests/test_pipeline.py`; all pass.

### Task E: wiring, config, demo_check, docs

**Files:** Modify `main.py`, `config.yaml`, `demo_check.py`, `CONTEXT.md`, `AGENTS.md`, `docs/FEATURE_STATUS.md`, `config.local.yaml.example`. Create `tests/test_room_main.py`.

**Consumes:** `RoomConfig`; Task A `TableView`, `default_rect`, `view_version`, `Zones`; Task B `RoomMemory.from_config(cfg, world, backend, table_rect)` and `.step(full)`.

**Produces:**
- `config.yaml`: new `room_memory:` section at the end with every RoomConfig key and a one-line comment each (enabled: false).
- `main.build`: when `room_memory.enabled` and not fake and not video: `FrameBuffer(camera, ring_s=rc.ring_s, opener=lambda src: core.capture.open_camera(src, *rc.capture_size))`, wrapped in `TableView(fb, rect)` where rect = `rc.table_view_rect` or `default_rect(...)` with a warning to measure it; after the detector exists, `room.room_memory = RoomMemory.from_config(cfg, world, detector.backend, rect)`.
- `Room.__init__`: `self.room_memory = None`. `Room.perceive`: after `world.update`, if `self.room_memory` and the frames source has `full_at`: `full = self.frames.full_at(frame.t)`; `room_events = self.room_memory.step(full)` inside try/except that logs and never breaks table perception; return value unchanged (`(dets, events + room_events)`).
- `demo_check.py`: `check_room_memory(rig)` before `check_room` in CHECKS: SKIP when disabled; FAIL when the zones file is missing, has no zones, or its view differs from `view_version(capture_size, zoom, rect)`, or `table_view_rect` is unset; PASS listing zones and their spoken names. File checks only (no camera).
- Docs: CONTEXT.md repo map rows for the new modules and a status line; AGENTS.md: `place` joins the WorldAPI list, the Freeze line notes it was lifted for spec 0009 by Anshul (Sat 26 Sep); FEATURE_STATUS.md: room memory M0 implemented, unit-tested, not on the rig; config.local.yaml.example: `room_memory: {enabled: true, table_view_rect: [x1, y1, x2, y2]}` commented example.

- [ ] Tests first (`tests/test_room_main.py`): Room.perceive calls room_memory.step with the full frame and appends its events; a raising step is logged and table events still return; room_memory None changes nothing; check_room_memory SKIP/FAIL/PASS cases with tmp zones files.
- [ ] Implement; run `tests/test_room_main.py tests/test_main.py tests/test_demo_check.py`; all pass.

### Task F: integration (after A-E)

- [ ] Full suite passes: `ASKROOM_NO_LOCAL_CONFIG=1 /Users/anshul/askroom/.venv/bin/python -m pytest -q`.
- [ ] End-to-end test `tests/test_room_e2e.py`: a real `World` + `RoomMemory` with a fake backend and synthetic full frames: keys carried off the table (Scene) then detected in the shelf zone on 2 visits -> `answers.answer` for "where are my keys" says "on the bookshelf"; a hand box over the shelf for 3 visits leaves them fresh-or-last-seen, never "can't see".
- [ ] Commit per task (conventional prefixes, no AI attribution lines).
