# 0006: Room pointing without depth (dot map + pixel-space closed loop)

Status: Phases 1, 2 and 2c are built on branch `room-pointing` and **off by default** (`room.enabled: false`). The work is sim-tested only: nothing has been measured on the rig. It is a roadmap item after the Sat Sep 26 6 PM EDT freeze, not the demo path. Phase 2b needs owner sign-off for `core/detect.py` and `core/world.py`. This spec supersedes the ordering of spec 0005 (tags first) and keeps its one-sided approach, Broyden update and 13-parameter model (the model is Phase 3 here).

## Problem

The laser works on the table plane only. The table homography is one plane, and `LaserFit` maps table cm to pulses. Anything off the table (a shelf, the floor, a counter) maps to the wrong pulses, and `detect_filter.table_margin_cm` drops detections off the table.

## Idea

If the camera sees the laser dot inside an object's image box, the dot is on that object, whatever its depth: the camera can't see the background at a pixel where the object is in the way. Pointing at a *visible* object therefore needs no depth. It needs three things:

1. A first guess of servo pulses for a pixel. This is the **room dot map**.
2. A pixel-space closed loop that finishes the job. This is **`Laser.aim_px`**.
3. A dot finder that still works at 3–4 m. This is **`Laser.find_dot_px`**, which averages on/off pairs.

The pixel→pulse map is nearly depth-independent when the pan-tilt pivot sits within ~5 cm of the lens. The miss from depth is about b·Δz/z: with b = 3 cm and the map at 2 m, that's 1.5 cm at 1 m and 3 cm at 4 m, below the SG90's own slop. The loop removes whatever is left.

## What is built

| Piece | Where | Notes |
|---|---|---|
| Dot map sweep | `act/room_map.py` `sweep()` | Serpentine pulse grid over `servo_limits` (`room.grid`, default 20×15), one-sided approach (`Laser.move_to`), `n_pairs` on/off pairs per look. A refine pass samples midpoints where neighbouring dots jump more than 2.5× the median, or where one was seen and the other missed (depth edges, silhouettes). `stop()` is checked before every point and raises `SweepAborted`. The laser ends off. |
| Inverse lookup | `RoomMap.pulses_for_px` | Anchors on the nearest seen dot and refuses beyond `max_map_gap_px`. Takes samples within 2.5 grid steps *in pulse space*, fits a Gaussian-weighted affine map, and drops the worst sample while its residual exceeds max(4 px, 0.25 × dot spacing), never the anchor (`fold=True`: a depth edge). Falls back to a global affine fit. The step is capped at 2 grid steps. Returns pulses, the local Jacobian (px/µs), `fold`, `gap_px`. |
| Dot finding | `Laser.find_dot_px(n_pairs, gate)` | Averages `dot_score` over the pairs and thresholds at `laser_diff_thr/√n` (SNR grows as √n, so the false-alarm rate stays the same). With `gate`, it keeps the brightest blob with aspect ≤ 4 and fill ≥ 0.3, and takes the score-weighted centre. `find_dot` (table) is `find_dot_px(1, gate=False)` + `px_to_cm`, so the table behaviour is unchanged. |
| Pixel aim | `Laser.aim_px(target_px, box_px)` → `PxAim` | Feedforward from the map, then a P step (gain 0.7) through the local Jacobian, a Broyden update when \|Δu\| > `deadband_us`, and a step cap. Stops when the dot is inside `box_px` shrunk 20% or within `tol_px`. Reasons: `in_box`, `within_tol`, `max_tries`, `not_seen` (3 misses, never seen), `lost`, `jumped` (the dot moved `jump_px` + 1.5× more than predicted: it landed on something nearer), `unmapped`. Only `in_box` and `within_tol` are `on_target`. |
| Routing | `voice/visual.py` `_pointed`, `main.py` `Room._aim_room` | A Grok point off the table (or with no table calibration) becomes `Answer(action="room:u,v")`, following the `sweep:left` convention, so `core/types.py` doesn't change. `main.aim` runs the gates, then `aim_px`; if the dot isn't confirmed on target, the laser goes off immediately. The dwell is `min(laser_timeout_s, room_dwell_s)`. An optional object box is passed as `room:u,v,x1,y1,x2,y2`. |
| Safety gate (2c) | `main.Room._aim_room`, `act/room_map.beam_blocked`, `people_boxes_hog` | It refuses when: room pointing is off or there's no map; the map resolution doesn't match the camera; the target is outside every zone polygon (`require_zone`); or a hand box from the last `hand_recent_s`, or a HOG person box, grown by 20 px, overlaps the target box or the head→target image segment (`head_px`). The sweep CLI needs nobody seen for `person_clear_s` and aborts when someone appears. |
| Zones | `RoomMap.zones`, `python -m act.room_map --zone NAME --poly x,y ...` | Image polygons of floor and furniture tops. A re-sweep keeps them. |
| Check | `demo_check.py` check 10 `room` | Skipped when off. Otherwise: the map matches the camera and has ≥ 10 dots and some zones; three mapped dots spread over the room are re-aimed, and the **first** look (the map's open-loop guess) must land within 2×`tol_px`. The loop would converge anyway, so the first look is what catches a moved camera or head. |
| Sim | `act/sim.py` `RoomScene`, `RoomHead`, `RoomCamera`, `RoomRig` | A 640×360 pinhole camera 2.3 m up, pitched 25°. Boxes: floor, three walls, table (75 cm), coffee table (45 cm), shelf (140 cm) with a bottle, and a backpack. The head pivot is offset b from the lens. The dot is drawn only where the camera can see the beam's hit (raycast), with brightness and size falling with distance and scaled by reflectance; fresh noise per frame. |

Config: the `room:` section at the end of `config.yaml`. The map path is the existing `room_map` key (gitignored).

## Phase 2b: seeing objects off the table (proposal for the owners of `core/detect.py`, `core/world.py`)

Today only Grok's visual answers reach room pointing, because detections off the table are filtered out. To make "where's my backpack?" point on the floor:

- **`core/detect.py`:** in the `table_margin_cm` filter, keep a box whose footprint pixel (bottom-centre) falls inside a room zone polygon (`RoomMap.zone_at`), limited to classes in `room_targets` (the existing key).
- **`core/world.py`:** give these entities `zone = <zone name>` (the field exists and defaults to `"table"`), keep `box_px` (already stored in `World._box_px`), and set `pos_cm = None`. Expose the zone and box to readers through `state_json` so `main.aim` can send `room:u,v,x1,y1,x2,y2` for an entity whose `zone != "table"`, and answer templates can say "on the bookshelf".
- **`voice/answers.py`:** "It's on the {zone}." for room entities. When `aim_px` returns `on_target=False`, the answer should say it can't reach it with the pointer and the dashboard should highlight the box instead.

## Limits

- **Accuracy is set by the SG90:** 5–10 µs dead band, 1–2° backlash. Expect ~2.5–5 cm at 3 m once converged. That's enough for mug-, bottle- or backpack-sized objects, not for picking between two small things 5 cm apart. Better servos (MG90S, STS3215) are the only real fix.
- **Small objects:** keys and pills beyond ~3 m are 8–27 px and can't be detected reliably, so they stay table-only.
- **Dot visibility:** dark, glossy or saturated-white surfaces and far floors may hide the dot. `aim_px` then reports `not_seen` and the laser goes off; it never claims success.
- **Moved furniture** leaves the map stale where it moved. The loop still converges when the head is close to the lens; re-sweep after rearranging. `check_room` catches a moved camera or head.
- **Person detection:** HOG finds standing people only and misses seated or partial ones. OpenCV 5 (the laptop venv) has no HOG, so `people_boxes_hog` returns None ("unknown", never "nobody"), room aims log that they're gated by zones and hands only, and the sweep CLI requires `--room-is-clear`. The rig's container has OpenCV 4.11 with HOG. Zones exclude eye level by construction, and the dwell cap and the hardware kill switch stay the last line.
- **`main.py --fake`** doesn't simulate a room: the dashboard sim camera is the table. Room behaviour is tested with `RoomRig` directly and in `demo_check --fake` check 10.

## Acceptance tests (all in `.venv/bin/python -m pytest -q`)

`tests/test_room_map.py`:

- The sweep maps floor, table and back wall, refines edges, and leaves the laser off. Pivot offsets b = 3, 10 and 25 cm.
- The map inverts to a median < 2 px on mapped surfaces (backlash accounted).
- `pulses_for_px` refuses an unmapped gap and flags a fold, using the consistent side.
- `aim_px` lands on the floor, table, coffee table, back wall and the bottle on the shelf (in its box), for all three offsets. With a 3° wrong guess and a Jacobian 2× or 0.5× off, it converges in 2–8 tries.
- A blocked dot gives `not_seen` / `on_target=False` after 3 tries; an unmapped target is refused without moving.
- 8 averaged pairs find a dim dot that 1 pair misses, with no false dots. The shape gate ignores a brighter streak.
- The sweep aborts when a person appears and leaves the laser off. `beam_blocked` geometry. HOG finds nobody in the empty room, or returns None without HOG.
- `main.Room` routes `room:u,v` to `aim_px` and turns off after `room_dwell_s`. It refuses outside zones, with a hand on the target, or when disabled (no servo calls), and turns off when the dot is never seen.
- A visual-Q&A point off the table becomes `room:u,v` only when `room.enabled`.

`tests/test_demo_check.py`: check 10 skips when off, passes on the sim map, and fails when the map is 80 px off.

Sim numbers (laptop, `python -m act.room_map --sim --b-cm B --aims 40`): a 16×12 grid + refine is ~260 points and ~140 s simulated; 39–40/40 aims on target, median true error 1.5–1.7 px, p90 2.5–3.0 px, for b = 3, 10 and 25 cm. The misses are targets whose beam hit is hidden from the camera. These are **sim numbers, not rig accuracy.**

## Rig protocol (not done yet)

1. **Phase 0 (mount):** pan-tilt pivot within ~5 cm of the lens, high in a corner, tilted 30–40° down. Lock exposure, gain and white balance low (`scripts/camera_setup.sh`). Set `servo_limits` to cover the view. Set `head_px` if the head is visible.
2. `python -m act.room_map --sweep` with nobody in view (~2–3 min at 20×15), then draw zones: `--zone floor --poly ...`, `--zone shelf --poly ...`.
3. Twelve sticky notes at ~1, 2, 3 and 4 m on at least 3 surfaces. Aim at each note's pixel. Measure the miss with a ruler, plus tries and time. Report the median and p90 per depth band in cm and in degrees, over the full pan/tilt range.
4. Right-object rate: 10 real objects, each with distractors 6–10 cm away (Vogel & Balakrishnan grid of depth × width).
5. Pass: median ≤ 3 cm and p90 ≤ 6 cm at ≤ 3 m; right object ≥ 90% for objects ≥ 8 cm wide; `python demo_check.py` passes check 10 with `room.enabled: true`.

## Phase 3 (optional, not built)

- Intrinsics (ChArUco, spec 0005 step 0).
- The 13-parameter pan-tilt "inverse camera" fitted to dot-map samples on ≥ 2 depths (single-plane data is degenerate), with scale from the table homography.
- Triangulated dots give a sparse room point cloud: zone heights, the `laser_max_height_cm` check, and "about 2 m away" answers.
- DAv2-S depth scaled by those anchors, loaded on demand only (~650–700 MB on the Orin).
