# 0005: Room calibration with laser-guided tag placement

Status: proposed (draft 2, revised after a prior-art check). Roadmap item after the Sat Sep 26 6 PM EDT freeze; nothing here is built. Needs owner sign-off for `core/table.py` and `core/world.py` changes, and probably better servos (see Risks).

## Problem

Calibration today covers one flat surface. One AprilTag (36h11, id 0, 16 cm) anywhere on the table gives a pixel to cm homography (`core/table.py`), and the laser is a second-order polynomial from table cm to servo pulses (`act/laser.py`, `act/calibrate.py`). That model breaks for a room:

- A room is several surfaces at different heights: table, counter, floor, shelf. One homography describes one plane.
- Accuracy falls off with distance from the tag, faster than linearly, so where the tag goes matters, and nobody should have to work that out by hand.
- The laser fit is only valid on the plane it was fitted on.

The unused stretch keys in `config.yaml` (`floor_zones`, `room_map`, `room_targets`, `laser_max_height_cm`, `edge_drop_cm`, `search_k_of_n`, `search_timeout_s`) were placeholders for this.

## Goal

A guided setup where the rig works out where each tag should go, points the laser at that spot, confirms the tag once it is placed, and ends with every zone calibrated for both the camera and the laser. Target: laser error at most 3 cm on table-distance zones (the Trello #11 target) and at most 6 cm on far floor zones unless the servos are upgraded, measured by the rig itself at each zone's far edge.

## Non-goals

- A moving camera or a camera on its own pan-tilt head. One fixed camera; "whole room" means whatever it sees from a high corner mount.
- Walls and vertical surfaces. Zones are horizontal: floor and furniture tops.
- Objects off the zone planes (a tall object's top still maps a few cm off, as today).

## Prior art (checked Sep 26)

Nobody has published this exact combination: an error model plans where tags go, a pan-tilt laser shows a person each spot, one fixed camera confirms, and the result is a multi-plane room plus laser calibration. The pieces exist separately:

| Piece | Closest prior art | What we take |
|---|---|---|
| Laser shows a person where to put a calibration target, camera confirms | Hunter Engineering ADAS patents (US 11,145,084; 11,544,870; 11,763,486; 12,287,190): gimbal lasers mark floor spots for vehicle calibration targets, and 11,544,870 recalibrates the gimbal from camera-seen spot error | Mark a short line, not just a dot, so the tag's rotation is shown too. Active patents: fine for a hackathon or research, check before any commercial use |
| Guided target placement order | Epson projector patent US 10,015,457 | Easiest spots first, so a half-finished run is still usable |
| Guided lens calibration | AprilCal (Richardson, Strom, Olson, IROS 2013); Calibration Wizard (Peng & Sturm, ICCV 2019); Rojtberg & Kuijper (ISMAR 2018, `github.com/paroj/pose_calib`) | Suggest the next board pose instead of free waving; stop on max expected reprojection error |
| Marker placement planning | arXiv 2211.01513 (greedy placement), arXiv 2509.17345 (ArUco placement study) | Greedy placement with a predicted-error score |
| Multi-tag mapping | TagSLAM (ROS + GTSAM), apriltag_ros bundles (MATLAB), MarkerMapper (C++, moving camera) | None fit a Python 3.10 Jetson stack; use OpenCV `solvePnPGeneric` + a small `scipy.optimize.least_squares` |
| Uncalibrated visual servoing | Jägersand 1996-97; Piepmeier 2002; Shademan ICRA 2010; Sutanto 1998 | Broyden Jacobian with deadband skip, forgetting factor, step cap, re-probe |
| Pan-tilt laser to camera model | Applied Optics 2009 (12-parameter galvo model); arXiv 1812.00232 (pan-tilt axis offsets) | Geometric model with nonlinear pulse to angle, fitted jointly on pixel reprojection error |
| Laser-pointer assistive robots | Kemp et al., El-E (HRI 2008, "A Clickable World" IROS 2008) | Narrow-band filter makes the dot far easier to see (their dot came from the person, ours from the rig) |

Hobby laser turrets (LazerPaw, pi-turret and similar) map linearly or run PID on pixels; none fit a geometric model.

## Design

### Step 0: lens intrinsics (once per camera)

Guided ChArUco calibration (`cv2.aruco.CharucoDetector` + `cv2.calibrateCamera`): the dashboard shows the next suggested board pose (Rojtberg / AprilCal style) until the expected reprojection error is low enough.

- 20-30 views, tilted up to about 45°, covering the image corners; squares at least 20 px in the image.
- Pass at RMS reprojection error below 0.5 px (0.15-0.3 px is good).
- `CALIB_RATIONAL_MODEL` for 90-130° lenses; `cv2.fisheye` only above about 140°.
- Save `camera_intrinsics.json` (K, distortion, image size). If the stream size differs but the aspect ratio matches, scale K; otherwise refuse.

Needed so a tag sighting gives a 3D pose, and it removes the lens-distortion error the calibration review flagged.

### Step 1: zones

On a dashboard still, the user outlines each surface as a polygon, names it ("coffee table", "floor by the couch") and marks it floor or furniture top. Saved as image polygons. The existing table zone is just the first zone. The user also enters the camera's rough height, which with the intrinsics gives each zone's scale for planning before any tag is down.

### Step 2: plan tag spots

Predicted error at a zone point, from the nearest tag of side `s` px at distance `d` px:

`err ≈ σ · (a · d/s + b · (d/s)²)`

The linear term is the tag's rotation and scale error; the squared term is the perspective error, which dominates when extrapolating far from a small tag. `σ` is corner noise (0.2-0.5 px with refinement); `a` and `b` are fitted once on the rig from check points measured with a ruler. Once intrinsics exist, a Monte Carlo replaces the formula: jitter the corners by `σ`, re-solve the pose, project the zone polygon, take the 90th-percentile error.

The planner is greedy:

- Start with one tag at the zone's centroid. Add a tag at the worst-predicted point until the far edge is under budget or the zone has 3 tags.
- Size the print from the predicted far-edge error, not only from detection. About 70-80 px is the detection floor, not an accuracy guarantee; far floor zones will want 25-30 cm prints.
- Order spots easiest first (nearest the laser's centre and the camera's centre), so a half-finished run is usable.
- Nothing above `laser_max_height_cm`: zones marked furniture top or floor only; checked again after the pose solve.
- Spots must be reachable and the dot visible; checked in step 3, with fallback to the next best spot.

### Step 3: laser-guided placement (image-space servoing)

The laser cannot aim in cm before anything is calibrated, so it aims in pixels.

Dot detection:

- Lock exposure, gain and white balance (as `scripts/camera_setup.sh` does), lower the exposure during setup.
- Average 3-5 on/off frame pairs from `dot_px_diff` (noise falls with √N), dropping frames inside the measured servo and camera latency. `dot_px_hsv` stays the fallback.
- Optional: a narrow-band filter over the lens for setup only, or a green laser (brighter on Bayer sensors). At 3 m with a 90° lens on 1280 px, one pixel is about 4.7 mm, so the dot is only 1-3 px.

Servoing:

1. Seed the pulse to pixel Jacobian with ± probe moves on each axis around the servo centre. Measure the servo deadband in pixels while doing this.
2. For each target pixel: move, find the dot, update the Jacobian (Broyden with a forgetting factor; skip the update when the move was under the deadband), cap the step size, re-probe if the error stops falling. Stop within `max(5 px, deadband px)` or after 10 steps.
3. Approach every final position from the same side (overshoot about 20 µs, then come back) to cancel backlash.
4. Draw a short line (small pulse oscillation on one axis) through the spot, and say "Put tag N where the line is, top edge along the line."
5. Wait until tag N is seen for 15 consecutive frames within 30 px of the target. Say "Got it", record `(pan, tilt) pulses, dot px, tag corners px`.
6. If the dot is never found (too far, dark or shiny floors reflect it away): skip laser guidance for that spot and highlight it on the dashboard still instead. Expect this on shiny floors.

Each zone uses its own tag ids (id 0 stays the table). Recommended: leave the tags down permanently, so drift can be checked while running.

### Step 4: solve each zone

- **Pose of one tag:** `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` returns both pose candidates. Small or far tags are nearly affine and the two fit almost equally well (the IPPE flip). Reject the sighting if the ratio of the two reprojection errors is above 0.2, unless one candidate's normal faces away from the camera or isn't roughly upward (zones are horizontal); then keep the other. Average corners over 15 frames first, with sub-pixel corner refinement.
- **Several tags on one zone:** their positions within the zone are unknown (nobody measures them), so an 8-corner homography is not possible directly. Joint solve per zone with `scipy.optimize.least_squares`: unknowns are the plane pose (6) plus (x, y, θ) of every tag after the first, minimising the reprojection error of all corners over all frames. About 20 lines.
- Per zone, store the px to zone-cm homography (same axis convention as today: +x left to right across the image), the plane pose (R, t) in camera coordinates (so all zones share the camera frame as the room frame), the tag layout, the zone polygon, the clipped tracked area and size, and the residual.

### Step 5: laser model

Step 3 already recorded dot sightings on known planes at different depths. Add 5-10 extra sweep points per zone, all approached one-sided.

Model: pivot position (3), orientation (3), per-axis pulse to angle as offset + scale + cubic term (6), tilt-before-pan axis offset and beam misalignment (1-2). About 13-14 parameters.

Fit all zones together with `scipy.optimize.least_squares` on pixel reprojection error: for each recorded pulse pair, intersect the modelled ray with that zone's plane, project with K, compare to the observed dot pixel. Points at two or more depths are required, otherwise pivot and orientation cannot be separated; with only the table zone, keep today's poly2 fit.

A per-zone residual correction is a constant offset or affine only (too few points per zone for poly2). Aiming: `zone cm → room 3D point → pan/tilt → pulses`, always approached from the same side.

### Step 6: verify

- The laser visits each zone's far-edge corners and a held-out tag if there is one, not only the tag centres (the fit is best there, so tag centres are biased).
- The camera measures the miss in pixels, mapped to the zone's cm. Report per-zone error against the Goal targets.
- `demo_check.py` gets a room check that fails when any zone's tag is visible and off target, and says "not in view" instead of passing silently when it is not.

### Drift while running

With tags left down, compare current corners (averaged over a few still frames, no hands in view) to the stored ones every 10 s. Mean corner shift above 2-3 px:

- Every tag moved the same way: the camera moved. Re-solve all zones and warn that the laser needs a re-verify.
- One tag moved: that tag was bumped. Flag it and ignore it until re-placed.

This extends the frame probe `main.py` already runs for the table.

## Data

`room_map.json` (gitignored, like `table_cal.json`):

```json
{"t": 0, "intrinsics": "camera_intrinsics.json", "zones": {
  "coffee_table": {"kind": "furniture", "tags": {"0": [0, 0, 0]}, "poly_px": [], "H": [], "R": [], "t": [],
                   "size_cm": [0, 0], "err_cm": 0, "corners_px": {}}},
 "laser": {"model": "pantilt_v1", "params": [], "residual": {}, "deadband_us": 0}}
```

`floor_zones` in `config.yaml` stays for configured defaults (tag ids and names per zone); computed values live in `room_map.json`. Entities gain a zone name next to their cm position; `World.observe_external` already takes a zone.

## Changes by file

| File | Change | Owner |
|---|---|---|
| `act/room_cal.py` (new) | planner, servoing, guided placement, joint zone solve, laser fit, verify | us |
| `act/laser.py` | pan-tilt model next to `LaserFit`; `aim_room(zone, cm)` with one-sided approach; averaged on/off dot detection | us |
| `scripts/intrinsics.py` (new) | guided ChArUco calibration | us |
| `core/table.py` | zones as several `Table`-like planes, optional intrinsics, sub-pixel corner refinement | P (sign-off) |
| `core/world.py` | zone on every position | owner sign-off |
| `main.py` | load `room_map.json`, per-zone recalibrate, multi-tag drift probe | us |
| dashboard | zone drawing on a still, next-pose hints for step 0, placement progress | us |
| `demo_check.py` | room check | us |
| `.gitignore` | `room_map.json`, `camera_intrinsics.json` | us |

## Acceptance tests

Unit tests, no hardware:
- The planner puts one tag in a small near zone and two or three in a long far one, predicted far-edge error under budget, easiest spots first, nothing above the height limit.
- Synthetic camera and planes: the zone pose is within 1 cm and 1°. A far, small tag set up to flip is rejected or disambiguated. The joint solve recovers an unmeasured second tag within 1 cm.
- Pan-tilt fit: recovers a known pivot within 1 cm and the cubic terms from noisy dots on two planes. It refuses (keeps poly2) with one plane.
- Servoing converges within 10 steps on a simulated laser with a skewed Jacobian, deadband and backlash, and gives up cleanly when the dot is never found.

Rig tests:
- Intrinsics RMS below 0.5 px.
- Two zones (table and floor): guided placement end to end, then verify at far edges against the Goal targets.
- Bump the camera: the drift probe flags it within 10 s. Move one tag: only that tag is flagged.

## Risks

- **Servos are the accuracy limit, not the algorithm.** 3 cm at 3 m is 0.57°. SG90/MG90S: about 0.1° per µs, 5-10 µs deadband (2.5-5 cm at 3 m), 1-2° backlash (5-10 cm). `act/laser.py` already assumes 1.5-1.8° deadband. Closed-loop correction and one-sided approach make 3 cm borderline at best. For room scale use digital servos (1-4 µs deadband) or Feetech STS3215 bus servos (0.088° resolution, ≤0.5° backlash, about 0.17° repeatability), or a 2:1 reduction. Otherwise accept the 6 cm far-zone target.
- **Grazing angles:** on a floor hit at about 30°, aim error stretches about 2x along the beam.
- **Dot visibility:** a low-power dot at 3 m on a dark or shiny floor may be invisible. The fallback is dashboard highlighting, which loses the laser guidance for that spot.
- **Tag size at distance:** a 16 cm tag is about 70-80 px at 3-4 m with a wide lens on 1280 px, which is the detection floor. Far zones need bigger prints.
- **Laser safety:** sweeping a room with people in it. The face check (#14) is still missing and blocks any room sweep.
- **Patents:** the laser-shows-target idea is in active Hunter Engineering patents; check before any commercial use.
- **Owned files:** the zone work in `core/table.py` and `core/world.py` needs their owners.

## Open questions

- Where the camera is mounted for the room (height, lens field of view).
- Servo upgrade or the relaxed far-zone target.
- Whether the tags stay down permanently.
- How the phone map shows several zones.
