# 0011: Stepper turret and pointing at camera finds

Status (Sun Sep 27, ~3 AM EDT): the turret hardware, firmware, driver and pointing math are built and checked on the bench from the Mac, on branch `turret-steppers` (not merged). The turret is **not yet on the Jetson**, the rig still runs `actuator: fake`, and pointing at camera finds is **not working yet**. The blockers below list what stands between here and "where's the remote?" lighting it up.

## What is built

| Piece | Where | Checked how |
|---|---|---|
| Firmware: degrees over serial, both axes planned together (20 kHz Timer2 step ISR, trapezoidal moves, retarget mid-move), soft limits ±90° on both axes, laser on D8, 2 s laser watchdog | `firmware/turret/turret.ino` | Host simulation of the firmware (moves land on target, ≤5 microsteps of overshoot on long moves); bench runs |
| Bring-up sketch | `firmware/servo42d_test/servo42d_test.ino` | Bench |
| Wiring, driver settings, gear ratios, flashing, protocol | `firmware/README.md` | — |
| Serial client, heartbeat keeps the laser on only while the host talks | `act/turret.py` | `tests/test_turret.py`; bench: the laser stayed on past the 2 s watchdog |
| `TurretActuator` (`actuator: turret`): servo-equivalent pulses (1500 + 10 µs/deg), so `act/laser.py`, `act/calibrate.py` and `act/room_map.py` run unchanged; `aim_deg()`, `point_at()`, parks at 0,0 on close | `act/actuator.py`, `turret:` in `config.yaml` | `tests/test_turret.py` (protocol emulator); bench: 7-pose test through the driver with the beam on |
| 3D point → pan/tilt with laser-offset parallax; pinhole camera helpers | `act/pointing.py` | `tests/test_pointing.py`: 2000 random targets with a laser offset hit within 1e-6 m; microstep rounding costs ≤0.4 mm at 1 m, ≤3.8 mm at 10 m |
| Seven-object check (`--laser`: beam on only while holding) | `scripts/turret_point_test.py` | Bench, Sun ~2:30 AM: all 7 poses reported on target, dots looked right by eye |
| The container can open the Uno | `scripts/dock.sh` (`--device /dev/ttyACM*`, cgroup rules 166/188) | `tests/test_room_app_sh.py`, `tests/test_demo_check.py`; not yet run with the Uno on the Jetson |

Measured on the bench: pan 4:1 (±90° swings exact), + pans right; tilt 3.2:1, + tilts up. The tilt ratio comes from the build guide; the 45° and 90° tilt poses looked right by eye but weren't measured.

## Blockers

| # | Blocker | Why it matters | Next action |
|---|---|---|---|
| B1 | The Uno still runs firmware with the old limits (pan ±170°, tilt −45° to +90°) | It can't tilt below −45°, and it disagrees with `act/pointing.FIRMWARE_LIMITS_DEG` | Plug the Uno into the Mac and flash `firmware/turret` (the Jetson has no arduino-cli or avrdude). The Python side clamps to `servo_limits` anyway, so this isn't unsafe meanwhile |
| B2 | The turret isn't on the Jetson, and the rig runs `actuator: fake` | Nothing points | Plug the Uno's USB into the Jetson; in `~/askroom_room/config.local.yaml` set `actuator: turret`, `turret: {port: /dev/ttyACM0}`, and `servo_limits` for the part of the room the camera sees; `scripts/room_app.sh restart` |
| B3 | The laser head is ~1.5 m below the Brio | A pixel is a direction, not a point: with the laser 1.5 m away, a wrong depth guess misses by up to ~14° (half a meter at 2 m). The dot-map sweep (spec 0006) needs no depth, but was only simulated with the pivot 3–25 cm from the lens | Build ray-plane pointing (plan below), or mount the head next to the camera |
| B4 | From ~1 m high, the laser can't hit tops above its own height | The counter top is probably unreachable (the camera sees it from above; the laser only grazes its edge) | Point only at zones below the head's height (couch seat, side table), or mount the head higher |
| B5 | Eye safety: laser class unknown, and the beam crosses the room at seated head height | A beam at ~1 m travels through seated head height | Find the class (on the module or its listing). Keep `room.require_zone`, the person check and `room_dwell_s`, and aim only below the seat line. Above Class 2 (1 mW), mount the head higher |
| B6 | Ray-plane pointing isn't implemented | Without it, or without the sweep, the laser can't follow camera finds | Plan below, about an hour with tests |
| B7 | Measurements missing: the laser's offset from the lens (1.5 m down plus any forward or side offset), camera height and downward tilt, surface heights (couch seat, side table, counter), and the turret's heading at 0,0 | Ray-plane pointing needs them. Each 1° of angle error is ~5 cm at 3 m | Measure them, or read the camera pose from the table's ArUco calibration; do one alignment check (aim at a known spot) |
| B8 | Drivers at Ma = 400 mA | Motion is weak, and tilt now carries the laser | Raise to ~800 mA (SERVO42D menu → Ma), or 70–80% of the motor's rated current |
| B9 | No feedback from the drivers | With the battery off, or with a loose EN wire (tilt's D2 came out while the laser was connected), the Uno still reports every move done | Before a run, check that both axes feel stiff once enabled, and that the driver screens show no error |
| B10 | The zero is wherever the mount points when the port opens | Opening the port resets the Uno and releases the motors briefly, so tilt can sag under the laser before 0,0 is set | Line it up at every start; `park_on_close` returns it to 0,0. Longer term: keep one long-lived connection, or add a home switch |
| B11 | Battery + on the Uno's VIN locks up the ATmega328P | The Uno goes silent until USB is unplugged | Power the Uno from USB only (the Jetson supplies it). Root cause not found |
| B12 | Laser module wiring unconfirmed | If the module has only S and −, the beam draws ~30–40 mA through D8 (the pin's absolute max is 40 mA) | Check the module's pins; for long on-times, drive it through a transistor |

## Rig state

On Sun Sep 27 ~2:45 AM, `act/actuator.py`, `act/turret.py`, `act/pointing.py`, `scripts/dock.sh` and `scripts/turret_point_test.py` were copied into `~/askroom_room`, and the `turret:` section was appended to its `config.yaml` (the rig's `config.yaml` is ahead of main, so it wasn't replaced). Backup: `~/askroom_room/.bak-turret-20260927-0243`. `config.local.yaml` is unchanged (`actuator` still defaults to fake). The app wasn't running and wasn't restarted.

## Plan: ray-plane pointing (B3, B6)

Everything we point at rests on a known flat surface, so the camera ray's depth is where it meets that surface.

1. **Pixel → ray.** Undistort pixel (u, v) with the Brio's intrinsics at the demo zoom, and turn it into a unit ray in the camera frame.
2. **Ray → 3D point.** Rotate the ray into the room frame with the camera pose (height, pitch, roll, yaw). Intersect it with the horizontal plane of the object's zone (`y = surface_height_m`; the floor is 0) and reject hits behind the camera or beyond `max_range_m`.
3. **3D point → pan/tilt.** Express the point in the turret frame (camera pose + `laser_offset_m`, the turret's 0,0 heading) and call `act.pointing.solve_aim`. `TurretActuator.aim_deg` moves there.
4. **Closed loop, where the camera sees the dot.** Hand the geometric aim to `Laser.aim_px` as the feedforward, in place of the sweep map's `pulses_for_px`, and let it finish the last centimetres. Where the dot can't be seen (dark fabric), the geometric aim stands alone.

Proposed config: a new `turret_geometry:` section at the end of `config.yaml` with `camera: {height_m, pitch_deg, roll_deg, yaw_deg, hfov_deg, dist_coeffs}`, `laser_offset_m`, `zone_heights_m: {couch, side_table, counter, floor}` and `max_range_m`. Per rig values go in `config.local.yaml`.

### Acceptance tests

- Synthetic room: a camera at 2.3 m, 30° down, with the laser 1.5 m below it; random points on the floor and on 0.45 m and 0.75 m planes, projected to pixels. Pixel + zone → pan/tilt puts the beam within 1 mm of each point (noise-free), and within 5 cm at 3 m with 1° of pose error.
- Depth matters: the same pixel on two planes gives different aims whose difference matches the parallax formula.
- A ray that misses its plane, hits behind the camera, or needs an aim outside `servo_limits` is refused with a reason; the laser stays off.
- On the rig: from the couch seat and the side table, three objects each, the dot within 10 cm of the object with no dot feedback, and inside the object's box with it.
- Safety: never fires outside a zone, or with a person or hand in view (the existing `room` gates).

## Not in scope

Depth cameras, per-object height models, and moving the laser onto the camera's mount (that's the alternative fix for B3–B5).
