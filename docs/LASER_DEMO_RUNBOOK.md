# Laser demo window: the runbook (validated on the rig, Sun 27 Sep 03:55–04:40)

Goal: "Room, where is the pill bottle?" is spoken and answered, and the red dot lands on the object (stepper
turret, `actuator: turret`). Every command below ran on the rig tonight; timings are measured. Safety rules are
in `docs/LASER_SAFETY.md`; this is the procedure. The live app runs the freeze build in `~/askroom_room`
(actuator fake); the laser build lives in `~/askroom_laser` (branch `ws/laser-demo`) and never touches it.

Roles: **U** the user (hands on the hardware, eyes on the room), **L** the laser operator (ssh from the Mac),
**D** the deploy owner (the only one who starts or stops the app; stops containers **only by exact ID**; a stop
by image name killed a running sweep tonight).

## 0. Checked steps before anything opens the serial port (U)

- [ ] Stepper **battery ON** (tonight a sweep ran with it off: the head stayed level while the firmware believed
      −11..−60°, so its tilt cap let the beam blink level across the empty room; the stall check now catches it).
- [ ] Head **re-homed by hand**: battery off, head level (phone level) and facing the same way as the camera,
      battery on. Opening the port makes that pose 0,0 and the −10° lit-tilt cap is relative to it. Re-home after
      anything that stopped the turret without parking (a killed container, a hub reset, a crash).
- [ ] The Uno on the Jetson (`ls /dev/ttyACM0`). It shares the external hub with the Brio: stepper noise reset
      that hub once (both re-enumerated); the software fails safe (laser locked off after an Uno reset).
- [ ] Nobody in front of the turret for steps 2–4 (HOG misses seated people; the empty room is the protection).

## 1. Watched dark range (L, 15 s; U watches)

```
ssh guru@10.90.84.178
cd ~/askroom_laser && ASKROOM_IMAGE=askroom:audio scripts/dock.sh python3 scripts/laser_bringup.py range --steps 3 --yes
```
U confirms "it moved": pan swung both ways, tilt went **down** only, then it parked. No "it moved", no light.

## 2. One lit test point (L, 20 s): does the camera see the dot?

```
ASKROOM_IMAGE=askroom:audio scripts/dock.sh python3 scripts/laser_testpoint.py 0 -22 --blinks 3
```
Tonight: dot px (1127, 889) on the couch seat, score 103 (threshold 40), firmware `POS 0.00 -22.01 0 1`. A tilt
of −40° or lower lands below the camera's view (the floor under the corner): aim −20..−25° for this test.

## 3. The sweep (L, ~4 min, 12×9, laser blinking; detached, own statement)

```
cd ~/askroom_laser && rm -f room_map.partial.json
ASKROOM_IMAGE=askroom:audio setsid nohup scripts/dock.sh python3 -m act.room_map --sweep --grid 12 9 \
    --camera /dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0 > data/sweep.log 2>&1 < /dev/null &
```
Launch it **as its own command** (no `a && b &`: that backgrounds the chain in a subshell that holds the ssh
session). Watch `data/sweep.log` (`10/108 ... wrote room_map.json: N points, M dots seen`). If it stops
(a person, a stall, a docker stop), it saves `room_map.partial.json` and `data/sweep_abort_HHMMSS.jpg`
(what it saw); after re-homing, rerun with `--resume` **only** if the partial came from this same verified
session. Tonight: 132 points, 91 dots; per zone: table 18, couch 6, doorway 2, side table 1, counter 1.

Rig config that made it work (`~/askroom_laser/config.local.yaml`, appended after `~/askroom_room`'s):
```
actuator: turret
turret: {port: /dev/ttyACM0, tilt_max_deg: -10}
servo_limits: {pan: [1050, 1950], tilt: [1050, 1390]}   # ±45° pan, −45..−11° tilt (−60 is below the view;
                                                          #  the top row must be inside the −10° lit cap)
room: {enabled: true, head_px: camera, room_dwell_s: 4, max_map_gap_px: 300}
laser_room: {ignore_px: [[850, 1250, 1300, 1440]]}       # the turret itself, at the frame's bottom edge
```

## 4. Zones, and densify the thin ones (L, ~2 min each)

```
ASKROOM_IMAGE=askroom:audio scripts/dock.sh python3 -m act.room_map --zones-from room_zones.json
ASKROOM_IMAGE=askroom:audio setsid nohup scripts/dock.sh python3 -m act.room_map --densify couch,side_table \
    --camera /dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0 > data/densify.log 2>&1 < /dev/null &
```
`--densify` sweeps only the pan/tilt range whose dots fell in those zones, ~40 px apart, and merges the result
into `room_map.json` (a stopped run resumes with `--resume`).

## 5. The laser app (D, ~40 s)

From `~/askroom_laser`, the usual run command and image, plus
`ASKROOM_DOCKER_ARGS="-v /home/guru/askroom_room/models:/home/guru/askroom_room/models -v /home/guru/askroom_room/third_party/whisper.cpp:/askroom/third_party/whisper.cpp:ro"`.
Voice: the GPU whisper-server can't share the GPU with the app's extra CUDA session (NvMap OOM tonight): run
the CPU whisper-server on the host at :8178 (the app reuses it), or start the app with `--no-voice`. The log
must show `room map: N dots, zones: ...`; "laser not calibrated (laser_cal.json missing)" is expected (that is
the table-only fit; aims go through the room map).

## 6. Aims (U asks, L watches)

- An aim needs a **tracked** object with a box: it must be VISIBLE on `/state` (a `thing:N` with its name, or a
  prop). Put objects down and leave them ~30 s, hands away. Grok room-look answers ("the deodorant on the side
  table") and stale "I last saw" answers never aim, by design.
- Ask by its name on `/state`: "Room, where is the pill bottle?" (typed: `curl -s -X POST
  http://10.90.84.178:8080/ask -H 'Content-Type: application/json' -d '{"text":"where is the pill bottle?"}'`).
- Watch `docker logs -f <id> | grep "laser ->"`: `in_box after 1 tries, err 11.2 px` is a hit (lit ≤ 4 s);
  `unmapped` (no map near the target), `jumped`/`stalled`/`unsafe`/`budget` are safe stops (dark).
- Tonight: pill bottle on the table, 2 of 3 in the box on the first try (10.9 and 11.2 px), 1 safe stop (a jump).

## 7. Restore (D)

Stop the laser app by its exact ID (its shutdown parks the head; confirm, or re-home next time), then launch the
freeze build in `~/askroom_room` (actuator fake).
