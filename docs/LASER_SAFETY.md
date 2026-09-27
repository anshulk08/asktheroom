# Laser safety: before the laser is ever powered

The room laser (stepper turret, `actuator: turret`) must not be powered, in any room with people, until every
item below is checked and written down. The code enforces what it can (see "What the software does"); the
rest is hardware and procedure. Owner: whoever powers it.

## Before power (hardware)

1. **The module is Class 2 (< 1 mW)**, documented: a photo of its label or its datasheet, kept with the rig
   notes. Unknown green modules can exceed their label and leak infrared. No documentation, no power.
2. **A 10 kΩ pull-down** on the laser switch's gate or base (the D9 side). The Uno's pins float during reset and
   the ~1 s bootloader, which happen every time the serial port opens.
3. **The switch defaults off without the Uno**: with USB unplugged (the Uno unpowered) the laser must be dark.
4. **Firmware from `ws/laser` cfdc1fc or later** is flashed (`firmware/README.md`): off after 1 s without a
   command, off 5 s after it was lit, off on any ERR, a 500 ms watchdog. The fe7f144 firmware has none of these.

## First power-on (an empty room, nobody in the beam's reach)

5. **Dark through resets**: with the laser module powered, check the dot stays dark through a port open (the app
   or `scripts/laser_bringup.py check`), a USB unplug and replug, and a watchdog reset.
6. **Cut-offs**: `scripts/laser_bringup.py jog --laser --eye-safe-confirmed --max-on-s 4` lights it; check it goes
   dark by itself at 4 s, and that killing the process (Ctrl-C, `kill -9`) darkens it within 1 s (firmware silence).
7. **The room map sweep** (`python -m act.room_map --sweep`, the laser blinks across the whole servo range) and
   `act.calibrate --rig` run only in the empty room.

## Rig configuration

8. `room.head_px` is set (the laser head's position in the full camera frame): without it every aim is refused.
9. `servo_limits` are narrowed to the demo spots (start at `[1050, 1950]`, ±45° on the turret; the jog prints them).
10. No stale `laser_cal.json`: a fit not made with the turret is ignored on it; refit with `act.calibrate --rig`.
11. Nothing else runs on the turret's port while the app does (bring-up, calibrate, sweep): the port is exclusive,
    and a second opener resets the Uno. After an unexpected reset the app refuses the laser until it restarts.

## What the software does

- The head moves only with the laser off; the dot is lit only at the target and stays lit only on a confirmed hit.
- Before every aim, and before every look within one, the perception thread looks for people on the full frame
  (YOLOE's person and clothing classes, any size); a person or recent hand near the target or the beam's path from
  `room.head_px` refuses it, as do a camera frame older than 0.5 s or no detector answer (HOG only if
  `laser_room.allow_hog`). An object in someone's hand is never pointed at; edge sweeps are off by default.
- An aim stops at a jump of the dot (something nearer in the beam), and is lit at most `laser_room.max_on_s` (4 s)
  from its first light; the actuator's own timer and the firmware's 5 s cap back it up.
- Off at startup, after any failed aim, on close; the actuator's off timer survives serial errors.
