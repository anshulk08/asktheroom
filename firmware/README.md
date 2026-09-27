# Stepper turret firmware

The laser's pan-tilt head, built from the Instructables "Remotely Operated WEBCAM / Laser" mount:
two NEMA17 steppers, each on an MKS SERVO42D closed-loop driver (STEP/DIR), driven by an Arduino
Uno R3. The host (the Jetson, or a laptop) talks to the Uno over USB serial. In the app, set
`actuator: turret` and `turret.port` in `config.local.yaml`; `act/actuator.py` `TurretActuator`
does the rest.

| Sketch | Use |
|---|---|
| `turret/turret.ino` | The real firmware: degrees over serial, both axes at once, laser control |
| `servo42d_test/servo42d_test.ino` | Bring-up only: `1` = pan test move, `2` = tilt, `x` = release |

## Wiring

Each driver's right-hand six-terminal connector (screen facing you, buttons at the bottom), top to
bottom, is V+, GND, COM, EN, STP, DIR. Identify the wires in the driver's cable by position, not
colour: its red wire is V+ (battery voltage), not 5 V.

| Driver terminal | Pan driver | Tilt driver |
|---|---|---|
| V+ | battery + | battery + |
| GND | battery − (shared with Uno GND) | battery − (shared with Uno GND) |
| COM | Uno 5V | Uno 5V |
| EN | D5 | D2 |
| STP | D6 | D3 |
| DIR | D7 | D4 |

The pan driver is on D5–D7 and tilt on D2–D4. The original diagram had them the other way round,
and the firmware follows the wiring as built. The laser module's signal goes to D9 (a placeholder
until the module is wired; `LASER_PIN` in `turret.ino`).

**Power.** The battery powers the two drivers only. The Uno takes its power over USB from the
host. Keep the Uno GND ↔ battery − wire, and leave the Uno's VIN pin unconnected: with battery +
on VIN, the ATmega328P locked up as soon as the battery came on. Its serial went silent, avrdude
reported `not in sync`, and only unplugging USB brought it back.

## Driver settings (SERVO42D menu, both drivers)

| Setting | Value |
|---|---|
| Mode | CR_vFOC |
| Ma | 400 mA as a first try; moves are weak at 400, so go up to about 800 mA or 70–80% of the motor's rated current |
| MStep | 16 |
| En | L (the Uno drives EN low to enable) |
| Dir | CW |
| MPlyer | Enable |
| Protect | Enable |

Run CAL on each driver with the belt off before first use, and again after any motor rewiring.

## Mechanics

| Axis | Gear | Positive | Status |
|---|---|---|---|
| Pan | 20T → 80T, 4:1 | right | measured: ±90° swings exact |
| Tilt | 20T → 64T, 3.2:1 | up | direction confirmed; ratio from the build guide, not yet measured |

These live in `GEAR_RATIO` and `INVERT_DIR` in `turret.ino`. The firmware's soft limits are pan
±170° and tilt −45° to +90° (`MIN_DEG`/`MAX_DEG`); `act/pointing.py` `FIRMWARE_LIMITS_DEG` mirrors
them. Opening the serial port resets the Uno, and wherever the mount points then becomes 0,0, so
line it up level and facing forward first.

## Flashing

```bash
arduino-cli core install arduino:avr
arduino-cli compile --fqbn arduino:avr:uno firmware/turret
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno firmware/turret   # Mac: /dev/cu.usbmodem*
```

## Protocol

115200 baud, one command per line, degrees at the axis. The full list is at the top of
`turret.ino`; `python -m act.turret <port>` opens a prompt for typing commands.

| Command | Does |
|---|---|
| `A <pan> <tilt>` | aim; replies `OK A <pan> <tilt>` (after the limits), then `DONE` when it arrives |
| `R <dpan> <dtilt>` | relative to the current target |
| `L 0/1`, `B 0-255` | laser off/on, brightness |
| `S`, `X` | smooth stop; emergency stop (laser off, motors released) |
| `V <deg/s>`, `C <deg/s²>` | max speed (default 120), acceleration (default 600) |
| `P` | `POS <pan> <tilt> <moving> <laser>` |

The laser switches itself off after 2 s without a command; `act/turret.py` sends `P` twice a
second while the laser is on.

## Checks

- `python scripts/turret_point_test.py <port>`: aims at seven imaginary objects (45° right,
  straight up, ...) through `TurretActuator.point_at`, holds each for 4 s so you can check it by
  eye, and confirms the board landed where the math asked.
- `python -m act.turret <port> demo`: pan 90° each way, tilt 45° up.
