# DUMMY DATA: simulated room, not a real recording

**Everything in this folder is dummy data.** No camera, detector, person or real table produced it.
It came from the repo's room simulator (`python -m server.sim`), which plays a scripted tabletop story
through the real world model. Do not use it as evidence that perception, tracking or answers work on
the rig, and never quote numbers from it as real accuracy or latency (AGENTS.md, "Quality bar").

Use it for: iOS previews and tests, protocol and parser checks, and demos of the app with no rig.

## How it was made

Recorded on Sat Sep 26 2026, around 06:30 EDT, on the Jetson:

- The simulator ran in the container (`scripts/dock.sh python3 -m server.sim`) and served `localhost:8000`.
- `ble_bridge.py` (the `main` version) relayed it over Bluetooth to an iPhone running the app.
- `GET /state` was polled every 2 s for 100 s, covering one full loop of the story. Then `GET /events`
  was read once, and the question and answer lines were copied from `data/ble_bridge.log`.
- `phone_states.jsonl` was computed afterwards on a Mac with `bleproto.compact_state` from
  `teammate-tasks` (table 90 × 60 cm). It was not sniffed off the air.

The story loops, as `server/sim.py` `story()` defines it: all objects on the table, then the keys go
into the box, the pill bottle goes under the notebook, the wallet is picked up and put back, the remote
is moved, and the phone is carried off the right edge. The room holds that state for 25 s, then
starts again.

## Files

| File | What it is |
|---|---|
| `room_states.jsonl` | 50 raw `GET /state` bodies, 2 s apart: `state` (the room app's WorldState), `answers`, `last_answer`, `server_t` |
| `phone_states.jsonl` | The same 50 states in the compact form the phone receives on the state characteristic (`mobile/PROTOCOL.md` on `teammate-tasks`) |
| `events.json` | 81 events from `GET /events` (PICKED_UP, PUT_INSIDE, EXITED_VIEW…). `snapshot_url` points at simulator images on the Jetson and was not copied |
| `answers.jsonl` | 11 questions asked from the phone, with the answer text, `point_at`, `target` (table cm) and bridge time in ms |

## Caveats

- **Timestamps** are Unix seconds from that run (`t`, `ls`, `wall`), so every age is in the past.
  Shift them if a test needs "just now".
- **The first line of `answers.jsonl`** (`ok:false`, "The room isn't running right now.") was asked
  before the simulator started. It is the real bridge reply for "room app down", and is kept on purpose.
- **Answer times** (5–30 ms) are simulator times with no laser, no speech and no detector. The real rig
  is much slower.
- **No notices or room answers.** No rig notices (`src:"notice"`) and no voice, dashboard or SMS answers
  happened during the capture, so there are none here. The app's mock mode covers those (`-mockNotice` in `MockRoom.swift`).
- **Nothing personal.** Object names are the eight demo objects, and no audio or images are included.
