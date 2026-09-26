# Ask the Room: project context

Start here if you're new to the repo, and point the `n8n/` chat bot at this file first. It covers what
the project is, how the code fits together, and what's done. Interfaces and pass tests live in the spec:
https://claude.ai/artifact/ETtzPXaFAWWkCJoXjuKiAP (section numbers like 3.5, V6 and H4 in the code refer to it).
Open tasks for teammates are in `TEAMMATE.md`.

## What it is

A HackGT 13 project (36 h, live demo judging). An overhead camera watches a tabletop and keeps track
of eight objects (keys, pill bottle, wallet, glasses, phone, remote, a box and a notebook), including
ones it can't currently see. You press a clicker and ask "where are my keys?". It answers out loud
("under the notebook, you slid it over them 2 minutes ago") and a pan-tilt laser points at the spot.

The hard part is object permanence. A detector only says what is visible right now, so the world model
keeps a belief for every object: VISIBLE, HELD by a hand, UNDER a cover, INSIDE a container, or GONE off
an edge of the table. Hidden objects inherit their parent's position through a chain
(keys → notebook → table), so moving the notebook moves where the laser points.

## Hardware

- Jetson Orin Nano (JetPack 6.2, Python 3.10, CUDA 12.6), run in MAXN_SUPER power mode.
- icSpring USB camera mounted overhead: MJPG 1280x720 at 30 fps, manual exposure,
  `CAP_PROP_BUFFERSIZE=2` (a value of 1 halves the fps). `scripts/camera_setup.sh` locks these settings. It needs a lamp in dim rooms.
- Four ArUco markers (0–3, DICT_4X4_50) on the table corners define table centimetres.
- Pan-tilt servo head with a laser diode (PCA9685 or serial/bus-servo driver).
- Presentation clicker (push to talk), microphone and speaker.

## Data flow

```
camera ─▶ core/capture.FrameBuffer (30 fps thread)
        ─▶ core/detect.Detector (YOLO, TensorRT, ~14 ms) ─▶ core/hands.HandTracker (stable hand ids)
        ─▶ core/world.World.update()  ── rules from spec 3.5 ──▶ core/events.EventLog (SQLite + JPEG snapshots)
                     │
question ─▶ voice/intents.parse ─▶ voice/pipeline.make_ask
                     ├─ WHERE / HANDLED / CHANGES / ... ─▶ voice/answers (offline templates)
                     └─ OTHER, and online only ─────────▶ voice/llm (Grok grok-4.3, reasoning none)
          ─▶ Answer(speech, point_at, action)
                     ├─ voice/tts: ElevenLabs when online, Piper offline
                     └─ act/laser.Laser.aim_object: closed-loop aim, corrects on the camera's view of the dot
server/app.py (FastAPI): dashboard, MJPEG overlay, WebSocket state, POST /ask, /sms (Twilio)
```

Core answers never need the network. Only open-ended (OTHER) questions go to Grok, and only question
text plus a compact world-state JSON leave the device.

## Repo map

| Path | What it holds |
|---|---|
| `core/types.py` | Shared data contracts: Frame, Detection(s), Entity, Event, Intent, Answer. **Change only as a team.** |
| `core/config.py`, `config.yaml` | Config loaded as a plain dict (`load_config()`); the world also reads a typed `Config` view. All thresholds are here. |
| `core/capture.py` | Camera `FrameBuffer`, plus `VideoFileSource` with the same API for replays. |
| `core/table.py` | ArUco homography mapping pixels to table cm (`table_cal.json`). |
| `core/detect.py` | YOLO-World (zero-shot, path A) or fine-tuned YOLO11 (path B, the plan), exported to TensorRT. |
| `core/hands.py` | Stable `hand:N` ids across frames. |
| `core/world.py`, `core/relations.py`, `core/geom.py` | Deterministic, rule-based world model (covers, containers, holds, edges, parent chains). About 200 tests. |
| `core/events.py` | EventLog: SQLite event history, questions table and snapshots. |
| `core/fakeworld.py` | Stand-in world with the same read API, for tests and `--fake` runs. |
| `voice/` | `intents` (rule parser), `answers` (spoken templates), `llm` (Grok tools), `pipeline` (router), `tts`. STT and the clicker are not built yet. |
| `act/` | `actuator` (servo drivers + fake), `laser` (poly2 fit + closed-loop aim), `calibrate`, `sim` (simulated rig). |
| `server/` | FastAPI dashboard (`app.py`), frame overlay, `sim.py` (full demo on a synthetic camera). |
| `eval/` | Trial recording, synthetic trials, replay against baselines (last-seen, nearest-object, current-frame) and the report. |
| `net.py` | Online/offline monitor. Readers check `.online`, which never blocks. |
| `scripts/` | `dock.sh` (run inside the Jetson Ultralytics container), camera setup, markers PDF, servo sweep, Grok smoke test. |
| `tests/` | About 475 tests. None need hardware. |
| `n8n/` | A chat workflow that answers questions about this repo (see `n8n/README.md`). |

## Conventions

- Positions are **table centimetres**: origin at marker 0, x to the right, y down. A field is in pixels only if its name says so.
- Code must run on **Python 3.10** (JetPack 6). No `match` statements and no 3.11+ stdlib.
- Add new config keys in new sections at the end of `config.yaml`. Don't rename existing keys.
  Tune thresholds from replays, never by guessing during a live run.
- Readers of the world use only `get / resolve / history / state_json` (`WorldAPI`).
- Spoken answers are 1–2 short sentences with no markdown. Pill-bottle wording stays neutral (never "taken").

## Running it

```bash
python -m pytest -q                 # all tests, ~10 s, no hardware
python -m server.sim                # dashboard at http://localhost:8000 on a scripted synthetic camera
python -m server.sim --check        # headless: print each step's events and final beliefs
python -m act.calibrate --sim       # laser calibration against the simulated rig
python -m eval.synth --out /tmp/t --per-category 5 && python -m eval.replay --trials /tmp/t --system all
scripts/dock.sh python3 -m core.detect --export   # on the Jetson: build the TensorRT engine in the container
```

The Jetson is at `guru@192.168.55.1` over USB-C. Build TensorRT engines inside the same container that
runs them. Ask before running anything heavy there, because the Orin Nano runs out of memory easily.

## Status (Fri 25 Sep night)

Done and tested: the world model (with the hand-slides-cover and carried-in-view rules), event log,
config, capture (29.9 fps live), table calibration, YOLO-World detection (14 ms engine), hand tracking,
intents, answers, Grok fallback, TTS, laser math and calibration against the sim, dashboard, and the eval
harness. The synthetic eval scores 255/300. That number is synthetic and says nothing about how well the
real detector works.

Open (see `TEAMMATE.md`): speech input (Silero VAD + whisper.cpp) and the clicker, `main.py` wiring all
threads together, the fine-tuning pipeline (zero-shot YOLO-World mistook the Jetson case for a phone),
and `demo_check.py`. Stretch goals (floor search camera, room map) wait until checkpoints D8/D9 pass.

## Decisions worth knowing

- **Rules, not a learned tracker**, for the world model. It's deterministic and testable, and it explains itself on the dashboard.
- **Fine-tune YOLO11** on our own overhead frames. YOLO-World is only the baseline and the auto-labeller.
- **Hands come from the detector's `hand` class.** MediaPipe was dropped: it runs CPU-only on the Jetson and misses hands that hold objects.
- **The laser corrects in a closed loop** from the camera's view of the dot. Frame-diff dot finding has to allow for camera latency.
- **Grok model: `grok-4.3` with reasoning `none`** for speed. Older Grok model names are retired.
- **The eval compares against a "nearest object to the last-seen spot" baseline**, so any win is measured against something reasonable.
