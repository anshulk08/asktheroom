# Ask the Room: project context

Start here if you're new to the repo, and point the `n8n/` chat bot at this file first. It covers what
the project is, how the code fits together, and what's done. Interfaces and pass tests live in the spec:
https://claude.ai/artifact/ETtzPXaFAWWkCJoXjuKiAP (section numbers like 3.5, V6 and H4 in the code refer to it).
Contributor and agent rules are in `AGENTS.md`, the plan in `PLANS.md`, per-feature status in
`docs/FEATURE_STATUS.md`. This file stays the short version (the n8n ask-the-repo bot reads it by URL).

## What it is

A HackGT 13 project (36 h, live demo judging). An overhead camera watches a tabletop and keeps track
of eight objects (keys, pill bottle, wallet, glasses, phone, remote, a box and a notebook), including
ones it can't currently see. You just ask "where are my keys?" (the mic is always on; a clicker
works too). It answers out loud
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
- Microphone (always on), speaker, and a presentation clicker ("listen now" and interrupt).

## Data flow

```
camera ─▶ core/capture.FrameBuffer (30 fps thread)
        ─▶ core/detect.Detector (YOLO, TensorRT, ~14 ms) ─▶ core/hands.HandTracker (stable hand ids)
        ─▶ core/world.World.update()  ── rules from spec 3.5 ──▶ core/events.EventLog (SQLite + JPEG snapshots)
                     │
always-on mic ─▶ voice/stt (Silero VAD + whisper.cpp; audio only in memory; muted while the rig speaks)
          ─▶ voice/understand: meant for the rig? (keyword gate, then wake word "room" or a question
             opening; chatter is dropped, never logged) ─▶ voice/intents.parse, then Qwen3 1.7B
             (llama.cpp on the Jetson, scripts/qwen_server.sh) for what the rules can't read
          ─▶ voice/pipeline.make_ask
                     ├─ WHERE / HANDLED / CHANGES / ... ─▶ voice/answers (templates)
                     └─ OTHER ─▶ voice/local_llm (templates for the common ones, else one Qwen call
                                 with the world state; same pill-wording filter)
          ─▶ Answer(speech, point_at, action)
                     ├─ voice/tts: ElevenLabs when online, Piper offline
                     └─ act/laser.Laser.aim_object: closed-loop aim, corrects on the camera's view of the dot
server/app.py (FastAPI): dashboard, MJPEG overlay, WebSocket state, POST /ask, /sms (Twilio)
main.py ─▶ n8n webhook (laptop): a log of every spoken question, plus a 5-minute health check
```

Every answer is worked out on the Jetson, online or not. Privacy, honestly: video and audio stay on
the device and audio is never written to disk; event snapshots (JPEGs) are kept 24 h (pruned at
start); overheard speech that isn't a question for the rig is dropped without being logged. What
leaves the device: answer text to ElevenLabs for the voice when online, texts via Twilio for /sms,
and the question log to the team's own n8n on the laptop. Grok is not on the voice path (team
decision); it is planned only as a detector helper (`docs/specs/0003-grok-detection-assist.md`).

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
| `voice/` | `intents` (rule parser), `answers` (spoken templates), `understand` (overheard filter + Qwen reads what the rules can't), `local_llm` (open questions, local), `pipeline` (router), `tts`, `stt` (Silero VAD + whisper.cpp), `trigger` (clicker), `llm` (world-state helpers and pill filter; its Grok call is off the voice path). |
| `act/` | `actuator` (servo drivers + fake), `laser` (poly2 fit + closed-loop aim), `calibrate`, `sim` (simulated rig). |
| `server/` | FastAPI dashboard (`app.py`), frame overlay, `sim.py` (full demo on a synthetic camera). |
| `eval/` | Trial recording, synthetic trials, replay against baselines (last-seen, nearest-object, current-frame) and the report. |
| `net.py` | Online/offline monitor. Readers check `.online`, which never blocks. |
| `scripts/` | `dock.sh` (run inside the Jetson Ultralytics container), camera setup, markers PDF, servo sweep, `qwen_server.sh`, `eval_understand.py` (interpreter accuracy per model), `overheard_test.py` (false triggers on a hall recording), `gen_n8n_workflow.py`. |
| `tests/` | About 610 tests. None need hardware. `understand_eval.json`: 64 spoken-style commands for the interpreter. |
| `n8n/` | `ask-the-room.json`: a live log of every spoken question plus a 5-minute health check. `ask-the-repo.json`: a chat bot about this repo. See `n8n/README.md`. |

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

## Status (Sat 26 Sep, before the 6 PM freeze)

Done and tested: the world model (with the hand-slides-cover and carried-in-view rules), event log,
config, capture (29.9 fps live), table calibration, YOLO-World detection (14 ms engine), hand tracking,
intents, answers, TTS, laser math and calibration against the sim, dashboard, and the eval
harness. The synthetic eval scores 255/300. That number is synthetic and says nothing about how well the
real detector works.

Also built, tested on the laptop, not yet on the Jetson: speech input (Silero VAD + whisper.cpp) and the
clicker, `main.py` wiring all threads together, the fine-tuning scripts (zero-shot YOLO-World mistook the
Jetson case for a phone), `demo_check.py`, Qwen question understanding and local open answers, and the always-on mic.
Interpreter eval on the laptop (`scripts/eval_understand.py`): rules alone 48/64, rules + Qwen3 1.7B
58/64 (overheard 16/16), median 114 ms. Still to do on the Jetson: build llama.cpp, time Qwen next to
YOLO and whisper (`tegrastats`), and the 10-minute hall-noise test (`scripts/overheard_test.py`). Stretch goals (floor search camera, room map) wait until checkpoints D8/D9
pass.

## Decisions worth knowing

- **Rules, not a learned tracker**, for the world model. It's deterministic and testable, and it explains itself on the dashboard.
- **Fine-tune YOLO11** on our own overhead frames. YOLO-World is only the baseline and the auto-labeller.
- **Hands come from the detector's `hand` class.** MediaPipe was dropped: it runs CPU-only on the Jetson and misses hands that hold objects.
- **The laser corrects in a closed loop** from the camera's view of the dot. Frame-diff dot finding has to allow for camera latency.
- **Local Qwen, not Grok, for speech.** Qwen3 1.7B beat Qwen2.5 1.5B on speed and open answers at the same intent accuracy. The model never writes coordinates, only picks an object from an enum.
- **Templates stay for the core questions**; the model only covers what they can't. They're exact, tested and instant.
- **"Was that for me?" is decided by rules, not the model**: Qwen got 8/16 overheard lines right, the gate + wake word + question-opening checks 16/16.
- **The eval compares against a "nearest object to the last-seen spot" baseline**, so any win is measured against something reasonable.
