# Ask the Room

An overhead camera tracks eight objects on a tabletop, including objects hidden under or inside other things. You can ask it out loud where something is. It answers in one or two sentences, and a pan-tilt laser points at the spot.

Built at HackGT 13 (Atlanta, Sep 25–27, 2026) on a Jetson Orin Nano.

<!-- Demo GIF: Demo/shell-game.gif (see Demo/README.md). -->
> Demo GIF: coming soon (`Demo/shell-game.gif`, listed in [`Demo/README.md`](Demo/README.md)).

## Why it matters

Losing everyday objects costs people time and stress, and much more so for someone with memory loss. Cameras and vision models can only report what is in view right now. They can't answer the common case: keys under a notebook, or a notebook that has since been put in a box and slid across the table. Ask the Room keeps track of objects while they are hidden, and it answers by pointing at the actual spot instead of describing it.

## What it does

- Tracks keys, a pill bottle, a wallet, glasses, a phone, a remote, a box (a container) and a notebook (a cover). The list is in `config.yaml` under `objects`.
- Keeps a belief for each object even when it can't be seen: VISIBLE, HELD by a hand, UNDER a cover, INSIDE a container, GONE off an edge of the table, or UNKNOWN.
- Hidden objects move with their parent. Keys under a notebook that is inside a box follow the box.
- Listens all the time. It answers only speech meant for it and drops everything else without logging it. A clicker means "listen now" and interrupts the current answer.
- Answers out loud (ElevenLabs online, Piper offline) and aims the laser at the same time: a point for visible objects, a circle when unsure, and a sweep along the edge when an object left the table.
- Answers "what happened to my keys?", "did anyone touch my pills?" and "what did I miss?" from a SQLite event log.
- Runs a dashboard with live video, the belief graph and an event timeline, and answers text messages through Twilio.

## Architecture

```
camera ─▶ core/capture.FrameBuffer (30 fps thread)
        ─▶ core/detect.Detector (YOLO, TensorRT) ─▶ core/hands.HandTracker (stable hand ids)
        ─▶ core/world.World.update()  (rules) ─▶ core/events.EventLog (SQLite + JPEG snapshots, 24 h)

mic (always on, audio in RAM only)
  ─▶ voice/stt: Silero VAD ─▶ whisper.cpp base.en
  ─▶ voice/understand: "was that for me?" (keyword gate + addressed check; else dropped, not logged)
        ─▶ voice/intents rule parser ─▶ Grok (online) for what the rules can't read
  ─▶ voice/pipeline.make_ask
        ├─ about what the camera sees ─▶ voice/visual: Grok with the frame (set-of-marks) or saved frames
        ├─ WHERE / HISTORY / HANDLED / CHANGES ─▶ voice/answers (templates)
        └─ OTHER ─▶ voice/llm.ask_other (templates, then Grok with the world state and tools)
  ─▶ Answer(text, point_at, action)
        ├─ voice/tts: ElevenLabs online, Piper offline (mic stays shut while speaking + a short tail)
        └─ act/laser: closed-loop aim on the camera's view of the dot; circle; edge sweep

server/app.py (FastAPI): dashboard, MJPEG overlay, WebSocket state, POST /ask, POST /sms (Twilio)
main.py ─▶ n8n webhook on the team laptop: question log + 5-minute health check
```

Walk through the key files in the order data flows:

1. `core/capture.py` reads the overhead camera (MJPG 1280x720 at 30 fps) into a ring buffer. `VideoFileSource` has the same API for replays.
2. `core/table.py` maps pixels to table centimetres through a homography fitted on ArUco markers 0–3.
3. `core/detect.py` runs YOLO-World v2 with text prompts (path A) or a fine-tuned YOLO11 (path B), exported to TensorRT. It keeps the best box per object and every hand box.
4. `core/hands.py` gives hands stable `hand:N` ids across frames.
5. `core/world.py` and `core/relations.py` hold the deterministic world model: debounce, hand contact, covers, container dwell, background change, parent chains and decay. See [`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md).
6. `core/events.py` stores events, question logs and snapshots in SQLite. On startup it prunes snapshot JPEGs and state snapshots older than 24 h.
7. `voice/stt.py` detects speech with Silero VAD and transcribes it with whisper.cpp, using a prompt that lists the object names.
8. `voice/understand.py` decides whether overheard speech was meant for the rig, then runs the rule parser (`voice/intents.py`) and falls back to Grok when the rules can't read it (offline, the rules alone answer).
9. `voice/pipeline.py` routes the intent. `voice/answers.py` fills templates for the core intents. `voice/local_llm.py` answers open questions.
10. `voice/tts.py` speaks. `act/laser.py` aims, circles or sweeps. `main.py` wires the threads together and reports each answered question to n8n.

## Demo flow (the shell game)

A judge runs it, and it resets in under a minute:

1. All eight objects start on the table. Run `python demo_check.py` first: it should print all green. A red `clock` line means the Jetson booted offline with a stale clock: join the phone hotspot so NTP sets it, or `sudo date -s "..."` on the host.
2. The judge puts the keys on the table, slides the notebook over them, puts the notebook into the box, and slides the box across the table.
3. The judge asks "where are my keys?". The rig answers something like "Your keys are under the notebook, which is inside the box" and the laser points at the box.
4. Follow-up questions: "what happened to my keys?" (the history), and "did anyone touch my pills?" (the pill-bottle wording stays neutral).
5. Carry the wallet off the left edge and ask for it. The rig says it was carried off the left side, and the laser sweeps that edge.
6. Reset: put the objects back and say "room, reset" (or type "reset" into the dashboard's question box).

## Current stack

| Part | Choice |
|---|---|
| Compute | Jetson Orin Nano 8 GB, JetPack 6.2, Python 3.10, CUDA 12.6, MAXN_SUPER |
| Camera | icSpring USB, overhead, MJPG 1280x720 at 30 fps, manual exposure (`scripts/camera_setup.sh`) |
| Table frame | 4 ArUco markers (DICT_4X4_50), homography to table cm |
| Detection | YOLO-World v2 (`models/yolov8s-worldv2-askroom.engine`), YOLO11 fine-tune planned; Ultralytics container (`scripts/dock.sh`) |
| World model | Rule-based, deterministic (`core/world.py`, `core/relations.py`) |
| Speech in | Silero VAD (ONNX) + whisper.cpp base.en (`pywhispercpp` on the laptop, `whisper-cli` on the Jetson) |
| Understanding | Rule parser, then Grok (`grok-4.3`, reasoning none) for what the rules can't read; offline the rules and templates answer (use a hotspot); local Qwen3-1.7B via `llama-server` is optional, not installed on the Jetson (`understand.backend: qwen`, or `auto`) |
| Speech out | ElevenLabs `eleven_flash_v2_5` online, Piper `en_US-lessac-medium` offline |
| Laser | Pan-tilt servos (PCA9685, serial or bus servo) + laser diode, 2nd-order poly fit + closed-loop correction |
| Server | FastAPI + uvicorn, vanilla JS dashboard, served locally so it works offline |
| Ops | n8n on the laptop (question log, health check), Twilio SMS |

## Prerequisites

- Python 3.10 (the Jetson's JetPack 6 version). The laptop setup uses [uv](https://github.com/astral-sh/uv).
- For the rig: a Jetson Orin Nano with JetPack 6.2 and Docker (the Ultralytics JetPack 6 image), a USB camera, a pan-tilt head with a laser, a USB mic and speaker, and a presentation clicker.
- `XAI_API_KEY` in `.env` for Grok, which reads questions the rules can't and answers open and visual questions. Local Qwen (`understand.backend: qwen`, needs `llama-server` from llama.cpp, see `scripts/qwen_server.sh`) is optional and not installed on the rig.
- Optional: `ELEVENLABS_API_KEY` (plus `ELEVENLABS_VOICE_ID`) for the online voice, and `TWILIO_AUTH_TOKEN` for SMS.

Nothing in the test suite needs hardware.

## Setup

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python numpy opencv-contrib-python pyyaml requests fastapi uvicorn \
    onnxruntime sounddevice pywhispercpp piper-tts twilio pytest
scripts/get_piper_voice.sh          # models/piper/en_US-lessac-medium.onnx
scripts/qwen_server.sh              # downloads models/qwen/Qwen3-1.7B-Q4_K_M.gguf, serves on :8081
```

There is no lock file yet. The list above covers the laptop path (`--fake`, the tests and voice). On the rig, detection runs inside the Ultralytics container (`scripts/dock.sh`), and the servo drivers need `adafruit-circuitpython-servokit` or `pyserial`. `evdev` reads the clicker on Linux. The Silero VAD model and the Whisper model download into `models/` on first use.

Rig setup, once per venue:

```bash
python scripts/make_markers.py                         # print ArUco 0-3, tape them to the table corners
scripts/camera_setup.sh                                # lock exposure (again after replugging the camera)
python -m core.table                                   # fit table_cal.json from the markers
scripts/dock.sh python3 -m core.detect --export        # build the TensorRT engine in the container
python scripts/sweep.py                                # check servo limits cover the table
```

Then calibrate the laser: on the rig, `act.calibrate.calibrate(Laser(...))` is called from Python with the live camera and table (the CLI explains how). `python -m act.calibrate --sim` runs the same procedure against the simulated rig.

## Configuration

Everything lives in `config.yaml`. All thresholds are starting values: tune them from replays, never live.

What differs per device goes in a gitignored `config.local.yaml` next to it, merged over `config.yaml` at startup (nested sections merge key by key). Copy `config.local.yaml.example`: on the rig it sets `actuator: pca9685`, since the committed default is `fake` and `main.py` warns at startup when the servos are fake. The sections you are most likely to touch:

| Section | What it controls |
|---|---|
| `objects`, `prompts`, `synonyms`, `display_names` | The 8 objects, their detector prompts and how they are spoken and heard |
| `detect` | Detector model path, image size, confidence floor |
| world keys (`present_k_of_n` … `answer_hedge`) | World model rule thresholds and confidences |
| `table`, `servo_limits`, `actuator`, `laser_*` | Table size and markers, pan-tilt driver and limits |
| `stt` | Whisper backend and VAD settings |
| `understand` | Model on/off, `backend: grok \| qwen \| auto` (default grok; auto: Grok online, Qwen offline), llama-server URL and model (qwen), intent timeout 1.5 s |
| `listen` | `mode: always \| wake \| click` (default `always`), `wake_words: [room]`, `idle_s`, `echo_tail_s` |
| `n8n` | `webhook_url` for the question log (empty turns it off), shared `token` |
| `demo` | `thinking_cue_s` (a short "Let me look." when an answer takes longer than this, default 1 s; 0 = off), `thinking_phrases`, `hold_notices` (true: reminders and the morning report never speak unasked) |
| `sms` | `whitelist` of E.164 numbers allowed to text questions |
| `paths` | Event DB, snapshot folder, calibration files |

The `llm` section configures Grok (`grok-4.3`): it reads questions the rules can't (`understand.backend: grok`) and answers open questions (4 s budget, `llm.timeout_s`). `narration` and `visual_memory` configure Grok with images.

Secrets come from environment variables (`.env`, passed into the container by `scripts/dock.sh`): `XAI_API_KEY` (Grok), `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `TWILIO_AUTH_TOKEN`, `ASKROOM_PUBLIC_URL` (for the Twilio signature check behind a tunnel).

## Running the system

```bash
.venv/bin/python -m pytest -q          # all tests, no hardware
python -m server.sim                   # dashboard at http://localhost:8000 on a scripted synthetic camera
python -m server.sim --check           # headless: each step's events and the final beliefs
python main.py --fake                  # whole program, no hardware: sim camera, sim laser, Enter = clicker
python main.py --fake --listen click   # same, but only the clicker/Enter opens the mic
python main.py                         # the rig (camera, detector, servos, mic, clicker)
python main.py --no-voice              # dashboard and /ask only
python demo_check.py                   # pre-demo check: one green or red line per check
```

Put `XAI_API_KEY` in `.env`. Offline (or without the key), the rule parser reads questions alone, open questions get a fixed fallback sentence, and questions about what the camera sees say they need the connection. Where things are and what happened to them always work.

Tools for single parts:

```bash
python -m voice.understand "ugh where did I put my specs"          # transcript -> intent
python -m voice.understand --overheard "I'll grab my keys later"   # always-on filter (IGNORE)
python -m voice.local_llm "what's in the box"                      # open question on the demo world
python scripts/eval_understand.py                                  # interpreter accuracy per set
python scripts/overheard_test.py hall.wav                          # false triggers on a hall recording
```

## API contracts

Served by `server/app.py` on `server.port` (8000).

**`POST /ask`**: ask a question as text (dashboard and testing). Answers from the dashboard are spoken and aimed too.

```http
POST /ask
Content-Type: application/json

{"text": "where are my keys?"}
```
```json
{"text": "Your keys are under the notebook, which is inside the box.",
 "point_at": "keys", "action": "point", "latency_ms": 212}
```
`action` is `"point"`, `"circle"`, `"sweep:<edge>"` or `null`. Text is cut to 500 characters. Missing or empty `text` returns 400. The server stops waiting after 10 s and answers "Sorry, that took too long. Please ask again."

**`GET /state`**: the world as JSON.

```json
{"state": {"t": 1790000000.0, "online": true, "fps": 14.8,
           "entities": [{"name": "keys", "kind": "target", "status": "UNDER", "parent": "notebook",
                         "pos_cm": [41.0, 22.5], "resolved_cm": [60.2, 38.0], "confidence": 0.85,
                         "candidates": [], "last_seen": 1789999950.0, "zone": "table", "edge": null}],
           "edges": [["keys", "UNDER", "notebook"], ["notebook", "INSIDE", "box"], ["box", "ON", "table"]],
           "laser": {"on": false, "target": null, "err_cm": null}},
 "last_answer": {"question": "...", "text": "...", "point_at": "keys", "action": "point",
                 "latency_ms": 212, "source": "dashboard", "t": 1790000001.0},
 "server_t": 1790000002.0}
```

**`GET /healthz`**: `{"ok": true}`.

**`POST /sms`**: Twilio's incoming-message webhook (form fields `From` and `Body`). The server checks the `X-Twilio-Signature` header against `TWILIO_AUTH_TOKEN`, and returns 403 if it is missing or wrong. It answers only numbers in `sms.whitelist`. Other numbers get an empty TwiML `<Response/>`. The reply is TwiML `<Response><Message>answer text</Message></Response>`. Texts are answered but never spoken or aimed. `ASKROOM_SMS_INSECURE=1` skips the signature check (local testing only).

Also served: `GET /` (the dashboard), `GET /video` (MJPEG with the overlay), `GET /frame.jpg`, `WS /ws` (state at `server.push_hz` plus new events), `GET /events?since=` and `GET /snapshots/{name}`.

## Repository layout

```
act/          pan-tilt actuators, laser fit + closed-loop aim, calibration, simulated rig
bench/        world-model timing
core/         capture, table homography, detector, hands, world model, relations, event log, types, config
data/print/   printable ArUco markers
eval/         trial recording, synthetic trials, baselines, replay scoring, report
n8n/          ask-the-room.json (question log + health check), ask-the-repo.json (repo chat bot)
scripts/      camera setup, Jetson container, Qwen server, interpreter eval, overheard test, markers, sweep, fine-tuning
server/       FastAPI app, overlay, simulated camera, web dashboard
tests/        pytest suite (no hardware), understand_eval.json, stt_questions.json
voice/        stt, understand, intents, answers, pipeline, local_llm, llm (state helpers), tts, trigger
docs/         FEATURE_STATUS.md and numbered specs
Demo/         diagram, GIF and eval charts (placeholders for now)
main.py       the whole program; demo_check.py the pre-demo check; net.py the online monitor
CONTEXT.md    compact project map (the n8n ask-the-repo bot reads it by URL)
AGENTS.md     rules for contributors and coding agents; PLANS.md plan and checkpoints
TECHNICAL_DESIGN.md  world model, perception, voice pipeline, laser loop
```

## Prototype boundaries and operational notes

**Privacy, stated plainly.** Audio stays on the device and is never written to disk, and speech not meant for the rig is dropped unlogged. Accepted questions (text) and event snapshots are kept; snapshots and saved table frames for 24 hours. When online: the text of a question the rules can't read, or an open question with a compact world state, goes to Grok (xAI); a question about what the camera sees sends the current table frame (and for "earlier" questions up to 6 saved frames) to Grok; answer text goes to ElevenLabs for the voice; SMS answers go through Twilio. The iPhone app reaches the rig only over Bluetooth and turns dictated questions into text on the phone; if a helper turns on "Read answers aloud" with the Grok or rig voice, the phone sends each answer's text to xAI or ElevenLabs with a key kept in the phone's Keychain (the iPhone voice sends nothing). Pictures chosen for things stay on the phone. The camera looks straight down at the tabletop, so frames show the table, objects and hands.

More detail:
- Audio exists only in memory and is never written to disk. Overheard speech the rig decides is not for it is dropped without being logged or stored. Only accepted questions (text, intent, answer, latency) go to the `questions` table.
- JPEG snapshots are written when events happen. `EventLog` deletes snapshot files and state snapshots older than 24 h on start (`prune_on_start_h=24`). Event rows (object, type, positions, time) are kept, with the snapshot path cleared.
- Each answered question (transcript, intent, answer, timings) is posted to the team's own n8n instance on the laptop when `n8n.webhook_url` is set.
- Understanding and answering run on the device. With no network the rig still answers everything, using the Piper voice.

**Safety.**
- Use a laser diode under 1 mW (Class 1/2). The servo limits must cover only the table, never head height. The actuator turns the laser off after `laser_timeout_s` (10 s), on close and at exit. `demo_check.py` check 8 confirms a hardware kill switch cuts laser power before each demo.
- Pill-bottle answers never say or imply that medication was taken. The rig reports only where the bottle is and when it moved.

**Limits.**
- Eight known objects on one table. Objects outside that list are not detected (see spec 0003 for the planned helper).
- Every object is assumed to lie on the table plane, so the top of a tall object maps a few cm off.
- The synthetic eval (255/300) measures the world rules on generated detections. It says nothing about how well the real detector works.
- Grok timings are from the laptop on campus Wi-Fi: about 0.85 s to read a question, 1.3–3.2 s for an open answer, about 1 s for a look at the table.
- A loud hall can defeat the keyword gate. Fallbacks are `listen.mode: wake` or `click`.

## Prior art and how we differ

- **Project Memoria** (2026): a dementia assistant with room cameras, a local LLM and cloud vision. It shows highlights on a phone, and reminders arrive about a minute after the event.
- **Google Project Astra**: answers "where did I leave my glasses?" from memory. It runs in the cloud and answers in words.
- **reCall** (Hack the North): a camera lifelog for finding keys.
- **Watch-Bot** (Cornell): a pan-tilt laser that points at objects, using the same iterative closed-loop aiming.
- **Georgia Tech Healthcare Robotics Lab, "Clickable World"**: a person points a laser to direct a robot, the reverse direction.

Other systems remember what they saw. Ask the Room knows where things are when no one can see them, and shows you.

## Roadmap snapshot

Details and owners are in [`PLANS.md`](PLANS.md), and per-feature status is in [`docs/FEATURE_STATUS.md`](docs/FEATURE_STATUS.md).

- Before the freeze (Sat 6 PM): Jetson timing for Qwen and whisper (`tegrastats`), the 10-minute hall-noise test, real recorded trials, and a live scoreboard.
- Specs: [0001 local interpreter](docs/specs/0001-local-interpreter.md) (done, Jetson timing pending), [0002 always listening](docs/specs/0002-always-listening.md) (done, hall test pending), [0003 Grok detection assist](docs/specs/0003-grok-detection-assist.md) (spec only, measure first), [0004 demo scoreboard](docs/specs/0004-demo-scoreboard.md).
- After the event: YOLO11 fine-tune, `trace`/`tour`/`find_new` laser actions, detector throttling while Qwen runs, streaming action-first answers, a lamp-head enclosure, and a floor search camera.

## Evaluation appendix

**Interpreter** (`scripts/eval_understand.py` on `tests/understand_eval.json`, 64 items). Measured on an M-series MacBook. Jetson numbers are TBD.

| Set | Items | Rules only | Rules + Qwen3-1.7B | Rules + Grok (Sat) |
|---|---|---|---|---|
| stt20 (the 20 recorded test questions) | 20 | 20/20 | 20/20 | 20/20 |
| loose (free phrasing) | 28 | 14/28 | 22/28 | 22/28 |
| overheard (IGNORE or not) | 16 | 14/16 | 16/16 | 16/16 |
| **all** | 64 | **48/64** | **58/64** | **58/64** |

Grok (`grok-4.3`, reasoning none, the default since Sat) was asked 15 times: median 853 ms, max 1.1 s. Rules alone now score 47/64: taught names count as sure, so "where are my kiss" stays a question about something called "kiss". Qwen2.5-1.5B-Instruct also scored 58/64. The median Qwen call took 114 ms for Qwen3-1.7B and 142 ms for Qwen2.5-1.5B, on the laptop. Qwen3-1.7B is the default because it was faster at the same accuracy.

**World model vs baselines** (`eval.replay` and `eval.report`). The baselines are current-frame, last-seen and nearest-object-to-the-last-seen-spot.

| Measure | Synthetic trials | Real recorded trials |
|---|---|---|
| Overall accuracy (full system) | 255/300 (synthetic, rules only) | TBD |
| Hidden objects (covered, inside, inside_box_moved) vs nearest-object baseline | see `eval.report` | TBD |
| Median laser error (cm) | n/a | TBD |
| Question to laser latency (s) | n/a | TBD |

Only real trials recorded with `eval.record` go in the right-hand column (spec 0004). Don't quote the synthetic number as real accuracy.
