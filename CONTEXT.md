# Ask the Room: project context

Start here if you're new to the repo, and point the `n8n/` chat bot at this file first. It covers what
the project is, how the code fits together, and what's done. Interfaces and pass tests live in the spec:
https://claude.ai/artifact/ETtzPXaFAWWkCJoXjuKiAP (section numbers like 3.5, V6 and H4 in the code refer to it).
Contributor and agent rules are in `AGENTS.md`, the plan in `PLANS.md`, per-feature status in
`docs/FEATURE_STATUS.md`. This file stays the short version (the n8n ask-the-repo bot reads it by URL).

## What it is

A HackGT 13 project (36 h, live demo judging). An overhead camera watches a tabletop and keeps track
of the objects on it, including ones it can't currently see: eight known props (keys, pill bottle,
wallet, glasses, phone, remote, a box and a notebook) plus anything else put down, tracked as an
unnamed `thing:N` until someone names it ("this is my charger"). The eight props are the reliability
fallback for the demo. You just ask "where are my keys?" (the mic is always on; a clicker
works too). It answers out loud
("under the notebook, you slid it over them 2 minutes ago") and a pan-tilt laser points at the spot.

The hard part is object permanence. A detector only says what is visible right now, so the world model
keeps a belief for every object: VISIBLE, HELD by a hand, UNDER a cover, INSIDE a container (the box, or
any large unnamed thing such as a tub or a bag), or GONE off
an edge of the table. Hidden objects inherit their parent's position through a chain
(keys → notebook → table), so moving the notebook moves where the laser points. For unnamed things,
identity is causal first (it went under the box, so what comes out is probably it); appearance only
confirms, and when the rig isn't sure it says so (UNKNOWN, `maybe_same_as`) instead of merging.

Questions the world model can't answer from its state go to Grok with the camera frame: "what colour
is my mug?", "what does the note say?", "was there a red mug here this morning?" (saved keyframes).
For "where", Grok picks one of the tracked objects drawn as numbered boxes (set-of-marks), so the laser
still points at a tracked entity. "Where is my red mug?" for a name the rig doesn't know asks Grok only
which box it is and what it is (`VisualQA.pick`); the world model says where, and an unnamed thing keeps
the name, so the next ask needs no Grok call.

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
             opening; chatter is dropped, never logged) ─▶ voice/intents.parse, then Grok (online,
             ~0.85 s) for what the rules can't read
          ─▶ voice/care (reminders, profile facts, morning report, follow-ups) ─▶ voice/pipeline.make_ask
                     ├─ visual questions ─▶ voice/visual: Grok look (set-of-marks) / recall (saved frames)
                     ├─ WHERE / HANDLED / CHANGES / ... ─▶ voice/answers (templates)
                     └─ OTHER ─▶ voice/llm.ask_other (templates for the common ones, else Grok with
                                 the world state and lookup tools; same pill-wording filter)
          ─▶ Answer(speech, point_at, action)
                     ├─ voice/tts: ElevenLabs when online, Piper offline
                     └─ act/laser.Laser.aim_object: closed-loop aim, corrects on the camera's view of the dot
                        (`room:u,v` actions off the table: Laser.aim_px on the room dot map, spec 0006)
server/app.py (FastAPI): dashboard, MJPEG overlay, WebSocket state, POST /ask, /sms (Twilio)
mobile/bridge (BLE GATT on the Jetson) ◀─▶ iPhone app (mobile/ios): ask, state, answers, notices over BLE, no internet needed; optional read-aloud on the phone sends answer text to xAI or ElevenLabs
main.py ─▶ n8n webhook (laptop): a log of every spoken question, plus a 5-minute health check
```

Every answer is worked out on the Jetson, online or not. Privacy, honestly: video and audio stay on
the device and audio is never written to disk; event snapshots (JPEGs) are kept 24 h (pruned at
start); overheard speech that isn't a question for the rig is dropped without being logged. What
leaves the device: answer text to ElevenLabs for the voice when online, texts via Twilio for /sms,
and the question log to the team's own n8n on the laptop. Grok (xAI) does all LLM/VLM work when
online: visual questions send the current frame (and for "earlier" questions a few saved frames),
narration sends short clips' keyframes, and each new unnamed object's close-up goes once for a guessed name
(`core/auto_name.py`: "where's my deodorant?" then works after it was hidden, hedged "I think"). Visual questions are on in `config.yaml` (Sat), narration is
off; the dashboard shows a disclosure for whatever is on. Grok also gets the text of questions the
rules can't read and of open questions (with a compact world state).

## Repo map

| Path | What it holds |
|---|---|
| `core/types.py` | Shared data contracts: Frame, Detection(s), Entity, Event, Intent, Answer. **Change only as a team.** |
| `core/config.py`, `config.yaml` | Config loaded as a plain dict (`load_config()`); the world also reads a typed `Config` view. All thresholds are here. A gitignored `config.local.yaml` (per device, e.g. the rig's `actuator: pca9685`) is merged over it; tests skip it (`ASKROOM_NO_LOCAL_CONFIG`). |
| `core/capture.py` | Camera `FrameBuffer`, plus `VideoFileSource` with the same API for replays. |
| `core/table.py`, `core/table_area.py` | ArUco / one-tag homography mapping pixels to table cm (`table_cal.json`); the operator's tabletop outline, where objects may appear (`table_area.json`, `python -m core.table --outline`). |
| `core/detect.py` | YOLO-World (zero-shot, path A) or fine-tuned YOLO11 (path B, the plan), exported to TensorRT. |
| `core/hands.py` | Stable `hand:N` ids across frames. |
| `core/world.py`, `core/relations.py`, `core/geom.py` | Deterministic, rule-based world model (covers, containers, holds, edges, parent chains). About 200 tests. |
| `core/events.py` | EventLog: SQLite event history, questions table and snapshots. |
| `core/fakeworld.py` | Stand-in world with the same read API, for tests and `--fake` runs. |
| `core/things.py`, `core/proposals.py`, `core/embed.py`, `core/crops.py` | Open world: unnamed `thing:N` identity (spec 0008: a thing lost within 60 s and seen again at its spot is itself, `thing_identity:`), object proposals (change detection, YOLOE prompt-free), DINOv2 re-id embedder (off by default), close-up crops. |
| `core/auto_name.py` | Automatic names: one Grok look at each new `thing:N`'s close-up, kept as a soft guess (not an alias) that questions fall back to, hedged. |
| `core/narration*.py`, `core/visual_memory.py`, `core/clip_tokenizer.py` | Grok clip narration and the keyframe archive with MobileCLIP2 text search. |
| `core/grok_check.py` | Grok settle check (spec 0007, off by default): Grok checks the tracked marks when the table settles; verdict rows in `grok_checks`; sightings answer "where is my X" when the world has no position; one object per kind and the label belief (spec 0008). YOLO stays Stage 1. |
| `core/xai.py` | The one client for every Grok call (xAI's API over plain requests, one shared warm connection; no openai package). `main.py` warms it at start and whenever the network comes back. |
| `core/reminders.py`, `core/reports.py`, `core/profile.py` | Care layer: event-triggered reminders, morning report, profile facts (ideas from Project Memoria, MIT). |
| `mobile/` | BLE bridge (`bridge/`), wire protocol (`PROTOCOL.md`), iPhone app (`ios/`). |
| `assets/` | Small licensed data files the code needs (CLIP BPE vocabulary). `models/` is never committed. |
| `voice/` | `visual` (Grok look/recall, routing), `teach` ("this is my X"), `care` + `conversation` (reminders, profile, follow-ups), `intents` (rule parser), `answers` (spoken templates), `understand` (overheard filter + Grok reads what the rules can't), `local_llm` (optional local Qwen answers, not deployed), `pipeline` (router), `tts`, `stt` (Silero VAD + whisper.cpp), `trigger` (clicker), `llm` (Grok open answers, world-state helpers, pill filter). |
| `act/` | `actuator` (servo drivers + fake), `laser` (poly2 fit + closed-loop aim; `aim_px` for room pointing), `calibrate`, `room_map` (room dot map, zones, beam gate; spec 0006, off by default), `sim` (simulated rig and room). |
| `server/` | FastAPI dashboard (`app.py`), frame overlay, `sim.py` (full demo on a synthetic camera). |
| `eval/` | Trial recording, synthetic trials, replay against baselines (last-seen, nearest-object, current-frame) and the report. `score_clip`: replay a guided clip (`data/clips/<id>`) through the production pipeline and score it (false births, identity changes, checkpoints, questions); on the Jetson `scripts/dock.sh python3 -m eval.score_clip data/clips/<id>`. |
| `net.py` | Online/offline monitor. Readers check `.online`, which never blocks. |
| `scripts/` | `dock.sh` (run inside the Jetson Ultralytics container), camera setup, markers PDF, servo sweep, `qwen_server.sh`, `eval_understand.py` (interpreter accuracy per model), `overheard_test.py` (false triggers on a hall recording), `gen_n8n_workflow.py`. |
| `tests/` | About 1,300 tests. None need hardware. `understand_eval.json`: 64 spoken-style commands for the interpreter. |
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
python -m pytest -q                 # all tests, ~90 s, no hardware
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

Also built, tested on the laptop, not yet on the Jetson: the clicker, `main.py` wiring all threads
together, the fine-tuning scripts (zero-shot YOLO-World mistook the Jetson case for a phone; the plan is
model-free background-difference labels plus copy-paste synthesis), `demo_check.py`, and the always-on mic.
Grok now reads what the rules can't (58/64 on the interpreter eval, the same as Qwen, median 853 ms) and
answers open questions (1.3–3.2 s); the rig is Grok-only (offline: rules and templates; the plan is a phone hotspot). Local Qwen is optional (`understand.backend: qwen` / `auto`) and not installed on the Jetson. On the Jetson: whisper.cpp base.en on the
GPU, 22/22 test questions, median 139 ms; DINOv2 re-id at 3.7–4.7 ms per crop.

Built overnight (Fri → Sat), unit-tested: open-world `thing:N` identity and teaching by voice, object
proposals, Grok visual questions (set-of-marks look: 20/23 on desk photos with real Grok, about 1 s;
recall over saved frames), Grok narration, reminders, morning report, profile facts, conversation
follow-ups, and the BLE bridge + iPhone app. Not yet on the real table: the D17 open-world check (teach,
hide, move the box, ask; at least 4/5 or the eight-object demo is the headline), real eval trials, and
laser calibration. Stretch goals (floor search camera, room map) are deprioritized.

Sat afternoon, unit-tested, not yet measured on the rig: one object, one `thing:N` (spec 0008). On the
rig, 39 things were born in 13 min, 31 of them within 5 cm of an earlier thing. Now a thing lost
recently and seen again at its spot is itself (on by default). When Grok names a new thing like a lost
one, the two are folded into one (`merge_same`, needs the settle check). A summed Grok label belief
names things and retires clutter (off by default).

Room pointing (spec 0006, branch `room-pointing`, after the freeze): the laser can point at visible
things off the table without depth, using a swept dot map and a pixel-space loop that stops once the camera
sees the dot inside the object's box. It's off by default (`room.enabled: false`) and tested in a sim
room only, not on the rig.

## Decisions worth knowing

- **Rules, not a learned tracker**, for the world model. It's deterministic and testable, and it explains itself on the dashboard.
- **Fine-tune YOLO11** on our own overhead frames. YOLO-World is only the baseline and the auto-labeller.
- **Hands come from the detector's `hand` class.** MediaPipe was dropped: it runs CPU-only on the Jetson and misses hands that hold objects.
- **The laser corrects in a closed loop** from the camera's view of the dot. Frame-diff dot finding has to allow for camera latency.
- **Grok for all LLM/VLM work** (Fri night, xAI track), replacing the earlier local-Qwen decision. Rules and templates still answer first and are the offline fallback. The model never writes coordinates: for "where" it picks a numbered mark (set-of-marks beat asking Grok for boxes, 5/5 vs 0/5 on a desk photo).
- **Open world is core**, with the eight known props as the fallback. Unnamed things keep UNKNOWN rather than guessing an identity.
- **Templates stay for the core questions**; the model only covers what they can't. They're exact, tested and instant.
- **"Was that for me?" is decided by rules, not the model**: Qwen got 8/16 overheard lines right, the gate + wake word + question-opening checks 16/16.
- **The eval compares against a "nearest object to the last-seen spot" baseline**, so any win is measured against something reasonable.
