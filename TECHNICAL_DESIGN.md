# Ask the Room: Technical Design

> This describes the code as it is on branch `teammate-tasks`. Planned work is marked as such and linked to its spec. Thresholds named here are keys in `config.yaml`.

## Architecture summary

One process (`main.py`) runs five threads:

1. **Capture** (30 fps): `core/capture.FrameBuffer` keeps the newest frame plus a short ring of recent frames.
2. **Perception** (10–15 fps, `main.perception_max_fps`): detector → hand tracker → `World.update()`. The world writes events to `core/events.EventLog`.
3. **Voice** (always on): mic → VAD → Whisper → overheard filter → rules/Qwen → answer → speech and laser.
4. **Net** (every 5 s): `net.NetMonitor` probes the network. Readers check `.online`, which never blocks.
5. **Server** (5 Hz push): `server/app.py` (FastAPI) serves the dashboard, `/ask`, `/state` and `/sms`.

All answers are worked out on the Jetson. The network only changes the voice (ElevenLabs vs Piper).

## Key contracts

`core/types.py` holds the shared types. Change them only as a team.

- `Detection(cls, conf, box_px, center_cm, box_cm)`. Hands have `cls` like `hand:3`.
- `Entity(name, kind, status, parent, pos_cm, box_cm, last_seen, confidence, candidates, edge, zone, ...)`. `kind` is `target`, `container` or `cover`.
- `Status`: `VISIBLE`, `HELD`, `UNDER`, `INSIDE`, `GONE`, `UNKNOWN`.
- `EventType`: `PICKED_UP`, `PUT_BACK`, `MOVED`, `COVERED`, `UNCOVERED`, `PUT_INSIDE`, `TAKEN_OUT`, `EXITED_VIEW`, `LOST_TRACK`, `CORRECTED`, `FOUND`.
- `Intent(kind, obj, raw)`. The kinds are `WHERE`, `HISTORY`, `HANDLED`, `CHANGES`, `RESET`, `RECAL`, `OTHER`, plus `IGNORE` from `voice/understand.py` for overheard speech.
- `Answer(text, point_at, action)`. `action` is `point`, `circle`, `sweep:<edge>` or `None`.
- World readers use only `get`, `resolve`, `history` and `state_json` (`WorldAPI`). `core/fakeworld.FakeWorld` has the same API.

## Perception

- **Camera.** icSpring USB, MJPG 1280x720 at 30 fps. `CAP_PROP_BUFFERSIZE=2` (a value of 1 halves the fps). Exposure is manual, locked by `scripts/camera_setup.sh`.
- **Table frame.** `core/table.py` finds ArUco markers 0–3 (DICT_4X4_50) and fits a pixel→cm homography (`table_cal.json`). Everything is assumed to lie on the table plane.
- **Detector.** `core/detect.py`. Path A is YOLO-World v2 with the text prompts from `config.yaml` baked in, exported to TensorRT (about 14 ms per frame on the Orin Nano). Path B, the plan, is a YOLO11 fine-tuned on our own overhead frames (`scripts/finetune/`). Both keep the best box per object and every hand box, and convert them to table cm.
- **Hands.** The detector's `hand` class, tracked by `core/hands.HandTracker`. It matches by IoU, then by the nearest centre, and drops a track unseen for 0.5 s. MediaPipe was dropped: it runs CPU-only on the Jetson and misses hands holding objects.

## World model rules

`core/world.py` is deterministic and rule-based. It is not a learned tracker, so it is testable and every belief can be explained on the dashboard. `World.update(dets, frame)` runs these steps in order, once per detection batch:

1. **Decay.** Every belief that isn't VISIBLE has its confidence multiplied by `decay_per_min` (0.99) for each elapsed minute. This runs before this batch's rules.
2. **Debounce.** For each object, keep the best detection at or above `conf_threshold`. It counts as present at `present_k_of_n` (6 of 10) and absent at `absent_k_of_n` (at most 1 of 10). Position refreshes on every detection.
3. **Observation wins.** An object that becomes present is VISIBLE at confidence 1.0, with its parent cleared. If it was HELD, it emits `MOVED` (moved at least `moved_min_cm` from where it was picked up) or `PUT_BACK`. If it was hidden or lost, it emits `UNCOVERED`, `TAKEN_OUT` or `CORRECTED`.
4. **Contacts.** A hand box covering at least `contact_overlap` of an object's box records a touch.
5. **Moved in plain view.** A carried object that stays detected leaves its rest spot by `moved_min_cm` with a recent touch, which emits `PICKED_UP`. Once it is untouched for `settle_s` and has moved less than `settle_cm`, it emits `MOVED` or `PUT_BACK`. A shift with no hand is treated as a new rest spot. A box that only shrank inside the one the object rests in is the object partly hidden (an arm over part of it), not moved. Its centre shifts, but it is not a pick-up.
6. **Disappearance.** An object that stops being present is checked in this order:
   - *Still there?* If the pixels still match its remembered appearance patch (`appearance_match`), it was a detector miss, and nothing changes.
   - *Slid over by hand.* If every hand that touched it now rests on a cover lying over it, it is UNDER that cover. The cover rule wins over the hand rule here.
   - *Picked up.* A touch within `contact_window_s` of last seen makes it HELD by that hand (`conf_held`). Other touching hands become candidates, with `ambiguity_penalty` applied.
   - *Covered.* A cover that moved within `cover_moved_window_s` and now overlaps at least `cover_overlap` of the object's last box means UNDER that cover (`conf_under`). Containers the hand visited become rival candidates.
   - *Background changed.* If the bare-table background under its box has changed, something undetected covers it: UNDER `unknown` (`conf_under_unknown`).
   - Otherwise *lost*: UNKNOWN at its last position (`LOST_TRACK`).
7. **Container dwell.** `relations.DwellTracker` records a hand whose centre stays inside a visible container's box for at least `container_dwell_s` and then leaves.
8. **HELD objects.** If the holding hand's latest container visit ended at least `reappear_wait_s` ago and the object hasn't reappeared, it is INSIDE that container (`conf_inside`). If the hand is unseen for `hand_lost_s` near a frame edge (`edge_margin`), the object is GONE with that edge (`EXITED_VIEW`). If the hand is lost elsewhere, or the object has been held for more than `held_timeout_s`, it is UNKNOWN.
9. **Lifted cover.** If a child's cover moves off it (overlap below `lifted_overlap_max`) or leaves view, and the child doesn't reappear within `reappear_wait_s`, the child is lost, with `lifted_cover_penalty` applied.
10. **Image refresh.** At most every `bg_update_every_s`, `relations.BackgroundModel` updates its median of the bare table (object and hand boxes excluded, downscaled to 320 px). Appearance patches are refreshed for visible objects that have no hand over them.

**Parent chains.** Children never move themselves. `resolve(name)` follows parent links up to `max_nesting` (3) and returns the outermost entity's position and the chain, for example keys → notebook → box. Moving the box moves where the laser points for the keys. A chain stops at `hand:N` or `unknown`.

**Confidence.** A heuristic between 0 and 1, not a probability. Answers say "probably" below `answer_plain` (0.7) or when there are candidates. Below `answer_hedge` (0.5) they say "I lost track" and circle the last-seen spot.

**Events.** Every rule outcome is an `Event` stored in SQLite (`core/events.py`). A JPEG snapshot is encoded on a background thread. On start, `EventLog` deletes snapshot JPEGs and `state_snapshots` older than 24 h (`prune_on_start_h=24`). Event rows are kept, with the snapshot path cleared.

The rules are covered by `tests/test_world_rules.py`, `tests/test_world_core.py` and `tests/test_relations.py`, and scored end to end by `eval/` against three baselines: current-frame, last-seen, and nearest-object-to-the-last-seen-spot.

## Voice pipeline

```
mic ─▶ Silero VAD ─▶ whisper.cpp base.en ─▶ overheard filter ─▶ rules │ Qwen ─▶ router ─▶ answer ─▶ TTS + laser
```

1. **Listening** (`main.Room.voice_loop`, `listen` config).
   - `always` (the default) listens continuously.
   - `wake` only accepts speech containing a wake word (`wake_words: [room]`).
   - `click` only listens after a clicker press.
   - In every mode, a clicker press means "listen now", and it stops the current answer (barge-in).
   - The mic is not read while the rig is speaking, and stays shut for `echo_tail_s` (0.4 s) afterwards, so the rig never answers itself.
2. **Speech to text** (`voice/stt.py`).
   - Silero v5 VAD (ONNX, no torch) marks the start and end of speech: it ends after `silence_ms` below the off threshold. A 0.3 s pre-roll is kept before the first speech.
   - whisper.cpp `base.en` transcribes, with an initial prompt listing the object names and an `audio_ctx` sized to the clip. The backend is `pywhispercpp` on the laptop and `whisper-cli` on the Jetson.
   - Audio exists only as numpy arrays in memory. In always-on mode, transcripts are not written to the log.
3. **Was that for me?** (`voice/understand.py`, overheard speech only). No model is involved.
   - *Keyword gate*: the speech must contain an object name or synonym, a command word, or the wake word.
   - *Addressed*: it must contain the wake word, or open like a question or request ("where", "did", "can", "show") after fillers like "okay so".
   - After parsing, RESET and RECAL need the wake word. OTHER needs the wake word or an object.
   - Anything that fails becomes `IGNORE`. It is dropped with no log, no storage and no n8n report.
4. **Understanding** (`voice/understand.Understander`).
   - The rule parser (`voice/intents.py`) answers when it found both the question type and the object it needs.
   - Otherwise it asks Qwen3-1.7B through llama-server. The request uses the `json_schema` response format `{kind, object}` with enums, `enable_thinking: false` (llama.cpp skips grammar enforcement while thinking), `temperature: 0` and a static system prompt so the prompt cache holds.
   - An object the rules recognised wins over Qwen's. Qwen's object must pass `sounds_like`, which checks it against the words actually said, so a model can't invent an object.
   - RESET and RECAL come from the rules only.
   - If Qwen is down, slower than `understand.timeout_s` (1.5 s), or returns bad output, the rules' answer stands.
5. **Routing** (`voice/pipeline.make_ask`).
   - WHERE, HISTORY, HANDLED and CHANGES go to the templates in `voice/answers.py`, which are deterministic and instant.
   - OTHER goes to `voice/local_llm.ask_local`. It first tries templates for common open questions: what is in or under something, what is hidden, what is on the table, privacy, help, and "did I take my meds". For the rest it makes one Qwen call. The prompt holds `compact_state` plus the last 12 events from the past 30 minutes. The schema is action-first `{action: point|circle|none, point_at: <object enum>|none, text}`.
   - The model's output goes through `voice/llm.to_answer`, which caps it at two sentences, strips markdown and applies the pill-wording filter. The laser moves only if the sentence names the object it points at.
   - Any failure returns a fixed fallback sentence.
   - Accepted questions are logged to the `questions` table.
6. **Output.**
   - `voice/tts.py` streams ElevenLabs `eleven_flash_v2_5` when online. It falls back to Piper if there is no network or the first byte takes over 1.5 s.
   - Speech and aiming start together on two threads. Aiming never blocks speech: an uncalibrated laser is logged and the answer still plays.
7. **Reporting.** `main.Room.report` posts each answered question to the n8n webhook in the background.

Grok (`voice/llm.ask_grok`) is not on this path. The planned detection-side role is in `docs/specs/0003-grok-detection-assist.md`.

## Laser loop

`act/laser.py` and `act/actuator.py`.

- **Model.** A 2nd-order polynomial per axis maps table cm (x, y) to servo pulses (pan, tilt). `act/calibrate.py` fits it by least squares from a grid over `servo_limits`, with the dot found in camera frames by frame difference.
- **Closed-loop aim.** Predict the pulses, move, wait 0.15 s, find the dot in the camera and correct with the polynomial's local Jacobian (gain 0.7). Repeat until the error is under 1 cm or 8 tries have been used. Dot finding uses frames from after the move, accounting for `camera_latency_s`. Shiny objects (`shiny_objects`) get an offset of `shiny_offset_cm` toward the table centre, so the dot lands next to them instead of glinting off them.
- **Actions.**
  - `point` aims at `resolve(name)`.
  - `circle` traces a 5 cm circle twice. It is used for UNKNOWN or low-confidence answers and for hidden objects in open answers.
  - `sweep:<edge>` runs the dot along the table edge an object left by.
- **Safety.** Every actuator clamps to `servo_limits`, eases its moves, and turns the laser off after `laser_timeout_s` (10 s), on `close()` and at exit. `demo_check.py` check 8 confirms that a hardware kill switch cuts power.
- **Simulation.** `act/sim.py` models a non-linear pan-tilt head with backlash, jitter and camera latency, so calibration and aiming are tested without hardware (`tests/test_laser_sim.py`).

## Critical design rules

- Rules, not a learned tracker, for the world model.
- The LLM never outputs coordinates, only an object from an enum and an action from a small set.
- Templates answer wherever they can. The LLM writes only the long tail.
- Nothing a visitor says leaves the device for understanding or answering.
- Tune from replays, never live.

## Reference documents

- `CONTEXT.md`: the compact project map (read by the n8n bot)
- `AGENTS.md`: contributor rules
- `PLANS.md`: plan and checkpoints
- `docs/FEATURE_STATUS.md`: per-feature status
- `docs/specs/`: numbered specs with acceptance tests
- `n8n/README.md`: the question log, health check and repo bot
