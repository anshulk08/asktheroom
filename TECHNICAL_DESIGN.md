# Ask the Room: Technical Design

> This describes the code as it is on `main`, as run on the rig (Logitech Brio, one AprilTag, Grok for the LLM work). Planned work is marked as such and linked to its spec. Thresholds named here are keys in `config.yaml`.

## Architecture summary

One process (`main.py`) runs five threads:

1. **Capture** (30 fps): `core/capture.FrameBuffer` keeps the newest frame plus a short ring of recent frames.
2. **Perception** (10–15 fps, `main.perception_max_fps`): detector → hand tracker → `World.update()`. The world writes events to `core/events.EventLog`.
3. **Voice** (always on): mic → VAD → Whisper → overheard filter → rules/Grok → answer → speech and laser.
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

- **Camera.** Logitech Brio, opened by its by-id path (`/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0`; `video-index2` is the IR node), MJPG 1280x720 at 30 fps. `CAP_PROP_BUFFERSIZE=2` (a value of 1 halves the fps). Exposure, gain, focus, zoom and white balance are manual, locked by `scripts/camera_setup.sh 166 <device> 80 10 160 3200`. `core/capture.py` reopens a camera that drops off USB and writes its snapshotted controls back (`core/v4l2ctl.py`).
- **Table frame.** `core/table.py` in one-tag mode (`table_tag.enabled`, the rig's setting): a single printed AprilTag 36h11 (id 0, `table_tag.size_cm` as printed) averaged over 15 frames gives the pixel→cm homography (`table_cal.json`), and the tracked area is the camera's footprint on the table plane. The four-ArUco mode (markers 0–3, DICT_4X4_50) remains for `table_tag.enabled: false`. The tabletop outline (`python -m core.table --outline`, `core/table_area.py`) limits births to the table. Everything is assumed to lie on the table plane.
- **Detector.** `core/detect.py`. The rig runs path B: `models/askroom-yolo26s-brio.engine`, a YOLO26s fine-tuned on Brio captures of the demo table (26 Sep; `scripts/finetune/`), set in the rig's gitignored `config.local.yaml`, with YOLOE-26s prompt-free proposals (`proposals.kind: yoloe`) for things it has no class for. Path A, the committed default in `config.yaml`, is YOLO-World v2 with the text prompts baked in, exported to TensorRT (about 14 ms per frame on the Orin Nano). Both keep the best box per object and every hand box, and convert them to table cm.
- **Hands.** The detector's `hand` class, tracked by `core/hands.HandTracker`. It matches by IoU, then by the nearest centre, and drops a track unseen for 0.5 s. MediaPipe was dropped: it runs CPU-only on the Jetson and misses hands holding objects.

## World model rules

`core/world.py` is deterministic and rule-based. It is not a learned tracker, so it is testable and every belief can be explained on the dashboard. `World.update(dets, frame)` runs these steps in order, once per detection batch:

1. **Decay.** Every belief that isn't VISIBLE has its confidence multiplied by `decay_per_min` (0.99) for each elapsed minute. This runs before this batch's rules.
2. **Debounce.** For each object, keep the best detection at or above `conf_threshold`. It counts as present at `present_k_of_n` (6 of 10) and absent at `absent_k_of_n` (at most 1 of 10). Position refreshes on every detection. Two label checks drop a detection first, because a detector fine-tuned on a few classes calls unknown things by those names: a label read on a thing that was already in place while the object was seen elsewhere (no hand there), visible or hidden where it lay, is that thing, not the object moving onto it; and a target's label read clear of its box (`moved_min_cm` away, IoU < 0.3) while its remembered pixels still match where it lies is a neighbour called by its name.
3. **Observation wins.** An object that becomes present is VISIBLE at confidence 1.0, with its parent cleared. If it was HELD, it emits `MOVED` (moved at least `moved_min_cm` from where it was picked up) or `PUT_BACK`. If it was hidden or lost, it emits `UNCOVERED`, `TAKEN_OUT` or `CORRECTED`.
4. **Contacts.** A hand box covering at least `contact_overlap` of an object's box records a touch.
5. **Moved in plain view.** A carried object that stays detected leaves its rest spot by `moved_min_cm` with a recent touch, which emits `PICKED_UP`. Once it is untouched for `settle_s` and has moved less than `settle_cm`, it emits `MOVED` or `PUT_BACK`. A shift with no hand is treated as a new rest spot. A box that only shrank inside the one the object rests in is the object partly hidden (an arm over part of it), not moved. Its centre shifts, but it is not a pick-up.
6. **Disappearance.** An object that stops being present is checked in this order:
   - *Still there?* If the pixels still match its remembered appearance patch (`appearance_match`), or a class-agnostic proposal still outlines a configured object's box (IoU 0.5) with no hand on it, it was a detector miss, and nothing changes. On blanket_1t the phone's screen lit up, the fine-tuned detector stopped calling it 'phone' (it named the tape roll instead) and the patch no longer matched, so the phone was lost and reborn as a thing though YOLOE still proposed its box.
   - *Slid over by hand.* If every hand that touched it now rests on a cover lying over it, it is UNDER that cover. The cover rule wins over the hand rule here.
   - *Laid over by hand (a cover no detector knows).* A blanket, a jacket or a napkin is laid by hands that touch what it covers, so the hand rule alone read each touched object as picked up, then lost when the hands left (blanket_1t). `core/surround.py` remembers, in colour, the band of table `unknown_cover.ring_cm` (4 cm) wide around each object, once it has looked the same for `adopt_s` with the object detected and no hand box over it. A real pick-up leaves that band as it was once the hand is off the spot; a cover leaves it changed all round. If at least `sides_min` (3) of its four sides are changed (hand boxes left out; half a side's pixels, and the three average `mean_changed`, 0.75), and the band is at rest (as it was `settle_s`, 0.5 s, before: a cover lies still, an arm keeps moving), the object is UNDER the cover, back at the spot it rested in: a named cover that qualifies, else a thing laid over it (below), else `unknown` (`conf_under_unknown`). While the visible band is all changed but hands hide too much of it, or while it settles, the absence waits (at most `wait_max_s`). If any side shows table, the hand rule decides as before. On place_1 a dim sleeve lying across the phone changed about half of three sides, which the average keeps out. Objects seen again from under `unknown` at their spot were found, not put down, so "this is my X" does not name them. Grey could not do this: on the rig a red blanket on brown wood differs from the table by too little, and the background model never saw the table under objects there since start-up.
   - *Picked up.* A touch within `contact_window_s` of last seen makes it HELD by that hand (`conf_held`). Other touching hands become candidates, with `ambiguity_penalty` applied.
   - *Covered.* A cover that moved within `cover_moved_window_s` and now overlaps at least `cover_overlap` of the object's last box means UNDER that cover (`conf_under`). Containers the hand visited become rival candidates.
   - *Background changed.* If the bare-table background under its box has changed, something undetected covers it: UNDER `unknown` (`conf_under_unknown`).
   - *Briefly unseen.* If none of these explains the absence and the object was detected, or its patch matched, less than `lost_grace_s` (2 s) ago, it stays VISIBLE where it was and the checks repeat on each batch. Seen again meanwhile, nothing is logged. The detector drops objects while an arm is near them, and arms in dim light are often not detected as hands, so without this untouched objects flickered `LOST_TRACK` then `CORRECTED` (shell_1). Covers and the background rule still decide at once during the wait, but a hand that arrives only after the absence began is not taken to have picked the object up.
   - Otherwise *lost*: UNKNOWN at its last position (`LOST_TRACK`), `lost_grace_s` after it was last seen, unless its band then looks covered: an object no hand touched is judged by its band only when the grace runs out, because an undetected arm lying across it looks alike for a while.
7. **Container dwell.** `relations.DwellTracker` records a hand whose centre stays inside a visible container's box for at least `container_dwell_s` and then leaves. The containers are the configured ones and any large thing (see *Things as containers* below).
8. **HELD objects.** First the unknown-cover check of rule 6: a held object whose band settles covered was covered by what the hand was laying, not carried (`COVERED`). If the holding hand's latest container visit ended at least `reappear_wait_s` ago and the object hasn't reappeared, it is INSIDE that container (`conf_inside`). For a thing container, the visit must not have crossed it, and the configured box wins over a thing the hand was in at the same time. If the hand is unseen for `hand_lost_s` near a frame edge (`edge_margin`), the object is GONE with that edge (`EXITED_VIEW`). If the hand is lost elsewhere, or the object has been held for more than `held_timeout_s`, it is UNKNOWN.
9. **Lifted cover.** If a child's cover moves off it (overlap below `lifted_overlap_max`) or leaves view, and the child doesn't reappear within `reappear_wait_s`, the child is lost, with `lifted_cover_penalty` applied. A thing cover's box is its cover box. Under `unknown`, the cover is lifted when the child's band looks as remembered again (at most one side changed): if the child's own remembered pixels are back, it is there (`UNCOVERED`); if it is not seen within `reappear_wait_s`, it is lost the same way.
10. **Image refresh.** At most every `bg_update_every_s`, `relations.BackgroundModel` updates its median of the bare table (object and hand boxes excluded, downscaled to 320 px). Appearance patches are refreshed for visible objects that have no hand over them.

**Confirmation latency.** The world adds little to a put-down. A configured object is VISIBLE on its `present_k`-th detection, about 0.33 s at 15 fps, even with a hand still on it. A new thing is admitted on the first batch at least `openworld.still_s` (0.5 s) after its first still sighting, about 0.53 s. On the rig clips most of the "cue → confirmed" time is the person. place_1 wallet: cue 4.84 s, hand off 9.75 s, VISIBLE 10.10 s. place_1 put-down: cue 17.84 s, set down about 20.5 s, confirmed 20.98 s. shell_1 case: cue 4.79 s, let go 8.15 s, APPEARED 8.64 s. Even an instant world could not bring these under 3 s from the cue. Replays rejected every faster setting. Scored on the Jetson, `still_s` 0.33–0.4 s saved 0.13–0.2 s on shell_1 (still 3.65 s) but cost it an identity change: the phone was reborn as a new thing. In offline replays of the rig's detections, `present_k_of_n` [5, 10] added 3 things in hands_1, and [4, 10] added 1–4 things in every clip but place_1. The rig's 0.6 wallet / phone cut-off stays. Tests in `tests/test_world_core.py` and `tests/test_openworld.py` lock this budget in.

**Things as containers.** A tub, a bag, a hat or a mug the detector has no class for is an unnamed `thing:N`, and it can hold things like the configured box (`core/things.py`, `thing_containers:` in `config.yaml`). The evidence is the container rule's own: the hand holding the object dwells inside the thing's box, leaves, and the object is not seen again within `reappear_wait_s`, so the object is `PUT_INSIDE` that thing. Its position then follows the thing (`resolve()`), carried or slid, and it is `TAKEN_OUT` when it is seen next to it after a hand was there. Which things qualify, and what counts:
   - *Gate.* A thing visible on the table (so not inside or under anything itself) whose footprint is at least `min_area_cm2` (100 cm²: a mug or larger). While an arm hides part of it, the box it rests in counts.
   - *The configured box wins.* If the hand left a configured container while it was inside the thing (a box standing on a tray), the configured container is the parent.
   - *Crossing is not dropping.* A hand whose centre entered the thing on one side and left on the other (entry and exit on opposite sides of its middle, at least half its size apart) was carrying something past it, over a placemat or a sheet of paper, not into it. That visit is ignored, so keys carried across a mat and off the table are `EXITED_VIEW`. A hand that only passes over a thing, with nothing held, changes nothing, because only a held object can be put inside.
   - *Nesting.* The thing may not lie inside the object, and the new chain may not be deeper than `max_nesting`.
   - *Identity.* A hand resting inside a tub lies wholly within the tub's box. That proposal is still the tub, not an arm (`hand_contain` is not applied to a proposal over a thing container's box), so a long dwell is not a pick-up of the tub. A proposal much larger or smaller than a hidden thing (more than `size_ratio_max`) is not that thing coming out, so the tub seen again after being carried is not its own contents.
   - *Open containers.* From overhead an open box or tub shows what lies in it, so those objects stay VISIBLE and never go INSIDE. `open_container_of(name)` finds the container (configured, or a thing container) holding at least 80% of a visible object's box, and "where" answers say "Your keys are in the box." For the shell-game trick itself the container must hide the object: a box with a lid or closed flaps, or a cup turned over it.
   - *Answers* say "inside the toy bin" for a taught name, "inside the plastic tub" for an automatic guess, else "inside a container", and the laser points at the thing.

**Things as covers.** An object UNDER `unknown` with a thing lying over it is UNDER that thing (`COVERED` again, `conf_under`): a napkin the proposer sees whole. The thing must cover `cover_overlap` of the object's box, be at least 1.5 times its footprint, have been put down no earlier than just before the object was last seen (a mat it already lay on does not count), and be established: untouched for 2 s since. On blanket_1t a piece of the blanket still in a hand, taken as the cover, made its child "come out" at the next piece of blanket the proposer saw. A blanket much larger than YOLOE's `max_area_frac` (0.15 of the frame) is only seen in pieces, so it stays `unknown`, and answers say "under something".

**Parent chains.** Children never move themselves. `resolve(name)` follows parent links up to `max_nesting` (3) and returns the outermost entity's position and the chain, for example keys → notebook → box. Moving the box moves where the laser points for the keys. A chain stops at `hand:N` or `unknown`.

**Confidence.** A heuristic between 0 and 1, not a probability. Answers say "probably" below `answer_plain` (0.7) or when there are candidates. Below `answer_hedge` (0.5) they say "I lost track" and circle the last-seen spot.

**Events.** Every rule outcome is an `Event` stored in SQLite (`core/events.py`). A JPEG snapshot is encoded on a background thread. On start, `EventLog` deletes snapshot JPEGs and `state_snapshots` older than 24 h (`prune_on_start_h=24`). Event rows are kept, with the snapshot path cleared.

The rules are covered by `tests/test_world_rules.py`, `tests/test_world_core.py` and `tests/test_relations.py`, and scored end to end by `eval/` against three baselines: current-frame, last-seen, and nearest-object-to-the-last-seen-spot.

## Voice pipeline

```
mic ─▶ Silero VAD ─▶ whisper.cpp base.en ─▶ overheard filter ─▶ rules │ Grok ─▶ router ─▶ answer ─▶ TTS + laser
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
   - Otherwise it asks Grok (`understand.backend: grok`, the default; `voice/understand.Grok`) when online. The request uses the `json_schema` response format `{kind, object}` with enums and a static system prompt. Local Qwen3-1.7B through llama-server (`backend: qwen`, or `auto` for Grok online and Qwen offline) is optional and not installed on the Jetson; it uses `enable_thinking: false` (llama.cpp skips grammar enforcement while thinking) and `temperature: 0`.
   - An object the rules recognised wins over the model's. The model's object must pass `sounds_like`, which checks it against the words actually said, so a model can't invent an object.
   - RESET and RECAL come from the rules only.
   - If the model is down or offline, slower than its timeout, or returns bad output, the rules' answer stands.
5. **Routing** (`voice/pipeline.make_ask`).
   - WHERE, HISTORY, HANDLED and CHANGES go to the templates in `voice/answers.py`, which are deterministic and instant.
   - Questions about what the camera sees go to `voice/visual.py` first (set-of-marks look, pick, recall). With the Grok settle check on (spec 0007), "where is my X" for something the world has no position for answers from Grok's last sighting.
   - OTHER goes to `voice/llm.ask_other`. It first tries templates for common open questions: what is in or under something, what is hidden, what is on the table, privacy, help, and "did I take my meds". For the rest it makes one Grok call when online (local Qwen through `voice/local_llm.ask_local` with `backend: qwen`), else the fallback sentence. The prompt holds `compact_state` plus the last 12 events from the past 30 minutes. The schema is action-first `{action: point|circle|none, point_at: <object enum>|none, text}`.
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
