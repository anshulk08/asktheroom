# 0010: The room demo: what it takes to win with it

Status: plan, written Sat 26 Sep 8:30 PM EDT, after the room memory 5-run check passed on the rig.
Decision (Anshul, Sat ~8 PM): the demo is the **room**, not the table. An older person doesn't keep
everything on one surface; "where are my keys?" answered across a room is the product. Everything below
serves that one demo. Time left: hacking ends **Sun 8 AM** (Devpost before that), expo judging
**9:00–11:15 AM at Klaus**, judges rotate, so the demo must reset in under a minute (PLANS.md).

## 1. The demo we are building

A 60-second script a judge can run, with the rig in the living-room corner and the coffee table as the
"table":

1. The judge picks up an object from the coffee table (say the wallet) and puts it somewhere in the room:
   the couch, the side table, the kitchen counter.
2. They walk away and ask, out loud or on the phone: "where's my wallet?"
3. The rig answers in a few seconds: "Your wallet, I think, is on the kitchen counter. It appeared there
   20 seconds ago." The phone shows the room with the spot marked.
4. They move it again, ask again. Then they bring it back to the table: "Your wallet is on the table."
5. Stretch: they hide it under the notebook on the table, or the laser points at it across the room.

What makes this land with judges: it works for *their* object choice from a short list, it is fast, it
explains itself ("appeared there 20 seconds ago"), it hedges honestly ("I think"), and it resets in seconds.

## 2. Where we are (facts, not hopes)

**Works on the rig now (Sat 7:45 PM, `room-memory-m0` @ `afca68c`, suite 1880 passing):**
- Room memory end to end with the **remote**: 5/5 handoffs (couch 8-10 s, side table 24 s, counter 20 s),
  5/5 table returns, via the spoken trial driver (`scripts/room_trials.py`).
- Camera: Brio high in the corner, zoom 100, 1080p; the coffee table is the table view; four zones drawn.
- Identity: YOLOE proposals + Grok names ("is it one of these?" against what just left the table).
- Dashboard `/full.jpg` shows the whole room with zones and room places.

**Does not work / not done:**
- **Speech.** The rig has no speaker (HDMI only) and its mic is the Brio in the ceiling corner. Every
  answer so far came through `POST /ask`.
- **Only the remote is proven.** Keys, wallet, glasses, pill bottle: untested from this camera.
- **The table's hidden-object rules are probably broken from this angle.** To make the room work, the
  fine-tuned prop detector's labels were turned off on the rig (it read the remote as a wallet); the
  cover/container rules need "notebook" and "box" recognised. Not measured (D17 not run).
- **Far zones are slow** (20-24 s to hand off vs a judge asking within ~5 s).
- **The decoy test is not run** (a second remote-like object must not steal the identity).
- **Reset** between judges is a full app restart (~40 s) and the world forgets everything.
- **Laser:** off. The green module's class is unknown; spec 0009 gates all room pointing on a documented
  module.
- **Team state:** the teammates' app is stopped; their `~/askroom` (main) is untouched; this branch is
  not pushed or merged and touches `world.py`, `answers.py`, `main.py`.
- Jetson runs 10-13 fps with room memory on; `NvMapMemAlloc error 12` lines appear at startup.

## 3. What we learned today (and must not forget)

1. **The fine-tuned detector only knows the overhead table view.** From the corner it mislabels; a
   class-agnostic proposer + a VLM name is the only thing that works from a new viewpoint without
   retraining.
2. **Open naming of a small far object fails; a targeted question works.** "What is this?" on a 40 px
   remote gave "phone", "eyeglasses case", "no usable name". "Is this the remote control?" with the object
   boxed in red inside a 240 px patch of context succeeds. Set-of-marks style beats crops.
3. **Grok leans to "yes".** A match needs confidence >= 0.7 and Grok's own description to fit the name.
4. **Static clutter re-proposes forever** (stove vents, counter items). A handoff must require that the
   pixels changed where the object appeared, and clutter Grok can't name must not block anything.
5. **A hand placing an object makes the table re-birth it several times**; same-named departed things
   are one object (the latest one carried).
6. **Departures are decided ~2 s after the object left.** Date them at the last table evidence, or a zone
   next to the table sees the object before the table "knows" it left.
7. **Answer with the freshest matching object**, not the best-scoring name.
8. **Test on the rig with a script, not by hand.** The spoken trial driver found 6 bugs in 40 minutes that
   1880 unit tests didn't. Every rig bug got a unit test the same hour.
9. **Everything is a demo assumption:** one instance per object name. Say "I think"; never assert.

## 4. External research that bears on tonight

- **xAI rate limits** (docs.x.ai/developers/rate-limits, Sep 2026): `grok-4.3` is 3 requests/s at Tier 0
  (10M tokens/min). Our earlier 6/min cap was self-imposed; 20/min is fine and could go higher. Vision
  calls measured today: ~0.6-1.5 s each. Latency, not limits, is the constraint.
- **Jetson Orin Nano and 4K MJPEG:** NVIDIA forums report `nvjpegdec` issues and no hardware *encoder*
  on the Orin Nano; hardware MJPEG decode through GStreamer is possible but not reliable inside our pip
  OpenCV container. Conclusion: **stay at 1080p tonight**; small objects (keys) are handled by choosing
  demo objects and zones, not by 4K.
- **Laser safety:** unknown green modules can exceed their label and leak infrared (NIST measured this);
  the spec's rule stands: no room pointing without a datasheet. Class 2 is required, not sufficient.
- **VLM grounding:** set-of-marks (Yang et al. 2023) is what made both visual Q&A and today's handoff
  verification work: mark the candidate, ask a closed question.

## 5. Workstreams, in priority order

Each has an owner slot, a done test, an estimate, and a fallback. **P0 items make or break the demo; P1
make it good; P2 are stretch.** Estimates assume subagents for code and a person at the rig for physical
steps; rig time is the scarce resource (one camera, one app).

### P0-1 Speech: ask out loud, hear the answer (est. 1-2 h, mostly physical)

The demo is dead silent today. Two paths; do **both**, the phone is the safety net.

- **Phone (safety net, ~30 min):** the iPhone app already sends questions over BLE and reads answers
  aloud (xAI/ElevenLabs TTS on the phone). Needs: the BLE bridge running on the Jetson alongside this
  build (check `mobile/bridge` starts with `main.py`; positions are table cm, so add the room zone name
  to the state/answer JSON, `PROTOCOL.md` v1 has `say`/text fields to carry it). Done when: a question
  typed or spoken on the phone gets the room answer read aloud.
- **Room voice (the wow, ~1 h):** a USB speaker + a USB or Bluetooth mic placed at chest height near
  where the judge stands (the Brio mic in the corner will not hear a judge over expo noise). Whisper
  runs on the Jetson already (22/22 test questions, 139 ms). Needs: the mic device name in
  `config.local.yaml` (`stt.input_device`), `tts.output_device`, and the always-listening gate tested
  in expo-like noise (`scripts/overheard_test.py`). Done when: "where's my wallet" said from 2 m away gets
  the spoken answer within 4 s, 5/5.
- Fallback: phone only. Both fail: the dashboard `/ask` box on a laptop.

### P0-2 Three objects that work, not one (est. 1.5 h rig time)

Pick the demo set now and tune for exactly those. Proposal: **wallet, glasses (case), pill bottle, remote**
(keys are ~15-20 px at couch distance at 1080p: demo them on the near end of the couch only, or skip).
For each object, run the spoken driver `scripts/room_trials.py --object <name>` on couch + side table +
counter. Done when: each object passes 3/3 handoffs and returns; total failures across the set <= 1.
Expected fixes: Grok name synonyms (`match_score` phrases), size limits in `YOLOEConfig` for a glasses
case, the pill bottle's neutral wording (already enforced). Fallback: drop the worst object from the
judge's list.

### P0-3 Speed on far zones (est. 45 min code + 20 min rig)

Today: couch 8-10 s, side table/counter 20-24 s. Target: <= 8 s everywhere. Levers, cheapest first:
1. Visit zones more often while a handoff is open: `room_every_n` 5 -> 2 while `room_handoff_hints()`
   is non-empty (the CPU cost is one extra YOLOE pass on a crop; measure fps).
2. Confirm on 1 visit instead of 2 when the track's spot changed (an arrival), keep 2 for static.
3. Raise `names_per_minute` (xAI allows 180/min) and send the verify call the moment a changed track is
   confirmed, before the round-robin comes back.
Done when: the 5-run check reports every handoff <= 8 s. Fallback: script the demo on the couch.

Built Sat night (branch `p03-speed`, unit-tested in `tests/test_room_speed.py`, not yet measured on the rig):
`room_every_n_hot` (1: a zone every frame while `room_handoff_hints()` is non-empty), `confirm_visits_arrival`
(1: a track whose spot changed confirms on its first visit; static keeps 2), `names_per_minute` 60, verification
jobs ahead of open naming and only arrivals sent to Grok while a handoff is open, and `RoomNamer.on_named`: the
World decides the track the moment its name lands (a one-track visit of its zone), not on the zone's next visit.
The verify call already runs at `reasoning_effort: none` (the auto-namer's provider is built from
`visual_memory`). Rig to-do: measure fps with a handoff open (hot mode is one more prop + YOLOE pass per frame)
and re-run the 5-run check.

### P0-4 Decoy and reset (est. 45 min)

- **Decoy:** a second remote-like object already on the couch; carry the remote to the side table; ask.
  Must answer side table, never the couch decoy (departure + arrival-change + name gates should hold; the
  test proves it). Then the reverse: the decoy arrives while the real one stays on the table: must not be
  handed off. Done when: 3/3 each way.
- **Reset:** a spoken "room, reset" / phone reset that clears room state and departures without a restart
  (today RESET clears the table world; extend `World.reset` -> `_reset_room` is already there; the
  tracker and namer queues need `RoomMemory.reset()`). Done when: reset -> next run passes in < 30 s.

### P0-5 One build, pushed, team informed (est. 30 min, Anshul)

- Push `room-memory-m0` to GitHub (full suite passes = the push rule).
- Tell the team: camera re-aimed (zoom 100, corner), their app is stopped, the demo build lives in
  `~/askroom_room`, run command, and that `world.py`/`answers.py`/`main.py` changed.
- Merge plan: merge `origin/main` into `room-memory-m0` tonight (collisions expected in `main.py`,
  `voice/answers.py`), run the suite, redeploy to `~/askroom_room`. Do not force-push anything.
- Update `PLANS.md` (the headline demo is now the room), `CONTEXT.md`, `docs/FEATURE_STATUS.md`.

### P1-1 The table's hidden objects from the corner (est. 2-3 h, parallel to rig work)

The original wow (keys under the notebook) needs the notebook and box recognised from this angle. Two
routes; run the first, decide by midnight:
- **Retrain the prop detector on this view** with the existing fast path (`scripts/finetune/README.md`:
  capture ~100-200 frames of the props on the coffee table and in the zones from the Brio, `bglabel`
  + `synthesize` + `train.py` on the Mac, export the engine in the container, ~1-2 h). Done when: on the
  rig, notebook/box/keys detected >= 0.6 on the coffee table view and the shell game passes 3/3.
- **Or**: name covers by Grok (a thing Grok calls "notebook"/"box" becomes a cover/container role). Spec
  0009 lists this as open (roles need interaction evidence); riskier tonight.
Fallback: the demo has no hiding; the room handoff is the whole show.

### P1-2 Phone room map (est. 1 h, teammate with Xcode)

The state JSON already carries `room` (zone, say, box_px, tentative). The app shows "on the couch" text
today (zone string). A room photo with the object's box drawn is a 30-minute view if the app can fetch
`/full.jpg` over Wi-Fi, or receive `box_px` over BLE and draw on a stored room photo. Done when: the
phone shows the room with the marked spot after each answer.

### P1-3 Robustness for a 2-hour expo (est. 1 h)

- Memory: watch `free -m` and the `NvMapMemAlloc` lines; keep RAM under 6.5 GB; if it climbs, lower the
  visual-memory archive and narration off (they are off/on per config).
- The room namer's queue must never grow (bounded at 8; verify under a busy room).
- A `demo_check.py` pass in the morning: camera view version, zones, Grok reachable, mic/speaker.
- Hotspot plan: Grok needs internet; the venue Wi-Fi is the risk. Offline the rig still answers from
  rules (the table) but room naming stops. Bring a phone hotspot and test the switch once.

### P2-1 Laser pointing across the room (only with a documented laser)

If someone finds a labelled Class 2 module tonight: the jog-mode calibration (spec 0009 M5) is ~2 h of
code + 30 min at the rig, and the dot-visibility test first. Otherwise: **skip**, say "the pointer is
the next step" and show the phone map instead. Do not bring an unlabelled laser to the expo.

### P2-2 Scoreboard and story

- The trial driver's JSON is real data: put "N/N room handoffs today" on the dashboard (spec 0004
  scoreboard) and in the Devpost.
- Devpost (before 8 AM): the 60-second script, a GIF of a run, the honesty line ("I think": the rig
  never asserts identity), open-source credits.

## 6. Tonight's schedule (EDT)

| When | Rig (one person + me driving) | Code (subagents) | Team |
|---|---|---|---|
| 8:30-9:15 | Speaker/mic in place; phone BLE up | P0-3 speed patches; P0-4 reset | Push branch; tell team; pick the 4 objects |
| 9:15-10:15 | P0-2: wallet, glasses, pill bottle trials | fixes from trials | Phone: zone text -> room map (P1-2) |
| 10:15-11:00 | P0-4 decoy runs; P0-3 speed re-check | P1-1 capture frames for retrain | Merge origin/main (P0-5) |
| 11:00-1:00 | Full 60-s script x 5 with voice | P1-1 train + export; P1-3 robustness | Devpost draft |
| 1:00-3:00 | Shell game on the coffee table if retrain worked | polish | sleep shifts |
| 3:00-7:00 | Sleep. One person on the rig at 7:00: `demo_check.py`, reset test, hotspot test | | |
| 7:00-8:00 | Final runs, Devpost submitted | | |

**Decision points:**
- **9:15 PM:** speech path chosen (phone / room voice / both).
- **Midnight:** retrain go/no-go: if the new engine doesn't beat 0.6 on notebook/box from the corner,
  drop hiding from the script.
- **1 AM:** freeze the demo build. After that only config, docs, and the morning checks.

## 7. Risks and what we do about them

| Risk | Odds | Mitigation |
|---|---|---|
| Expo noise defeats always-listening | high | wake word "room" mode + clicker; phone as the primary ask device |
| Venue Wi-Fi drops (Grok unreachable) | medium | hotspot tested tonight; offline the rig says "I'm offline; ask me on the table" |
| Judge's object is small/dark (keys) | medium | the object list on a card; keys only on the near couch |
| A second similar object in the room | medium | decoy runs tonight; the gates (arrival change, name, one-to-one) |
| Jetson memory pressure over 2 hours | medium | robustness pass; restart script ready; reset without restart |
| Team merge conflict on Sunday morning | high if left | merge tonight (P0-5), one build in `~/askroom_room` |
| Someone moves the camera | low | view-version check in `demo_check`; tape the mount; zones redrawn in 5 min if needed |

## 8. Morning checklist (7:00 AM)

1. `python demo_check.py` all green (camera view, zones, room memory, network, audio, clock).
2. Reset works; one full 60-s run with voice and one with the phone.
3. Hotspot switch tested once; Wi-Fi back.
4. Object card on the table (the 4 objects), the notebook/box only if P1-1 passed.
5. Dashboard `/full.jpg` on the laptop screen for judges; scoreboard shows today's counts.
6. Laser off and unplugged unless P2-1 passed. Kill switch checked.
