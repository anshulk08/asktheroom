# Recording the room-demo identity clips (spec 0010)

Tonight's failure: one real object on the coffee table became dozens of `thing:N` identities (feet at the
table edge, clutter, people on the couch). These eight guided clips record it once, on the rig, so it can
be replayed and scored on the Mac as often as needed (`eval/scorecard.py`). Total camera time: about 20
minutes, including stopping and restarting the app.

**Who does what.** The rig owner (WS7) stops and restarts the live app; nobody else touches it. One person
at the Mac runs the commands (the Mac speaks every cue). One or two people act in the room.

## Scenarios

| Clip | Length | What happens | Scored for |
|---|---|---|---|
| `room_still` | 103 s | the six props (wallet A, keys B, phone C, box BOX, notebook NB, pill bottle PB) put down one by one, then nobody near for 60 s | one entity per object, 0 phantom births/min still, position within 5 cm |
| `room_clutter` | 103 s | same, with an open laptop and a cable pile already on the table | the same, with clutter (`scene_objects`) |
| `room_couch` | 129 s | props down, then people on the couch: feet up at the table edge, hands near the edge, sit still, walk past, leave | phantom births/min with people (target at most 1), body-part / clothing things (target 0) |
| `room_carry` | 138 s | wallet put down, then carried to the couch, back, the side table, back, the kitchen counter, back | room handoff to the right zone, identity on return |
| `room_move` | 96 s | wallet, phone, notebook put down; phone and wallet picked up and put down elsewhere; notebook slid | one entity per object after moves |
| `room_remove` | 123 s | the six props put down, then taken away one by one, then nobody near | removed props not ghosted, no phantom births |
| `room_straight` | 81 s | wallet and notebook put down; then the phone put straight onto the couch and the keys onto the floor, never on the table | room placement seen, no false handoff of a table prop, no table phantom |
| `room_block` | 73 s | wallet, phone, pill bottle put down; someone sits or crouches in front of the pill bottle ~10 s, then moves away | the pill bottle keeps its identity through the occlusion |

Every clip starts by putting each prop down on a spoken cue: that tells the scorer which identity is which
prop, with no annotation afterwards.

## Before (rig owner, WS7)

```bash
ssh guru@10.90.84.178
cd ~/askroom_room
scripts/room_app.sh status        # note the args it was started with
scripts/room_app.sh stop          # or, for a hand-launched app: docker stop <its container name>
```

The recorder needs the camera. `eval.guided` refuses to start while any container runs `main.py` and never
stops the app itself. The rig's `~/askroom_room` needs this branch's `eval/raw_record.py`, `eval/clip.py`
and `eval/guided.py` (it records at `room_memory.capture_size`, 2560x1440, and writes the view and zones
into `meta.json`); copy just those three files if the rig checkout is older. Camera settings are left as
the app set them (room clips never run `camera_setup.sh`).

## Recording (Mac, in this checkout)

```bash
export ASKROOM_JETSON=guru@10.90.84.178 ASKROOM_RIG_DIR=askroom_room
# optional: ASKROOM_IMAGE=askroom:audio if askroom:latest is not on the rig (dock.sh's image)
python -m eval.guided --list
python -m eval.guided room_still   --id room_still_1
python -m eval.guided room_clutter --id room_clutter_1
python -m eval.guided room_couch   --id room_couch_1
python -m eval.guided room_carry   --id room_carry_1
python -m eval.guided room_move    --id room_move_1
python -m eval.guided room_remove  --id room_remove_1
python -m eval.guided room_straight --id room_straight_1
python -m eval.guided room_block   --id room_block_1
```

Each run prints the setup and waits for Enter, says "Get ready", then speaks each cue. Each clip lands in
`data/clips/<id>/` on the rig and the Mac (video.mp4 at 2560x1440, frames.json, meta.json, truth.json).
The last line gives frames, fps and how many frames the writer dropped: a few is fine (frames.json has the
real times), hundreds means the Jetson could not encode 1440p fast enough; say so.

## What the people in the room do

- **room_still.** Start with an empty table, holding all six props. On each cue put that prop down (spread
  out, a hand-width apart) and pull your hand back. Then stay out of the camera's view of the table for the
  full minute: nobody walks past.
- **room_clutter.** Before Enter: open a laptop on the table and drop a loose pile of cables next to it.
  Then as room_still, putting the props among the clutter without touching it.
- **room_couch.** As room_still for the props. Then one or two people sit on the couch; on the cue put your
  feet up on the table edge near the props (not touching them), later lean forward with hands near the
  edge, then feet down and sit still, then stand and walk past the table, then leave.
- **room_carry.** Phone, notebook and box on the table beforehand; hold the wallet. Put it down on the cue.
  On each carry cue take it to the couch / side table / kitchen counter, put it where the camera can see it,
  and step out of the way; on the next cue bring it back to the table.
- **room_move.** Box and pill bottle on the table beforehand; hold wallet, phone, notebook. Put each down on
  its cue; later pick up the phone (hold it up), put it on the other side; same with the wallet; slide the
  notebook to a new spot.
- **room_remove.** Put all six down on the cues, then on each cue take that prop off the table and out of
  sight (pocket, bag, behind your back), then stay away from the table.
- **room_straight.** Put the wallet and notebook down on the cues. Then walk the phone round the table (not
  over it) and put it on the couch where the camera can see it; later put the keys on the floor beside the
  table. Neither ever touches the table. Then everyone steps away.
- **room_block.** Put the pill bottle near the table edge by the couch, then the others, on the cues. On the
  block cue sit or crouch in front of the pill bottle so the camera can't see it, keep still without
  touching it; on the next cue move away.

## After (rig owner, WS7)

```bash
cd ~/askroom_room && scripts/room_app.sh restart     # the args of the last start
scripts/room_app.sh status
```

## Scoring (Mac, any time, no rig)

```bash
# the Mac has no TensorRT: hands off (no fixed-class detector) and the YOLOE .pt the rig's engine was made from
python -m eval.scorecard data/clips/room_*_1 --hands-off --yoloe-model models/yoloe-26s-seg-pf.pt \
    --save-trace --json card.json
python -m eval.scorecard data/clips/room_*_1 --from-trace          # rescore without replaying
python -m eval.scorecard data/clips/room_carry_1 --hands-off --yoloe-model models/yoloe-26s-seg-pf.pt --grok
                                                                  # names (body parts, handoffs) need Grok
python -m eval.score_clip data/clips/room_move_1 --hands-off --yoloe-model models/yoloe-26s-seg-pf.pt
                                                                  # the older PASS/FAIL report
# another tracker on the same clips, same code path: any config key (dotted, YAML value), repeatable
python -m eval.scorecard data/clips/room_*_1 --hands-off --yoloe-model models/yoloe-26s-seg-pf.pt \
    --mode registry --json card-registry.json        # --mode M is --set permanence.mode=M
python -m eval.scorecard data/clips/room_*_1 --set permanence.mode=registry --set proposals.yoloe.conf=0.25
```

Name the traces per tracker when saving them (`--save-trace --trace-name trace-registry.json`), since
`--from-trace` rescores whatever trace it finds. The card header lists the overrides it replayed with.

Replay cuts each 1440p frame to the table view exactly as the app does (`TableView`, the recorded
`table_view_rect`) and runs room memory on the full frame with the recorded zones. Offline, nothing is
named: body-part things and the naming hook read n/a, and thing handoffs (which need a Grok name) only
happen with `--grok` (live calls, so two replays can differ). On the Jetson (stop the app first),
`scripts/dock.sh python3 -m eval.scorecard data/clips/<id>` replays with the recorded engines and hands.
