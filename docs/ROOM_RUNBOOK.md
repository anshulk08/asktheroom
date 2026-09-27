# Room runbook: the room demo (spec 0010)

The demo is the **room** (spec 0010 §1): the Brio high in the living-room corner (zoom 100, 1080p), the coffee
table as the table view, four drawn zones (couch, side table, counter, stove). A judge moves an object from
the table into the room and asks where it is. This runbook is how to deploy, check and run that build.
The table-only runbook (overhead camera, shell game, table laser) is `docs/RIG_RUNBOOK.md`; it does not apply
to the corner camera.

Conventions:

- Jetson: `guru@10.90.84.178` (Wi-Fi; the USB link is down). The live build is `~/askroom_room`; its local
  files (`config.local.yaml`, `room_zones.json`, `table_cal.json`, `.env`, `models/`, `data/`) are **not** in
  git and are never overwritten by a deploy.
- Dashboard: `http://10.90.84.178:8080` (`/full.jpg` the whole room with zones, `/state`, `POST /ask`).
- The Brio, always by path: `CAM=/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0`.
- Only one thing may hold the camera: the room app, `demo_check.py`, or a capture. Stop the app first.
- The room app has priority over everything else on the Jetson; keep `free -m` "available" above 1.5 GB.

```bash
ssh guru@10.90.84.178
cd ~/askroom_room
CAM=/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0
RIG=http://10.90.84.178:8080
```

## 1. Deploy `room-demo-integration` to `~/askroom_room`

From a Mac with the repo (Anshul applies this at his P0-5 merge):

```bash
git fetch origin && git checkout room-demo-integration            # or merge it into room-memory-m0 first
ssh guru@10.90.84.178 'cp -a ~/askroom_room/config.local.yaml ~/askroom_room/room_zones.json ~/askroom_room/table_cal.json /tmp/ 2>/dev/null; true'   # belt and braces
git archive HEAD | ssh guru@10.90.84.178 'tar -x -C ~/askroom_room && echo "$(date +%T) room-demo-integration '"$(git rev-parse --short HEAD)"'" >> ~/askroom_room/DEPLOYED'
```

`git archive | tar -x` overwrites tracked files only: the local files above stay. Then restart the app (§2).

## 2. Start, stop, restart the room app

```bash
# stop
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'               # the room app is the askroom container
docker stop <that container>

# start (voice on; for dashboard/phone only add --no-voice)
cd ~/askroom_room && nohup scripts/dock.sh python3 -u main.py --camera $CAM --port 8080 \
    > data/room/app.log 2>&1 < /dev/null &
tail -f data/room/app.log                                            # wait for the dashboard line
```

*Restart script and reset-without-restart: from the room/check and room/voice branches, pasted here when merged.*

## 3. Morning checklist (spec 0010 §8), 7:00 AM

| # | Check | Command | Pass |
|---|---|---|---|
| 1 | Pre-demo check | app stopped, then `scripts/dock.sh python3 demo_check.py --camera $CAM` | every line green (camera, zones drawn at this view, room memory, network, audio, clock) |
| 2 | Start the app | §2 start | `/healthz` answers: `curl -s $RIG/healthz` |
| 3 | Reset | `curl -s -X POST $RIG/ask -H 'content-type: application/json' -d '{"text":"ask the room, reset"}'` | answered in < 5 s, then one full run passes |
| 4 | One full 60-s run with voice | the §1 script of spec 0010, spoken from the judge's spot | spoken answer names the right zone |
| 5 | One full run with the phone | the iPhone app over BLE | the answer is read aloud on the phone |
| 6 | Hotspot switch | §5 | answers keep coming; Wi-Fi back after |
| 7 | Object card | wallet, glasses (case), pill bottle, remote (keys only on the near couch) | card on the coffee table; notebook/box only if the retrain passed |
| 8 | Judges' screen | open `$RIG/full.jpg` on the laptop | the room with zones and the last places |
| 9 | Laser | off and unplugged (spec 0010 P2-1: no documented module) | |
| 10 | Memory | `free -m` | available > 1.5 GB with the app up; no growth over 10 min |

Scripted handoff check any time (from a laptop, stdlib only):

```bash
python3 scripts/room_trials.py --rig $RIG --object remote --zones couch side_table couch counter couch --out data/room/trials.json
# then put the session on the dashboard scoreboard (room_trials overwrites --out and its records carry no
# object name or time, so upload after every session, naming the object):
curl -X POST -H 'content-type: application/json' --data @data/room/trials.json "$RIG/scoreboard/trials?object=remote"
```

The dashboard shows "room handoffs today" and a per-object / per-zone table from these uploads (real trials
only; this is the Devpost number too, `docs/DEVPOST.md`).

## 4. Mic and speaker bring-up

The full procedure (device discovery, level checks, the wake-word gate in expo noise, the own-speaker echo
test, spoken trials with `scripts/voice_trials.py`) is **`docs/runbooks/room-voice.md`** (room/voice branch;
in this build once that branch is merged). The rig-side prerequisites, in `~/askroom_room/config.local.yaml`:

```yaml
stt: {input_device: "<USB mic name part>"}      # not the corner Brio's mic: it can't hear a judge over the hall
tts: {output_device: "<USB speaker name part>"} # null = HDMI, which is silent on the rig
listen: {wake_words: [ask the room, askroom, ask room]}
demo: {hold_notices: true}                      # nothing speaks unasked while judges are there
```

Then run the app **with voice** (§2 start, without `--no-voice`) and follow the voice runbook's checks.

## 5. Hotspot switch and restart script

*From the room/check branch. Pasted here when merged.*

## 6. Retrain slot (P1-1, Anshul's 10:15 PM rig slot)

The full checklist is **`docs/corner_retrain.md`** (room/detector branch; in this build once that branch is
merged). In short:

- **A** (no camera): a scratch copy `~/askroom_ft` of room/detector on the Jetson with `~/askroom_room`'s
  `config.local.yaml` and `room_zones.json` copied in; props and 4 distractors ready; demo lighting.
- **B-C** (the only camera time, Anshul's call): stop the room app, capture the table view as the room build
  sees it (`scripts/finetune/capture.py --room table ...`), zoom stays 100.
- **D-F** (Mac): label, merge with the public hand frames, train YOLO26s.
- **G** (Jetson, with the lock and >= 1.5 GB free): export the engine in the container.
- **Midnight go/no-go:** notebook, box and keys at confidence >= 0.6 in >= 80% of the demo layout's frames
  (`conf_sweep --images ... --require`). If not: no hiding in the demo script (spec 0010 P1-1 fallback).
- Put the room app back afterwards (§2 start) and check `$RIG/full.jpg`.

## 7. Restore notes

- **Camera view.** Zones are drawn at one camera view (`room_zones.json`). If the Brio is bumped or its zoom
  changes, `demo_check` check 10 fails: re-aim, `scripts/camera_setup.sh` with the room settings (zoom 100),
  and redraw the zones: `scripts/dock.sh python3 -m core.room --zone NAME --say TEXT --poly x,y x,y x,y ...`.
  The table view: `python -m core.room --measure-rect` into `config.local.yaml` (`room_memory.table_view_rect`).
- **Local files.** Keep a copy of `config.local.yaml`, `room_zones.json` and `table_cal.json` from a working
  state: `mkdir -p ~/room_frozen && cp -p ~/askroom_room/{config.local.yaml,room_zones.json,table_cal.json} ~/room_frozen/`.
  Restore: copy back and restart the app.
- **Code.** The previous build is whatever `DEPLOYED` names before the last line; redeploy that commit with §1.
- **Other stacks.** `baby-tau` (not ours) was stopped Sat 26 Sep ~20:00 with `docker compose stop`; bring it
  back only after the expo: `cd ~/git/autonomous-intelligence/baby-tau && docker compose start`.
