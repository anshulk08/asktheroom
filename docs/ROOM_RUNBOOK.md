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

`scripts/room_app.sh` runs the app in one container (`askroom_room_app`): starting it twice never launches a
second app, the log is appended (`data/room/app-<time>.log`, with `data/room/app.log` pointing at the
latest), and `start` refuses while another container runs `main.py` or `demo_check.py`, or something holds
the camera. It prints how to stop that one and never stops it itself.

```bash
cd ~/askroom_room
scripts/room_app.sh status                                   # FIRST, once: its flock / bash paths are unverified on the Jetson
scripts/room_app.sh start --camera $CAM --port 8080          # voice on; add --no-voice for dashboard/phone only
scripts/room_app.sh restart                                  # no args: the ones the last start used (data/room/app.args)
scripts/room_app.sh stop
tail -f data/room/app.log
```

"started" is printed only once the dashboard answers (60 s), else the log's last lines. The **first switch**
from the hand-launched app (e.g. `upbeat_curie`) to the script is Anshul's call: `docker stop <that container>`,
then `start`.

**Before 7 AM, run the old launch command once too** (this build's `scripts/dock.sh` binds `/dev` with a V4L2
cgroup rule so a replugged Brio reappears; if the container or its terminal misbehaves, fall back with
`ASKROOM_DEV_BIND=0`):

```bash
docker stop askroom_room_app 2>/dev/null
cd ~/askroom_room && nohup scripts/dock.sh python3 -u main.py --camera $CAM --port 8080 > data/room/app-manual.log 2>&1 < /dev/null &
# misbehaves? stop it and: ASKROOM_DEV_BIND=0 nohup scripts/dock.sh python3 -u main.py --camera $CAM --port 8080 ...
```

**Reset without a restart:** a spoken "ask the room, reset" (or the `curl` in §3 row 3) clears the table and
room memory (spec 0010 P0-4); restart only if reset doesn't bring the next run back.

## 3. Morning checklist (spec 0010 §8), 7:00 AM

| # | Check | Command | Pass |
|---|---|---|---|
| 1 | Pre-demo check, app running | `scripts/dock.sh python3 demo_check.py --live` | green: 6 network, 9 clock, 10 zones at this camera view, 12 room app (fps, `/full.jpg`), 13 Grok round trip, 14 mic and speaker by name, 15 RAM >= 1 GB and no NvMapMemAlloc / tracebacks in the app log, 16 namer queue <= 8, 17 table calibration fits this view and a tabletop outline is set (§8). It never opens the camera or mic |
| 2 | App up | `scripts/room_app.sh status` and `curl -s $RIG/healthz` | running, dashboard answers |
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
test, spoken trials with `scripts/voice_trials.py`) is **`docs/runbooks/room-voice.md`**. The rig-side prerequisites, in `~/askroom_room/config.local.yaml`:

```yaml
stt: {input_device: "<USB mic name part>"}      # not the corner Brio's mic: it can't hear a judge over the hall
tts: {output_device: "<USB speaker name part>"} # null = HDMI, which is silent on the rig
listen: {wake_words: [ask the room, askroom, ask room]}
demo: {hold_notices: true}                      # nothing speaks unasked while judges are there
room_check: {pulse_sink: bluez}                 # once the Bluetooth speaker is paired (askroom:audio image, PulseAudio):
                                                # demo_check --live check 14 then fails if the sink fell back to HDMI
```

A Bluetooth speaker plays through the host's PulseAudio (the `askroom:audio` image; `scripts/dock.sh` passes the
socket through): set `tts.output_device: pulse` (or `default`) and see `docs/runbooks/room-voice.md` §11.

Then run the app **with voice** (§2 start, without `--no-voice`) and follow the voice runbook's checks.

## 5. Hotspot switch

The procedure is **`docs/HOTSPOT.md`**: save the phone hotspot once with `nmcli` (autoconnect off), switch with
`sudo nmcli connection up askroom-hotspot`, verify with `scripts/dock.sh python3 demo_check.py --live --only 6 9 13`
and `scripts/room_app.sh status`, switch back with `sudo nmcli connection up "<venue connection NAME>"`.
Do it at the rig's keyboard or over USB-C: Wi-Fi SSH drops and the Jetson's address changes on the hotspot.
The app keeps running; it notices within 5 s and re-warms Grok. Offline, table answers still work, but no
new room object gets a name.

## 6. Retrain slot (P1-1, Anshul's 10:15 PM rig slot)

The full checklist is **`docs/corner_retrain.md`**. In short:

- **A** (no camera): a scratch copy `~/askroom_ft` of room/detector on the Jetson with `~/askroom_room`'s
  `config.local.yaml` and `room_zones.json` copied in; props and 4 distractors ready; demo lighting.
- **B-C** (the only camera time, Anshul's call): stop the room app, capture the table view as the room build
  sees it (`scripts/finetune/capture.py --room table ...`), zoom stays 100.
- **D-F** (Mac): label, merge with the public hand frames, train YOLO26s.
- **G** (Jetson, with the lock and >= 1.5 GB free): export the engine in the container.
- **Midnight go/no-go:** notebook, box and keys at confidence >= 0.6 in >= 80% of the demo layout's frames
  (`conf_sweep --images ... --require`). If not: no hiding in the demo script (spec 0010 P1-1 fallback).
- Put the room app back afterwards (§2 start) and check `$RIG/full.jpg`.

## Troubleshooting

- **demo_check 12 can't reach the app:** it looks at `room_check.app_url`, else the `--port` that
  `room_app.sh start` saved, else `server.port` (8000). A hand-launched app on 8080: set
  `room_check: {app_url: http://127.0.0.1:8080}` in `config.local.yaml`.
- **demo_check 14 fails:** `stt.input_device` / `tts.output_device` must be part of the device's **name**
  (`cat /proc/asound/cards` lists them), not an index (indexes shift on replug). A device the app holds open
  (RUNNING) passes. `null` fails on purpose: the corner Brio's mic and silent HDMI.
- **demo_check 15 says "no app log":** it reads `data/room/app.log`, which `room_app.sh` maintains; for a
  hand-launched app point `room_check.app_log` at its log.
- **Presence acts odd** (objects flip present/absent too fast or too slow after the time-based debounce):
  `presence: {hz: null}` in `config.local.yaml` restores the old frame-count debounce; restart the app.
- **The camera is gone after a replug:** `ls /dev/v4l/by-id/` on the host; if it's there but not in the
  container, restart the app (`room_app.sh restart`), or with `ASKROOM_DEV_BIND=0` if the `/dev` bind is the
  suspect.

## 7. Restore notes

- **Camera view.** Zones are drawn at one camera view (`room_zones.json`). If the Brio is bumped or its zoom
  changes, `demo_check` check 10 fails: re-aim, `scripts/camera_setup.sh` with the room settings (zoom 100),
  and redraw the zones: `scripts/dock.sh python3 -m core.room --zone NAME --say TEXT --poly x,y x,y x,y ...`.
  The table view: `python -m core.room --measure-rect` into `config.local.yaml` (`room_memory.table_view_rect`).
  Then recalibrate the table and redraw the tabletop (§8).
- **Local files.** Keep a copy of `config.local.yaml`, `room_zones.json`, `table_cal.json` (with `table_cal_view.png`)
  and `table_area.json` from a working state:
  `mkdir -p ~/room_frozen && cp -p ~/askroom_room/{config.local.yaml,room_zones.json,table_cal.json,table_cal_view.png,table_area.json} ~/room_frozen/`.
  Restore: copy back and restart the app.
- **Code.** The previous build is whatever `DEPLOYED` names before the last line; redeploy that commit with §1.
- **Other stacks.** `baby-tau` (not ours) was stopped Sat 26 Sep ~20:00 with `docker compose stop`; bring it
  back only after the expo: `cd ~/git/autonomous-intelligence/baby-tau && docker compose start`.

## 8. Recalibrate the table and draw the tabletop

For the user and WS7. Do it when the camera, its zoom or resolution, or the coffee table moves (`demo_check`
17 FAILs), and **once now**: the 26 Sep 18:30 calibration predates 1440p and the table move (its tag outline sits
on the table's far edge), and there is no tabletop outline, so the phone's map is the camera's whole view of the
table plane (100.8 x 73.1 cm), not the table. The app keeps running throughout: it holds the camera and does
the fit. `python -m core.table` can't (no camera while the app runs, and `--image` can't do the one-tag fit,
which averages 15 frames).

1. **Tag size.** `config.local.yaml` says `table_tag: {size_cm: 14.5}` (the iPad stand-in). With the printed
   tag (AprilTag 36h11, id 0), measure its black square and set `size_cm` to that (16.0 as printed), then
   `scripts/room_app.sh restart`: config is read at start.
2. **Lay the tag flat on the tabletop**, near the middle, not on a stand and not held: the fit assumes the tag
   lies in the table's plane. All of it inside the table view (the blue box on `$RIG/full.jpg`); no hands or feet
   on the table.
3. **Recalibrate** through the app (one-tag fit, about a second):

   ```bash
   curl -s -X POST $RIG/ask -H 'content-type: application/json' -d '{"text":"ask the room, recalibrate"}'
   grep -E "table calibrat|recalibration" data/room/app.log | tail -3    # "table calibrated from tag 0: tracked area W x H cm"
   ```

   or say "ask the room, recalibrate". "I couldn't recalibrate": the tag wasn't seen; move it nearer the middle,
   flatter, out of glare, and ask again. "Recalibrating by voice is turned off": `perception_guard:
   {voice_recalibrate: true}` in `config.local.yaml`, restart, ask again. The fit writes `table_cal.json` (now
   with the view it was made at) and `table_cal_view.png` beside it.
4. **Remove the tag.**
5. **Draw the tabletop** (a new calibration voids any old outline). Take the whole view with nothing drawn on
   it (`?raw=1`: 2560 x 1440, the capture's own px), read the tabletop's corners off it in order in a viewer
   that shows pixel coordinates (a little inside the edge; where the table runs out of the table view, the
   blue box on `$RIG/full.jpg`, put the corner on the box's edge), and give them as `--outline-full`:

   ```bash
   curl -s -o full.jpg "$RIG/full.jpg?raw=1"
   scripts/dock.sh python3 -m core.table --outline-full 176,1104 712,1100 492,1408 4,1408 4,1266 --image full.jpg
   ```

   The example is the coffee table on the 27 Sep 00:31 frame: far-left, far-right (under a hand then,
   extrapolated), near-right, and the two points where the left and near edges leave the view. Read yours off
   your frame. Open `full_outline.jpg`: the green outline must sit on the tabletop. It saves `table_area.json`.
6. `scripts/room_app.sh restart`: the new tracked area and the outline are read at start.
7. `scripts/dock.sh python3 demo_check.py --live --only 17` must say:

   ```text
   [PASS] 17 table view   calibrated at this view; compared with the app's /full.jpg; tag spot matches (0.9x); view shift 0.x%; tabletop outline 5 corners
   ```

   `[WARN] no view recorded: recalibrate`: `table_cal.json` is still the old one (step 3 didn't save).
   `[WARN] no tabletop outline ...`: step 5. `[FAIL] calibrated at another view (...)`: the config's zoom or
   `table_view_rect` changed since the fit; `[FAIL] the scene at the tag spot changed` or `the view shifted`:
   the camera or table moved. Either way, from step 2. The match threshold is `demo_check.patch_min_ncc`
   (lighting doesn't lower it). Then freeze the files (§7 Local files).
8. **On the phone, pick the seat** ("I sit here"), so "your left" and the map match where the user sits.
