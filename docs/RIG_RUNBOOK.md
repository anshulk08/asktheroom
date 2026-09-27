# Rig runbook: the table shell-game demo

For the teammate at the rig. Goal (PLANS.md): a judge plays the shell game (keys under the notebook, notebook
into the box, box slid across), asks where the keys are, and the rig answers correctly with the laser on the
box, every time, even offline. Work top to bottom; each step says what "done" looks like. Write every number
you measure into the log table at the end and send it back.

Conventions:

- The Jetson is `guru@10.90.84.178` (Wi-Fi) or `guru@192.168.55.1` (USB). Every command below runs **on the
  Jetson, in the deploy directory**, unless it says "Mac".
- The deploy directory is `~/askroom_rig`: a clean copy of the integration branch, prepared for you (it has
  `config.local.yaml` from `~/askroom`, and `.env` and `models/` linked to `~/askroom`'s). Calibration files
  (`table_cal.json`, `table_area.json`, `laser_cal.json`) are written there.
- Python runs in the app container: `D=scripts/dock.sh` below, so `$D python3 -m ...`.
- The camera is the Brio, always by its stable path:
  `CAM=/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0` (`video-index2` is its IR node; never
  use a bare index, it moves on replug).
- Only one thing may hold the camera at a time: the app, `demo_check`, `eval.record` or `core.table`.

```bash
cd ~/askroom_rig
D=scripts/dock.sh
CAM=/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0
```

## 1. Hygiene (only after the orchestrator's OK, given at that moment)

These stop other people's processes. Do not run them on your own. (Sat 26 Sep ~20:00: the room app and
baby-tau were stopped with the user's OK; `~/askroom_rig` was prepared; the USB link was still down, so the
Jetson is on Wi-Fi only. Check `docker ps` and skip what is already done.)

```bash
docker ps --format '{{.Names}}  {{.Image}}  {{.Status}}'      # note what runs
docker top <room container> -o pid,etime,args                   # the room app: main.py --no-voice, from ~/askroom_room
docker stop <room container>                                    # frees the Brio and ~2.8 GB
( cd <baby-tau compose dir> && docker compose stop )            # or: docker stop baby-tau-ollama-1 baby-tau-whisper-1 baby-tau-piper-tts-1 baby-tau-python-1
free -m                                                         # expect > 4.5 GB available
ls -d ~/askroom_*                                               # stale scratch copies; remove only the ones the orchestrator lists
```

- Reconnect the USB cable (Jetson to laptop) so SSH works without Wi-Fi; step 6 needs it:
  `ping -c1 192.168.55.1` from the Mac.
- Clock: the Jetson has no RTC battery. `date` on the Jetson and the Mac must agree within a few seconds.
  If not: `sudo date -s "$(ssh <mac> date -u +%FT%TZ)"` or, online, `sudo systemctl restart systemd-timesyncd`.
- Done when: only what the demo needs is running, `free -m` shows > 4.5 GB available, `date` is right.

## 2. Light, exposure, table frame, outline, then freeze

1. **Even light.** Light the table evenly: no sun patch or hard shadow across it, no glare on the box. Keep
   the same light for every later step; if it changes, redo steps 2 and 3.
2. **Lock the camera** (auto exposure and autofocus off; values measured over this table on Sat 26 Sep):

   ```bash
   scripts/camera_setup.sh 166 $CAM 80 10 160 3200   # exposure 16.6 ms, gain 80, focus 10, zoom 160, 3200 K
   ```

   Too dark or too bright in venue light: change only the gain (3rd value) in steps of 20 and rerun. Run it
   again after any replug (the app also restores the controls itself when the camera drops off USB).
3. **Table frame (one AprilTag).** Lay the printed tag (36h11, id 0) flat near the middle of the table.
   `table_tag.size_cm` in `config.local.yaml` must be the printed black square's side: 16.0 for the printed
   tag, 14.5 for the iPad stand-in. Then:

   ```bash
   $D python3 -m core.table --device $CAM
   ```

   Done when it prints `calibrated: True`, a tracked area, and `saved table_cal.json`. Take the tag away: the
   app loads `table_cal.json` at startup.
4. **Tabletop outline** (objects are only born inside it; an arm or the floor at the frame edge is not):

   ```bash
   $D python3 -m core.table --outline                  # prints these steps
   # grab a 1280x720 frame (app stopped):
   $D python3 -c "import cv2; c = cv2.VideoCapture('$CAM', cv2.CAP_V4L2); c.set(3, 1280); c.set(4, 720); [c.read() for _ in range(30)]; cv2.imwrite('frame.jpg', c.read()[1])"
   # Mac: scp guru@10.90.84.178:askroom_rig/frame.jpg . and open it in Preview (Tools > Show Inspector shows px)
   # read the tabletop's corners in px (a little inside the real edge), in order around the table, then:
   $D python3 -m core.table --outline-px X1,Y1 X2,Y2 X3,Y3 X4,Y4 --image frame.jpg
   ```

   Check the drawn `frame_outline.jpg`: the outline hugs the tabletop. Restart the app afterwards.
5. **Freeze.** From here on nothing may move the table frame: not the camera, not the table, and nobody says
   "recalibrate" (it refits and invalidates the outline and the laser fit). Keep a copy to restore:

   ```bash
   mkdir -p calib_frozen && cp -p table_cal.json table_area.json calib_frozen/
   # (after step 3 also: cp -p laser_cal.json calib_frozen/)
   # restore: cp -p calib_frozen/*.json . && restart the app
   ```

## 2b. Brio regression clips (guided, about 25 min)

The replay clips the fixes are measured on were recorded on the old camera; these replace them. Record them
right after step 2 (same light, camera settings, table frame and outline: each clip stores its calibration and
outline), **before** step 3 (a laser dot in the frame would be a new object), with the app stopped.

`eval.guided` runs on a **Mac next to the table**: the Mac speaks each cue out loud (`say`), you do what it
says, and the Jetson records. Nobody annotates afterwards: the cues are the ground truth. It needs the
integration branch on that Mac (`~/asktheroom/wt-integration` on the orchestrating Mac) and SSH to the Jetson.

```bash
# Mac, in the integration worktree
export ASKROOM_JETSON=guru@10.90.84.178 ASKROOM_REMOTE_DIR=askroom_rig
PY=~/asktheroom/askroom/.venv/bin/python
$PY -m eval.guided --list                          # each clip's setup and length
$PY -m eval.guided shell --id brio_shell_1 --no-setup
```

`--no-setup` keeps the camera exactly as step 2.2 locked it. Each run prints the setup: lay the props out as it
says, then press nothing: it says "Get ready", then the cues. It ends with "Done." and copies the clip to
`data/clips/<id>` on the Mac (and leaves it in `~/askroom_rig/data/clips/<id>` on the Jetson).

Props (the same every clip): A wallet, B a small solid object (the keys), C the phone, NB the notebook, BOX the
open box (open side up).

| # | Command | Length | What the cues ask |
|---|---|---|---|
| 1 | `$PY -m eval.guided still --id brio_still_1 --no-setup` | 32 s | Notebook, wallet, keys, phone, box spread out, not touching; hands away the whole time |
| 2 | `$PY -m eval.guided hands --id brio_hands_1 --no-setup` | 32 s | Same layout; wave a hand over the table, rest a forearm on an empty part, hands away |
| 3 | `$PY -m eval.guided place_name_pickup --id brio_place_1 --no-setup` | 28 s | Only the notebook down, wallet in hand: put the wallet in the middle, pick it up and hold it, put it down on the right |
| 4-6 | `$PY -m eval.guided shell --id brio_shell_1 --no-setup` (then `brio_shell_2`, `brio_shell_3`) | 40 s each | Box and notebook apart, keys in hand: keys in the middle; slide the notebook over them; lift it, keys into the box, notebook aside; slide the box to a new spot. Vary the hand and where the box ends up |
| 7 | `$PY -m eval.guided shell --id brio_shell_dim_1 --no-setup` | 40 s | The same game with the room lights partly off (**do not** rerun camera_setup: the point is a darker picture at the locked exposure). Lights back on after |
| 8 | `$PY -m eval.guided blanket --id brio_blanket_1 --no-setup` | 30 s | Keys, phone and notebook spread out, blanket in hand: lay it over everything; lift it off and take it away |

Between clips: reset the props (about 1-2 min). If a cue was missed or a hand stayed in view, rerun with the
same `--id` (it overwrites). Done when: 8 clips, each printed `... frames, ... s (~30 fps)`. Tell the
orchestrator; the replays move to these clips.

## 3. Laser (F6): calibrate, ruler check, kill switch

*From the laser/demo_check workstream (fix/laser-check).* **Needs:** fix/laser-check merged; the table
calibrated and its outline set (step 2); the app stopped (it holds the Brio). Keep the kill switch within
reach and nobody's eyes at table height. The laser turns itself off after 10 s idle.

### 3.0 Image and driver (once)

1. Put servokit's wheels into `docker/wheels`: the `pip3 download adafruit-circuitpython-servokit==1.3.24
   Jetson.GPIO==2.1.11 ...` command is in the header of `docker/Dockerfile`.
2. Rebuild: `docker build -t askroom:latest docker/` (adds espeak-ng too if the build has network, e.g. a hotspot).
3. On the host, `sudo i2cdetect -y -r 7` must show `40` (the board on I2C bus 7).
4. `$D python3 -c "import board, adafruit_servokit; print(board.board_id)"` must print the board.
5. In `config.local.yaml` set `actuator: pca9685`.

If servokit is missing or the board isn't found, main.py logs `LASER DISABLED ...` and answers by voice only
(it no longer crashes).

### 3.1 Calibrate

```bash
$D python3 -m act.calibrate --rig        # from a real terminal (ssh -t): the jog reads single keys
```

- **Jog.** The dot starts mid-travel. `a`/`d` (or left/right) pan, `w`/`s` (or up/down) tilt, `[`/`]` change
  the step (2-50 µs a press; start at 10). Drive the dot to each corner of the tabletop in order (top-left,
  top-right, bottom-right, bottom-left, as the camera sees it) and press Enter at each. `u` undoes the last
  corner, `l` toggles the laser, `q` quits. If the dot goes out (10 s auto-off), any key lights it again.
- **Then it runs by itself for 1-2 min:** latency measurement, the grid fit (dots off the tabletop outline
  are dropped), 10 held-out aims, the centre aim.
- **It prints** `fit: N points ... error median X cm`, then `F6 gate: fit median X < 1.5 cm PASS/FAIL; centre
  ... Y cm < 3 cm PASS/FAIL`, and a `servo_limits:` / `camera_latency_s:` block: copy that block into
  `config.local.yaml`.
- It saves `laser_cal.json`, which also stores the table homography, so a later table recalibration is
  remapped. Recalibrate the laser only if the camera or the laser head moves. Then
  `cp -p laser_cal.json calib_frozen/`.
- Redo without jogging: `$D python3 -m act.calibrate --rig --limits PAN_LO PAN_HI TILT_LO TILT_HI` with the
  printed limits.
- "only N dots seen": the room is too bright or exposure isn't locked (step 2.2); re-jog the corners.

### 3.2 Ruler check (10 spots)

```bash
$D python3 -m act.calibrate --rig --check
```

- Use blue or green sticky notes, not yellow or pink (the camera's red channel saturates on those and the dot
  can vanish). Draw a small cross at each note's centre.
- At each prompt: put one new note down, take your hand fully out of view, press Enter; the laser aims at the
  note. Measure from the dot's centre to the cross with a ruler and type the distance in cm (Enter relights
  the dot; `x` = no dot).
- Spread the 10 spots: 4 near the corners (about 10 cm in), 4 mid-edge, the centre, and where the box sits in
  the shell game. Leave the old notes down.
- It prints the table below (paste it into the log) and saves `data/trials/laser_spots_<time>.json`.

| spot | x cm | y cm | camera cm | open-loop cm | ruler cm | ok (< 3 cm) |
|---|---|---|---|---|---|---|
| 1-10 | | | | | | |

Gate: ruler median < 1.5 cm and max < 3 cm.

### 3.3 Kill switch and the pre-demo laser check

```bash
$D python3 demo_check.py --only 4 8
```

- Check 4: fit median < 1.5 cm, and the aim at the tabletop's centre lands < 3 cm away.
- Check 8: the laser turns on; press the kill switch, then Enter; answer `y` only if the dot went out. Release
  the switch and run `--only 4` again to confirm the laser comes back.
- Every demo_check step has a deadline: a wedged speaker or camera shows `FAIL ... timed out`, not a hang.

### 3.4 Camera replug (once)

`$D bash`, unplug and replug the Brio, then `ls /dev/v4l/by-id/` inside the container must list it again
(dock.sh mounts /dev with a V4L2 cgroup rule). If the shell or tty misbehaves, run with
`ASKROOM_DEV_BIND=0 scripts/dock.sh ...` (the old behaviour) and report it.

Done when: fit median < 1.5 cm, centre < 3 cm, the 10-spot table meets its gate, the kill switch cuts the dot.
Log: fit median/max and point count, centre error, `servo_limits` and `camera_latency_s`, the 10-spot table,
kill switch pass/fail.

## Demo config.local.yaml overrides

Once the table is frozen (step 2.5), the rig runs judging with these in `~/askroom_rig/config.local.yaml`
(on top of what is already there), then restart the app. Each line says what it depends on.

```yaml
detect:
  model: models/askroom-yolo26s-brio.engine      # already live (main); the fine-tuned Brio detector
proposals:
  kind: yoloe                                    # already live (main)
  yoloe: {model: models/yoloe-26s-seg-pf-reduced.engine}
perception_guard:
  voice_recalibrate: false                       # needs fix/perception: a spoken "recalibrate" cannot move the table frame mid-demo
demo:
  hold_notices: true                             # main: no reminders or morning report spoken unasked during judging
actuator: pca9685                                # needs fix/laser-check and step 3.0 (servokit in the image)
# servo_limits: / camera_latency_s:             # paste the block act.calibrate --rig printed (step 3.1)
# tts: {output_device: "<name part>"}           # step 5.1: the speaker, not HDMI
# listen: {mode: wake}                          # only if step 5.2 had more than 1 false trigger
```

## 4. Real trials: 10 shell games and 10 place/hide

Each trial is one clip recorded with `eval.record`. Record with the app **stopped** (it holds the camera).
The question is asked at the end of the clip; `--truth` is what is true at that moment. Trial ids are
unique: shell games 101-110, place/hide 201-210.

```bash
# shell game: keys under the notebook, notebook into the box, box slid across. Truth: the keys are in the box
$D python3 -m eval.record --camera $CAM --trial-id 101 --category inside_box_moved --object keys --truth box \
    --question "where are my keys" --no-preview           # Enter to stop at the end of the game
# ... 102 to 110 the same, varying the hand, speed and where the box ends up

# place/hide: alternate these, 2-3 each
$D python3 -m eval.record --camera $CAM --trial-id 201 --category covered --object keys --truth notebook --no-preview
$D python3 -m eval.record --camera $CAM --trial-id 202 --category inside --object wallet --truth box --no-preview
$D python3 -m eval.record --camera $CAM --trial-id 203 --category moved --object phone --truth visible --no-preview
$D python3 -m eval.record --camera $CAM --trial-id 204 --category carried_away --object glasses --truth left --no-preview
```

Then detect, replay and score:

```bash
for t in trials/*/; do $D python3 -m eval.record --trial-id $(basename $t) --detect-only; done
$D python3 -m eval.replay --trials trials/ --system all
$D python3 -m eval.report --trials trials/ --out report.md
```

Done when: `report.md` has 20 trials under `full`, and the shell-game row (`inside_box_moved`) is 10/10 with
the laser error median < 3 cm. These are the scoreboard's real numbers; never quote the synthetic ones.
Copy `report.md` into the log. A wrong trial: keep it (do not re-record over it) and note what happened.

## 5. Hall noise and the rig's own speaker

1. **Audio devices.** The container's default output is HDMI. Find the speaker and mic, and pin them:

   ```bash
   $D python3 -m voice.tts --devices          # then in config.local.yaml:  tts: {output_device: "<name part>"}
   arecord -l                                  # the mic card; listen.input_device if it isn't the default
   ```

2. **10 minutes of hall noise** at the venue, with people talking but nobody talking to the rig:

   ```bash
   arecord -D plughw:<card>,0 -f S16_LE -r 16000 -c 1 -d 600 ~/hall.wav
   cp ~/hall.wav tests/stt_audio/hall.wav                          # git-ignored; never commit it
   $D python3 scripts/overheard_test.py tests/stt_audio/hall.wav
   ```

   Done when: 0 or 1 false triggers. More than 1: run it again with `--mode wake`; if that is 0-1, set
   `listen: {mode: wake}` in `config.local.yaml` (the judge then says "room, where are my keys").
3. **Own-speaker echo test.** Start the app with voice (step 7's command), stand at the judge's spot and ask
   5 questions ("where are my keys", "where is my wallet", "what did I move", "what's on the table", "where is
   the notebook"), waiting for each answer.

   ```bash
   python3 -c "import sqlite3,time;[print(time.strftime('%H:%M:%S',time.localtime(t)),q,'->',a) for t,q,a in sqlite3.connect('data/events.db').execute('select t,text,answer from questions order by t desc limit 8')]"
   ```

   Done when: exactly 5 new rows, one per question you asked, none whose text is the rig's own answer.

## 6. Offline rehearsal (E3)

Over the USB link (step 1), not Wi-Fi: turning Wi-Fi off drops a Wi-Fi SSH session.

```bash
ssh guru@192.168.55.1
cd ~/askroom_rig && nmcli radio wifi off        # and unplug Ethernet if any
# start the app (step 7); the dashboard shows "offline" within a few seconds and the voice is Piper
```

Play 3 shell games and ask each question out loud, including "show me my keys". Done when: every answer is
correct, spoken in the Piper voice, with the laser on the box, and nothing waits on the network (no answer
takes > 2 s). Then `nmcli radio wifi on`.

## 7. Demo check (E5), then 5 clean games in a row (E2)

Start the app for the demo:

```bash
$D python3 main.py --camera $CAM
# dashboard: http://10.90.84.178:8000
```

Before every judge session, with the app **stopped**:

```bash
$D python3 demo_check.py --camera $CAM          # every line green; it asks you to confirm the tone and the kill switch
```

Then five shell games in a row, each reset in under 1 minute (props back home, say "room, reset" if the
board is off). Done when: 5/5 correct, laser on the box each time, resets < 1 min. A failure restarts the
count; write down what went wrong.

## Log (send this back)

| Step | Measure | Value |
|---|---|---|
| 1 | `free -m` available / clock right | |
| 2 | camera_setup gain used; tracked area (cm) | |
| 2b | Brio clips recorded (ids) | |
| 3 | laser fit error cm / centre error cm / 10 ruler spots (cm each) / kill switch ok | |
| 4 | shell games correct /10; place-hide correct /10; laser error median cm | |
| 5 | hall noise false triggers (always / wake); echo test rows = 5? | |
| 6 | offline games correct /3 | |
| 7 | demo_check all green?; clean games in a row | |

## Restore after the demo

What step 1 stopped on Sat 26 Sep, and how to bring it back (only when its owner wants it back):

```bash
# the room app (room mode, no voice), launched from its own scratch copy; holds the Brio and ~2.8 GB
cd ~/askroom_room && nohup scripts/dock.sh python3 -u main.py \
    --camera /dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0 --no-voice --port 8080 \
    > data/room/app.log 2>&1 < /dev/null &

# baby-tau (not ours: ollama, whisper, piper, python); stopped with `docker compose stop`, nothing removed,
# restart policy unless-stopped left as it was
cd ~/git/autonomous-intelligence/baby-tau && docker compose start
```
