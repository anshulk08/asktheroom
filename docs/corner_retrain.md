# Corner-view detector retrain: the 10:15 PM checklist (spec 0010 P1-1)

Goal: the prop detector recognises **notebook, box, keys and hands** on the coffee table **as the room build
sees it** (Brio high in the corner, zoom 100, 1920x1080, table view = `table_view_rect` cut to 1280x720), so
the table's hide rules (keys under the notebook, notebook into the box) work in the room demo.
Midnight go/no-go: notebook, box and keys at confidence ≥ 0.6 in ≥ 80% of the demo layout's frames.

Code: branch `room/detector` (off `room-memory-m0`). New here: `capture.py --room VIEW` (captures exactly
what the room build's detector sees) and `--scene`, `merge_sets.py`, `conf_sweep --images … --require`.
Mac tools are ready: `~/asktheroom/train-venv` (ultralytics 8.4.163, torch 2.14, MPS), the COCO start
weights `~/asktheroom/ft/yolo26s.pt`, the EgoHands frames `~/asktheroom/ft/hands_public` (pubhand, class 8 = hand).

Who: **Anshul + the teammate at the rig** (steps A–C, the only camera time), then **the Mac** (D–F),
then **the Jetson** once more for the engine (G), with asktheroom-60's lock.

## A. Before the slot (5 min, no camera)

1. A scratch copy of `room/detector` on the Jetson, next to the live build, never in `~/askroom`:
   `~/askroom_ft` (git archive of `room/detector`), with `~/askroom_room`'s `config.local.yaml` and
   `room_zones.json` copied in (they carry `room_memory.enabled`, `table_view_rect`, the zones).
2. Props on hand: keys, notebook, box (the demo ones), remote, wallet, phone, glasses case, pill bottle;
   4 distractors, one word each, none of our classes: `mug charger airpods deodorant` (or similar).
3. Lamp on, the room lit as for the demo. Nobody walking behind the coffee table during captures.

## B. Capture (the 10:15–11:00 slot; ~30 min of it)

The room app holds the camera, so it is stopped for the capture (Anshul's call; it's his slot):
`docker stop upbeat_curie`. The Brio stays at zoom 100 (don't run `camera_setup.sh` with other values).
All from `~/askroom_ft` on the Jetson host, `CAM=/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0`:

```
PY=~/askroom/.venv/bin/python          # the host venv (cv2 + numpy + yaml); only its python is used

# B1. Table view, shell-game props: 8 poses each + 60 s of hands (~12 min)
$PY scripts/finetune/capture.py --room table --device $CAM --data data/ft-corner-table \
    --only keys notebook box hand --poses 8 --hand-seconds 60

# B2. Table view, the other props (4 poses) + distractors (3 poses) (~12 min)
$PY scripts/finetune/capture.py --room table --device $CAM --data data/ft-corner-table \
    --only remote wallet phone glasses pill_bottle mug charger airpods deodorant \
    --distractors mug charger airpods deodorant --poses 4 --distractor-poses 3

# B3. The go/no-go frames: the demo layout, hands in now and then, 60 s (~2 min). Not trained on.
$PY scripts/finetune/capture.py --room table --device $CAM --data data/ft-corner-table --scene 60
```

- Each prompt says where to put the object ("in the middle", "near a corner", "rotated 90°", ...); place
  it **alone** on the empty coffee table, take hands away, press Enter. A retake prints why (arm in view,
  object split in two, touching the frame edge). The box: once upright, once on its side, lid on and off.
  The notebook: closed, open, and at an angle. Keys: flat, bunched, near the edge.
- B1's hands: one hand, then both, slowly over the empty table from every side, open, fist, pointing,
  touching the table; sleeves as the judges will wear them.
- If people or the floor show changes, add `--roi x1,y1,x2,y2` (the tabletop, table-view px of the 1280x720 frame).
- Look at `data/ft-corner-table/qa_capture.jpg`: every box on its object, the empty table and the
  distractors unboxed. Redo one object with `--only <name>` (same command as its pass).
- B4 (only if 10+ min remain; optional): one zone, for later, not for tonight's go/no-go:
  `--room couch --data data/ft-corner-couch --only remote wallet glasses pill_bottle --poses 3 --no-hands`.
- Restart the room app: `docker start upbeat_curie`.

Expected: ~24 + 20 + 12 real labelled frames, 4 empty-table negatives, ~100-150 hand frames, ~200 scene frames.

## C. Data to the Mac (1 min)

`rsync -a guru@10.90.84.178:askroom_ft/data/ft-corner-table/ ~/asktheroom/ft/corner-table/`
(and `ft-corner-couch` if B4 ran). Open `~/asktheroom/ft/corner-table/qa_capture.jpg` once more.

## D. Synthesize and merge on the Mac (1 min)

From a checkout of `room/detector` (e.g. `~/asktheroom/wt-room-det`), `T=~/asktheroom/train-venv/bin/python`:

```
$T scripts/finetune/synthesize.py --data ~/asktheroom/ft/corner-table --n 600      # pastes on this view's backgrounds; check qa_synth.jpg
$T scripts/finetune/merge_sets.py --out ~/asktheroom/ft/corner-all \
    table=$HOME/asktheroom/ft/corner-table pubhand=$HOME/asktheroom/ft/hands_public
```

## E. Train on the Mac (~45 min for 30 epochs)

```
PYTORCH_ENABLE_MPS_FALLBACK=1 $T scripts/finetune/train.py --data ~/asktheroom/ft/corner-all \
    --model ~/asktheroom/ft/yolo26s.pt --device mps --batch 8 --epochs 30 --val-trials cap3 \
    --name askroom-yolo26s-corner
```

Measured on this Mac (M5 Pro, 24 GB): 2.1 it/s at batch 8, 640 px, 5.4 GB GPU memory. ~1,300 images
(600 synthetic + ~700 EgoHands + the real captures) is ~160 iterations, ~75 s per epoch: 30 epochs ≈ 40 min
plus validation. Short on time: `--epochs 20` (≈ 27 min). It prints the `best.pt` path.

## F. Midnight go/no-go on the Mac (1 min, no Jetson)

```
B=<the best.pt path train.py printed>
$T -m eval.conf_sweep --images ~/asktheroom/ft/corner-table/scene --detect-model $B \
    --require notebook box keys --require-conf 0.6 --min-rate 0.8 --json ~/asktheroom/ft/gate.json
$T -m eval.conf_sweep --images ~/asktheroom/ft/corner-table/scene --detect-model $B \
    --require hand --require-conf 0.35 --min-rate 0.4          # hands are in view about half the scene
```

**GO** (exit 0): notebook, box and keys reach 0.6 in ≥ 80% of the demo layout's frames. Then G. The table
prints every class's rate at every cut: pick each class's `conf_threshold` as the highest cut that keeps
it in ≥ 90% of the frames where it lies (and above the rate the distractors reach).
**NO-GO**: drop hiding from the script (spec 0010 decision point); the room handoff is the show.

## G. Engine on the Jetson (~10 min, asktheroom-60's lock)

TensorRT's build takes about 2-2.5 GB of RAM on top of what runs (an estimate: measure with `free -m`).
The live room app needs 1.5 GB free, so build with the app stopped or with `free -m` showing ≥ 4 GB available.
```
scp $B guru@10.90.84.178:askroom_room/models/askroom-yolo26s-corner.pt
cd ~/askroom_room && scripts/dock.sh yolo export model=models/askroom-yolo26s-corner.pt format=engine half=True imgsz=640
```
Then in `~/askroom_room/config.local.yaml`: `detect: {model: models/askroom-yolo26s-corner.engine}` and the
per-class `conf_threshold` from F (drop the old `wallet: 0.6` / `phone: 0.6` unless F shows the need).
Keep `room_memory.room_prop_conf` as the rig has it: the prop labels stay **off in the zones** (YOLOE + Grok
does the room); the new model's labels are for the **table view**. Restart the app; its log's
`detector weights …` line must list the 9 classes with no "has no class for" warning, and `/state`
carries it (`state.perception.model`). Then the shell game on the coffee table, 3/3 (spec 0010 P1-1).

## Why `remote` stays a class

The room build turned the prop labels off because the overhead model read the remote as a wallet. Dropping
`remote` would not stop that: a detector calls anything it wasn't taught by its nearest class (the first
model scored an AirPods case 0.53 'phone' before distractors were added). Training the remote, wallet,
phone and glasses from this view, with their own poses (B2), is what teaches them apart. The class list
must stay exactly the config objects plus `hand` anyway (`core/detect.py` maps model classes to objects by
name). What tonight's model is **not** for: the room zones (far, small, other backgrounds), where the
YOLOE + Grok path is the proven one (5/5); keep zone prop labels off unless a zone set (B4) is captured,
trained and swept (`conf_sweep --images` on its frames) with no cross-labels.
