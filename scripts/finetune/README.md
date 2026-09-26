# Fine-tuning the detector (spec P7)

## Fast path (no hand labelling)

Labels come from background subtraction: each object lies **alone** on the empty table, so what changed
is that object (`bglabel.py`). YOLO-World isn't used (it scored the wallet 0.00 and glasses <0.2 from
overhead). `synthesize.py` then pastes the cutouts into multi-object, occluded, hand-over-object scenes
with exact boxes. Validation is real captures only: group `cap3` (poses 4 and 8 of every object, a
quarter of the hand frames), which is never pasted. Synthetic `synth_*` images always train.

**Jetson** (`ssh guru@192.168.55.1`, `cd ~/askroom`). The app holds the camera, so stop it first:
```
docker ps --filter ancestor=askroom:latest --format '{{.ID}} {{.Command}}' | grep main.py | cut -d' ' -f1 | xargs -r docker stop
scripts/camera_setup.sh                          # lamp on: labels need a lit, still table
.venv/bin/python scripts/finetune/capture.py     # ~15 min: empty table, 8 objects x 8 poses, 40 s of hands
nohup scripts/dock.sh python3 -u main.py --no-voice > main.log 2>&1 &    # app back up (old model)
```
Capture prompts for each pose, waits until the view has been still for 0.7 s (hand gone), and retakes
the shot with a reason if the change is missing, split in pieces, too big/small, or touches the frame
edge (arm in view). Redo objects with `--only glasses keys`. Look at `data/finetune/qa_capture.jpg`.

**Mac** (`cd ~/askroom`):
```
rsync -a guru@192.168.55.1:askroom/data/finetune/ data/finetune/           # ~1 min
open data/finetune/qa_capture.jpg                                          # every box right?
.venv/bin/python scripts/finetune/synthesize.py --n 400                    # ~15 s; check qa_synth.jpg
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/finetune/train.py \
    --model models/yolo26s.pt --device mps --batch 8 --epochs 30 --val-trials cap3 --name askroom-yolo26s
cp <the "best weights:" path it prints> models/askroom-yolo26s.pt          # usually ~/runs/detect/runs/askroom/askroom-yolo26s/weights/best.pt
scp models/askroom-yolo26s.pt guru@192.168.55.1:askroom/models/
```
Training on the M-series Mac runs ~1.4 s/iteration at 640 px, batch 8 (batch 16 swaps on 16 GB), so
30 epochs of ~550 images is roughly 50-60 min; a cloud GPU is much faster. If YOLO26 gives any trouble,
use `--model yolo11s.pt`.

**Jetson** again:
```
scripts/dock.sh yolo export model=models/askroom-yolo26s.pt format=engine half=True imgsz=640   # ~10 min
# config.yaml -> detect: model: models/askroom-yolo26s.engine
docker ps --filter ancestor=askroom:latest --format '{{.ID}} {{.Command}}' | grep main.py | cut -d' ' -f1 | xargs -r docker stop
nohup scripts/dock.sh python3 -u main.py --no-voice > main.log 2>&1 &
```
**P6 check** on a recorded clip (`eval/record.py`), inside the container for real engine speed:
```
scripts/dock.sh python3 -m eval.replay_video --video trials/<id>/video.mp4 --model models/askroom-yolo26s.engine
```
It reports fps, per-object detection rate and every disappearance event (PICKED_UP, COVERED,
PUT_INSIDE, EXITED_VIEW, LOST_TRACK) of an object no hand touched; pass = none at >= 10 fps.

## Slow path (label trial-video frames by hand)

Classes are exactly the config objects in order, then `hand`:
`keys pill_bottle wallet glasses phone remote box notebook hand`. `core/detect.py` maps a model's
class names to objects by name, so don't rename them.

1. **Frames:** `python scripts/finetune/extract.py` pulls 400–600 varied frames from
   `trials/*/video.mp4` into `data/finetune/images/` as `<trial>_<frame>.jpg`, dropping near-duplicates.
2. **Pre-labels:** `python scripts/finetune/autolabel.py` runs `models/yolov8s-worldv2-askroom.pt`
   (YOLO-World with the config prompts) and writes `data/finetune/labels/*.txt` plus `data.yaml`.
   Labels that already exist are kept, so fixed labels survive a re-run (`--overwrite` replaces them).
3. **Fix labels.** Keys and pill bottles need the most work. The quickest route:
   - **Roboflow:** create an Object Detection project and drag in `images/`, `labels/` and
     `data.yaml` together. It reads YOLO format directly. Fix the boxes, then export as "YOLOv8"
     and copy the `.txt` files back into `labels/`.
   - **Label Studio:** `pip install label-studio label-studio-converter`, then
     `label-studio-converter import yolo -i data/finetune -o tasks.json --image-root-url /data/local-files/?d=images`.
     Import `tasks.json`, fix the boxes, and export as YOLO.
4. **Train** on a GPU (Colab is fine): `python scripts/finetune/train.py --data data/finetune --device 0`.
   YOLO11s, 640 px, 80 epochs by default. About 20% of the *videos* are held out for validation
   (`--val-trials 3 11` picks them). `--split-only` writes `train.txt`, `val.txt` and
   `dataset.yaml` without training.
5. **Engine (Jetson, when told it's free):** copy `best.pt` to `models/askroom-yolo11s.pt`, then
   build the engine in the container. `core.detect --export` is YOLO-World only (it calls `set_classes`),
   so use:
   ```
   scripts/dock.sh yolo export model=models/askroom-yolo11s.pt format=engine half=True imgsz=640
   ```
   Then set `detect.model: models/askroom-yolo11s.engine` in `config.yaml`.

**Done when:** the new model passes P6 on a held-out video, with no false "disappeared" events
at 10 fps or more.
