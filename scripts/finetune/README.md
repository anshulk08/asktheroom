# Fine-tuning the detector (spec P7)

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
