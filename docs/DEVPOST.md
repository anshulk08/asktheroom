# Ask the Room: Devpost draft

Draft for the HackGT 13 submission (due before Sun 8 AM). Every number here has a source, named in
brackets for the editor and to be deleted before submitting. `[TBD: …]` marks a number from tonight's
trials: fill it from the dashboard scoreboard (`GET /scoreboard`, real `scripts/room_trials.py` runs) or
leave the sentence out. Never quote the synthetic eval (255/300) as accuracy.

## Inspiration

Losing everyday things costs everyone time, and it costs much more for someone with memory loss, who
also loses the confidence that they will find it. Cameras and vision models can say what is in view right
now. They cannot say where your wallet went after you carried it across the room and walked off. We
wanted to ask the room itself, out loud, and get an honest answer.

## What it does

A camera high in the corner of a living room watches the coffee table and the places things usually end
up: the couch, the side table, the kitchen counter. Pick up an object, put it somewhere else, walk away,
and ask "where's my wallet?". The rig answers in a sentence, "Your wallet, I think, is on the kitchen
counter. It appeared there 20 seconds ago.", and the dashboard shows the whole room with the zones and
where things are. [TBD: "and the phone shows the room with the spot marked", only if spec 0010 P1-2 lands.]

**The 60-second demo** (spec 0010 §1):

1. Pick an object from the coffee table (say the wallet) and put it somewhere in the room: the couch, the
   side table, the kitchen counter.
2. Walk away and ask, out loud or on the phone: "where's my wallet?"
3. The rig answers in a few seconds: "Your wallet, I think, is on the kitchen counter. It appeared there
   20 seconds ago." The phone shows the room with the spot marked.
4. Move it again and ask again. Then bring it back to the table: "Your wallet is on the table."
5. Stretch: hide it under the notebook on the table.

[TBD: reset between visitors, "room, reset" in under 30 s, only if spec 0010 P0-4 lands; today it is an
app restart of about 40 s.]

**The honesty line.** The rig never asserts who an object is. It says "I think", names the place and how
long ago it saw it arrive, and would rather say it lost track than guess. Pill bottles get neutral
wording on every path: the rig never says or implies that medication was taken.

## How we built it

- **Camera.** A Logitech Brio high in the room's corner at 1080p. The coffee table's region is cut out
  and resized for the table pipeline; the rest of the room is covered by zones drawn as polygons
  (`python -m core.room --zone couch --say "the couch" --poly …`).
- **Finding objects without knowing them.** A detector fine-tuned on the overhead table view mislabels
  things from a corner, so objects are found class-agnostically: a prompt-free YOLOE model proposes a box
  for anything object-like, on the table view and, round-robin, on a full-resolution crop of each zone.
- **Naming by asking a closed question.** Open naming of a 40-pixel remote failed ("phone", "eyeglasses
  case"). What works is set-of-marks prompting: the candidate is boxed in red inside a patch of context
  and Grok is asked "is this the remote control?", against the objects that just left the table. A match
  needs confidence of at least 0.7 and Grok's own description to fit the name.
- **A deterministic world model.** Everything the rig says comes from explicit rules over tracked
  events, each explainable in one sentence: an object leaving the table, a matching object arriving in a
  zone, a hand picking it up, a cover laid over it. Answers are filled-in templates, so they are short,
  consistent and never made up; Grok is used for naming, visual questions and open questions only.
- **Voice.** whisper.cpp for speech recognition with Silero VAD to find where a question starts and
  stops, a rule parser (Grok when it can't read a question), and speech out through ElevenLabs online or
  Piper offline. The iPhone app asks over Bluetooth and reads answers aloud.
- **Hardware.** An NVIDIA Jetson Orin Nano (8 GB) runs everything but the cloud calls, in an Ultralytics
  JetPack 6 container with TensorRT engines.
- **Tested like a product.** A spoken trial driver (`scripts/room_trials.py`) talks a person through runs
  and scores the rig's answers; it found six bugs in 40 minutes that the unit tests didn't. Each rig bug
  got a unit test the same hour. [TBD: final test count, e.g. "1,9xx tests" from `pytest -q` on the
  submitted build; 1,880 at `afca68c`, spec 0010 §2.]

## What's real (measured on the rig)

- Room handoffs with the remote: **5 of 5** (couch 8–10 s, side table 24 s, counter 20 s) and **5 of 5**
  returns to the table, run by the spoken trial driver. [spec 0010 §2, Sat 7:45 PM]
- Tonight's trials across the demo objects: [TBD: "N/N handoffs, M/M table returns, median X s" per object
  and zone, from the dashboard scoreboard.]
- Speech recognition: 22 of 22 test questions right, 139 ms per question on the Jetson. [spec 0010 §5
  P0-1]
- The Jetson runs the vision loop at 10–13 frames per second with room memory on. [spec 0010 §2]
- [TBD: spoken end-to-end latency, "where's my wallet" to answer, if measured with the room mic and
  speaker (target under 4 s, spec 0010 P0-1).]

Laptop and synthetic numbers are not accuracy and are not quoted here.

## Challenges we ran into

- **A new viewpoint breaks a fine-tuned detector.** Moving the camera from overhead to a corner made
  the prop detector read the remote as a wallet. We switched to class-agnostic proposals plus targeted
  naming, which work from any viewpoint without retraining.
- **Vision models lean towards "yes".** Asked "is this the remote?", Grok agreed too easily; a match now
  needs a confidence floor and its own description has to fit.
- **Static clutter.** Stove vents and counter items are proposed over and over. A handoff needs the
  pixels to have changed where the object appeared, and clutter nobody can name blocks nothing.
- **Timing.** A departure is only certain about 2 s after the object left, so a zone next to the table
  could see it arrive before the table knew it had gone; departures have to be dated at the last table
  evidence. [TBD: confirm this is in the submitted build, spec 0010 §3 lesson 6.]
- **Far zones are slow**: handing off to the side table and counter took 20–24 s against a judge asking
  within about 5 s. [TBD: after tonight's speed work.]

## Accomplishments that we're proud of

- The room answers where things are across a real room, from one camera, with honest hedging.
- Every rig bug became a unit test within the hour.
- [TBD: the scoreboard line, "N/N room handoffs today", once tonight's trials are uploaded.]

## What we learned

Mark the candidate and ask a closed question. Test on the rig with a script, not by hand. Say "I think".

## What's next

Pointing a laser at the spot across the room, which waits on a laser module with a documented safety
class. Hidden objects on the table from the corner view (keys under the notebook), which needs the prop
detector retrained on this view. More than one instance per object name.

## Privacy

Adapted for the room build from the README's statement, which still describes the overhead table camera
and needs the same update before submission.

Audio stays on the device and is never written to disk, and speech not meant for the rig is dropped
unlogged. Accepted questions (text) and event snapshots are kept, snapshots and saved frames for 24 hours.
The camera sees the room from a corner, so frames can show the people in it; frames stay on the rig
except as follows, and only while online. When a new object is confirmed in a zone or on the table, one
close-up crop of it is sent to Grok (xAI) to name it, at most 20 a minute and only while a handoff is
possible; a name you teach always wins. The text of a question the rules can't read, or an open question
with a compact world state, goes to Grok; a question about what the camera sees sends the current frame
(and for "earlier" questions up to 6 saved frames). Answer text goes to ElevenLabs for the voice; SMS
answers go through Twilio. The iPhone app reaches the rig only over Bluetooth and turns dictated questions
into text on the phone; with "Read answers aloud" on and the Grok or rig voice chosen, the phone sends
each answer's text to xAI or ElevenLabs with a key kept in the phone's Keychain (the iPhone voice sends
nothing). Offline, nothing leaves the rig: the table still answers from rules, and room naming waits.

## Built with

Python, NVIDIA Jetson Orin Nano (JetPack 6, TensorRT), OpenCV, FastAPI, Swift (iPhone app), xAI Grok.

## Open-source credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0): detection, the fine-tuned
  prop detector, the JetPack 6 container image.
- [YOLOE](https://github.com/THU-MIG/yoloe) (prompt-free, via Ultralytics): class-agnostic object
  proposals.
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World): zero-shot detection and auto-labelling for the
  fine-tune.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with OpenAI's Whisper `base.en` model: speech
  recognition.
- [Silero VAD](https://github.com/snakers4/silero-vad): voice activity detection.
- [Piper](https://github.com/rhasspy/piper) and its `en_US-lessac-medium` voice: offline speech.
- [MobileCLIP2](https://github.com/apple/ml-mobileclip) (Apple): image embeddings for the visual memory.
- [OpenCV](https://opencv.org) with ArUco markers: table calibration and image work.
- [ONNX Runtime](https://onnxruntime.ai), [NumPy](https://numpy.org), [FastAPI](https://fastapi.tiangolo.com)
  and Uvicorn.
- [vis-network](https://github.com/visjs/vis-network): the dashboard's "what's where" graph.
- [Atkinson Hyperlegible Next](https://www.brailleinstitute.org/freefont/) (Braille Institute, OFL): the
  dashboard font.
- [EgoHands](http://vision.soic.indiana.edu/projects/egohands/) (Bambach et al., ICCV 2015): public hand
  images for fine-tuning the hand class (`scripts/finetune/public_hands.py`). [TBD: keep only if the
  shipped detector was trained with them.]
- Optional, off in the demo: [DINOv2](https://github.com/facebookresearch/dinov2) re-identification
  embeddings, and a local [Qwen3](https://github.com/QwenLM/Qwen3) through
  [llama.cpp](https://github.com/ggml-org/llama.cpp) (not installed on the Jetson).

Services: xAI Grok (naming, visual and open questions), ElevenLabs (voice), Twilio (SMS).
