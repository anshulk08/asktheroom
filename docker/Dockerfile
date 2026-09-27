# Ask the Room runtime: the Ultralytics JetPack 6 image (CUDA, TensorRT 10.7, torch) + the app's packages,
# installed offline from wheels downloaded on the Mac (no numpy/onnxruntime wheels: the image's own are kept;
# piper-tts goes in with --no-deps because the image ships onnxruntime as onnxruntime-gpu). Build on the Jetson
# under a new tag, so askroom:latest stays as the rollback, and run it by name:
#   docker build -t askroom:demo docker/
#   ASKROOM_IMAGE=askroom:demo scripts/dock.sh python3 main.py      # promote: docker tag askroom:demo askroom:latest
# The servo driver (actuator: pca9685) is adafruit-circuitpython-servokit; with Blinka's deps and Jetson.GPIO
# every wheel is pure Python or has a cp310 aarch64 build, fetched on the Mac with:
#   pip3 download adafruit-circuitpython-servokit==1.3.24 Jetson.GPIO==2.1.11 --platform manylinux2014_aarch64 \
#     --platform linux_aarch64 --platform any --python-version 3.10 --implementation cp --only-binary=:all: -d docker/wheels
# If it is missing anyway, main.py logs "LASER DISABLED" and answers by voice only (act.actuator.make_actuator_or_fake).
FROM ultralytics/ultralytics:latest-jetson-jetpack6
COPY libs/ /usr/lib/aarch64-linux-gnu/
COPY wheels/ /tmp/wheels/
RUN ldconfig \
 && pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/pins.txt \
 && pip install --no-index --find-links /tmp/wheels --no-build-isolation -c /tmp/pins.txt \
      fastapi uvicorn httpx python-multipart websockets twilio sounddevice pyserial evdev \
      adafruit-circuitpython-servokit Jetson.GPIO \
 && pip install --no-index --find-links /tmp/wheels --no-deps piper-tts \
 && rm -rf /tmp/wheels \
 && python3 -c "import fastapi, uvicorn, sounddevice, piper, evdev, serial, adafruit_pca9685; print('askroom deps ok')"
# espeak-ng: voice/tts.py's last-resort voice when Piper can't load. apt needs network (the phone hotspot) at
# build time; an offline build still succeeds, without it.
RUN (apt-get update && apt-get install -y --no-install-recommends espeak-ng && rm -rf /var/lib/apt/lists/* \
     && espeak-ng --version) \
 || echo "WARNING: espeak-ng not installed (no network during the build?): no fallback voice if Piper fails"
