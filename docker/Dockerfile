# Ask the Room runtime: the Ultralytics JetPack 6 image (CUDA, TensorRT 10.7, torch) + the app's packages,
# installed offline from wheels downloaded on the Mac (no numpy/onnxruntime wheels: the image's own are kept;
# adafruit-circuitpython-servokit left out until the servo hardware is chosen (Blinka's deps need more
# offline wheels); piper-tts goes in with --no-deps because the image ships onnxruntime as onnxruntime-gpu). Build on the Jetson:
#   docker build -t askroom:latest docker/
FROM ultralytics/ultralytics:latest-jetson-jetpack6
COPY libs/ /usr/lib/aarch64-linux-gnu/
COPY wheels/ /tmp/wheels/
RUN ldconfig \
 && pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/pins.txt \
 && pip install --no-index --find-links /tmp/wheels --no-build-isolation -c /tmp/pins.txt \
      fastapi uvicorn httpx python-multipart websockets openai twilio sounddevice pyserial evdev \
 && pip install --no-index --find-links /tmp/wheels --no-deps piper-tts \
 && rm -rf /tmp/wheels \
 && python3 -c "import fastapi, uvicorn, sounddevice, piper, evdev, serial; print('askroom deps ok')"
