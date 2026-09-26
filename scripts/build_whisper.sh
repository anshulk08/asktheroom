#!/usr/bin/env bash
# Build whisper.cpp's whisper-server (plus whisper-cli and whisper-bench for benchmarking) for voice/stt.py.
#
#   Mac:    scripts/build_whisper.sh                 -> third_party/whisper.cpp/build-mac (Metal)
#   Jetson: scripts/dock.sh scripts/build_whisper.sh -> third_party/whisper.cpp/build   (CUDA, sm_87)
#
# Build on the Jetson INSIDE the askroom container so the binary links against the runtime's CUDA.
# The container has nvcc but no cmake, and the Jetson has no internet, so cmake comes from an aarch64
# wheel downloaded on the Mac into third_party/wheels (pip download cmake ninja --platform
# manylinux2014_aarch64 --only-binary=:all: --python-version 3.10 -d third_party/wheels) and is
# installed into third_party/tools on each run (the container is --rm).
# -j2 on purpose: -j6 runs the 8 GB Orin Nano out of memory while nvcc compiles the CUDA kernels.
# Measured 2026-09-25 (whisper.cpp d09f61a, with the app running): 40 min on the Jetson, 32 s on an M5 Mac.
# Static libs so the binary does not depend on the build tree's .so files.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=third_party/whisper.cpp
JOBS="${JOBS:-2}"
TARGETS=(--target whisper-server whisper-cli whisper-bench)

if [ "$(uname -s)" = "Darwin" ]; then
  cmake -S "$SRC" -B "$SRC/build-mac" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=ON -DWHISPER_BUILD_TESTS=OFF
  cmake --build "$SRC/build-mac" -j "${JOBS}" --config Release "${TARGETS[@]}"
  exit 0
fi

if ! command -v cmake >/dev/null; then
  TOOLS="$PWD/third_party/tools"
  if [ ! -x "$TOOLS/bin/cmake" ]; then
    python3 -m pip install -q --no-index --find-links third_party/wheels --target "$TOOLS" cmake ninja
  fi
  export PATH="$TOOLS/bin:$PATH" PYTHONPATH="$TOOLS${PYTHONPATH:+:$PYTHONPATH}"
fi
export PATH="/usr/local/cuda/bin:$PATH"
cmake --version | head -1
nvcc --version | tail -1

cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DWHISPER_BUILD_TESTS=OFF
time cmake --build "$SRC/build" -j "${JOBS}" --config Release "${TARGETS[@]}"
ls -la "$SRC/build/bin"
