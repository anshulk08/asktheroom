#!/usr/bin/env bash
# Serve Qwen2.5 1.5B Instruct for voice/understand.py with llama.cpp's llama-server (OpenAI-compatible,
# port 8081). Run it next to main.py; main.py falls back to the rule parser while it's down.
#
#   scripts/qwen_server.sh                     # models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf, all layers on the GPU
#   QWEN_SIZE=0.5b scripts/qwen_server.sh      # 0.5B (~0.4 GB, not ~1.1 GB) only if memory forces it: it read
#                                              # loose questions no better than the rules alone
#   QWEN_HOST=0.0.0.0 scripts/qwen_server.sh   # reachable from the laptop (n8n's health check)
#
# llama-server: `brew install llama.cpp` on a Mac. On the Jetson, build llama.cpp once, and ask the
# team first: it's a big build, and together with the TensorRT engine build it can run out of memory.
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build -j2 --target llama-server
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${QWEN_SIZE:-1.5b}"
FILE="qwen2.5-${SIZE}-instruct-q4_k_m.gguf"
MODEL="models/qwen/$FILE"
if [ ! -f "$MODEL" ]; then
  mkdir -p models/qwen
  REPO="Qwen/Qwen2.5-$(echo "$SIZE" | tr b B)-Instruct-GGUF"
  echo "downloading $REPO/$FILE"
  curl -fL -o "$MODEL.part" "https://huggingface.co/$REPO/resolve/main/$FILE" && mv "$MODEL.part" "$MODEL"
fi
BIN="${LLAMA_SERVER:-$(command -v llama-server || echo ../llama.cpp/build/bin/llama-server)}"
# One slot, small context: questions are one sentence and the system prompt is ~350 tokens.
exec "$BIN" -m "$MODEL" --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-8081}" \
  -c 1024 --parallel 1 -ngl 99 --no-webui
