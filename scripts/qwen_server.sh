#!/usr/bin/env bash
# Serve the local Qwen for voice/understand.py (question -> intent) and voice/local_llm.py (open
# questions) with llama.cpp's llama-server (OpenAI-compatible, port 8081). Run it next to main.py;
# while it's down, questions use the rule parser and open questions get the fallback sentence.
#
#   scripts/qwen_server.sh                       # Qwen3 1.7B Q4_K_M (~1.1 GB), all layers on the GPU
#   QWEN_MODEL=qwen2.5-1.5b scripts/qwen_server.sh   # the previous default: same intent score on
#                                                # tests/understand_eval.json, slower, worse open answers
#   QWEN_HOST=0.0.0.0 scripts/qwen_server.sh     # reachable from the laptop (n8n's health check)
#
# Compare models: start each on its own port and run scripts/eval_understand.py --url ... against it.
# (Qwen2.5 0.5B read loose questions no better than the rules alone; Qwen3 4B only if tegrastats
# shows room next to YOLO and whisper.)
#
# llama-server: `brew install llama.cpp` on a Mac. On the Jetson, build llama.cpp once, and ask the
# team first: it's a big build, and together with the TensorRT engine build it can run out of memory.
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build -j2 --target llama-server
set -euo pipefail
cd "$(dirname "$0")/.."
case "${QWEN_MODEL:-qwen3-1.7b}" in
  qwen3-1.7b)   REPO=unsloth/Qwen3-1.7B-GGUF;          FILE=Qwen3-1.7B-Q4_K_M.gguf ;;
  qwen2.5-1.5b) REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF;  FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf ;;
  *) echo "QWEN_MODEL: qwen3-1.7b or qwen2.5-1.5b" >&2; exit 2 ;;
esac
MODEL="models/qwen/$FILE"
if [ ! -f "$MODEL" ]; then
  mkdir -p models/qwen
  echo "downloading $REPO/$FILE"
  curl -fL -o "$MODEL.part" "https://huggingface.co/$REPO/resolve/main/$FILE" && mv "$MODEL.part" "$MODEL"
fi
BIN="${LLAMA_SERVER:-$(command -v llama-server || echo ../llama.cpp/build/bin/llama-server)}"
# One slot. 2048 tokens: the open-question prompt carries the world state and recent events.
exec "$BIN" -m "$MODEL" --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-8081}" \
  -c 2048 --parallel 1 -ngl 99 --no-webui
