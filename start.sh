#!/usr/bin/env bash
set -euo pipefail

mkdir -p /comfyui/input /comfyui/output /runpod-volume/models/loras

export COMFY_HOST="${COMFY_HOST:-127.0.0.1:8188}"
export PYTHONUNBUFFERED=1

python -u /comfyui/main.py \
  --disable-auto-launch \
  --disable-metadata \
  --listen 127.0.0.1 \
  --port 8188 \
  --extra-model-paths-config /app/extra_model_paths.yaml \
  ${COMFY_PERFORMANCE_ARGS:---fast fp16_accumulation} &
COMFY_PID=$!

cleanup() {
  kill "${COMFY_PID}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

python -u /app/handler.py
