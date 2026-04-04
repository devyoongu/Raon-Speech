#!/bin/bash
# Run the RAON Gradio demo with a local model.
#
# Edit the preset variables below, then run:
#   bash demo/run_gradio_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="python3"
MODEL_PATH="KRAFTON/Raon-Speech-9B"
CONFIG=""  # default: config/infer.yaml
DEVICE="cuda"
DTYPE="bfloat16"
HF_TOKEN=""
HOST="0.0.0.0"
PORT="7860"
SHARE="false"

cd "$REPO_DIR"
ARGS=(
    demo/gradio_demo.py
    --model-path "$MODEL_PATH"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --host "$HOST"
    --port "$PORT"
)

if [ -n "$CONFIG" ]; then
    ARGS+=(--config "$CONFIG")
fi
if [ -n "$HF_TOKEN" ]; then
    ARGS+=(--hf-token "$HF_TOKEN")
fi
if [ "$SHARE" = "true" ]; then
    ARGS+=(--share)
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
