#!/bin/bash
# Run RAON full-duplex inference on JSONL metadata files.
#
# Edit the preset variables below, then run:
#   bash scripts/duplex_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="/path/to/duplex-model"
DATA_DIR="/path/to/data-dir"
OUTPUT_DIR="${REPO_DIR}/output/duplex-inference"
CONFIG=""  # default: config/duplex_infer.yaml
SPEAKER_AUDIO=""
DEVICE="cuda"
DTYPE="bfloat16"
ATTN_IMPLEMENTATION="sdpa"
EXTRA_ARGS=()

cd "$REPO_DIR"
exec python -m raon.duplex_generate \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    ${CONFIG:+--config "$CONFIG"} \
    ${SPEAKER_AUDIO:+--speaker_audio "$SPEAKER_AUDIO"} \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    "${EXTRA_ARGS[@]}"
