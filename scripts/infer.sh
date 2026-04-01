#!/bin/bash
# Run RAON inference on JSONL data (TTS / STT / SpeechQA).
#
# Usage:
#   bash scripts/infer.sh [MODEL_PATH] [DATA_DIR] [OUTPUT_DIR]
#
# All arguments are optional and have sensible defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configurable parameters ---
MODEL_PATH="${1:-/path/to/pretrained/model}"
DATA_DIR="${2:-/path/to/data/dir}"
OUTPUT_DIR="${3:-${REPO_DIR}/output/inference}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
echo "=== RAON Inference ==="
echo "Model:       ${MODEL_PATH}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Device:      ${DEVICE}"
echo "Dtype:       ${DTYPE}"
echo "Attn impl:   ${ATTN_IMPLEMENTATION}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "GPU:         ${CUDA_VISIBLE_DEVICES}"
fi
echo "======================"

python -m raon.generate \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}"
