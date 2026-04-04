#!/bin/bash
# Run RAON inference on JSONL data (TTS / STT / Speech-Chat / TextQA).
#
# Edit the preset variables below, then run:
#   bash scripts/infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="/path/to/pretrained-model"
DATA_DIR="/path/to/data-dir"
OUTPUT_DIR="${REPO_DIR}/output/speechllm-inference"
CONFIG=""  # default: config/infer.yaml
BATCH_SIZE="1"
DEVICE="cuda"
DTYPE="bfloat16"
ATTN_IMPLEMENTATION="sdpa"
EXTRA_ARGS=()

echo "=== RAON Inference ==="
echo "Model:       ${MODEL_PATH}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Device:      ${DEVICE}"
echo "Dtype:       ${DTYPE}"
echo "Attn impl:   ${ATTN_IMPLEMENTATION}"
if [ -n "${CONFIG}" ]; then
    echo "Config:      ${CONFIG}"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "GPU:         ${CUDA_VISIBLE_DEVICES}"
fi
echo "======================"

exec python -m raon.generate \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    ${CONFIG:+--config "${CONFIG}"} \
    "${EXTRA_ARGS[@]}"
