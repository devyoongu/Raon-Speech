#!/bin/bash
# Run RAON full-duplex fine-tuning with HuggingFace Trainer.
#
# Edit the preset variables below, then run:
#   bash scripts/duplex_train.sh
#
# Environment variables kept for low-level runtime control:
#   NCCL_TIMEOUT    NCCL timeout in seconds (default: 1800)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="/path/to/duplex-model"
DATA_DIR="/path/to/data-dir"
OUTPUT_DIR="${REPO_DIR}/output/duplex-finetune"
MAX_STEPS="100"
SAVE_STEPS="50"
BATCH_SIZE="1"
LEARNING_RATE="1e-5"
DTYPE="bfloat16"
ATTN_IMPLEMENTATION="sdpa"
NPROC_PER_NODE="1"
MASTER_PORT="29500"
EXTRA_ARGS=()

if [ "${BATCH_SIZE}" != "1" ]; then
    echo "duplex training is fixed to --batch-size 1" >&2
    exit 2
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
fi
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

echo "=== RAON Duplex Fine-tuning ==="
echo "Model:          ${MODEL_PATH}"
echo "Data dir:       ${DATA_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Max steps:      ${MAX_STEPS}"
echo "Save steps:     ${SAVE_STEPS}"
echo "Batch size:     ${BATCH_SIZE} (fixed)"
echo "Learning rate:  ${LEARNING_RATE}"
echo "Dtype:          ${DTYPE}"
echo "Attn impl:      ${ATTN_IMPLEMENTATION}"
echo "Num GPUs:       ${NPROC_PER_NODE}"
echo "Master port:    ${MASTER_PORT}"
echo "NCCL timeout:   ${NCCL_TIMEOUT}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "GPU:            ${CUDA_VISIBLE_DEVICES}"
fi
echo "==============================="

COMMON_ARGS=(
    -m raon.duplex_train
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --max_steps "${MAX_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --batch_size "1"
    --learning_rate "${LEARNING_RATE}"
    --dtype "${DTYPE}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
)

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    exec torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    exec python "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
