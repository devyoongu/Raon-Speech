#!/bin/bash
# Run RAON full-duplex fine-tuning with HuggingFace Trainer.
#
# Usage:
#   bash scripts/duplex_train.sh [MODEL_PATH] [DATA_DIR] [OUTPUT_DIR] [EXTRA_ARGS...]
#
# All arguments are optional and have sensible defaults.
# Extra arguments (position 4+) are forwarded directly to the Python script.
#
# Environment variables:
#   NPROC_PER_NODE  Number of GPUs (default: 1)
#   MASTER_PORT     torchrun master port (default: 29500)
#   MAX_STEPS       Training steps (default: 100)
#   SAVE_STEPS      Checkpoint interval (default: 50)
#   BATCH_SIZE      Ignored. Duplex training is fixed to batch size 1.
#   LEARNING_RATE   Learning rate (default: 1e-5)
#   DTYPE           Model dtype (default: bfloat16)
#   NCCL_TIMEOUT    NCCL timeout in seconds (default: 1800)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configurable parameters ---
MODEL_PATH="${1:-/path/to/duplex/model}"
DATA_DIR="${2:-/path/to/data/dir}"
OUTPUT_DIR="${3:-${REPO_DIR}/output/duplex-finetune}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-50}"
BATCH_SIZE="1"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DTYPE="${DTYPE:-bfloat16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
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
)

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        "${COMMON_ARGS[@]}" \
        "${@:4}"
else
    python "${COMMON_ARGS[@]}" "${@:4}"
fi
