#!/bin/bash
# Run RAON fine-tuning with HuggingFace Trainer.
#
# Edit the preset variables below, then run:
#   bash scripts/train.sh
#
# Environment variables kept for low-level runtime control:
#   NCCL_TIMEOUT    NCCL timeout in seconds (default: 1800)
#   CUDA_VISIBLE_DEVICES  GPUs to use (default: 0..NPROC_PER_NODE-1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="/path/to/pretrained-model"
DATA_DIR="/path/to/data-dir"
OUTPUT_DIR="${REPO_DIR}/output/speechllm-finetune"
MAX_STEPS="1000"
SAVE_STEPS="500"
BATCH_SIZE="1"
LEARNING_RATE="1e-5"
DTYPE="bfloat16"
USE_SPEAKER_EMBEDDING="true"
ATTN_IMPLEMENTATION="sdpa"
NPROC_PER_NODE="1"
MASTER_PORT="29500"
EXTRA_ARGS=()

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
fi
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

echo "=== RAON Fine-tuning ==="
echo "Model:          ${MODEL_PATH}"
echo "Data dir:       ${DATA_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Max steps:      ${MAX_STEPS}"
echo "Save steps:     ${SAVE_STEPS}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Learning rate:  ${LEARNING_RATE}"
echo "Dtype:          ${DTYPE}"
echo "Speaker embed:  ${USE_SPEAKER_EMBEDDING}"
echo "Attn impl:      ${ATTN_IMPLEMENTATION}"
echo "Num GPUs:       ${NPROC_PER_NODE}"
echo "Master port:    ${MASTER_PORT}"
echo "NCCL timeout:   ${NCCL_TIMEOUT}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "GPU:            ${CUDA_VISIBLE_DEVICES}"
fi
echo "========================"

COMMON_ARGS=(
    -m raon.train
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --max_steps "${MAX_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --dtype "${DTYPE}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
)

if [[ "${USE_SPEAKER_EMBEDDING}" == "true" ]]; then
    COMMON_ARGS+=(--use_speaker_embedding)
else
    COMMON_ARGS+=(--no-use_speaker_embedding)
fi

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    # Multi-GPU: use torchrun with NCCL
    exec torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    # Single-GPU: run directly without distributed overhead
    exec python "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
