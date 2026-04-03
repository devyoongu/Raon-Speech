#!/bin/bash
# Run RAON full-duplex inference on an input audio file.
#
# Usage:
#   bash scripts/duplex_infer.sh [MODEL_PATH] [AUDIO_INPUT] [OUTPUT_DIR] [CONFIG_PATH]
#
# All arguments are optional and have sensible defaults.
# Sampling parameters (temperature, top_p, top_k, etc.) are configured
# in config/duplex_infer.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configurable parameters ---
MODEL_PATH="${1:-/path/to/duplex/model}"
AUDIO_INPUT="${2:-/path/to/input.flac}"
OUTPUT_DIR="${3:-${REPO_DIR}/output/duplex-inference}"
CONFIG="${4:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
SPEAKER_AUDIO="${SPEAKER_AUDIO:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

echo "=== RAON Full-Duplex Inference ==="
echo "Model:      ${MODEL_PATH}"
echo "Audio:      ${AUDIO_INPUT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Device:     ${DEVICE}"
echo "Dtype:      ${DTYPE}"
echo "Attn impl:  ${ATTN_IMPLEMENTATION}"
if [ -n "${CONFIG}" ]; then
    echo "Config:     ${CONFIG}"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
fi
if [ -n "${SPEAKER_AUDIO}" ]; then
    echo "Speaker:    ${SPEAKER_AUDIO}"
fi
echo "=================================="

python -m raon.duplex_generate \
    --model_path "${MODEL_PATH}" \
    --audio_input "${AUDIO_INPUT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    ${CONFIG:+--config "${CONFIG}"} \
    ${SPEAKER_AUDIO:+--speaker_audio "${SPEAKER_AUDIO}"}
