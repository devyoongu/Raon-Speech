#!/bin/bash
# Export a RAON HuggingFace checkpoint to SGLang bundle format.
#
# Usage:
#   bash scripts/export_hf_to_sglang.sh [INPUT_PATH] [OUTPUT_PATH]
#
# All arguments are optional and have sensible defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configurable parameters ---
INPUT_PATH="${1:-/path/to/hf/checkpoint}"
OUTPUT_PATH="${2:-${REPO_DIR}/output/sglang-bundle}"
DTYPE="${DTYPE:-bfloat16}"

echo "=== RAON HF to SGLang Export ==="
echo "Input:      ${INPUT_PATH}"
echo "Output:     ${OUTPUT_PATH}"
echo "Dtype:      ${DTYPE}"
echo "================================"

python -m raon.export \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --dtype "${DTYPE}" \
    "${@:3}"
