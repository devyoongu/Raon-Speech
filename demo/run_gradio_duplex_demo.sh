#!/bin/bash
# Run the RAON full-duplex Gradio demo shell.
#
# Usage:
#   bash demo/run_gradio_duplex_demo.sh [--port 7861] [--api-base http://127.0.0.1:7861]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="${PYTHON:-python3}"

# Keep realtime compile warmup enabled for duplex demo startup.
export FD_ENABLE_COMPILE_AUDIO_MODULES=1
export FD_COMPILE_MAX_SEQUENCE_LENGTH="${FD_COMPILE_MAX_SEQUENCE_LENGTH:-8192}"

cd "$REPO_DIR"
exec "$PYTHON_BIN" demo/gradio_duplex_demo.py "$@"
