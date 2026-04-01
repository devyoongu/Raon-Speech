#!/bin/bash
# Run the RAON Gradio demo with a local model.
#
# Usage:
#   bash demo/run_gradio_demo.sh --model /path/to/model [--port 7860] [--share]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="${PYTHON:-python3}"

cd "$REPO_DIR"
exec "$PYTHON_BIN" demo/gradio_demo.py "$@"
