#!/usr/bin/env bash
set -euo pipefail

HF_CACHE="/home/posicube/.cache/huggingface"
HF_MODEL_PATH="${HF_CACHE}/hub/models--KRAFTON--Raon-SpeechChat-9B/snapshots/d844aee2c3da129b92fce3e0193c07eb98b88443"
IMAGE_NAME="raon-realtime"
PORT="${1:-7861}"
EXPORT_DIR="$(pwd)/output/sglang-bundle"

# -----------------------------------------------------------
# Step 1: Build Docker image (with demo dependencies)
# -----------------------------------------------------------
echo "=== Step 1: Building Docker image (raon-realtime) ==="
docker build -t "${IMAGE_NAME}" -f Dockerfile.realtime .

# -----------------------------------------------------------
# Step 2: Export model to SGLang bundle (if not already done)
# -----------------------------------------------------------
if [ ! -d "${EXPORT_DIR}/text_model" ]; then
    echo ""
    echo "=== Step 2: Exporting Raon-SpeechChat-9B to SGLang bundle ==="
    docker run --rm \
        --gpus all \
        -v "${HF_MODEL_PATH}:/model:ro" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        -v "$(pwd)/output:/app/output" \
        "${IMAGE_NAME}" \
        python -m raon.export \
            --input_path "/model" \
            --output_path "output/sglang-bundle" \
            --dtype bfloat16
    echo "Export complete: ${EXPORT_DIR}"
else
    echo ""
    echo "=== Step 2: SGLang bundle already exists, skipping export ==="
fi

# -----------------------------------------------------------
# Step 3: Run Gradio duplex demo server
# -----------------------------------------------------------
echo ""
echo "=== Step 3: Starting Gradio Realtime Demo on port ${PORT} ==="
echo "    Access: http://<server-ip>:${PORT}"
echo ""
docker run --rm \
    --gpus all \
    -p "${PORT}:7861" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/wav:/app/wav" \
    -v "$(pwd)/data:/app/data" \
    "${IMAGE_NAME}" \
    python demo/gradio_duplex_demo.py \
        --host 0.0.0.0 \
        --port 7861 \
        --model-path "output/sglang-bundle" \
        --result-root "output/fd_gradio_demo" \
        --speaker-audio "wav/femail_achernar.wav" \
        --compile-audio-modules false