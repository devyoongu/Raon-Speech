#!/usr/bin/env bash
set -euo pipefail

HF_CACHE="/home/posicube/.cache/huggingface"
IMAGE_NAME="raon-speech-test"

# -----------------------------------------------------------
# Step 2: Build Docker image
# -----------------------------------------------------------
echo ""
echo "=== Step 2: Building Docker image ==="
docker build -t "${IMAGE_NAME}" .

# -----------------------------------------------------------
# Step 3: Run inference test
# -----------------------------------------------------------
echo ""
echo "=== Step 3: Running inference test ==="
docker run --rm \
    --gpus all \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -v "$(pwd)/output:/app/output" \
    "${IMAGE_NAME}" \
    python scripts/test_inference.py \
        --model_path "KRAFTON/Raon-Speech-9B" \
        --audio "wav/femail_achernar.wav" \
        --output_dir "output/test"
