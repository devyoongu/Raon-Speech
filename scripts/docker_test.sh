#!/usr/bin/env bash
set -euo pipefail

HF_CACHE="/home/posicube/.cache/huggingface"
IMAGE_NAME="raon-speech-test"

# -----------------------------------------------------------
# Step 1: Download Raon-Speech-9B model (if not already cached)
# -----------------------------------------------------------
echo "=== Step 1: Downloading KRAFTON/Raon-Speech-9B ==="
docker run --rm \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    python:3.11-slim \
    bash -c "pip install -q huggingface_hub && python -c \"from huggingface_hub import snapshot_download; snapshot_download('KRAFTON/Raon-Speech-9B')\""

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
