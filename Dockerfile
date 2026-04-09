FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        libsndfile1 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.8 (compatible with driver 575 / CUDA 12.9)
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project and install
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/test_inference.py"]
