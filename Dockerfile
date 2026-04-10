FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        git curl libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[demo]" 2>/dev/null || \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -e ".[demo]"

# Copy project source
COPY . .

# Re-install in editable mode with full source
RUN pip install --no-cache-dir -e ".[demo]"

EXPOSE 7861

ENTRYPOINT ["python", "demo/gradio_duplex_demo.py"]
CMD ["--host", "0.0.0.0", "--port", "7861"]
