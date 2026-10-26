# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

# Ollama release to install (https://github.com/ollama/ollama/releases).
ARG OLLAMA_VERSION=0.40.11
# Set by BuildKit (amd64 or arm64); dpkg is the fallback.
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/home/vscode/.cache/huggingface \
    NLTK_DATA=/home/vscode/nltk_data \
    OLLAMA_MODELS=/home/vscode/.cache/ollama/models \
    OLLAMA_HOST=127.0.0.1:11434

# Only what the course needs at runtime: git (+ OpenMP for LightGBM),
# plus curl, CA certificates and zstd to download and unpack Ollama.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git libgomp1 curl ca-certificates zstd \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -ms /bin/bash vscode

# Ollama (local LLM server, e.g. Mistral for LangChain's ChatOllama).
# Official Linux tarball, unpacked to /usr like the official install.sh.
# CPU-only: skip the GPU libraries (~2.4 GB of CUDA/Vulkan/ROCm).
# This layer sits before requirements.txt, so package edits don't
# download Ollama again. Models are NOT baked in: they go to the
# ml-study-cache volume (OLLAMA_MODELS) via `ollama pull`.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN arch="${TARGETARCH:-$(dpkg --print-architecture)}" \
 && base="https://github.com/ollama/ollama/releases/download" \
 && curl -fsSL "${base}/v${OLLAMA_VERSION}/ollama-linux-${arch}.tar.zst" \
    | zstd -d | tar -x -C /usr \
        --exclude='lib/ollama/cuda_*' \
        --exclude='lib/ollama/vulkan*' \
        --exclude='lib/ollama/rocm*' \
 && test -x /usr/bin/ollama

# 1) CPU-only PyTorch first, from the PyTorch CPU index (no CUDA wheels).
# 2) Everything else from PyPI. BuildKit cache makes rebuilds fast.
# 3) ultralytics pulls GUI opencv-python; swap it back to headless.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --force-reinstall --no-deps \
        opencv-python-headless==5.0.0.93

# Pre-create volume targets so named volumes inherit vscode ownership.
RUN mkdir -p /home/vscode/.cache/ollama/models /home/vscode/nltk_data \
 && chown -R vscode:vscode /home/vscode

USER vscode
WORKDIR /workspace
