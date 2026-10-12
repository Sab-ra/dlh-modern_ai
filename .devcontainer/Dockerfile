# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/home/vscode/.cache/huggingface \
    NLTK_DATA=/home/vscode/nltk_data

# Only what the course needs at runtime: git (+ OpenMP for LightGBM).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -ms /bin/bash vscode

# 1) CPU-only PyTorch first, from the PyTorch CPU index (no CUDA wheels).
# 2) Everything else from PyPI. BuildKit cache makes rebuilds fast.
# 3) ultralytics pulls GUI opencv-python; swap it back to headless.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93

# Pre-create volume targets so named volumes inherit vscode ownership.
RUN mkdir -p /home/vscode/.cache /home/vscode/nltk_data \
 && chown -R vscode:vscode /home/vscode

USER vscode
WORKDIR /workspace
