# =============================================================================
# ai-document-indexer-mcp — Python 3.12 container with Git hydration
#
# Build examples:
#   CPU only:     docker build -f docker/Dockerfile .
#   NVIDIA GPU:   docker build -f docker/Dockerfile --build-arg GPU_RUNTIME=cuda .
#   AMD GPU:      docker build -f docker/Dockerfile --build-arg GPU_RUNTIME=rocm .
#
# GPU builds require the matching container toolkit on the host:
#   NVIDIA → nvidia-container-toolkit
#   AMD    → WSL2 requires librocdxg/libdxcore mounts. Native Linux requires /dev/kfd.
# =============================================================================
FROM ubuntu:24.04 AS base

ARG GPU_RUNTIME=

# ── System dependencies ────────────────────────────────────────────────
# Python 3.12 + git for repo hydration.
# Standardized on Ubuntu 24.04 (Noble) for native AMD/NVIDIA repository support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    git \
    curl \
    gnupg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# ── Optional: CUDA runtime for ONNX Runtime GPU (NVIDIA) ────────────
RUN if [ "$GPU_RUNTIME" = "cuda" ]; then \
      apt-get update && apt-get install -y --no-install-recommends wget && \
      wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
      dpkg -i cuda-keyring_1.1-1_all.deb && \
      apt-get update && apt-get install -y --no-install-recommends \
        cuda-runtime-12-6 \
        libcublas-12-6 \
        libcudnn9-cuda-12 \
      && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb; \
    fi

# ── Optional: ROCm runtime for ONNX Runtime GPU (AMD) ───────────────
# Installs ROCm core libraries natively in the container for portability.
# Note: ONNX Runtime often requires rocBLAS for hardware acceleration,
# so it is included alongside rocm-core.
RUN if [ "$GPU_RUNTIME" = "rocm" ]; then \
      mkdir -p --mode=0755 /etc/apt/keyrings && \
      wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
        tee /etc/apt/keyrings/rocm.asc > /dev/null && \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/debian noble main" \
        | tee /etc/apt/sources.list.d/rocm.list && \
      apt-get update && apt-get install -y --no-install-recommends \
        rocm-core \
        rocm-smi-lib \
        rocblas \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# ── Install uv (fast Python package installer) ───────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies (layer cached until pyproject.toml / uv.lock change) ──
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Optional: swap the ONNX Runtime backend for GPU acceleration ─────
#  CUDA → fastembed-gpu (bundles onnxruntime-gpu, replaces fastembed)
#  ROCm → keep standard fastembed, just swap onnxruntime → onnxruntime-rocm
RUN if [ "$GPU_RUNTIME" = "cuda" ]; then \
      uv pip uninstall fastembed -y && \
      uv pip install fastembed-gpu; \
    elif [ "$GPU_RUNTIME" = "rocm" ]; then \
      uv pip uninstall onnxruntime -y || true && \
      uv pip install onnxruntime-rocm; \
    fi

# ── Copy source code and config ──────────────────────────────────────────
COPY src/ ./src/
COPY config/ ./config/

# ── Create data directories ─────────────────────────────────────────
RUN mkdir -p /data/repo /data/chroma_db

# ── Run the startup script (Git hydration) then launch the server ────────
CMD ["uv", "run", "src/startup.py"]
