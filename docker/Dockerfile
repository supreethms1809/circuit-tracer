# Spline-CLT training image (CUDA).
#
# Build context is the repo root (see docker/docker-compose.yml).
# .dockerignore lives at the repo root because Docker reads it from the context.
#
# On a blank GPU cloud machine (NVIDIA driver already present):
#
#   git clone https://github.com/supreethms1809/circuit-tracer.git
#   cd circuit-tracer
#   git checkout claude/silly-feistel
#   bash docker/host_setup.sh
#   docker compose -f docker/docker-compose.yml up --build
#
# Nsight Systems short run (100 steps default; NSYS_STEPS=10 for a 10-step trace):
#   docker compose -f docker/docker-compose.yml -f docker/docker-compose.nsight.yml up --build
#
# Paper-eval / FSDP (v3 launch suite, 2 GPUs):
#   MODE=paper-eval NPROC_PER_NODE=2 \
#     PAPER_SUITE=experiments/paper_configs_v3/launch_suites/r5b_b3_gpt2_small_spline_feature_match.json \
#     docker compose -f docker/docker-compose.yml up --build
#
# Requires: NVIDIA driver >= 550 (CUDA 12.4) and compose GPU reservation.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,profiler \
    HF_HOME=/workspace/hf-cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf-cache \
    HF_DATASETS_CACHE=/workspace/hf-cache/datasets \
    TRANSFORMERS_CACHE=/workspace/hf-cache/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=/workspace/circuit-tracer \
    PAPER_OUTPUT_ROOT=/workspace/circuit-tracer/results/paper \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        git-lfs \
        wget \
        gnupg \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# Nsight Systems CLI (`nsys`). Arch-specific CUDA repo: x86_64 vs GH200/sbsa.
ARG TARGETARCH
RUN arch_repo="x86_64" \
    && if [ "${TARGETARCH}" = "arm64" ]; then arch_repo="sbsa"; fi \
    && wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${arch_repo}/cuda-keyring_1.1-1_all.deb" \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && (apt-get install -y --no-install-recommends nsight-systems \
        || apt-get install -y --no-install-recommends nsight-systems-cli) \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb \
    && nsys --version

# Install CUDA-enabled PyTorch first so the project resolver does not pull a CPU wheel.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ENV PIP_EXTRA_INDEX_URL=${TORCH_INDEX_URL}
RUN pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}"

WORKDIR /workspace/circuit-tracer

COPY pyproject.toml README.md ./
COPY attribution/ attribution/
COPY circuit_tracer/ circuit_tracer/
COPY spline_clt/ spline_clt/
COPY macag/ macag/
COPY eval/ eval/
COPY spline_sae/ spline_sae/
COPY rebuttal_eval/ rebuttal_eval/
COPY experiments/ experiments/
COPY scripts/ scripts/
COPY tests/ tests/
COPY docker/entrypoint.sh /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh \
    && pip install --no-cache-dir git+https://github.com/Blealtan/efficient-kan.git \
    && pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir datasets pyyaml wandb hf_transfer

RUN mkdir -p /workspace/circuit-tracer/data \
             /workspace/circuit-tracer/checkpoints \
             /workspace/circuit-tracer/profiles \
             /workspace/circuit-tracer/results \
             /workspace/hf-cache

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD []
