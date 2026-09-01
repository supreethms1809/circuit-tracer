#!/usr/bin/env bash
# Install Docker Engine + NVIDIA Container Toolkit on a blank Ubuntu/Debian GPU VM.
# Requires: NVIDIA driver already installed (nvidia-smi works).
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "=== Host preflight ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "ERROR: nvidia-smi not found. Install an NVIDIA GPU driver on the host first." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "=== Installing Docker Engine ==="
    curl -fsSL https://get.docker.com | ${SUDO} sh
else
    echo "Docker already installed: $(docker --version)"
fi

if [[ -n "${SUDO}" ]]; then
    ${SUDO} usermod -aG docker "${USER}" || true
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "=== Installing NVIDIA Container Toolkit ==="
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y --no-install-recommends curl ca-certificates gnupg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | ${SUDO} gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y nvidia-container-toolkit
else
    echo "NVIDIA Container Toolkit already installed."
fi

${SUDO} nvidia-ctk runtime configure --runtime=docker
if command -v systemctl >/dev/null 2>&1; then
    ${SUDO} systemctl restart docker
else
    ${SUDO} service docker restart || true
fi

echo "=== Verifying GPU in Docker ==="
${SUDO} docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo
echo "Host is ready. From the repo root:"
echo "  docker compose -f docker/docker-compose.yml up --build"
echo "  NSYS_STEPS=10 docker compose -f docker/docker-compose.yml -f docker/docker-compose.nsight.yml up --build"
if [[ -n "${SUDO}" ]]; then
    echo
    echo "If 'docker' is permission-denied, either:"
    echo "  newgrp docker"
    echo "  or log out and back in (you were added to the docker group),"
    echo "  or prefix commands with sudo."
fi
