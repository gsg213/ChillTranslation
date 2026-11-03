#!/usr/bin/env sh
# Run registre.generate inside the vLLM image on a Grid5000 GPU node.
#
#   IMAGE=vllm/vllm-openai:<tag> MODEL=<hf-model-id> scripts/g5k_generate.sh [extra generate.py args]
#
# Run it from the repo root, on a reserved GPU node, after enabling Docker on the node
# (Grid5000 provides a helper script for that; see the Docker page of their wiki).
# Choose IMAGE by the node's driver: `nvidia-smi` shows CUDA >= 13 → the plain tag works;
# CUDA 12.x → a -cu129 tag (plus NVIDIA_DISABLE_REQUIRE=true below, and Dockerfile.vllm's fix).
set -eu
: "${IMAGE:?set IMAGE, e.g. IMAGE=vllm/vllm-openai:v0.32.0}"
: "${MODEL:?set MODEL to the teacher's Hugging Face id}"

# Model weights and compile caches go to the node's local disk, not your (quota-limited,
# network-mounted) home directory.
CACHE=/tmp/registre-cache
mkdir -p "$CACHE/hf" "$CACHE/home"

# --user: write output files as you. Grid5000 homes are network mounts where root inside a
# container typically can't write. HOME/HF_HOME point to the writable local cache.
exec docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD":/app -w /app \
  -v "$CACHE/hf":/hf -v "$CACHE/home":/home/runner \
  -e HOME=/home/runner -e HF_HOME=/hf -e PYTHONPATH=/app/src \
  -e NVIDIA_DISABLE_REQUIRE="${NVIDIA_DISABLE_REQUIRE:-false}" \
  --env-file .env \
  --entrypoint python3 \
  "$IMAGE" -m registre.generate --backend vllm --model "$MODEL" "$@"
