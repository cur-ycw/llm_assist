#!/usr/bin/env bash
# Source this before running eureka_mujoco.py:
#   source ./env.sh
# You must export OPENAI_API_KEY yourself (and optionally OPENAI_API_BASE)
# BEFORE sourcing this file. Do NOT check secrets into this repo.

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.chatanywhere.tech/v1}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-eureka-mujoco}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[env.sh] WARNING: OPENAI_API_KEY is not set. Export it in your shell before sourcing this file."
fi

# IsaacGym lives outside conda; activate before importing isaacgym.
export ISAACGYM_PATH="${ISAACGYM_PATH:-/root/ycw/isaacgym}"
if [ -d "$ISAACGYM_PATH/python" ]; then
    export LD_LIBRARY_PATH="$ISAACGYM_PATH/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
fi

if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libpython3.8.so.1.0" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
