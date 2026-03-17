#!/bin/bash
# SCARE PyMARL 环境配置脚本 (Linux GPU)
# 基于 Windows pymarl 环境复刻，PyTorch 改为 CUDA 版本
#
# Usage:
#   chmod +x setup_pymarl_env.sh
#   ./setup_pymarl_env.sh

set -e

ENV_NAME="pymarl"
PYTHON_VERSION="3.8"

echo "=== Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "=== Installing PyTorch (CUDA 12.1) ==="
# 如果你的 CUDA 版本不同，改这里：
#   CUDA 11.8: pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
#   CUDA 12.1: pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing core dependencies ==="
pip install \
    numpy==1.24.4 \
    scipy==1.10.1 \
    pyyaml==6.0.3 \
    sacred==0.8.4 \
    tensorboard-logger==0.1.0 \
    matplotlib==3.7.5 \
    protobuf==3.20.3

echo "=== Installing StarCraft II related ==="
pip install \
    pysc2==3.0.0 \
    s2clientprotocol==5.0.15.95299.0 \
    s2protocol==5.0.15.95299.0 \
    mpyq==0.2.5 \
    portpicker==1.6.0

echo "=== Installing SMAC ==="
# smac 和 smacv2 在项目同级目录，本地安装
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -e "${PROJECT_ROOT}/smac"
pip install -e "${PROJECT_ROOT}/smacv2"

echo "=== Installing utility packages ==="
pip install \
    openai \
    cloudpickle \
    jsonpickle==4.1.1 \
    deepdiff==8.4.2 \
    tqdm \
    psutil \
    requests

echo ""
echo "=== Done! ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Verify PyTorch:  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo ""
echo "Remember to:"
echo "  1. Install StarCraft II Linux: https://github.com/Blizzard/s2client-proto#linux-packages"
echo "  2. Set SC2PATH:  export SC2PATH=/path/to/StarCraftII"
echo "  3. Copy SMAC maps to \$SC2PATH/Maps/SMAC_Maps/"
