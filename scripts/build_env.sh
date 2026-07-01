#!/usr/bin/env bash
# Build the dedicated `vtc` conda env for the VLM-token-compression project.
# P0 foundation: torch 2.8+ (cu128) + transformers + accelerate + eval/image deps.
# Base-model-specific deps (mmcv for LLaVA, flash-attn, qwen-specific) are ADDED in P1/P2
# once the base is chosen -- kept out here to avoid heavy/fragile builds upfront.
#
# Usage:   bash scripts/build_env.sh 2>&1 | tee runs/install_vtc.log
set -euo pipefail

ENV=vtc
PY=3.11

echo "[1/4] conda create -n $ENV (python $PY)"
conda create -y -n "$ENV" "python=$PY"

PYBIN=/home/dell/miniconda3/envs/$ENV/bin/python

echo "[2/4] torch (cu128 index -> ~2.10) + torchvision"
"$PYBIN" -m pip install --upgrade pip
"$PYBIN" -m pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "[3/4] transformers + training/eval stack"
"$PYBIN" -m pip install --no-cache-dir \
    "transformers>=4.57" "accelerate>=1.10" \
    datasets sentencepiece protobuf einops \
    pillow requests tqdm pyyaml \
    numpy scipy scikit-learn pandas matplotlib

echo "[4/4] verify"
"$PYBIN" - <<'PY'
import torch, transformers, accelerate, PIL
print("torch         ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("transformers  ", transformers.__version__)
print("accelerate    ", accelerate.__version__)
print("pillow        ", PIL.__version__)
print("GPU           ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

echo "DONE vtc"
