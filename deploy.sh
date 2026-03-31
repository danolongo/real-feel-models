#!/usr/bin/env bash
# deploy.sh — DigitalOcean GPU Droplet setup for real-feel-models
# Run once after SSH-ing into a fresh droplet:  bash deploy.sh
set -euo pipefail

REPO_URL="https://github.com/danolongo/real-feel-models.git"
PROJECT_DIR="$HOME/real-feel-models"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.8"          # H100 droplets ship with CUDA 12.8
DRIVER_VERSION="535"         # minimum driver for CUDA 12.x on Ubuntu

# ---------------------------------------------------------------------------
echo "=== [1/7] System packages ==="
apt-get update -q
apt-get install -y -q git curl build-essential

# ---------------------------------------------------------------------------
echo "=== [2/7] NVIDIA driver + CUDA check ==="

# Check if nvidia kernel module is loaded
if ! lsmod | grep -q nvidia; then
  echo "NVIDIA kernel module not found — installing driver..."
  apt-get install -y -q nvidia-driver-${DRIVER_VERSION}
  echo ""
  echo "⚠️  Driver installed. A reboot is required before continuing."
  echo "    Run: reboot"
  echo "    Then re-run this script: bash deploy.sh"
  exit 0
fi

# Driver is loaded — verify nvidia-smi works
if ! nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi failed even though kernel module is loaded."
  echo "Try: apt-get install -y nvidia-utils-${DRIVER_VERSION}"
  exit 1
fi

echo "NVIDIA driver OK:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
echo "=== [3/7] Install uv ==="
curl -Ls https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
echo "=== [4/7] Clone repo ==="
if [ -d "$PROJECT_DIR" ]; then
  echo "Repo already exists, pulling latest..."
  git -C "$PROJECT_DIR" pull
else
  git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
echo "=== [5/7] Install Python + dependencies ==="
uv python install "$PYTHON_VERSION"
uv sync

# Ensure PyTorch matches the installed CUDA version.
# DO H100 droplets run CUDA 12.8; install the cu128 wheel explicitly.
echo "Installing PyTorch for CUDA ${CUDA_VERSION}..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA is visible to PyTorch
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: torch.cuda.is_available() is False after installing cu128 wheel.')
    print('torch version :', torch.__version__)
    print('torch.version.cuda:', torch.version.cuda)
    sys.exit(1)
print('PyTorch CUDA OK:', torch.cuda.get_device_name(0))
"

# ---------------------------------------------------------------------------
echo "=== [6/7] Download dataset ==="
uv run python rf.v1.0.0/data_pipeline/download_data.py
echo "Dataset ready at rf.v1.0.0/datasets/cresci_2017_merged.csv"

# ---------------------------------------------------------------------------
echo "=== [7/7] Final GPU verification ==="
uv run python -c "
import torch
print('CUDA available :', torch.cuda.is_available())
print('GPU            :', torch.cuda.get_device_name(0))
print('VRAM           :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

echo ""
echo "Setup complete. Start training with:"
echo "  nohup python3 train.py \\"
echo "    --config production \\"
echo "    --data_path rf.v1.0.0/datasets/cresci_2017_merged.csv \\"
echo "    --output_dir ./trained_models \\"
echo "    > training.log 2>&1 &"
echo ""
echo "  echo \$! > training.pid   # save PID"
echo "  tail -f training.log"
