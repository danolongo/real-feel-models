#!/usr/bin/env bash
# deploy.sh — DigitalOcean GPU Droplet setup for real-feel-models
# Run once after SSH-ing into a fresh droplet:  bash deploy.sh
set -euo pipefail

REPO_URL="https://github.com/danolongo/real-feel-models.git"
PROJECT_DIR="$HOME/real-feel-models"
PYTHON_VERSION="3.10"

echo "=== [1/6] System packages ==="
apt-get update -q
apt-get install -y -q git curl build-essential

echo "=== [2/6] Install uv ==="
curl -Ls https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [3/6] Clone repo ==="
if [ -d "$PROJECT_DIR" ]; then
  echo "Repo already exists, pulling latest..."
  git -C "$PROJECT_DIR" pull
else
  git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

echo "=== [4/6] Install Python + dependencies ==="
uv python install "$PYTHON_VERSION"
uv sync

echo "=== [5/6] Download dataset ==="
uv run python rf.v1.0.0/data_pipeline/download_data.py
echo "Dataset ready at rf.v1.0.0/datasets/cresci_2017_merged.csv"

echo "=== [6/6] Verify GPU ==="
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "Setup complete. Start training with:"
echo "  nohup uv run python3 train.py \\"
echo "    --config production \\"
echo "    --data_path rf.v1.0.0/datasets/cresci_2017_merged.csv \\"
echo "    --output_dir ./trained_models \\"
echo "    > training.log 2>&1 &"
echo ""
echo "Tail logs: tail -f training.log"
