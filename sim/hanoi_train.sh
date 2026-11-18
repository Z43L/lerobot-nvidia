#!/usr/bin/env bash
# Fine-tune de SmolVLA para UN color de disco (un VLA por color), sobre sim/datasets/hanoi_<color>.
# Uso: sim/hanoi_train.sh <color> [pasos] [dispositivo: mps|cpu|cuda] [batch] [policy_base]
# Parte del checkpoint SO-101 (mismas cámaras camera1/2/3 y misma escala normalizada).
set -euo pipefail
cd "$(dirname "$0")/.."
COLOR="${1:?Falta el color: red|green|blue|cyan|brown}"
STEPS="${2:-2000}"
DEVICE="${3:-mps}"
BATCH="${4:-8}"
BASE="${5:-Harrysunshine/so101-smolvla-real-9task}"
DATASET_ROOT="$(pwd)/sim/datasets/hanoi_${COLOR}"
OUT="$(pwd)/sim/checkpoints/hanoi_${COLOR}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
./.venv312/bin/lerobot-train \
  --policy.path="$BASE" \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --dataset.repo_id="local/hanoi_${COLOR}" \
  --dataset.root="$DATASET_ROOT" \
  --output_dir="$OUT" \
  --job_name="so101_hanoi_${COLOR}" \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false
echo "Checkpoint en $OUT/checkpoints/last/pretrained_model"
