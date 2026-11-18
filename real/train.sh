#!/usr/bin/env bash
# Fine-tune de SmolVLA sobre un dataset grabado con real/record.sh.
# Uso: real/train.sh <nombre_dataset> [pasos] [dispositivo: cuda|mps|cpu] [policy_base]
# En Mac, mps/cpu son lentos: usa una GPU NVIDIA (la del evento o nube) para entrenar.
set -euo pipefail
cd "$(dirname "$0")/.."
source real/env.sh
NAME="${1:?Falta el nombre del dataset}"
STEPS="${2:-20000}"
DEVICE="${3:-cuda}"
BASE="${4:-lerobot/smolvla_base}"
OUT="real/checkpoints/${NAME}_$(date +%Y%m%d_%H%M)"
./.venv312/bin/lerobot-train \
  --policy.path="$BASE" \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --dataset.repo_id="local/${NAME}" \
  --dataset.root="real/datasets/${NAME}" \
  --output_dir="$OUT" \
  --job_name="so101_${NAME}" \
  --steps="$STEPS" \
  --batch_size=8 \
  --save_freq=5000 \
  --wandb.enable=false
echo "Checkpoint en $OUT/checkpoints/last/pretrained_model"
