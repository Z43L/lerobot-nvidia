#!/usr/bin/env bash
# Entrenamiento de ACT desde cero (sin checkpoint base) sobre un dataset LeRobot.
# Uso: real/train_act.sh <nombre_dataset> [pasos] [dispositivo: mps|cuda|cpu] [raiz_dataset]
# Por defecto lee real/datasets/<nombre>. Para sim, pasa sim/datasets/<nombre> como raíz.
# Medido en el Mac (mps, 240x320, batch 8): ~0,9 s por paso.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
NAME="${1:?Falta el nombre del dataset}"
STEPS="${2:-5000}"
DEVICE="${3:-mps}"
ROOT="${4:-real/datasets/${NAME}}"
OUT="real/checkpoints/act_${NAME}_$(date +%Y%m%d_%H%M)"
./.venv312/bin/lerobot-train \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --dataset.repo_id="local/${NAME}" \
  --dataset.root="$ROOT" \
  --output_dir="$OUT" \
  --job_name="act_${NAME}" \
  --steps="$STEPS" \
  --batch_size=8 \
  --save_freq=1000
echo "Checkpoint en $OUT/checkpoints/last/pretrained_model"
