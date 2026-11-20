#!/usr/bin/env bash
# Fine-tune de un detector YOLO de piezas de ajedrez en la GPU de Apple (mps).
# Uso: vision/train_pieces.sh <data.yaml> [epochs] [device]
# ultralytics ya está instalado en .venv312.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${1:?Uso: vision/train_pieces.sh <data.yaml> [epochs] [device]}"
EPOCHS="${2:-50}"
DEVICE="${3:-mps}"

export PYTORCH_ENABLE_MPS_FALLBACK=1

./.venv312/bin/yolo detect train \
  model=yolo11n.pt \
  data="$DATA" \
  epochs="$EPOCHS" \
  imgsz=640 \
  batch=16 \
  device="$DEVICE" \
  project=vision/runs name=pieces exist_ok=True
