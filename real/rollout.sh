#!/usr/bin/env bash
# Ejecuta un VLA en el brazo real. Antes de usarlo: el brazo despejado, parada de emergencia a mano
# (desconectar la alimentación) y la velocidad limitada con SO101_MAX_RELATIVE_TARGET.
# Uso: real/rollout.sh <ruta_o_repo_del_checkpoint> ["instrucción"] [segundos]
set -euo pipefail
cd "$(dirname "$0")/.."
source real/env.sh
POLICY="${1:?Falta el checkpoint}"
TASK="${2:-}"
TASK="${TASK:-Pick up a cube and place in the bin}"
DURATION="${3:-60}"
./.venv312/bin/lerobot-rollout \
  --strategy.type=base \
  --policy.path="$POLICY" \
  --robot.type=so101_follower \
  --robot.port="$SO101_FOLLOWER_PORT" \
  --robot.id="$SO101_FOLLOWER_ID" \
  --robot.max_relative_target="${SO101_MAX_RELATIVE_TARGET:-5}" \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${SO101_TOP_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${SO101_WRIST_CAM}, width: 640, height: 480, fps: 30} }" \
  --task="$TASK" \
  --duration="$DURATION"
