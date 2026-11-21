#!/usr/bin/env bash
# Teleopera el brazo con el líder y graba demostraciones en un dataset local (sin subir a Hugging Face).
# Uso: real/record.sh [num_episodios] [nombre_dataset] ["instrucción"]
set -euo pipefail
cd "$(dirname "$0")/.."
source real/env.sh
# Cámara de mesa opcional (SO101_TABLE_CAM >= 0).
CAM_TABLE=""
if [ "${SO101_TABLE_CAM:--1}" -ge 0 ]; then CAM_TABLE=", table: {type: opencv, index_or_path: ${SO101_TABLE_CAM}, width: 320, height: 240, fps: 30}"; fi
EPISODES="${1:-30}"
NAME="${2:-so101_demos}"
TASK="${3:-}"
TASK="${TASK:-Pick up a cube and place in the bin}"
REPO="local/${NAME}"
./.venv312/bin/lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$SO101_FOLLOWER_PORT" \
  --robot.id="$SO101_FOLLOWER_ID" \
  --robot.max_relative_target="${SO101_MAX_RELATIVE_TARGET:-5}" \
  --robot.use_degrees=true \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${SO101_TOP_CAM}, width: 320, height: 240, fps: 30}, wrist: {type: opencv, index_or_path: ${SO101_WRIST_CAM}, width: 320, height: 240, fps: 30}${CAM_TABLE} }" \
  --teleop.type=so101_leader \
  --teleop.port="$SO101_LEADER_PORT" \
  --teleop.id="$SO101_LEADER_ID" \
  --teleop.use_degrees=true \
  --dataset.repo_id="$REPO" \
  --dataset.root="real/datasets/${NAME}" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --display_data=false
