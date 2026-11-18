#!/usr/bin/env bash
# Calibra el brazo seguidor y el líder con LeRobot. Solo hace falta si no hay calibración guardada.
# Pide mover las articulaciones por sus límites cuando lo indique la consola.
set -euo pipefail
cd "$(dirname "$0")/.."
source real/env.sh
./.venv312/bin/lerobot-calibrate --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" \
  --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID"
