# Copia este fichero a real/env.local.sh y rellena los valores del puesto del evento.
# Luego: source real/env.local.sh
export SO101_FOLLOWER_PORT="${SO101_FOLLOWER_PORT:-}"
export SO101_LEADER_PORT="${SO101_LEADER_PORT:-}"
export SO101_FOLLOWER_ID="${SO101_FOLLOWER_ID:-so101_follower}"
export SO101_LEADER_ID="${SO101_LEADER_ID:-so101_leader}"
export SO101_TOP_CAM="${SO101_TOP_CAM:-0}"
export SO101_WRIST_CAM="${SO101_WRIST_CAM:-1}"
export SO101_TABLE_CAM="${SO101_TABLE_CAM:--1}"   # -1 = sin cámara de mesa; p. ej. 0 para la webcam del Mac
export SO101_VLA_DEVICE="${SO101_VLA_DEVICE:-mps}"  # GPU de Apple. Usa cuda solo si el puesto tiene GPU NVIDIA
export SO101_MAX_RELATIVE_TARGET="${SO101_MAX_RELATIVE_TARGET:-2}"  # grados por paso; no subir sin acuerdo
# ALLOW_REAL_MOTION: el brazo se mueve por defecto (política del 21-nov). export ALLOW_REAL_MOTION=0 lo para todo.
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && source "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"
