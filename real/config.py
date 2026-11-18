"""Configuración del SO-101 real. Todo se puede sobreescribir con variables de entorno.

Uso típico (en el portátil del evento):
  export SO101_FOLLOWER_PORT=/dev/tty.usbmodemXXXX   # brazo seguidor
  export SO101_LEADER_PORT=/dev/tty.usbmodemYYYY     # brazo líder (teleoperación)
  export SO101_TOP_CAM=0                             # índice de la cámara cenital
  export SO101_WRIST_CAM=1                           # índice de la cámara de muñeca
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "real" / "datasets"
CHECKPOINTS_DIR = ROOT / "real" / "checkpoints"

FOLLOWER_PORT = os.environ.get("SO101_FOLLOWER_PORT", "")
LEADER_PORT = os.environ.get("SO101_LEADER_PORT", "")
FOLLOWER_ID = os.environ.get("SO101_FOLLOWER_ID", "so101_follower")
LEADER_ID = os.environ.get("SO101_LEADER_ID", "so101_leader")

TOP_CAM = int(os.environ.get("SO101_TOP_CAM", "0"))
WRIST_CAM = int(os.environ.get("SO101_WRIST_CAM", "1"))
# Cámara de vista completa de la mesa (opcional). -1 = desactivada.
# Puede ser la webcam del Mac (suele ser el índice 0 si no hay otra cámara; confírmalo con check_hardware.py).
TABLE_CAM = int(os.environ.get("SO101_TABLE_CAM", "-1"))
CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS = 30

# Seguridad: máximo cambio de posición por paso de control (unidades normalizadas).
# LeRobot lo aplica en el seguidor; bájalo si el brazo se mueve de forma brusca.
MAX_RELATIVE_TARGET = float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "5"))

# Tareas del entrenamiento del checkpoint de referencia (texto exacto)
DEFAULT_TASK = os.environ.get("SO101_TASK", "Pick up a cube and place in the bin")

# Modelo base y checkpoint de referencia
BASE_POLICY = "lerobot/smolvla_base"
REFERENCE_POLICY = "Harrysunshine/so101-smolvla-real-9task"
