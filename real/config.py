"""Configuración del SO-101 real. Todo se puede sobreescribir con variables de entorno.

Uso típico (en el portátil del evento):
  export SO101_FOLLOWER_PORT=/dev/tty.usbmodemXXXX   # brazo seguidor
  export SO101_LEADER_PORT=/dev/tty.usbmodemYYYY     # brazo líder (teleoperación)
  export SO101_TOP_CAM=0                             # índice de la cámara cenital
  export SO101_WRIST_CAM=1                           # índice de la cámara de muñeca
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_LOCAL = ROOT / "real" / "env.local.sh"
# Ninguna variable está vetada: env.local.sh puede fijar también ALLOW_REAL_MOTION (0 = parada).
_NEVER_FROM_FILE: set[str] = set()


def load_local_env(path: Path = ENV_LOCAL) -> dict[str, str]:
    """Valores por defecto desde real/env.local.sh (sin ejecutar el shell). No pisa lo que ya esté en el entorno,
    así que `export` en la terminal o `env` en .mcp.json siguen mandando. Es la ÚNICA fuente de puertos y cámaras."""
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        m = re.match(r'\s*export\s+(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S*))', line)
        if m and m.group(1) not in _NEVER_FROM_FILE:
            value = next(g for g in m.groups()[1:] if g is not None)
            if value != "" and m.group(1) not in os.environ:
                os.environ[m.group(1)] = value
                loaded[m.group(1)] = value
    return loaded


load_local_env()
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

# Unidades de las articulaciones del brazo (la pinza va siempre 0..100).
# Fijo en GRADOS: el checkpoint de referencia se entrenó así. Sus estadísticas llegan a ±110
# en shoulder_lift y a 171 en wrist_roll, valores que RANGE_M100_100 no puede producir.
# No es una variable de entorno a propósito: cambiarlo rompería el checkpoint.
USE_DEGREES = True

# Seguridad: máximo cambio de posición por paso de control (en unidades de LeRobot:
# grados si USE_DEGREES, si no escala -100..100). LeRobot lo aplica en el seguidor.
MAX_RELATIVE_TARGET = float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2"))

# Tareas del entrenamiento del checkpoint de referencia (texto exacto)
DEFAULT_TASK = os.environ.get("SO101_TASK", "Pick up a cube and place in the bin")

# Modelo base y checkpoint de referencia
BASE_POLICY = "lerobot/smolvla_base"
REFERENCE_POLICY = "Harrysunshine/so101-smolvla-real-9task"
