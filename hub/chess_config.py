"""Configuración ÚNICA del ajedrez real. Nada de puertos, cámaras ni rutas de pesos escritos en el código de los módulos.

Orden de prioridad de cada valor: variable de entorno > real/env.local.sh (lo escribe so101-hub write_config) > defecto.
Los puertos NO tienen defecto: si faltan, el módulo que los necesita lo dice.

  Cámaras:   CHESS_YOLO_CAM (o SO101_TOP_CAM), CHESS_WRIST_CAM (o SO101_WRIST_CAM)
  Visión:    CHESS_BOARD_GEOMETRY (esquinas en píxeles), CHESS_YOLO_WEIGHTS
  ACT:       CHESS_ACT_CHECKPOINT (genérico, sin objetivo), CHESS_ACT_GOAL_CHECKPOINT (con objetivo), ACT_HF_30K, ACT_LOCAL_25K
  Brazo:     SO101_FOLLOWER_PORT, SO101_FOLLOWER_ID, CHESS_STEP_DEG (grados por paso), CHESS_HZ, CHESS_GRIP_OPEN
  Partida:   CHESS_GAME_FILE
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HUB = Path(__file__).resolve().parent
ROOT = HUB.parent
sys.path.insert(0, str(ROOT / "real"))
import config as _real_config  # noqa: E402  (carga real/env.local.sh en os.environ)

_real_config.load_local_env()


def _int(*names: str, default: int) -> int:
    for n in names:
        if os.environ.get(n, "") != "":
            return int(os.environ[n])
    return default


def _first_existing(*candidates: Path) -> Path:
    """Primera ruta con un checkpoint REAL (config.json y pesos de más de 1 MB: un puntero de Git LFS sin descargar no vale)."""
    for c in candidates:
        w = c / "model.safetensors"
        if (c / "config.json").exists() and (not w.exists() or w.stat().st_size > 1_000_000):
            return c
    return candidates[0]


YOLO_CAM = _int("CHESS_YOLO_CAM", "SO101_TOP_CAM", default=0)
WRIST_CAM = _int("CHESS_WRIST_CAM", "SO101_WRIST_CAM", default=1)
FOLLOWER_PORT = os.environ.get("SO101_FOLLOWER_PORT", "")
FOLLOWER_ID = os.environ.get("SO101_FOLLOWER_ID", "so101_follower")
HZ = float(os.environ.get("CHESS_HZ", "30"))
# Nunca por encima del límite global de seguridad (SO101_MAX_RELATIVE_TARGET, 2 por defecto).
MAX_STEP_DEG = float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2"))
STEP_DEG = min(float(os.environ.get("CHESS_STEP_DEG", MAX_STEP_DEG)), MAX_STEP_DEG)

GAME_FILE = Path(os.environ.get("CHESS_GAME_FILE", HUB / "chess_game.json"))
BOARD_GEOMETRY = Path(os.environ.get("CHESS_BOARD_GEOMETRY", ROOT / "vision" / "board_camara0.json"))
YOLO_WEIGHTS = Path(os.environ.get("CHESS_YOLO_WEIGHTS",
                                   ROOT / "runs/detect/vision/runs/piezas_camara_v2/weights/best.pt"))
# ACT genérico (SIN objetivo: 6 articulaciones + cámaras front y gripper a 640x360). Los pesos van en el repo con Git LFS
# (models/act/<nombre>/, `git lfs pull` en el host). Ver pipeline/MODELOS_ACT.md. Si no están, se buscan en las rutas
# donde se entrenaron/descargaron.
def _model(env: str, *candidates: Path) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else _first_existing(*candidates)


ACT_MODELS: dict[str, Path] = {
    # El mismo modelo de 30k (HF PIPEFD/act_tray_a100_100k_ckpts, paso 30000) puede estar en tres sitios, en este orden:
    # en el repo con Git LFS, descargado en el host (~/act_tray_ckpts) o en el submódulo de HF (act_tray_a100_100k_ckpts/).
    "hf-30k-cuda": _model("ACT_HF_30K", ROOT / "models/act/hf-30k-cuda",
                          Path.home() / "act_tray_ckpts/checkpoints/030000/pretrained_model",
                          ROOT / "act_tray_a100_100k_ckpts/checkpoints/030000/pretrained_model"),
    "local-25k-mps": _model("ACT_LOCAL_25K", ROOT / "models/act/local-25k-mps",
                            ROOT / "outputs/train/act_tray_clean_25000/checkpoints/025000/pretrained_model"),
}
ACT_GENERIC_CKPT = (Path(os.environ["CHESS_ACT_CHECKPOINT"]) if os.environ.get("CHESS_ACT_CHECKPOINT")
                    else _first_existing(*ACT_MODELS.values(), ROOT / "modeloACT/checkpoints/last/pretrained_model"))
ACT_GOAL_CKPT = Path(os.environ["CHESS_ACT_GOAL_CHECKPOINT"]) if os.environ.get("CHESS_ACT_GOAL_CHECKPOINT") else None

# Apertura de la pinza (0-100) al acercarse a la pieza. SIN medir con piezas reales. El ACT entrenado vio como mucho ~32.
GRIP_OPEN = float(os.environ.get("CHESS_GRIP_OPEN", "40"))

MOTION_FILE = ROOT / "real" / ".motion_ok"


def motion_permission() -> tuple[bool, str]:
    """Gate único de movimiento (MCP y ejecutor IK). Política del 21-nov (decisión del usuario): movimiento directo.
      - ALLOW_REAL_MOTION=0  -> parada: nada se mueve (interruptor de emergencia por software).
      - CHESS_REQUIRE_PERMISSION=1 -> vuelve el modo con permiso: ALLOW_REAL_MOTION=1 + real/authorize_motion.sh.
    Los límites mecánicos (rango, grados por paso, atasco) están en el código y no dependen de esto."""
    if os.environ.get("ALLOW_REAL_MOTION") == "0":
        return False, "parada activada (ALLOW_REAL_MOTION=0)."
    if os.environ.get("CHESS_REQUIRE_PERMISSION") != "1":
        return True, "movimiento directo (sin permiso)"
    if os.environ.get("ALLOW_REAL_MOTION") != "1":
        return False, "ALLOW_REAL_MOTION=1 no está puesto en el entorno."
    if not MOTION_FILE.exists():
        return False, "Modo con permiso (CHESS_REQUIRE_PERMISSION=1): falta real/authorize_motion.sh <minutos>."
    try:
        expiry = float(MOTION_FILE.read_text().strip())
    except ValueError:
        return False, "real/.motion_ok no es válido."
    if time.time() > expiry:
        return False, "El permiso de movimiento ha caducado."
    return True, "ok"


def require_port() -> str:
    if not FOLLOWER_PORT:
        raise RuntimeError("Falta SO101_FOLLOWER_PORT: ponla en real/env.local.sh (so101-hub write_config) o en el entorno.")
    return FOLLOWER_PORT
