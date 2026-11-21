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
    for c in candidates:
        if (c / "config.json").exists():
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
# ACT genérico (SIN objetivo: 6 articulaciones + cámaras front y gripper a 640x360). Modelos conocidos, en orden de
# preferencia. Ninguno está en git (los pesos pesan ~200 MB): se copian al host con rsync (ver pipeline/MODELOS_ACT.md).
ACT_MODELS: dict[str, Path] = {
    "hf-30k-cuda": Path(os.environ.get("ACT_HF_30K", Path.home() / "act_tray_ckpts/checkpoints/030000/pretrained_model")),
    "local-25k-mps": Path(os.environ.get("ACT_LOCAL_25K", ROOT / "outputs/train/act_tray_clean_25000/checkpoints/025000/pretrained_model")),
}
ACT_GENERIC_CKPT = (Path(os.environ["CHESS_ACT_CHECKPOINT"]) if os.environ.get("CHESS_ACT_CHECKPOINT")
                    else _first_existing(*ACT_MODELS.values(), ROOT / "modeloACT/checkpoints/last/pretrained_model"))
ACT_GOAL_CKPT = Path(os.environ["CHESS_ACT_GOAL_CHECKPOINT"]) if os.environ.get("CHESS_ACT_GOAL_CHECKPOINT") else None

# Apertura de la pinza (0-100) al acercarse a la pieza. SIN medir con piezas reales. El ACT entrenado vio como mucho ~32.
GRIP_OPEN = float(os.environ.get("CHESS_GRIP_OPEN", "40"))

MOTION_FILE = ROOT / "real" / ".motion_ok"


def motion_permission() -> tuple[bool, str]:
    """Gate único de movimiento: ALLOW_REAL_MOTION=1 (lo pone el usuario) + permiso con caducidad que crea el usuario
    con real/authorize_motion.sh. Lo usan el MCP y el ejecutor IK; antes cada uno tenía su copia.
    Por defecto (21-nov, decisión del usuario) el movimiento es directo: el permiso solo se exige con CHESS_REQUIRE_PERMISSION=1."""
    if os.environ.get("CHESS_REQUIRE_PERMISSION") != "1":
        return True, "movimiento directo (sin permiso)"
    if os.environ.get("ALLOW_REAL_MOTION") != "1":
        return False, "ALLOW_REAL_MOTION=1 no está puesto en el entorno."
    if not MOTION_FILE.exists():
        return False, ("Sin permiso del usuario. Debe ejecutar en su terminal: real/authorize_motion.sh <minutos> "
                       "(tras confirmar área libre, persona junto al brazo y corte de alimentación a mano).")
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
