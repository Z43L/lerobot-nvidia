"""Backend del SO-101 real para el MCP. Mismo nombre de métodos que sim/sim_arm.py.

Unidades (igual que el dataset de LeRobot y el checkpoint SmolVLA):
- 5 articulaciones del brazo: grados (config.USE_DEGREES, por defecto 1).
- gripper: escala 0 (cerrada) a 100 (abierta), siempre.
La respuesta indica "units": "degrees" o "normalized" según la configuración.
En sim, las articulaciones van en radianes y la conversión a grados está en sim/vla_policy.py.

Seguridad:
- LeRobot limita el cambio por paso con max_relative_target (SO101_MAX_RELATIVE_TARGET).
- move_joints recorta a los límites de unidades y avanza en tramos pequeños.
- run_skill solo arranca si ALLOW_REAL_MOTION=1 está puesto explícitamente.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TRANSIT_STEP = 4.0  # máximo cambio por tramo (unidades de LeRobot: ~4 grados)
CONTROL_HZ = 30
FRAMES_DIR = Path(__file__).resolve().parent / "frames"

# Envolvente de seguridad por articulación, en grados. Se deriva de los rangos que vio el
# checkpoint en su entrenamiento (estadísticas observation.state, min/max) con margen.
# Evita órdenes que lleven el brazo contra un tope mecánico. Ajústala con la calibración real.
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-100.0, 100.0),   # datos: -81..58
    "shoulder_lift": (-120.0, 120.0),  # datos: -110..110
    "elbow_flex": (-110.0, 110.0),     # datos: -97..98
    "wrist_flex": (-115.0, 115.0),     # datos: -98..107
    "wrist_roll": (-180.0, 180.0),     # datos: -83..171
}
GRIPPER_LIMIT = (0.0, 100.0)
UNITS = "degrees" if config.USE_DEGREES else "normalized"
LO = np.array([JOINT_LIMITS_DEG[n][0] for n in JOINT_NAMES[:5]] + [GRIPPER_LIMIT[0]])
HI = np.array([JOINT_LIMITS_DEG[n][1] for n in JOINT_NAMES[:5]] + [GRIPPER_LIMIT[1]])


class RealArm:
    def __init__(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.cameras.opencv import OpenCVCameraConfig

        if not config.FOLLOWER_PORT:
            raise RuntimeError("Falta SO101_FOLLOWER_PORT. Revisa real/env.local.sh.")
        cams = {
            "top": OpenCVCameraConfig(index_or_path=config.TOP_CAM, width=config.CAM_WIDTH, height=config.CAM_HEIGHT, fps=config.FPS),
            "wrist": OpenCVCameraConfig(index_or_path=config.WRIST_CAM, width=config.CAM_WIDTH, height=config.CAM_HEIGHT, fps=config.FPS),
        }
        if config.TABLE_CAM >= 0:
            cams["table"] = OpenCVCameraConfig(index_or_path=config.TABLE_CAM, width=config.CAM_WIDTH, height=config.CAM_HEIGHT, fps=config.FPS)
        self.robot = SO101Follower(SO101FollowerConfig(
            port=config.FOLLOWER_PORT,
            id=config.FOLLOWER_ID,
            cameras=cams,
            max_relative_target=config.MAX_RELATIVE_TARGET,
            use_degrees=config.USE_DEGREES,
        ))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise RuntimeError("El seguidor no tiene calibración. Ejecuta real/calibrate.sh o usa la del organizador.")
        self._last_obs = None
        self.has_table = config.TABLE_CAM >= 0
        self._vlas: dict[str, object] = {}

    def policy_for(self, disk: str) -> str:
        """Checkpoint del VLA de ese color: variable SO101_HANOI_VLA_<COLOR> (carpeta pretrained_model)."""
        path = os.environ.get(f"SO101_HANOI_VLA_{disk.upper()}")
        if not path:
            raise RuntimeError(f"Falta SO101_HANOI_VLA_{disk.upper()} con el checkpoint del VLA de {disk}")
        return path

    # ------------------------------------------------------------ estado
    def _obs(self) -> dict:
        self._last_obs = self.robot.get_observation()
        return self._last_obs

    def get_state(self) -> dict:
        obs = self._obs()
        joints = {n: round(float(obs[f"{n}.pos"]), 2) for n in JOINT_NAMES}
        limits = {n: list(JOINT_LIMITS_DEG[n]) for n in JOINT_NAMES[:5]}
        limits["gripper"] = list(GRIPPER_LIMIT)
        return {"units": UNITS, "joints": joints, "limits": limits}

    def get_scene_summary(self) -> dict:
        # En el real no hay estado de objetos: la cámara es la fuente de verdad.
        return {"state": self.get_state(), "cube_position_m": None, "note": "Sin posición de objetos en el real; usa get_camera_image."}

    def reset(self) -> dict:
        return self.home()

    # --------------------------------------------------------- movimiento
    @staticmethod
    def motion_allowed() -> bool:
        return os.environ.get("ALLOW_REAL_MOTION") != "0"

    BLOCKED = {"status": "blocked", "message": "Movimiento real desactivado. Pon ALLOW_REAL_MOTION=1 y confirma que el área está libre."}

    def move_joints(self, targets: list[float], max_chunks: int = 60) -> dict:
        if not self.motion_allowed():
            return dict(self.BLOCKED)
        if len(targets) != 6:
            raise ValueError(f"Hacen falta 6 valores ({UNITS} en el brazo, 0..100 en la pinza).")
        goal = np.clip(np.asarray(targets, dtype=float), LO, HI)
        reached = False
        for _ in range(max_chunks):
            cur = np.array([self._obs()[f"{n}.pos"] for n in JOINT_NAMES], dtype=float)
            if np.allclose(cur, goal, atol=1.0):
                reached = True
                break
            step = np.clip(goal - cur, -TRANSIT_STEP, TRANSIT_STEP)
            action = {f"{n}.pos": float(cur[i] + step[i]) for i, n in enumerate(JOINT_NAMES)}
            self.robot.send_action(action)
            time.sleep(1.0 / CONTROL_HZ)
        return {"units": UNITS, "requested": [float(x) for x in targets], "reached": reached, "state": self.get_state()}

    def home(self) -> dict:
        # Reposo: 0 grados = centro del rango calibrado (LeRobot). Confírmalo con el brazo y el espacio libre.
        return self.move_joints([0, 0, 0, 0, 0, 10])

    def grip_update(self, grip_rad: float) -> None:  # interfaz común con sim (no aplica en real)
        return None

    def get_hanoi_state(self) -> dict:  # interfaz común con sim (el real usa real/hanoi_board.py)
        raise NotImplementedError("El estado de la torre en real lo lleva real/hanoi_board.py")

    # ------------------------------------------------------------- cámara
    def render(self, camera: str = "top") -> Path:
        obs = self._obs()
        key = camera
        if key not in obs:
            raise ValueError(f"Cámara desconocida o no configurada: {camera}. Disponibles: {[k for k in ('top','wrist','table') if k in obs]}")
        FRAMES_DIR.mkdir(exist_ok=True)
        path = FRAMES_DIR / f"real_{camera}.png"
        Image.fromarray(np.asarray(obs[key])).save(path)
        return path

    # -------------------------------------------------------------- VLA
    def run_skill(self, instruction: str, seconds: float = 20.0, repo_id: str | None = None) -> dict:
        """Ejecuta el VLA en el brazo real, dentro de este proceso (real/real_vla.py).
        Requiere ALLOW_REAL_MOTION=1. Corta por tiempo; no hay sensor de éxito: la verificación es visual."""
        if not self.motion_allowed():
            return dict(self.BLOCKED)
        from real_vla import RealVLARunner

        repo = repo_id or os.environ.get("SO101_POLICY", config.REFERENCE_POLICY)
        if repo not in self._vlas:  # un VLA por color; se cargan al primer uso
            self._vlas[repo] = RealVLARunner(self.robot, repo, device=os.environ.get("SO101_VLA_DEVICE", "mps"))
        res = self._vlas[repo].run(instruction, seconds=seconds)
        return {"status": res.stopped_by, "steps": res.steps, "seconds": res.seconds,
                "instruction": instruction, "policy": repo}

    def close(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
