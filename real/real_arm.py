"""Backend del SO-101 real para el MCP. Mismo nombre de métodos que sim/sim_arm.py.

Unidades: el checkpoint SmolVLA se entrenó con posiciones en escala normalizada de LeRobot
(articulaciones en [-100, 100], gripper en [0, 100]). Por eso aquí todo va en esa escala y
la respuesta incluye "units": "normalized". En sim las unidades son radianes.

Seguridad:
- LeRobot limita el cambio por paso con max_relative_target (SO101_MAX_RELATIVE_TARGET).
- move_joints recorta a [-100, 100] y avanza en tramos pequeños.
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
TRANSIT_STEP = 4.0  # máximo cambio por tramo (normalizado)
CONTROL_HZ = 30
FRAMES_DIR = Path(__file__).resolve().parent / "frames"


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
        ))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise RuntimeError("El seguidor no tiene calibración. Ejecuta real/calibrate.sh o usa la del organizador.")
        self._last_obs = None

    # ------------------------------------------------------------ estado
    def _obs(self) -> dict:
        self._last_obs = self.robot.get_observation()
        return self._last_obs

    def get_state(self) -> dict:
        obs = self._obs()
        joints = {n: round(float(obs[f"{n}.pos"]), 2) for n in JOINT_NAMES}
        return {"units": "normalized", "joints": joints, "limits": {n: [-100, 100] if n != "gripper" else [0, 100] for n in JOINT_NAMES}}

    def get_scene_summary(self) -> dict:
        # En el real no hay estado de objetos: la cámara es la fuente de verdad.
        return {"state": self.get_state(), "cube_position_m": None, "note": "Sin posición de objetos en el real; usa get_camera_image."}

    def reset(self) -> dict:
        return self.home()

    # --------------------------------------------------------- movimiento
    def move_joints(self, targets: list[float], max_chunks: int = 60) -> dict:
        if len(targets) != 6:
            raise ValueError("Hacen falta 6 valores normalizados (uno por articulación).")
        goal = np.clip(np.asarray(targets, dtype=float), [-100] * 5 + [0], [100] * 5 + [100])
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
        return {"units": "normalized", "requested": [float(x) for x in targets], "reached": reached, "state": self.get_state()}

    def home(self) -> dict:
        # Posición de reposo: centro del rango. Ajusta según tu calibración y el espacio libre.
        return self.move_joints([0, 0, 0, 0, 0, 10])

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
    def run_skill(self, instruction: str, duration_s: int = 60) -> dict:
        """Lanza el VLA en el brazo real con real/rollout.sh. Requiere ALLOW_REAL_MOTION=1."""
        if os.environ.get("ALLOW_REAL_MOTION") != "1":
            return {"status": "blocked", "message": "Movimiento real desactivado. Pon ALLOW_REAL_MOTION=1 y confirma que el área está libre."}
        script = Path(__file__).resolve().parent / "rollout.sh"
        policy = os.environ.get("SO101_POLICY", config.REFERENCE_POLICY)
        self.robot.disconnect()  # rollout.sh abre su propia conexión
        proc = subprocess.Popen([str(script), policy, instruction, str(duration_s)])
        return {"status": "started", "pid": proc.pid, "instruction": instruction, "policy": policy, "duration_s": duration_s}

    def close(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
