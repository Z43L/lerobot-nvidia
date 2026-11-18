"""Wrapper de simulación del SO-101 en MuJoCo.

Expone una API pequeña (estado, mover articulaciones, cámara, reset) que luego
puede usar el MCP. Mismo contrato que tendrá el brazo real, para cambiar de
backend sin tocar la capa de Claude.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

SCENE = Path(__file__).parent / "so101" / "scene_vla.xml"  # cámaras top/wrist + caja
FRAMES_DIR = Path(__file__).parent / "frames"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Límite de cambio por llamada, en radianes. Evita saltos bruscos al brazo.
MAX_STEP_RAD = 0.5
# Pasos de simulación por segundo de movimiento interpolado.
SETTLE_STEPS = 200


class SimArm:
    def __init__(self, camera_size: tuple[int, int] = (480, 640), scene: Path = SCENE):
        self.viewer = None  # visor gráfico en vivo (ver attach_viewer)
        self._last_sync = 0.0
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.camera_size = camera_size
        self._renderer = None
        self.home_qpos = np.zeros(6)
        self.reset()

    # ------------------------------------------------------------------ estado
    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self.home_qpos
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self) -> dict:
        q = self.data.qpos[:6].copy()
        return {
            "joints": {n: round(float(v), 4) for n, v in zip(JOINT_NAMES, q)},
            "limits": {
                n: [round(float(lo), 4), round(float(hi), 4)]
                for n, (lo, hi) in zip(JOINT_NAMES, self.model.jnt_range[:6])
            },
        }

    def _cube_pos(self) -> list[float]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        return [round(float(v), 4) for v in self.data.xpos[bid]]

    def get_scene_summary(self) -> dict:
        """Estado del brazo y posición del cubo."""
        return {"state": self.get_state(), "cube_position_m": self._cube_pos()}

    # ----------------------------------------------------------- movimiento
    def move_joints(self, targets: list[float], max_chunks: int = 40) -> dict:
        """Mueve las 6 articulaciones a `targets` (radianes).

        Los valores se recortan a los límites de cada articulación. El movimiento
        se hace en tramos de como mucho MAX_STEP_RAD por tramo, para que sea suave,
        hasta llegar al objetivo o agotar `max_chunks`.
        """
        if len(targets) != 6:
            raise ValueError("Hacen falta exactamente 6 valores (uno por articulación).")

        target = np.asarray(targets, dtype=float)
        lo = self.model.jnt_range[:6, 0]
        hi = self.model.jnt_range[:6, 1]
        clipped = np.clip(target, lo, hi)
        was_clipped = not np.allclose(clipped, target)

        reached = False
        for _ in range(max_chunks):
            current = self.data.qpos[:6].copy()
            if np.allclose(current, clipped, atol=2e-3):
                reached = True
                break
            goal = current + np.clip(clipped - current, -MAX_STEP_RAD, MAX_STEP_RAD)
            self.data.ctrl[:6] = goal
            for _ in range(SETTLE_STEPS):
                mujoco.mj_step(self.model, self.data)

        reached = bool(np.allclose(self.data.qpos[:6], clipped, atol=2e-3))
        return {
            "requested": [round(float(x), 4) for x in target],
            "target_after_limits": [round(float(x), 4) for x in clipped],
            "was_clipped": bool(was_clipped),
            "reached": bool(reached),
            "state": self.get_state(),
        }

    def home(self) -> dict:
        return self.move_joints(self.home_qpos.tolist())

    # ------------------------------------------------------- visor en vivo
    def attach_viewer(self, realtime: bool = True):
        """Abre la ventana de MuJoCo (requiere mjpython en macOS, o el hilo principal)."""
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.realtime = realtime
        return self.viewer

    def sync_view(self, dt: float = 1.0 / 30) -> None:
        """Refresca la ventana tras cada paso de control y, si realtime, espera para ir a tiempo real."""
        import time

        if self.viewer is None or not self.viewer.is_running():
            return
        self.viewer.sync()
        if getattr(self, "realtime", True):
            now = time.time()
            wait = dt - (now - self._last_sync)
            if wait > 0:
                time.sleep(wait)
        self._last_sync = time.time()

    # --------------------------------------------------------------- cámara
    def render(self, camera: str = "top") -> Path:
        if self._renderer is None:
            h, w = self.camera_size
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=camera)
        img = self._renderer.render()
        FRAMES_DIR.mkdir(exist_ok=True)
        path = FRAMES_DIR / f"{camera}.png"
        Image.fromarray(img).save(path)
        return path
