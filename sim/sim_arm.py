"""Wrapper de simulación del SO-101 en MuJoCo.

Expone una API pequeña (estado, mover articulaciones, cámara, reset) que luego
puede usar el MCP. Mismo contrato que tendrá el brazo real, para cambiar de
backend sin tocar la capa de Claude.

Paridad con el brazo real:
- Velocidad: cada paso de control dura 1/30 s de reloj y avanza como mucho
  SO101_MAX_RELATIVE_TARGET grados, igual que el seguidor real.
- Calibración: si existe el fichero de calibración de LeRobot del seguidor
  (~/.cache/huggingface/lerobot/calibration/.../<SO101_FOLLOWER_ID>.json), los límites
  de cada articulación salen de sus rangos grabados con la misma fórmula de LeRobot:
  grados = (lectura - centro_del_rango) * 360 / 4095. Sin fichero, se usan los límites del modelo.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

SCENE = Path(__file__).parent / "so101" / "scene_vla.xml"  # cámaras top/wrist + caja
FRAMES_DIR = Path(__file__).parent / "frames"
STATE_FILE = FRAMES_DIR / "view_state.json"  # pose publicada para la ventana externa (--follow)
CAL_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
STS3215_RESOLUTION = 4096

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

CONTROL_HZ = 30
TICK_S = 1.0 / CONTROL_HZ
REACH_TOL_RAD = 0.01   # ~0,6°: se considera llegado
MAX_TICKS = 900        # 30 s como máximo por movimiento


def max_deg_per_tick() -> float:
    """Avance máximo por paso de control, en grados. Mismo límite que el seguidor real."""
    return float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2"))


def load_real_calibration(robot_id: str | None = None) -> dict | None:
    """Lee la calibración del seguidor real guardada por LeRobot. None si no existe."""
    rid = robot_id or os.environ.get("SO101_FOLLOWER_ID", "so101_follower")
    files = glob.glob(str(CAL_ROOT / "**" / f"{rid}.json"), recursive=True)
    if not files:
        return None
    try:
        return json.loads(Path(files[0]).read_text())
    except (OSError, json.JSONDecodeError):
        return None


class SimArm:
    def __init__(self, camera_size: tuple[int, int] = (480, 640), scene: Path = SCENE):
        self.viewer = None  # visor gráfico en vivo (ver attach_viewer)
        # Cerrojo: el hilo del MCP y el hilo principal (visor) no pueden tocar MuJoCo a la vez.
        # Sin él, viewer.sync() copia mjData mientras mj_step lo usa y el proceso muere.
        self.lock = threading.RLock()
        self._last_sync = 0.0
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.camera_size = camera_size
        self._renderer = None
        self.home_qpos = np.zeros(6)
        self.calibration = load_real_calibration()
        self.reset()

    # ------------------------------------------------------------------ límites
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Límites (rad) de las 6 articulaciones. Si hay calibración real, se usan sus rangos grabados."""
        lo = self.model.jnt_range[:6, 0].copy()
        hi = self.model.jnt_range[:6, 1].copy()
        if not self.calibration:
            return lo, hi
        for i, name in enumerate(JOINT_NAMES[:5]):
            cal = self.calibration.get(name)
            if not cal:
                continue
            mn, mx = float(cal["range_min"]), float(cal["range_max"])
            mid = (mn + mx) / 2
            deg_lo = (mn - mid) * 360 / (STS3215_RESOLUTION - 1)
            deg_hi = (mx - mid) * 360 / (STS3215_RESOLUTION - 1)
            lo[i] = max(lo[i], np.radians(deg_lo))
            hi[i] = min(hi[i], np.radians(deg_hi))
        return lo, hi

    # ------------------------------------------------------------------ estado
    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self.home_qpos
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self) -> dict:
        q = self.data.qpos[:6].copy()
        lo, hi = self.joint_limits()
        return {
            "joints": {n: round(float(v), 4) for n, v in zip(JOINT_NAMES, q)},
            "limits": {n: [round(float(a), 4), round(float(b), 4)] for n, a, b in zip(JOINT_NAMES, lo, hi)},
            "calibrated": self.calibration is not None,
        }

    def guard(self) -> None:
        """Hace que cada método público tome el cerrojo. Así las herramientas del MCP y la
        sincronización del visor nunca tocan MuJoCo a la vez.
        move_joints y home toman el cerrojo por paso, para que el visor pueda dibujar mientras se mueven."""
        skip = {"attach_viewer", "sync_view", "publish_state", "guard", "move_joints", "home"}
        for name in dir(type(self)):
            if name.startswith("_") or name in skip:
                continue
            attr = getattr(type(self), name, None)
            if not callable(attr):
                continue
            setattr(self, name, self._locked(getattr(self, name)))

    def _locked(self, fn):
        def wrapper(*args, **kwargs):
            with self.lock:
                return fn(*args, **kwargs)
        return wrapper

    def publish_state(self) -> None:
        """Escribe la pose completa en STATE_FILE para que una ventana externa (objects_env.py --follow)
        la muestre. Solo si SO101_PUBLISH_STATE=1 (lo pone el servidor MCP, no el visor). Escritura atómica."""
        if os.environ.get("SO101_PUBLISH_STATE") != "1":
            return
        STATE_FILE.parent.mkdir(exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"t": time.time(), "qpos": [float(v) for v in self.data.qpos]}))
        tmp.replace(STATE_FILE)

    def _cube_pos(self) -> list[float]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        return [round(float(v), 4) for v in self.data.xpos[bid]]

    def get_scene_summary(self) -> dict:
        """Estado del brazo y posición del cubo."""
        return {"state": self.get_state(), "cube_position_m": self._cube_pos()}

    # ----------------------------------------------------------- movimiento
    def move_joints(self, targets: list[float], max_ticks: int = MAX_TICKS) -> dict:
        """Mueve las 6 articulaciones a `targets` (radianes), a la velocidad del seguidor real.

        Cada paso de control (1/30 s de reloj) avanza como mucho max_deg_per_tick() grados por
        articulación. Los objetivos se recortan a los límites (calibración real si existe).
        """
        if len(targets) != 6:
            raise ValueError("Hacen falta exactamente 6 valores (uno por articulación).")

        target = np.asarray(targets, dtype=float)
        lo, hi = self.joint_limits()
        clipped = np.clip(target, lo, hi)
        was_clipped = not np.allclose(clipped, target)
        step_max = np.radians(max_deg_per_tick())
        substeps = max(1, round(TICK_S / self.model.opt.timestep))

        ticks = 0
        while ticks < max_ticks:
            with self.lock:
                cur = self.data.qpos[:6].copy()
                if np.allclose(cur, clipped, atol=REACH_TOL_RAD):
                    break
                step = np.clip(clipped - cur, -step_max, step_max)
                self.data.ctrl[:6] = cur + step
                for _ in range(substeps):
                    mujoco.mj_step(self.model, self.data)
                self.publish_state()
            ticks += 1
            time.sleep(TICK_S)  # reloj real, fuera del cerrojo para que el visor se actualice

        reached = bool(np.allclose(self.data.qpos[:6], clipped, atol=REACH_TOL_RAD))
        return {
            "requested": [round(float(x), 4) for x in target],
            "target_after_limits": [round(float(x), 4) for x in clipped],
            "was_clipped": bool(was_clipped),
            "reached": reached,
            "ticks": ticks,
            "seconds": round(ticks * TICK_S, 2),
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

    def sync_view(self, dt: float = TICK_S) -> None:
        """Refresca la ventana tras cada paso de control y, si realtime, espera para ir a tiempo real."""
        if self.viewer is None or not self.viewer.is_running():
            return
        # El visor solo se sincroniza desde el hilo principal y con el cerrojo tomado.
        if threading.current_thread() is threading.main_thread():
            with self.lock:
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
