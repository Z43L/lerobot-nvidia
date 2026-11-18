"""Conecta el VLA SmolVLA (checkpoint SO-101 de Hugging Face) con la simulación MuJoCo.

Flujo por paso de control:
  cámaras (top, wrist) + estado de articulaciones  ->  SmolVLA  ->  acción  ->  ctrl de MuJoCo

Convenciones (igual que el brazo real, real/real_arm.py):
  - observation.state y action: 6 valores; las 5 articulaciones en GRADOS (el valor por
    defecto de LeRobot, con el que se entrenó el checkpoint) y el gripper en [0, 100].
    La sim usa radianes internamente; aquí se convierte: deg = rad * 180 / pi.
  - cámaras: observation.images.camera1 = vista cenital ("top"),
             observation.images.camera2 = vista de muñeca ("wrist").

AVISO: el mapeo rad <-> escala normalizada usa los límites del modelo MuJoCo como
aproximación de la calibración real del brazo. El checkpoint no está validado en
hardware y se entrenó con otra cámara cenital, así que el resultado en sim es
una prueba de integración, no una medida del rendimiento real.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

# Checkpoint por defecto; con SO101_VLA_PATH se puede usar un fine-tune local (p. ej. el de Hanoi).
REPO_ID = os.environ.get("SO101_VLA_PATH", "Harrysunshine/so101-smolvla-real-9task")
CONTROL_HZ = 30
CAMERA_SIZE = (480, 640)  # alto, ancho (igual que en el dataset del checkpoint)
GRIPPER_IDX = 5


# Cámaras que espera el VLA: clave del checkpoint -> cámara de MuJoCo.
CAMS_PICK = {"observation.images.top": "top", "observation.images.wrist": "wrist"}
CAMS_HANOI = {"observation.images.camera1": "top", "observation.images.camera2": "wrist",
              "observation.images.camera3": "table"}


class VLARunner:
    def __init__(self, arm, cams: dict[str, str] | None = None, repo_id: str = REPO_ID):
        self.arm = arm
        self.repo_id = repo_id
        self.cams = cams or CAMS_PICK
        self.policy = None
        self.pre = None
        self.post = None
        self._renderer = None
        self.device = os.environ.get("SO101_VLA_DEVICE", "cpu")  # cpu | mps | cuda

    # ------------------------------------------------------------- carga
    def _load(self) -> None:
        if self.policy is not None:
            return
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        t0 = time.time()
        self.policy = SmolVLAPolicy.from_pretrained(self.repo_id)
        self.policy.config.device = self.device
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.repo_id,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        print(f"[vla] modelo cargado en {self.device} en {time.time() - t0:.1f}s", flush=True)

    # ------------------------------------------------------- conversiones
    def _limits(self):
        lo = self.arm.model.jnt_range[:6, 0]
        hi = self.arm.model.jnt_range[:6, 1]
        return lo, hi

    def rad_to_norm(self, q: np.ndarray) -> np.ndarray:
        """Radianes de la sim -> unidades del VLA/real: grados en las articulaciones, 0..100 en la pinza.
        Grados = mismo convenio que LeRobot con use_degrees=True (0 = centro del rango calibrado).
        La sim asume que su cero coincide con el centro calibrado del brazo real; confírmalo en el puesto."""
        lo, hi = self._limits()
        out = np.degrees(q).astype(np.float64)
        frac = (q - lo) / (hi - lo)
        out[GRIPPER_IDX] = np.clip(frac[GRIPPER_IDX], 0.0, 1.0) * 100.0
        return out

    def norm_to_rad(self, a: np.ndarray) -> np.ndarray:
        """Inverso de rad_to_norm: grados -> radianes; pinza 0..100 -> rango de la pinza en sim."""
        lo, hi = self._limits()
        q = np.radians(np.asarray(a, dtype=np.float64)).copy()
        frac = np.clip(np.asarray(a, dtype=np.float64)[GRIPPER_IDX] / 100.0, 0.0, 1.0)
        q[GRIPPER_IDX] = lo[GRIPPER_IDX] + frac * (hi[GRIPPER_IDX] - lo[GRIPPER_IDX])
        # límites de articulación de la sim, para no pasar del tope mecánico
        q[:5] = np.clip(q[:5], lo[:5], hi[:5])
        return q

    def _render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.arm.model, *CAMERA_SIZE)
        self._renderer.update_scene(self.arm.data, camera=camera)
        return self._renderer.render()

    def _observation(self, instruction: str) -> dict:
        state = self.rad_to_norm(self.arm.data.qpos[:6].copy()).astype(np.float32)
        obs = {"observation.state": torch.from_numpy(state), "task": instruction}
        for key, cam in self.cams.items():
            img = self._render(cam).astype(np.float32) / 255.0
            obs[key] = torch.from_numpy(img).permute(2, 0, 1)
        return obs

    # -------------------------------------------------------------- éxito
    def _cube_in_bin(self) -> bool:
        cube = self.arm.data.body("cube").xpos
        bin_xy = self.arm.data.body("bin").xpos[:2]
        dx, dy = cube[0] - bin_xy[0], cube[1] - bin_xy[1]
        return bool(abs(dx) < 0.05 and abs(dy) < 0.05 and cube[2] < 0.06)

    # ---------------------------------------------------------- ejecución
    def run(self, instruction: str, max_steps: int = 600, success=None) -> dict:
        """Ejecuta `instruction` en bucle. `success()` -> bool, si no se pasa usa 'cubo en la caja'."""
        check = success or self._cube_in_bin
        self._load()
        self.policy.reset()
        model, data = self.arm.model, self.arm.data
        substeps = max(1, int(round(1.0 / CONTROL_HZ / model.opt.timestep)))

        t0 = time.time()
        success = False
        steps_done = 0
        with torch.inference_mode():
            for step in range(max_steps):
                obs = self._observation(instruction)
                batch = self.pre(obs)
                action = self.post(self.policy.select_action(batch))
                a_norm = action.squeeze(0).detach().cpu().numpy()[:6]
                goal = self.norm_to_rad(a_norm)
                data.ctrl[:6] = goal
                if hasattr(self.arm, "grip_update"):  # pinza simulada (enganche), ver hanoi_env.py
                    self.arm.grip_update(float(goal[5]))
                for _ in range(substeps):
                    if hasattr(self.arm, "sync_held"):
                        self.arm.sync_held()
                    mujoco.mj_step(model, data)
                if hasattr(self.arm, "sync_view"):
                    self.arm.sync_view()
                steps_done = step + 1
                if check():
                    success = True
                    break

        return {
            "instruction": instruction,
            "success": success,
            "control_steps": steps_done,
            "seconds": round(time.time() - t0, 1),
            "device": self.device,
            "cube_position_m": [round(float(x), 4) for x in data.body("cube").xpos],
            "state": self.arm.get_state(),
            "note": "Sim con mapeo de unidades aproximado; checkpoint no validado en hardware.",
        }
