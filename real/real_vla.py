"""Ejecuta un VLA (SmolVLA) en el SO-101 real, dentro del mismo proceso que el MCP.

A diferencia de real/rollout.sh (que lanza lerobot-rollout y no devuelve nada hasta acabar),
aquí el MCP controla cada movimiento: se lee el estado y las cámaras, el modelo predice una
acción y se envía al seguidor, con límite de tiempo y parada inmediata por excepción.

Claves de cámara: el checkpoint de Hanoi espera camera1 (top), camera2 (wrist), camera3 (table).
Escala: articulaciones en grados (LeRobot use_degrees=True, como el entrenamiento) y gripper en [0, 100].
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMS_HANOI = {"observation.images.camera1": "top", "observation.images.camera2": "wrist",
              "observation.images.camera3": "table"}
CONTROL_HZ = 30


@dataclass
class VLAResult:
    steps: int
    seconds: float
    stopped_by: str  # "time" | "done" | "error"


class RealVLARunner:
    """Carga una vez el checkpoint y lo ejecuta sobre un SO101Follower ya conectado."""

    def __init__(self, robot, repo_id: str, device: str = "mps", cams: dict[str, str] | None = None):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.robot = robot
        self.repo_id = repo_id
        self.device = device
        self.cams = cams or CAMS_HANOI
        self.policy = SmolVLAPolicy.from_pretrained(repo_id)
        self.policy.config.device = device
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=repo_id,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def _observation(self, obs: dict, instruction: str) -> dict:
        state = np.array([obs[f"{n}.pos"] for n in JOINT_NAMES], dtype=np.float32)
        batch = {"observation.state": torch.from_numpy(state), "task": instruction}
        for key, cam in self.cams.items():
            img = np.asarray(obs[cam]).astype(np.float32) / 255.0  # HWC RGB
            batch[key] = torch.from_numpy(img).permute(2, 0, 1)
        return batch

    def run(self, instruction: str, seconds: float) -> VLAResult:
        """Ejecuta `instruction` durante `seconds` como máximo.

        Nunca envía nada si el robot no está conectado. Ante cualquier error, deja de enviar
        comandos y devuelve stopped_by='error' (el brazo queda en la última posición enviada)."""
        self.policy.reset()
        t0 = time.time()
        steps = 0
        stopped = "time"
        try:
            with torch.inference_mode():
                while time.time() - t0 < seconds:
                    obs = self.robot.get_observation()
                    batch = self.pre(self._observation(obs, instruction))
                    action = self.post(self.policy.select_action(batch)).squeeze(0).cpu().numpy()[:6]
                    self.robot.send_action({f"{n}.pos": float(a) for n, a in zip(JOINT_NAMES, action)})
                    steps += 1
                    time.sleep(max(0.0, 1.0 / CONTROL_HZ - 0.01))
        except KeyboardInterrupt:
            stopped = "error"
            raise
        except Exception:
            stopped = "error"
            raise
        return VLAResult(steps=steps, seconds=round(time.time() - t0, 1), stopped_by=stopped)
