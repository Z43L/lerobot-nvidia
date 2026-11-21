"""Carga y predicción offline del checkpoint ACT de modeloACT/.

No conecta con el robot ni envía comandos. Entrada y salida en grados (use_degrees),
gripper en 0..100. Pensado para ser llamado por scripts de prueba o por Claude vía act_cli.py.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = Path(os.environ.get("ACT_CKPT") or ROOT / "modeloACT" / "checkpoints" / "last" / "pretrained_model")
# Dataset local de SenseCraft (opcional, solo para comparar con el entrenamiento). Sin ruta por defecto: ACT_DATASET_ROOT.
DATASET_ROOT = Path(os.environ["ACT_DATASET_ROOT"]) if os.environ.get("ACT_DATASET_ROOT") else None
DATASET_REPO = "local/data play tray 1"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERAS = {"front": "observation.images.front", "gripper": "observation.images.gripper"}


class ACTOffline:
    def __init__(self, device: str | None = None, ckpt: Path = CKPT):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device or os.environ.get("SO101_VLA_DEVICE", "mps")
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        t0 = time.time()
        self.policy = ACTPolicy.from_pretrained(str(ckpt))
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.load_seconds = round(time.time() - t0, 2)

    def reset(self) -> None:
        self.policy.reset()

    def predict_chunk(self, state, images: dict, task: str = "pick and place") -> np.ndarray:
        """state: 6 floats en grados. images: {'front': HWC uint8, 'gripper': HWC uint8}.
        Devuelve array (chunk, 6) en grados, con la pinza en 0..100."""
        batch = {
            "observation.state": torch.tensor(np.asarray(state, dtype=np.float32)),
            "task": task,
        }
        for name, key in CAMERAS.items():
            img = np.asarray(images[name]).astype(np.float32) / 255.0
            batch[key] = torch.from_numpy(img).permute(2, 0, 1)
        with torch.inference_mode():
            self.policy.reset()
            # select_action devuelve una acción por paso; reunimos el chunk completo
            # llamando a predict_action_chunk sobre el batch ya preprocesado.
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)  # (1, chunk, 6) normalizado
            chunk = self.post(chunk.squeeze(0) if chunk.dim() == 3 else chunk)
        return np.asarray(chunk.detach().cpu().numpy()).reshape(-1, 6)
