"""Diagnóstico del harness (no mueve el brazo).

a) Roundtrip de normalización: desnormalizar(normalizar(acción real)) == acción real.
b) Sensibilidad a imágenes: ¿la predicción cambia si cambian las imágenes?
c) Sensibilidad al estado: ¿la predicción cambia si cambia el estado?
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from act_model import DATASET_REPO, DATASET_ROOT, JOINTS, ACTOffline  # noqa: E402
from test_act import to_uint8  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

m = ACTOffline()
if DATASET_ROOT is None:
    raise SystemExit("Falta ACT_DATASET_ROOT: ruta del dataset local de SenseCraft (ya no hay una ruta de usuario fija).")
ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, video_backend="pyav")
r = 5000
s = ds[r]
gt = np.asarray(ds.hf_dataset["action"][r:r + 100], dtype=np.float32)
imgs = {"front": to_uint8(s["observation.images.front"].numpy()),
        "gripper": to_uint8(s["observation.images.gripper"].numpy())}
state = s["observation.state"].numpy()

# a) roundtrip
norm_step = m.pre.steps[3]
unnorm_step = m.post.steps[0]
act_stats = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in unnorm_step.stats["action"].items()}
gt_t = torch.from_numpy(gt)
norm = (gt_t - act_stats["mean"]) / (act_stats["std"] + 1e-8)
back = norm * (act_stats["std"] + 1e-8) + act_stats["mean"]
print("a) roundtrip max abs err (deg):", float((back - gt_t).abs().max()))
print("   stats action mean:", [round(float(x), 2) for x in act_stats["mean"]])

# b) sensibilidad a imágenes
p0 = m.predict_chunk(state, imgs)
rng = np.random.default_rng(1)
p_noise = m.predict_chunk(state, {k: rng.integers(0, 255, v.shape, dtype=np.uint8) for k, v in imgs.items()})
p_black = m.predict_chunk(state, {k: np.zeros_like(v) for k, v in imgs.items()})
print("b) |Δ| medio con imágenes ruido  (deg):", np.abs(p0 - p_noise).mean(axis=0).round(2))
print("   |Δ| medio con imágenes negras (deg):", np.abs(p0 - p_black).mean(axis=0).round(2))

# c) sensibilidad al estado
p_state = m.predict_chunk(state + np.array([0, 0, 0, 0, 0, 0], dtype=np.float32) + 10.0, imgs)
print("c) |Δ| medio con estado +10° en todas (deg):", np.abs(p0 - p_state).mean(axis=0).round(2))
print("   estado real:", state.round(2), " acción real paso0:", gt[0].round(2))
