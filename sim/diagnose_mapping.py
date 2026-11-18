"""Diagnóstico del mapeo rad <-> escala normalizada del VLA.

Ejecuta unos pasos y registra: estado normalizado, acción predicha (normalizada),
error de seguimiento (ctrl vs qpos) y posición del cubo. Compara con los
cuantiles del dataset del checkpoint (estadísticas en la model card / preprocesador).
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).parent))
from sim_arm import SimArm
from vla_policy import VLARunner

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# Cuantiles q01/q99 del dataset (estadísticas de observation.state, en escala normalizada)
DATASET_Q01 = np.array([-21.88, -105.32, -33.65, 39.31, -6.75, 2.16])
DATASET_Q99 = np.array([12.05, 43.55, 94.87, 86.19, 20.94, 33.47])

arm = SimArm()
run = VLARunner(arm)
run._load()
run.policy.reset()
import torch
rec = {"state": [], "action": [], "track_err": []}
substeps = max(1, int(round(1.0 / 30 / arm.model.opt.timestep)))
with torch.inference_mode():
    for _ in range(STEPS):
        obs = run._observation("Pick up a cube and place in the bin")
        batch = run.pre(obs)
        a = run.post(run.policy.select_action(batch)).squeeze(0).cpu().numpy()[:6]
        rec["state"].append(run.rad_to_norm(arm.data.qpos[:6].copy()))
        rec["action"].append(a)
        arm.data.ctrl[:6] = run.norm_to_rad(a)
        for _ in range(substeps):
            mujoco.mj_step(arm.model, arm.data)
        rec["track_err"].append(np.abs(arm.data.ctrl[:6] - arm.data.qpos[:6]))

S = np.array(rec["state"]); A = np.array(rec["action"]); E = np.array(rec["track_err"])
names = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
print("joint        state_min  state_max | action_min action_max | dataset_q01 q99 | rad_err_mean")
for i, n in enumerate(names):
    print(f"{n:12s} {S[:,i].min():9.1f} {S[:,i].max():9.1f} | {A[:,i].min():9.1f} {A[:,i].max():9.1f} | "
          f"{DATASET_Q01[i]:7.1f} {DATASET_Q99[i]:6.1f} | {E[:,i].mean():.4f}")
print("first action vs first state (norm):", np.round(A[0] - S[0], 1))
print("mean |action - state| (norm):", np.round(np.abs(A - S).mean(0), 1))
print("cube:", np.round(arm.data.body("cube").xpos, 3).tolist())
json.dump({"state": S.tolist(), "action": A.tolist()}, open(Path(__file__).parent / "frames" / "diag.json", "w"))
