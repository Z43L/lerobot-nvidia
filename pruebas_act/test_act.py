"""Pruebas offline del checkpoint ACT (modeloACT/). No mueve el brazo.

Test 1 - carga y forward en MPS: forma del chunk, valores finitos, tiempo.
Test 2 - predicción frente a acciones reales del dataset de entrenamiento
         (error por articulación a distintos horizontes; baseline = repetir el estado).
Test 3 - rango: fracción de acciones predichas fuera del rango de entrenamiento.

Uso: ./.venv312/bin/python pruebas_act/test_act.py [--samples 60]
Escribe pruebas_act/results/latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from act_model import DATASET_REPO, DATASET_ROOT, JOINTS, ACTOffline  # noqa: E402

CHUNK = 100
HORIZONS = [0, 10, 50, 99]
# Umbrales de aprobación (grados). Definidos antes de ejecutar; ajustables con criterio.
MAX_MAE_STEP0_DEG = 5.0      # el primer paso debe estar cerca de la acción real
MAX_OUT_OF_RANGE = 0.02      # como máximo el 2 % de las acciones fuera del rango de entrenamiento


def to_uint8(img_chw: np.ndarray) -> np.ndarray:
    return (np.clip(img_chw, 0, 1) * 255.0).round().astype(np.uint8).transpose(1, 2, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    report: dict = {"tests": {}}
    model = ACTOffline(device=args.device)
    report["device"] = model.device
    report["load_seconds"] = model.load_seconds

    # ---------- Test 1: carga y forward
    ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, video_backend="pyav")
    s0 = ds[0]
    rng_imgs = {"front": to_uint8(s0["observation.images.front"].numpy()),
                "gripper": to_uint8(s0["observation.images.gripper"].numpy())}
    t = time.time()
    out = model.predict_chunk(s0["observation.state"].numpy(), rng_imgs)
    dt = round(time.time() - t, 2)
    t1 = {
        "shape_ok": out.shape == (CHUNK, 6),
        "shape": list(out.shape),
        "finite": bool(np.isfinite(out).all()),
        "seconds_per_chunk": dt,
        "device": model.device,
    }
    t1["pass"] = bool(t1["shape_ok"] and t1["finite"] and model.device == "mps")
    report["tests"]["1_load_forward"] = t1

    # ---------- Test 2: predicción vs dataset
    n_eps = ds.num_episodes
    ep_idx = np.linspace(0, n_eps - 1, 12).round().astype(int)
    ep_len = {}
    all_actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    ep_arr = np.asarray(ds.hf_dataset["episode_index"])
    per_ep = max(1, args.samples // len(ep_idx))
    rng = np.random.default_rng(0)
    errs = {h: [] for h in HORIZONS}
    base_errs = {h: [] for h in HORIZONS}
    step0_per_joint = []
    preds_all = []
    for e in ep_idx:
        rows = np.where(ep_arr == e)[0]
        if len(rows) < CHUNK + 1:
            continue
        starts = rng.choice(len(rows) - CHUNK, size=min(per_ep, len(rows) - CHUNK), replace=False)
        for st in starts:
            r = rows[st]
            sample = ds[int(r)]
            imgs = {"front": to_uint8(sample["observation.images.front"].numpy()),
                    "gripper": to_uint8(sample["observation.images.gripper"].numpy())}
            state = sample["observation.state"].numpy()
            pred = model.predict_chunk(state, imgs)
            gt = all_actions[r:r + CHUNK]
            preds_all.append(pred)
            for h in HORIZONS:
                errs[h].append(np.abs(pred[h] - gt[h]))
                base_errs[h].append(np.abs(state - gt[h]))
            step0_per_joint.append(np.abs(pred[0] - gt[0]))

    def summarize(d):
        return {str(h): {j: round(float(v), 2) for j, v in zip(JOINTS, np.mean(d[h], axis=0))}
                for h in HORIZONS}

    mae_step0_mean = float(np.mean(step0_per_joint))
    t2 = {
        "n_samples": len(preds_all),
        "mae_deg_model": summarize(errs),
        "mae_deg_baseline_repeat_state": summarize(base_errs),
        "mae_step0_mean_deg": round(mae_step0_mean, 2),
    }
    t2["pass"] = bool(mae_step0_mean < MAX_MAE_STEP0_DEG)
    report["tests"]["2_offline_vs_dataset"] = t2

    # ---------- Test 3: rango de salidas
    preds = np.concatenate(preds_all, axis=0)
    lo = all_actions.min(axis=0)
    hi = all_actions.max(axis=0)
    out_mask = (preds < lo) | (preds > hi)
    frac = out_mask.mean(axis=0)
    t3 = {
        "train_action_min": {j: round(float(v), 2) for j, v in zip(JOINTS, lo)},
        "train_action_max": {j: round(float(v), 2) for j, v in zip(JOINTS, hi)},
        "pred_min": {j: round(float(v), 2) for j, v in zip(JOINTS, preds.min(axis=0))},
        "pred_max": {j: round(float(v), 2) for j, v in zip(JOINTS, preds.max(axis=0))},
        "fraction_out_of_range": {j: round(float(v), 4) for j, v in zip(JOINTS, frac)},
        "fraction_out_of_range_mean": round(float(out_mask.mean()), 4),
    }
    t3["pass"] = bool(out_mask.mean() <= MAX_OUT_OF_RANGE)
    report["tests"]["3_output_range"] = t3

    report["all_pass"] = all(v["pass"] for v in report["tests"].values())
    report["thresholds"] = {"mae_step0_deg": MAX_MAE_STEP0_DEG, "out_of_range_fraction": MAX_OUT_OF_RANGE}
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    (outdir / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
