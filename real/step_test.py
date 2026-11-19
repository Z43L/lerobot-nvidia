"""Prueba de escalón para medir la dinámica del servo real y compararla con la simulación.

Mide una articulación (por defecto shoulder_lift): parte de la pose actual, pide un escalón de
--step grados, espera, vuelve al punto de partida y guarda las lecturas a 30 Hz.
Calcula: tiempo hasta 1° del objetivo, sobreimpulso y error final. Y lo compara con la simulación
con los mismos parámetros del modelo (sim/so101/so101_vla.xml).

Por defecto NO mueve nada (dry-run). Para mover el brazo hacen falta las dos cosas:
  ALLOW_REAL_MOTION=1 y --confirm, y la persona debe estar junto al brazo con el área libre.

  ./.venv312/bin/python real/step_test.py                       # dry-run, solo plan y simulación
  ALLOW_REAL_MOTION=1 ./.venv312/bin/python real/step_test.py --confirm --step 4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "real"))
sys.path.insert(0, str(ROOT / "sim"))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OUT_DIR = ROOT / "real" / "measurements"
HZ = 30


def sim_step(joint: str, step_deg: float, hold_s: float = 2.0) -> dict:
    """Misma prueba en la simulación con el modelo actual (kv y forcerange ya ajustados)."""
    import mujoco
    from sim_arm import SimArm

    a = SimArm()
    m, d = a.model, a.data
    idx = JOINTS.index(joint)
    a.reset()
    target = np.radians(step_deg)
    d.ctrl[idx] = target
    t1 = None
    peak = 0.0
    for k in range(int(hold_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
        q = d.qpos[idx]
        peak = max(peak, q)
        if t1 is None and abs(q - target) < np.radians(1):
            t1 = k * m.opt.timestep
    return {
        "t_1deg_s": None if t1 is None else round(t1, 3),
        "overshoot_pct": round((peak - target) / target * 100, 1) if target else None,
        "error_final_deg": round(float(np.degrees(abs(d.qpos[idx] - target))), 3),
    }


def real_step(joint: str, step_deg: float, port: str, rid: str, hold_s: float) -> dict:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    max_rel = float(os.environ.get("SO101_STEP_MAX_RELATIVE_TARGET", str(max(abs(step_deg), 2.0))))
    robot = SO101Follower(SO101FollowerConfig(port=port, id=rid, use_degrees=True,
                                              max_relative_target=max_rel))
    robot.connect(calibrate=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"step_{joint}_{int(step_deg)}deg_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    try:
        start = {n: float(robot.get_observation()[f"{n}.pos"]) for n in JOINTS}
        goal = dict(start)
        goal[joint] = start[joint] + step_deg
        phases = [("escalon", goal), ("vuelta", start)]
        t0 = time.time()
        for name, tgt in phases:
            t_phase = time.time()
            while time.time() - t_phase < hold_s:
                obs = robot.get_observation()
                rows.append({"t": round(time.time() - t0, 4), "fase": name,
                             "pos": round(float(obs[f"{joint}.pos"]), 3),
                             "objetivo": round(float(tgt[joint]), 3)})
                robot.send_action({f"{n}.pos": float(tgt[n]) for n in JOINTS})
                time.sleep(1.0 / HZ)
    finally:
        robot.disconnect()  # desactiva el par por defecto
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "fase", "pos", "objetivo"])
        w.writeheader()
        w.writerows(rows)
    esc = [r for r in rows if r["fase"] == "escalon"]
    tgt = start[joint] + step_deg
    t1 = next((r["t"] - esc[0]["t"] for r in esc if abs(r["pos"] - tgt) < 1.0), None)
    peak = max(r["pos"] for r in esc)
    return {
        "archivo": str(out),
        "t_1deg_s": None if t1 is None else round(t1, 3),
        "overshoot_pct": round((peak - tgt) / step_deg * 100, 1) if step_deg else None,
        "error_final_deg": round(abs(esc[-1]["pos"] - tgt), 3),
        "max_relative_target": max_rel,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="shoulder_lift", choices=JOINTS[:5])
    ap.add_argument("--step", type=float, default=4.0, help="escalón en grados (pequeño, 2..6)")
    ap.add_argument("--hold", type=float, default=2.0, help="segundos por fase")
    ap.add_argument("--confirm", action="store_true", help="confirma que el brazo se puede mover")
    args = ap.parse_args()

    if not 1.0 <= abs(args.step) <= 6.0:
        print("El escalón debe estar entre 1 y 6 grados.")
        return 2

    print("Simulación con el modelo actual:")
    print(" ", sim_step(args.joint, args.step, args.hold))

    port = os.environ.get("SO101_FOLLOWER_PORT", "")
    rid = os.environ.get("SO101_FOLLOWER_ID", "so101_follower")
    move_ok = os.environ.get("ALLOW_REAL_MOTION") == "1" and args.confirm
    if not move_ok:
        print("\nDry-run: no se mueve el brazo. Para medir el real: ALLOW_REAL_MOTION=1 y --confirm.")
        print(f"Plan: escalón de {args.step}° en {args.joint}, {args.hold}s, vuelta al inicio, CSV en {OUT_DIR}.")
        return 0
    import glob
    if not port or port not in glob.glob("/dev/cu.usb*"):
        print(f"Puerto del seguidor no disponible: '{port}'. Revisa el cable (usa so101-hub).")
        return 1
    print("\nMidiendo el brazo real:")
    print(" ", real_step(args.joint, args.step, port, rid, args.hold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
