"""Demo visual del generador de movimientos en la sim (sin brazo real).

Calcula un movimiento a punto y un saludo con hub/motion_gen.py y los ejecuta en la sim, en la
ventana de MuJoCo. Imprime velocidad, aceleración, jerk y el error de seguimiento.

  # Solo cálculo, sin ventana (comprueba tiempos y métricas)
  ./.venv312/bin/python hub/motion_demo.py --no-view

  # Con ventana (en macOS hay que usar mjpython)
  ./.venv312/bin/mjpython hub/motion_demo.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
import motion_gen as mg  # noqa: E402


def execute(sim, plan: dict, label: str) -> None:
    """Ejecuta el plan en la sim (una muestra por paso de control) y mide el seguimiento."""
    err = []
    for row, p in zip(plan["joints_deg"], plan["path"]):
        sim._tick(np.radians(row[:5]), float(row[5]))
        err.append(np.linalg.norm(sim.tcp() - p))
    err = np.array(err) * 1000
    print(f"[{label}] seguimiento en sim: medio {err.mean():.1f} mm, máximo {err.max():.1f} mm", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-view", action="store_true", help="solo calcula y mide, sin ventana")
    args = ap.parse_args()

    from objects_env import ObjectsSim

    sim = ObjectsSim()
    sim.reset_objects()
    if not args.no_view:
        sim.attach_viewer(realtime=True)
        time.sleep(1.0)

    q0 = sim.data.qpos[:5].copy()
    start = sim.tcp().copy()

    t0 = time.time()
    move = mg.move_plan(sim, start, np.array([0.20, 0.06, 0.07]), q0, seconds=1.5)
    print(f"[mover] calculado en {time.time() - t0:.2f} s, estiramiento {move['stretch']}x, "
          f"ik {move['ik_err_m'] * 1000:.1f} mm, vel máx {move['max_vel_deg_s']:.0f} °/s, "
          f"acc máx {move['max_acc_deg_s2']:.0f} °/s², jerk máx {move['max_jerk_deg_s3']:.0f} °/s³", flush=True)

    t0 = time.time()
    wave = mg.wave_plan(sim, move["path"][-1], amplitude_m=0.03, cycles=3, period_s=1.2,
                        q0_rad=move["joints_deg"][-1, :5] * np.pi / 180)
    print(f"[saludo] calculado en {time.time() - t0:.2f} s, estiramiento {wave['stretch']}x, "
          f"ik {wave['ik_err_m'] * 1000:.1f} mm, vel máx {wave['max_vel_deg_s']:.0f} °/s, "
          f"acc máx {wave['max_acc_deg_s2']:.0f} °/s², jerk máx {wave['max_jerk_deg_s3']:.0f} °/s³", flush=True)

    if args.no_view:
        return 0

    execute(sim, move, "mover")
    execute(sim, wave, "saludo")
    print("Demo terminada. Cierra la ventana para salir.", flush=True)
    while sim.viewer.is_running():
        sim.viewer.sync()
        time.sleep(0.03)
    return 0


if __name__ == "__main__":
    sys.exit(main())
