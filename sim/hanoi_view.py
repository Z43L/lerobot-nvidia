"""Ve en tiempo real la torre de Hanoi en la ventana de MuJoCo.

En macOS hay que lanzarlo con mjpython (viene con el paquete mujoco en .venv312):
    ./.venv312/bin/mjpython sim/hanoi_view.py            # experto: resuelve la Hanoi de 3 discos
    ./.venv312/bin/mjpython sim/hanoi_view.py --loops 3   # repite la solución
"""

from __future__ import annotations

import argparse
import time

from hanoi_env import HanoiSim, solve_plan
from hanoi_expert import HanoiExpert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--fast", action="store_true", help="sin esperar a tiempo real")
    args = ap.parse_args()

    sim = HanoiSim()
    sim.attach_viewer(realtime=not args.fast)
    expert = HanoiExpert(sim)
    time.sleep(1.0)
    for loop in range(args.loops):
        sim.reset_hanoi()
        expert.q5 = sim.data.qpos[:5].copy()
        for disk, src, dst in solve_plan(sim.get_hanoi_state(), "C"):
            print(f"{disk}: {src} -> {dst}", flush=True)
            expert.move_disk(disk, src, dst)
            if not sim.viewer.is_running():
                return
        print("resuelta:", sim.is_solved("C"), flush=True)
        time.sleep(2.0)
    while sim.viewer.is_running():
        sim.viewer.sync()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
