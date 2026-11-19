"""Experto scriptado (IK) que coge un disco de un pivote y lo deja en otro.

Se usa para:
  - generar demostraciones para entrenar el VLA (grabación con `record_episode`);
  - validar la escena (`python hanoi_expert.py --demo`).

Las acciones registradas usan la misma escala normalizada de LeRobot que el VLA del
SO-101 (articulaciones en [-100, 100], pinza en [0, 100]), igual que sim/vla_policy.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from hanoi_env import DISK_THICK, GRIP_CLOSED, GRIP_OPEN, PEGS, TRAVEL_Z, HanoiSim

CONTROL_HZ = 30
GRIP_IDX = 5


def rad_to_norm(sim: HanoiSim, q: np.ndarray) -> np.ndarray:
    """Radianes de la sim -> unidades de LeRobot/real: grados en las articulaciones y 0..100 en la pinza.
    Igual que VLARunner.rad_to_norm (sim/vla_policy.py): así los datos de Hanoi y el brazo real coinciden."""
    lo, hi = sim.model.jnt_range[:6, 0], sim.model.jnt_range[:6, 1]
    frac = (q - lo) / (hi - lo)
    out = np.degrees(q).astype(np.float64)
    out[GRIP_IDX] = np.clip(frac[GRIP_IDX], 0.0, 1.0) * 100.0
    return out


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


class HanoiExpert:
    """Ejecuta movimientos pick-and-place en HanoiSim.

    `on_step(obs_state, action)` se llama en cada paso de control ANTES de avanzar la
    física, para grabar observaciones y acciones.
    """

    def __init__(self, sim: HanoiSim, on_step: Callable | None = None, jitter: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.sim = sim
        self.on_step = on_step
        self.jitter = jitter  # ruido (m) en el punto de agarre, para robustez
        self.rng = rng or np.random.default_rng(0)
        self.substeps = max(1, int(round(1.0 / CONTROL_HZ / sim.model.opt.timestep)))
        self.q5 = sim.data.qpos[:5].copy()
        self.grip = GRIP_OPEN

    # ------------------------------------------------------------ bajo nivel
    def _tick(self, tcp_target: np.ndarray, grip: float) -> bool:
        """Un paso de control: IK hacia tcp_target, pinza a `grip`. Devuelve True si OK."""
        q5 = self.sim.ik(tcp_target, q0=self.q5)
        self.q5 = q5
        self.grip = grip
        ctrl = np.concatenate([q5, [grip]])
        sim = self.sim
        sim.data.ctrl[:6] = ctrl
        if self.on_step is not None:
            state_norm = rad_to_norm(sim, sim.data.qpos[:6].copy()).astype(np.float32)
            action_norm = rad_to_norm(sim, ctrl).astype(np.float32)
            self.on_step(state_norm, action_norm)
        for _ in range(self.substeps):
            sim.sync_held()
            mujoco.mj_step(sim.model, sim.data)
        sim.sync_view()
        return True

    def _segment(self, p0: np.ndarray, p1: np.ndarray, seconds: float,
                 g0: float, g1: float) -> None:
        n = max(1, int(round(seconds * CONTROL_HZ)))
        for i in range(1, n + 1):
            s = _smooth(i / n)
            self._tick(p0 + s * (p1 - p0), g0 + s * (g1 - g0))

    def _hold(self, p: np.ndarray, seconds: float, g0: float, g1: float) -> None:
        n = max(1, int(round(seconds * CONTROL_HZ)))
        for i in range(1, n + 1):
            s = _smooth(i / n)
            self._tick(p, g0 + s * (g1 - g0))

    # ----------------------------------------------------------- alto nivel
    def stack_count(self, peg: str) -> int:
        return len(self.sim.get_hanoi_state()[peg])

    def move_disk(self, color: str, src: str, dst: str) -> None:
        """Mueve el disco `color` (el de arriba de `src`) a `dst`.

        Agarre: al cerrar la pinza, si el TCP está sobre el disco, el disco se engancha
        al TCP (ver HanoiSim.hold). Al abrir, se suelta y cae sobre la pila de destino.
        """
        state = self.sim.get_hanoi_state()
        if not state[src] or state[src][-1] != color:
            raise ValueError(f"El disco {color} no está encima del pivote {src}: {state[src]}")
        n_src = len(state[src])
        n_dst = len(state[dst])
        sx, sy = PEGS[src]
        tx, ty = PEGS[dst]
        z_grasp = (n_src - 1) * DISK_THICK + DISK_THICK / 2
        z_place = n_dst * DISK_THICK + DISK_THICK / 2 + 0.002
        src_top = np.array([sx, sy, TRAVEL_Z])
        grasp = np.array([sx, sy, z_grasp])
        dst_top = np.array([tx, ty, TRAVEL_Z])
        place = np.array([tx, ty, z_place])

        self._segment(self.sim.tcp(), src_top, 1.5, GRIP_OPEN, GRIP_OPEN)      # aproximación
        self._segment(src_top, grasp, 1.2, GRIP_OPEN, GRIP_OPEN)               # bajada
        self._hold(grasp, 0.7, GRIP_OPEN, GRIP_CLOSED)                         # cierre
        disk = self.sim.disk_xy_z(color)
        if np.linalg.norm(disk[:2] - grasp[:2]) > 0.012 or abs(disk[2] - grasp[2]) > 0.012:
            raise RuntimeError(f"Agarre fallido sobre {color}: disco en {disk}, TCP en {grasp}")
        self.sim.hold(color)
        self._segment(grasp, src_top, 1.2, GRIP_CLOSED, GRIP_CLOSED)           # subida
        self._segment(src_top, dst_top, 2.0, GRIP_CLOSED, GRIP_CLOSED)         # transporte
        self._segment(dst_top, place, 1.2, GRIP_CLOSED, GRIP_CLOSED)           # bajada
        self.sim.release()
        self._hold(place, 0.6, GRIP_CLOSED, GRIP_OPEN)                         # soltar
        self._segment(place, dst_top, 1.0, GRIP_OPEN, GRIP_OPEN)               # retirada

    def go_home(self) -> None:
        q = self.sim.home_qpos[:5]
        for i in range(1, 2 * CONTROL_HZ + 1):
            s = _smooth(i / (2 * CONTROL_HZ))
            self.q5 = (1 - s) * self.q5 + s * q
            self._tick(self.sim.tcp(), GRIP_OPEN)


def run_demo(out_dir: Path) -> None:
    """Ejecuta una Hanoi completa de 3 discos (7 movimientos) y guarda fotogramas."""
    sim = HanoiSim()
    exp = HanoiExpert(sim)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = [("red", "A", "C"), ("green", "A", "B"), ("red", "C", "B"),
           ("blue", "A", "C"), ("red", "B", "A"), ("green", "B", "C"), ("red", "A", "C")]
    for i, (c, s, t) in enumerate(seq):
        exp.move_disk(c, s, t)
        print(i + 1, c, s, "->", t, sim.get_hanoi_state(), flush=True)
        sim.render("table")
        sim.render("top")
        (out_dir / f"step_{i + 1:02d}.png").write_bytes(Path(sim.render("table")).read_bytes())
    print("resuelta:", sim.is_solved("C"))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo(Path(__file__).parent / "frames" / "hanoi_expert")
