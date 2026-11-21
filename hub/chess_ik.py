"""Cinemática inversa del SO-101 REAL para el tablero calibrado (hub/chess_board_real.json). No mueve nada.

Usa el modelo de la sim (b._sim().ik) y los desfases de ángulo de la calibración por contacto:
    grados_reales = grados_IK_sim - joint_offsets_deg        (pan, lift, elbow, wrist_flex)
La IK de sim fuerza la pinza vertical; aquí se le deja inclinarse lo mínimo para llegar (filas 7-8 piden 30-47 grados).
wrist_roll NO lo calcula la IK: el ejecutor lo mantiene donde está.

    from chess_ik import Kin
    kin = Kin()
    sol = kin.solve("e2", dz=0.05)  ->  {"joints": [pan, lift, elbow, wrist_flex] reales, "tilt": grados, "margin": grados, "err_mm": ...}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HUB = Path(__file__).resolve().parent
ROOT = HUB.parent
sys.path.insert(0, str(HUB))
import board_touch_calib as b  # noqa: E402

BOARD_REAL = HUB / "chess_board_real.json"
CAL_FOLLOWER = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json"
J4 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
TILT_WEIGHTS = (0.5, 0.2, 0.08, 0.03, 0.0)  # peso de "pinza vertical", de más a menos exigente


class Kin:
    def __init__(self, board: Path = BOARD_REAL):
        if not Path(board).exists():
            raise RuntimeError("Falta hub/chess_board_real.json: calibra el tablero (hub/board_touch_fit.py).")
        self.board = json.loads(Path(board).read_text())
        self.squares = {k: tuple(v) for k, v in self.board["squares"].items()}
        self.z_contact = float(self.board["contact_z_m"])
        self.off = np.array([self.board["joint_offsets_deg"][k] for k in J4] + [0.0])
        cal = json.loads(CAL_FOLLOWER.read_text())
        # En LeRobot (grados) el rango queda simétrico alrededor de 0: +-(max-min)/2 * 360/4095
        self.limit = np.array([(cal[n]["range_max"] - cal[n]["range_min"]) / 2 * 360 / 4095 for n in J4])
        self.sim = b._sim()
        self.lo, self.hi = self.sim.model.jnt_range[:5, 0], self.sim.model.jnt_range[:5, 1]
        self._q = np.radians([0.0, 30.0, -10.0, 80.0, 0.0])

    # ----------------------------------------------------------------------------
    def _ik(self, target: np.ndarray, k: float, q0: np.ndarray) -> np.ndarray:
        sim = self.sim
        sim._d = mujoco.MjData(sim.model)
        sim._d.qpos[:] = sim.data.qpos
        q = q0.copy()

        def res(qq):
            r = sim._residual(qq, target, (1.0, 0.0), 0.0).copy()
            r[3:6] *= k
            r[6] = 0.0
            return r

        for _ in range(300):
            r = res(q)
            if np.linalg.norm(r) < 1e-5:
                break
            J = np.zeros((7, 5))
            for i in range(5):
                dq = q.copy()
                dq[i] += 1e-5
                J[:, i] = (res(dq) - r) / 1e-5
            q = np.clip(q + 0.5 * (J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(7), -r)), self.lo, self.hi)
        return q

    def _tilt(self, q: np.ndarray) -> float:
        d = self.sim._d
        d.qpos[:5] = q
        mujoco.mj_forward(self.sim.model, d)
        return float(np.degrees(np.arccos(np.clip(d.xmat[self.sim.body_id].reshape(3, 3)[2, 2], -1, 1))))

    def solve_xy(self, xy: tuple[float, float], dz: float, k_hint: float | None = None) -> dict:
        """IK para la punta en (x, y, z_contacto + dz). Devuelve None en 'joints' si no llega (error > 3 mm)."""
        target = np.array([xy[0], xy[1], self.z_contact + dz])
        for k in ([k_hint] if k_hint is not None else []) + list(TILT_WEIGHTS):
            q = self._ik(target, k, self._q)
            sim = self.sim
            sim._d.qpos[:5] = q
            mujoco.mj_forward(sim.model, sim._d)
            err = float(np.linalg.norm(sim._d.site_xpos[sim.tcp_site] - target))
            if err < 0.003:
                real = np.degrees(q) - self.off
                self._q = q
                return {"joints": [float(v) for v in real[:4]], "tilt": self._tilt(q), "k": k, "err_mm": err * 1000,
                        "margin": float(np.min(self.limit - np.abs(real[:4])))}
        return {"joints": None, "tilt": None, "k": None, "err_mm": None, "margin": None}

    def solve(self, square: str, dz: float = 0.0, k_hint: float | None = None) -> dict:
        return self.solve_xy(self.squares[square], dz, k_hint)


def steps_between(q0, q1, max_step: float = 2.6) -> int:
    """Pasos de control necesarios para ir de q0 a q1 con como mucho max_step grados por articulación y paso."""
    return int(np.ceil(np.max(np.abs(np.asarray(q1, float) - np.asarray(q0, float))) / max_step))
