"""Grabaciones desde el brazo líder -> movimientos reutilizables (hub/motions) listos para skills.

Flujo:
  1. Grabación: muestras (tiempo, 6 valores en grados / pinza 0..100) a 30 Hz, leídas del LÍDER
     (solo lectura: el líder no mueve el brazo). Se guardan en hub/recordings/<nombre>.json.
  2. Simplificación: Ramer-Douglas-Peucker en el espacio de articulaciones. Se conservan los
     puntos donde la trayectoria cambia de verdad (giros, paradas, apertura de la pinza) y se
     descartan los intermedios. Tolerancia por defecto: 1,5° (pinza 4 unidades).
  3. Movimiento: keyframes con tiempo (t, en segundos) y pose, en el mismo formato que
     hub/motion_lib.py, de modo que sirve para la simulación y para el real.
  4. Reproducción rápida: plan_playback() densifica la trayectoria a 30 Hz respetando el límite de
     SO101_MAX_RELATIVE_TARGET por paso, así el real sigue la grabación sin sobrepasar su velocidad.

Lo que NO hace aquí: mover el brazo real. Eso lo hace so101-motion con gates y confirmación.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECORDINGS_DIR = HERE / "recordings"
HZ = 30
TOL_DEG = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 4.0])  # tolerancia por articulación (pinza en 0..100)


def max_deg_per_tick() -> float:
    return float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2"))


# ------------------------------------------------------------------ grabación
def save_recording(name: str, t: list[float], deg: list[list[float]]) -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    p = RECORDINGS_DIR / f"{name}.json"
    p.write_text(json.dumps({"name": name, "hz": HZ, "t": [round(x, 4) for x in t],
                             "deg": [[round(float(v), 3) for v in row] for row in deg]}))
    return p


def load_recording(name: str) -> dict:
    return json.loads((RECORDINGS_DIR / f"{name}.json").read_text())


def record_from_leader(seconds: float, port: str, leader_id: str) -> tuple[list[float], list[list[float]]]:
    """Lee el brazo LÍDER durante `seconds` a 30 Hz. No envía órdenes a ningún brazo."""
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    leader = SOLeader(SOLeaderTeleopConfig(port=port, id=leader_id, use_degrees=True))
    leader.connect(calibrate=False)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    t_out, deg_out = [], []
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            a = leader.get_action()
            t_out.append(time.time() - t0)
            deg_out.append([float(a[f"{n}.pos"]) for n in names])
            time.sleep(1.0 / HZ)
    except KeyboardInterrupt:
        pass
    finally:
        leader.disconnect()
    return t_out, deg_out


# --------------------------------------------------------- simplificación
def _rdp(points: np.ndarray, tol: np.ndarray) -> list[int]:
    """Ramer-Douglas-Peucker con distancia escalada por articulación. Devuelve índices a conservar."""
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    scaled = points / tol
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = scaled[i], scaled[j]
        seg = b - a
        seg_len2 = float(seg @ seg)
        mid = scaled[i + 1:j]
        if seg_len2 == 0:
            d = np.linalg.norm(mid - a, axis=1)
        else:
            t = np.clip(np.sum((mid - a) * seg, axis=1) / seg_len2, 0, 1)
            proj = a + t[:, None] * seg
            d = np.linalg.norm(mid - proj, axis=1)
        k = int(np.argmax(d))
        if d[k] > 1.0:  # en unidades de tolerancia
            idx = i + 1 + k
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return list(np.flatnonzero(keep))


def simplify(t: list[float], deg: list[list[float]]) -> list[dict]:
    """Recording -> keyframes {t, deg}. Conserva inicio, fin y los cambios reales de trayectoria."""
    if len(t) < 2:
        return [{"t": round(t[0], 3) if t else 0.0, "deg": [round(v, 2) for v in deg[0]]}] if deg else []
    pts = np.asarray(deg, dtype=float)
    idx = _rdp(pts, TOL_DEG)
    return [{"t": round(float(t[i]) - float(t[0]), 3), "deg": [round(float(v), 2) for v in pts[i]]} for i in idx]


def to_motion(name: str, keyframes: list[dict], description: str = "") -> dict:
    """Formato de hub/motion_lib.py, con el tiempo de cada keyframe (campo extra 't')."""
    return {"name": name, "description": description, "units": "degrees",
            "keyframes": [{"deg": kf["deg"], "t": kf["t"], "note": ""} for kf in keyframes]}


# ----------------------------------------------------------- reproducción
def plan_playback(motion: dict, step_limit: float | None = None) -> np.ndarray:
    """Trayectoria a 30 Hz (N x 6, grados) que sigue los keyframes con interpolación lineal y
    el límite de avance por paso. Es lo que el real debe seguir paso a paso."""
    kfs = motion["keyframes"]
    limit = step_limit if step_limit is not None else max_deg_per_tick()
    times = np.array([kf.get("t", 0.0) for kf in kfs], dtype=float)
    targets = np.array([kf["deg"] for kf in kfs], dtype=float)
    if len(kfs) == 1:
        return targets.copy()
    grid = np.arange(0.0, times[-1] + 1e-9, 1.0 / HZ)
    ref = np.stack([np.interp(grid, times, targets[:, j]) for j in range(6)], axis=1)
    out = [ref[0].copy()]
    for k in range(1, len(ref)):
        step = np.clip(ref[k] - out[-1], -limit, limit)
        out.append(out[-1] + step)
    return np.asarray(out)


def playback_summary(motion: dict) -> dict:
    traj = plan_playback(motion)
    dur = len(traj) / HZ
    spans = np.abs(np.diff(traj, axis=0)).max(axis=0) if len(traj) > 1 else np.zeros(6)
    return {"name": motion["name"], "keyframes": len(motion["keyframes"]), "seconds": round(dur, 2),
            "ticks": len(traj), "max_step_deg_per_tick": round(float(spans.max()), 3) if len(traj) > 1 else 0.0,
            "range_deg": {n: [round(float(traj[:, i].min()), 1), round(float(traj[:, i].max()), 1)]
                          for i, n in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex",
                                                 "wrist_flex", "wrist_roll", "gripper"])}}
