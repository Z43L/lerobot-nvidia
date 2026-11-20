"""Generador de movimientos paramétricos del SO-101: la trayectoria se calcula en el momento.

No copia grabaciones. Una "skill" da un objetivo y un estilo (amplitud, velocidad, altura...) y este
módulo calcula, con cinemática inversa y un perfil min-jerk, la trayectoria de la punta de la pinza.
Así un mismo gesto sale parecido, no idéntico, y se adapta a donde esté el objeto o el destino.

Garantías:
  - Velocidad y aceleración continuas: perfil min-jerk en el espacio cartesiano.
  - Límite de velocidad por articulación: si una trayectoria supera SO101_MAX_RELATIVE_TARGET por
    paso de 30 Hz, se alarga la duración hasta cumplirlo. Nunca se recorta el movimiento.
  - Sin brazo: todo se calcula con el modelo de sim. El envío al real pasa por so101-motion.

Uso típico (desde sim/ o con sys.path apuntando a sim/):
    from motion_gen import wave_plan, move_plan, kinematics
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HZ = 30
HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "sim"
GRIP_OPEN = 1.5      # rad (sim)
GRIP_CLOSED = -0.1745
UP = np.array([0.0, 0.0, 1.0])  # pinza vertical: orientación de trabajo
MAX_IK_ERR_M = 0.003   # por encima de 3 mm el objetivo no es alcanzable con esa orientación


def speed_limit_deg_per_tick() -> float:
    """Avance máximo por paso de control. Mismo límite que el seguidor real (sim_arm.max_deg_per_tick)."""
    return float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2"))


def minjerk(tau: np.ndarray) -> np.ndarray:
    """Perfil min-jerk de 0 a 1: velocidad y aceleración nulas en los extremos."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _sim():
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))
    from objects_env import ObjectsSim

    return ObjectsSim()


def cartesian_path(start: np.ndarray, waypoints: list[np.ndarray], durations: list[float]) -> np.ndarray:
    """Puntos (N, 3) a HZ que recorren start -> waypoints[0] -> ... con min-jerk en cada tramo."""
    pts = [np.asarray(start, dtype=float)]
    p0 = pts[0]
    for p1, T in zip(waypoints, durations):
        p1 = np.asarray(p1, dtype=float)
        n = max(1, int(round(T * HZ)))
        s = minjerk(np.arange(1, n + 1) / n)
        seg = p0 + (p1 - p0) * s[:, None]
        pts.extend(seg)
        p0 = p1
    return np.array(pts[1:]) if len(pts) > 1 else np.array(pts)


def body_z(sim, q5_rad: np.ndarray) -> np.ndarray:
    """Eje z de la pinza (marco del cuerpo 'gripper') para 5 articulaciones en radianes."""
    import mujoco

    body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    d = mujoco.MjData(sim.model)
    d.qpos[:] = sim.data.qpos
    d.qpos[:5] = q5_rad
    mujoco.mj_forward(sim.model, d)
    return d.xmat[body].reshape(3, 3)[:, 2].copy()


def fast_ik(sim, target: np.ndarray, q0_rad: np.ndarray, iters: int = 40, tol: float = 1e-5,
            z_axis: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """IK de la punta con pinza hacia abajo: Levenberg-Marquardt con Jacobiano analítico de MuJoCo.

    Mismo residuo que ObjectsSim.ik (posición + eje de la pinza vertical). El amortiguamiento se adapta:
    si un paso empeora el error se rechaza y se amortigua más. Milisegundos por solución.
    Devuelve (q5 en radianes, error de posición en metros).
    """
    import mujoco

    d = getattr(sim, "_fast_ik_data", None)
    if d is None:
        d = mujoco.MjData(sim.model)
        sim._fast_ik_data = d
        sim._jacp = np.zeros((3, sim.model.nv))
        sim._jacr = np.zeros((3, sim.model.nv))
        sim._body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    d.qpos[:] = sim.data.qpos
    lo, hi = sim.model.jnt_range[:5, 0], sim.model.jnt_range[:5, 1]
    target = np.asarray(target, dtype=float)
    up = np.array([0.0, 0.0, 1.0]) if z_axis is None else np.asarray(z_axis, dtype=float)

    def residual(q):
        d.qpos[:5] = q
        mujoco.mj_forward(sim.model, d)
        p = d.site_xpos[sim.tcp_site]
        z = d.xmat[sim._body].reshape(3, 3)[:, 2]
        return np.concatenate([target - p, 0.5 * (up - z)]), z

    q = np.clip(np.asarray(q0_rad, dtype=float)[:5], lo, hi).copy()
    r, z = residual(q)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iters):
        if np.linalg.norm(r) < tol:
            break
        mujoco.mj_jacSite(sim.model, d, sim._jacp, sim._jacr, sim.tcp_site)
        Jr = np.stack([np.cross(sim._jacr[:, i], z) for i in range(5)], axis=1)
        J = np.vstack([-sim._jacp[:, :5], -0.5 * Jr])
        A = J.T @ J
        g = J.T @ r
        accepted = False
        for _try in range(8):
            step = -np.linalg.solve(A + lam * np.eye(5), g)
            q_new = np.clip(q + step, lo, hi)
            r_new, z_new = residual(q_new)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                q, r, z, cost = q_new, r_new, z_new, cost_new
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                break
            lam *= 5.0
        if not accepted:
            break
    err = float(np.linalg.norm(target - d.site_xpos[sim.tcp_site]))
    return q, err


def joints_from_path(sim, path: np.ndarray, q0_rad: np.ndarray, grip: np.ndarray, z_axis: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """IK punto a punto con arranque cálido. Devuelve (N, 6) en grados y el error máximo de IK (m)."""
    out = np.zeros((len(path), 6))
    q = np.asarray(q0_rad, dtype=float)[:5].copy()
    z_ref = UP if z_axis is None else np.asarray(z_axis, dtype=float)  # pinza vertical por defecto
    worst = 0.0
    for i, p in enumerate(path):
        q, _ = fast_ik(sim, p, q, z_axis=z_ref)
        out[i, :5] = np.degrees(q)
        out[i, 5] = grip[i]
        tip = _fk(sim, q)
        worst = max(worst, float(np.linalg.norm(tip - p)))
    return out, worst


def _fk(sim, q5_rad: np.ndarray) -> np.ndarray:
    """Punta de la pinza para 5 articulaciones (rad). Usa su propio MjData: no toca sim.data."""
    import mujoco

    d = getattr(sim, "_fk_data", None)
    if d is None:
        d = mujoco.MjData(sim.model)
        sim._fk_data = d
    d.qpos[:] = sim.data.qpos
    d.qpos[:5] = q5_rad
    mujoco.mj_forward(sim.model, d)
    return d.site_xpos[sim.tcp_site].copy()


def fit_speed(build, limit: float | None = None, max_tries: int = 12) -> tuple[np.ndarray, float, int]:
    """build(stretch) -> joints_deg (N, 6). Alarga la duración hasta cumplir el límite de velocidad.
    Devuelve (joints_deg, stretch_usado, intentos)."""
    limit = limit or speed_limit_deg_per_tick()
    stretch = 1.0
    for attempt in range(1, max_tries + 1):
        joints = build(stretch)
        if getattr(build, "last_err", 0.0) > MAX_IK_ERR_M:
            raise ValueError(f"objetivo inalcanzable con pinza vertical (error IK {build.last_err * 1000:.1f} mm)")
        step = np.abs(np.diff(joints[:, :5], axis=0)).max() if len(joints) > 1 else 0.0
        if step <= limit:
            return joints, stretch, attempt
        stretch *= max(1.1, step / limit)
    raise RuntimeError(f"no se cumple el límite de {limit}°/paso ni alargando {stretch:.1f}x")


def kinematics(joints_deg: np.ndarray, hz: int = HZ) -> dict:
    """Velocidad, aceleración y jerk máximos por articulación (grados, s, s², s³)."""
    x = joints_deg[:, :5]
    v = np.gradient(x, 1 / hz, axis=0)
    a = np.gradient(v, 1 / hz, axis=0)
    j = np.gradient(a, 1 / hz, axis=0)
    return {
        "max_vel_deg_s": float(np.abs(v).max()),
        "max_acc_deg_s2": float(np.abs(a).max()),
        "max_jerk_deg_s3": float(np.abs(j).max()),
        "max_step_deg_tick": float(np.abs(np.diff(x, axis=0)).max()),
        "seconds": round(len(joints_deg) / hz, 2),
    }


def move_plan(sim, start_xyz, target_xyz, q0_rad, seconds: float = 1.5, grip: float = GRIP_OPEN) -> dict:
    """Movimiento a punto con min-jerk en articulaciones (de q0 a la solución IK con pinza vertical).

    Un solo IK (en el destino), así que es rápido. La punta describe una curva suave, no una recta.
    """
    q0 = np.asarray(q0_rad, dtype=float)[:5]
    q_goal, err = fast_ik(sim, np.asarray(target_xyz, dtype=float), q0, z_axis=UP)
    if err > MAX_IK_ERR_M:
        raise ValueError(f"destino inalcanzable con pinza vertical (error IK {err * 1000:.1f} mm)")

    def build(stretch):
        n = max(2, int(round(seconds * stretch * HZ)))
        s = minjerk(np.arange(1, n + 1) / n)[:, None]
        q = q0 + (q_goal - q0) * s
        joints = np.zeros((n, 6))
        joints[:, :5] = np.degrees(q)
        joints[:, 5] = float(grip)
        build.last_path = np.array([_fk(sim, qi) for qi in q])
        build.last_err = err
        return joints

    joints, stretch, tries = fit_speed(build)
    return {"joints_deg": joints, "path": build.last_path, "ik_err_m": err,
            "stretch": round(stretch, 2), "tries": tries, **kinematics(joints)}


def wave_plan(sim, center_xyz, amplitude_m: float, cycles: int, period_s: float, q0_rad,
              grip: float = GRIP_OPEN, ramp_s: float | None = None) -> dict:
    """Saludo: la punta oscila en el eje y alrededor de center, con pinza vertical.

    Oscilación senoidal (jerk continuo) con rampas min-jerk de entrada y salida de ramp_s (por defecto, un ciclo).
    amplitude_m: desplazamiento a cada lado. period_s: un ciclo completo. cycles: número de ciclos.
    """
    c = np.asarray(center_xyz, dtype=float)
    ramp_base = period_s if ramp_s is None else ramp_s  # rampa de un ciclo: jerk de entrada bajo

    def build(stretch):
        T = period_s * stretch
        R = ramp_base * stretch
        total_s = cycles * T + 2 * R
        n = max(2, int(round(total_s * HZ)))
        t = np.arange(1, n + 1) / HZ
        env = np.minimum(minjerk(t / R), minjerk((total_s - t) / R))  # rampas de entrada y salida
        y = amplitude_m * env * np.sin(2 * np.pi * t / T)  # fase 0: empieza y acaba en el centro
        path = np.repeat(c[None, :], n, axis=0)
        path[:, 1] += y
        joints, err = joints_from_path(sim, path, q0_rad, np.full(n, float(grip)), z_axis=UP)
        build.last_path, build.last_err = path, err
        return joints

    joints, stretch, tries = fit_speed(build)
    return {"joints_deg": joints, "path": build.last_path, "ik_err_m": build.last_err,
            "stretch": round(stretch, 2), "tries": tries, **kinematics(joints)}


def linear_joint_baseline(sim, start_xyz, target_xyz, q0_rad, seconds: float = 1.5) -> np.ndarray:
    """Referencia: interpolación lineal en articulaciones entre dos IK (lo que hace hoy plan_playback)."""
    q_start = np.asarray(q0_rad, dtype=float)[:5]
    q_end = sim.ik(np.asarray(target_xyz, dtype=float), q0=q_start)
    n = int(round(seconds * HZ))
    alpha = np.linspace(0, 1, n)[:, None]
    q = (1 - alpha) * q_start + alpha * q_end
    out = np.zeros((n, 6))
    out[:, :5] = np.degrees(q)
    out[:, 5] = GRIP_OPEN
    return out
