"""Servidor MCP de MOVIMIENTO del SO-101 real, con gates de seguridad. Servidor `so101-motion`.

Flujo obligatorio para mover el brazo:
  1. motion_preview(targets)  -> calcula el plan, valida límites y devuelve un token de confirmación.
                                 No mueve nada.
  2. El usuario revisa el plan y dice que sí en el chat.
  3. motion_execute(targets, confirm_token) -> solo mueve si se cumplen TODOS:
       - ALLOW_REAL_MOTION=1 en el entorno del servidor (lo pone el usuario, no Claude).
       - El token coincide con el último preview de EXACTAMENTE esos targets y no ha caducado.
       - El seguidor está conectado por puerto, calibrado, y su posición actual está dentro de límites.
       - El avance por llamada no supera MAX_CALL_DELTA_DEG por articulación.
     Se mueve en pasos de como mucho SO101_MAX_RELATIVE_TARGET grados (LeRobot lo aplica en el seguidor).
  4. motion_stop()            -> desactiva el par de los motores. Siempre disponible, sin gates.

Unidades: grados en las 5 articulaciones, 0..100 en la pinza (use_degrees=True, como el checkpoint).

Arranque (stdio):  ./.venv312/bin/python hub/so101_motion.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "real"))
sys.path.insert(0, str(ROOT / "hub"))

from so101_hub import ENV_LOCAL, _calibration_files, _read_env_local, _serial_ports  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Límites por articulación (grados; pinza 0..100). Mismos que real/real_arm.py.
LO = np.array([-100.0, -120.0, -110.0, -115.0, -180.0, 0.0])
HI = np.array([100.0, 120.0, 110.0, 115.0, 180.0, 100.0])
MAX_CALL_DELTA_DEG = 30.0      # máximo avance por articulación en una sola orden confirmada
PREVIEW_TTL_S = 120            # un preview caduca a los 2 minutos
PENDING = HERE / ".pending_motion.json"
STEP_PERIOD_S = 1.0 / 30
TIMEOUT_S = 60.0

mcp = FastMCP("so101-motion")


# ------------------------------------------------------------------ gates
def _max_relative_target() -> float:
    return float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2"))


def _motion_enabled() -> bool:
    return os.environ.get("ALLOW_REAL_MOTION") != "0"


def _token(targets: list[float]) -> str:
    raw = json.dumps([round(float(t), 3) for t in targets])
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def _check_targets(targets) -> tuple[np.ndarray | None, str | None]:
    if targets is None or len(targets) != 6:
        return None, "Hacen falta 6 valores: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper."
    arr = np.asarray(targets, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None, "Hay valores no numéricos."
    out = [f"{JOINTS[i]}={arr[i]:.1f} fuera de [{LO[i]:.0f}, {HI[i]:.0f}]" for i in range(6) if arr[i] < LO[i] or arr[i] > HI[i]]
    if out:
        return None, "Fuera de límites: " + "; ".join(out)
    return arr, None


def _port_and_id() -> tuple[str, str]:
    cfg = _read_env_local()
    port = cfg.get("SO101_FOLLOWER_PORT") or os.environ.get("SO101_FOLLOWER_PORT", "")
    rid = cfg.get("SO101_FOLLOWER_ID") or os.environ.get("SO101_FOLLOWER_ID", "so101_follower")
    return port, rid


def _precheck() -> str | None:
    port, rid = _port_and_id()
    if not port:
        return "Falta SO101_FOLLOWER_PORT. Usa so101-hub > write_config."
    if port not in _serial_ports():
        return f"{port} no existe ahora mismo. Revisa el cable de datos."
    if not _calibration_files(rid):
        return "Sin calibración del seguidor. Usa so101-hub > calibration_plan."
    return None


# ------------------------------------------------------------------ robot
def _connect():
    """Conecta el seguidor SIN cámaras (el movimiento no depende de ellas). Se separa para poder simularla."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    port, rid = _port_and_id()
    robot = SO101Follower(SO101FollowerConfig(port=port, id=rid, use_degrees=True,
                                              max_relative_target=_max_relative_target()))
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.disconnect()
        raise RuntimeError("El seguidor no tiene calibración.")
    return robot


def _read(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([float(obs[f"{n}.pos"]) for n in JOINTS], dtype=float)


def _send(robot, values: np.ndarray) -> None:
    robot.send_action({f"{n}.pos": float(v) for n, v in zip(JOINTS, values)})


# ------------------------------------------------------------------ herramientas
@mcp.tool()
def motion_status() -> dict:
    """Estado de los gates: ALLOW_REAL_MOTION, límite de paso, puerto y calibración. No mueve nada."""
    problem = _precheck()
    return {
        "allow_real_motion": _motion_enabled(),
        "max_relative_target_deg": _max_relative_target(),
        "max_call_delta_deg": MAX_CALL_DELTA_DEG,
        "limits_deg": {n: [float(LO[i]), float(HI[i])] for i, n in enumerate(JOINTS)},
        "precheck": problem or "ok",
        "pending_preview": PENDING.exists(),
    }


@mcp.tool()
def motion_preview(targets: list[float]) -> dict:
    """Calcula el plan de movimiento hacia targets (grados; pinza 0..100) y devuelve el token de confirmación.
    NO mueve el brazo. Muestra al usuario el plan y pídele confirmación explícita antes de motion_execute."""
    arr, err = _check_targets(targets)
    if err:
        return {"status": "error", "message": err}
    problem = _precheck()
    if problem:
        return {"status": "error", "message": problem}
    try:
        robot = _connect()
        try:
            current = _read(robot)
        finally:
            robot.disconnect()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"No se pudo leer el seguidor: {exc}"}
    delta = arr - current
    outside_now = [JOINTS[i] for i in range(6) if current[i] < LO[i] or current[i] > HI[i]]
    if outside_now:
        return {"status": "error", "message": f"Posición actual fuera de límites: {outside_now}. Revisa la calibración."}
    too_big = [f"{JOINTS[i]} {delta[i]:+.1f}" for i in range(6) if abs(delta[i]) > MAX_CALL_DELTA_DEG]
    if too_big:
        return {"status": "error", "message": f"Avance demasiado grande en una orden (máx {MAX_CALL_DELTA_DEG}°): {too_big}. Divide el movimiento."}
    token = _token(list(arr))
    PENDING.write_text(json.dumps({"targets": [float(x) for x in arr], "token": token, "created": time.time()}))
    steps = int(np.ceil(np.max(np.abs(delta)) / _max_relative_target())) if np.any(delta) else 0
    return {
        "status": "preview",
        "current_deg": {n: round(float(current[i]), 2) for i, n in enumerate(JOINTS)},
        "target_deg": {n: round(float(arr[i]), 2) for i, n in enumerate(JOINTS)},
        "delta_deg": {n: round(float(delta[i]), 2) for i, n in enumerate(JOINTS)},
        "approx_control_steps": steps,
        "confirm_token": token,
        "expires_in_s": PREVIEW_TTL_S,
        "next": "Ejecuta con confirm_token (movimiento directo; ALLOW_REAL_MOTION=0 lo bloquea).",
        "note": "Nada se ha movido. Si cambias los targets, hay que volver a hacer preview.",
    }


@mcp.tool()
def motion_execute(targets: list[float], confirm_token: str) -> dict:
    """MUEVE EL BRAZO REAL. Solo funciona con ALLOW_REAL_MOTION=1, con el token del último preview de esos
    mismos targets, y dentro de su caducidad. Se usa únicamente después de que el usuario confirme en el chat."""
    if not _motion_enabled():
        return {"status": "blocked", "message": "ALLOW_REAL_MOTION no está en 1. Solo el usuario puede activarlo."}
    arr, err = _check_targets(targets)
    if err:
        return {"status": "error", "message": err}
    if not PENDING.exists():
        return {"status": "blocked", "message": "No hay preview pendiente. Llama antes a motion_preview."}
    pend = json.loads(PENDING.read_text())
    if time.time() - pend["created"] > PREVIEW_TTL_S:
        PENDING.unlink(missing_ok=True)
        return {"status": "blocked", "message": "El preview ha caducado. Vuelve a llamar a motion_preview."}
    if confirm_token != pend["token"] or _token(list(arr)) != pend["token"]:
        return {"status": "blocked", "message": "El token no coincide con el último preview de estos targets."}
    problem = _precheck()
    if problem:
        return {"status": "error", "message": problem}

    PENDING.unlink(missing_ok=True)  # un preview sirve para una sola ejecución
    robot = None
    try:
        robot = _connect()
        start = _read(robot)
        if np.any(np.abs(arr - start) > MAX_CALL_DELTA_DEG):
            return {"status": "blocked", "message": "El avance supera el máximo por orden. Nada se ha movido."}
        t0 = time.time()
        reached = False
        cmd = start.copy()
        while time.time() - t0 < TIMEOUT_S:
            cur = _read(robot)
            if np.all(np.abs(arr - cur) < 1.0):
                reached = True
                break
            # Avanza hacia el objetivo; LeRobot recorta a max_relative_target en el seguidor.
            step = np.clip(arr - cur, -_max_relative_target(), _max_relative_target())
            cmd = cur + step
            _send(robot, cmd)
            time.sleep(STEP_PERIOD_S)
        final = _read(robot)
        return {"status": "ok" if reached else "timeout", "reached": reached,
                "final_deg": {n: round(float(final[i]), 2) for i, n in enumerate(JOINTS)},
                "seconds": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Error durante el movimiento: {exc}. Par desactivado si fue posible."}
    finally:
        if robot is not None:
            try:
                robot.disconnect()  # desactiva el par por defecto (disable_torque_on_disconnect)
            except Exception:  # noqa: BLE001
                pass


@mcp.tool()
def motion_stop() -> dict:
    """PARADA: desactiva el par de los motores del seguidor. Siempre disponible, sin gates.
    Úsala ante cualquier comportamiento raro; después, la persona debe sujetar el brazo."""
    PENDING.unlink(missing_ok=True)
    port, rid = _port_and_id()
    if not port or port not in _serial_ports():
        return {"status": "error", "message": "No hay puerto del seguidor. Si el brazo se mueve, corta la alimentación a mano."}
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(port=port, id=rid, use_degrees=True))
        robot.connect(calibrate=False)
        robot.bus.disable_torque()
        robot.disconnect()
        return {"status": "ok", "message": "Par desactivado. Sujeta el brazo antes de seguir."}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"No se pudo desactivar el par: {exc}. Corta la alimentación a mano."}


# ------------------------------------------------------------------ trayectorias grabadas
# Una confirmación por TRAYECTORIA (no por keyframe). Mismos gates que motion_execute.
TRAJ_PENDING = HERE / ".pending_trajectory.json"
TRAJ_ERROR_DEG = 15.0      # error de seguimiento que se considera fuera de control
TRAJ_STREAK_LIMIT = 5      # pasos seguidos con ese error antes de abortar
APPROACH_TOL_DEG = 1.0     # tolerancia para dar por llegada la aproximación inicial
APPROACH_MAX_TICKS = 300   # 10 s como máximo para llegar al primer keyframe
TRAJ_STATE_TOL_DEG = 2.0   # cuánto puede haberse movido el brazo entre preview y ejecución


def _trajectory_plan(name: str) -> np.ndarray:
    import motion_lib
    import trajectory as T

    motion = motion_lib.load(name)
    problems = motion_lib.validate(motion)
    if problems:
        raise ValueError("; ".join(problems))
    return T.plan_playback(motion)


def _plan_token(name: str, plan: np.ndarray) -> str:
    raw = name.encode() + np.round(plan, 2).tobytes()
    return hashlib.sha256(raw).hexdigest()[:10]


@mcp.tool()
def trajectory_preview(name: str) -> dict:
    """Plan de una trayectoria grabada (hub/motions/<name>.json) para el brazo real. NO mueve nada.
    Muestra el plan completo al usuario y pide confirmación ÚNICA para toda la trayectoria."""
    problem = _precheck()
    if problem:
        return {"status": "error", "message": problem}
    try:
        plan = _trajectory_plan(name)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Trayectoria no válida: {exc}"}
    try:
        robot = _connect()
        try:
            current = _read(robot)
        finally:
            robot.disconnect()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"No se pudo leer el seguidor: {exc}"}
    if np.any(current < LO) or np.any(current > HI):
        return {"status": "error", "message": "La pose actual está fuera de límites. Revisa la calibración."}
    jump = np.abs(plan[0] - current)
    if np.any(jump > MAX_CALL_DELTA_DEG):
        return {"status": "error",
                "message": f"El primer keyframe está a {np.round(jump, 1).tolist()}° de la pose actual "
                           f"(máx {MAX_CALL_DELTA_DEG}°). Lleva el brazo antes cerca de ese punto."}
    token = _plan_token(name, plan)
    TRAJ_PENDING.write_text(json.dumps({"name": name, "token": token, "start": current.tolist(),
                                        "created": time.time()}))
    spans = np.abs(np.diff(plan, axis=0)).max(axis=0) if len(plan) > 1 else np.zeros(6)
    return {
        "status": "preview",
        "name": name,
        "seconds": round(len(plan) / 30, 2),
        "ticks": len(plan),
        "start_deg": {n: round(float(current[i]), 1) for i, n in enumerate(JOINTS)},
        "first_keyframe_deg": {n: round(float(plan[0][i]), 1) for i, n in enumerate(JOINTS)},
        "max_step_deg_per_tick": round(float(spans.max()), 2) if len(plan) > 1 else 0.0,
        "range_deg": {n: [round(float(plan[:, i].min()), 1), round(float(plan[:, i].max()), 1)]
                      for i, n in enumerate(JOINTS)},
        "confirm_token": token,
        "expires_in_s": PREVIEW_TTL_S,
        "next": "Ejecuta con confirm_token (el token garantiza que se ejecuta exactamente la trayectoria previsualizada).",
        "note": "Nada se ha movido. Si cambia la trayectoria o el brazo se mueve, hay que volver a previsualizar.",
    }


@mcp.tool()
def trajectory_execute(name: str, confirm_token: str) -> dict:
    """MUEVE EL BRAZO REAL siguiendo una trayectoria grabada, a 30 Hz. Requiere ALLOW_REAL_MOTION=1, el
    token del último preview de esa trayectoria y que el brazo siga en la pose del preview. Aborta y
    desactiva el par si el error de seguimiento se mantiene. Úsala solo tras el sí del usuario."""
    if not _motion_enabled():
        return {"status": "blocked", "message": "ALLOW_REAL_MOTION no está en 1. Solo el usuario puede activarlo."}
    if not TRAJ_PENDING.exists():
        return {"status": "blocked", "message": "No hay preview de trayectoria. Llama antes a trajectory_preview."}
    pend = json.loads(TRAJ_PENDING.read_text())
    if time.time() - pend["created"] > PREVIEW_TTL_S:
        TRAJ_PENDING.unlink(missing_ok=True)
        return {"status": "blocked", "message": "El preview ha caducado. Vuelve a llamar a trajectory_preview."}
    if pend["name"] != name or confirm_token != pend["token"]:
        return {"status": "blocked", "message": "El token no corresponde al último preview de esta trayectoria."}
    try:
        plan = _trajectory_plan(name)
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "message": f"La trayectoria ha cambiado: {exc}"}
    if _plan_token(name, plan) != pend["token"]:
        return {"status": "blocked", "message": "La trayectoria ha cambiado desde el preview. Vuelve a previsualizar."}
    problem = _precheck()
    if problem:
        return {"status": "error", "message": problem}

    TRAJ_PENDING.unlink(missing_ok=True)  # un preview sirve para una sola ejecución
    robot = None
    try:
        robot = _connect()
        start = _read(robot)
        if np.any(np.abs(start - np.asarray(pend["start"])) > TRAJ_STATE_TOL_DEG):
            return {"status": "blocked", "message": "El brazo se ha movido desde el preview. Nada se ha movido; "
                                                    "vuelve a previsualizar."}
        period = 1.0 / 30
        step_max = _max_relative_target()
        # 1) aproximación al primer keyframe, despacio
        for _ in range(APPROACH_MAX_TICKS):
            cur = _read(robot)
            if np.all(np.abs(plan[0] - cur) < APPROACH_TOL_DEG):
                break
            _send(robot, cur + np.clip(plan[0] - cur, -step_max, step_max))
            time.sleep(period)
        # 2) la trayectoria, a 30 Hz, vigilando el error de seguimiento
        t0 = time.time()
        streak = 0
        max_err = 0.0
        for k, goal in enumerate(plan):
            tick = time.time()
            _send(robot, goal)
            cur = _read(robot)
            err = float(np.max(np.abs(cur - goal)))
            max_err = max(max_err, err)
            streak = streak + 1 if err > TRAJ_ERROR_DEG else 0
            if streak >= TRAJ_STREAK_LIMIT:
                return {"status": "aborted", "tick": k, "max_error_deg": round(max_err, 2),
                        "message": "Error de seguimiento sostenido: par desactivado. Revisa el brazo."}
            time.sleep(max(0.0, period - (time.time() - tick)))
        final = _read(robot)
        final_err = float(np.max(np.abs(final - plan[-1])))
        return {"status": "ok" if final_err < 3.0 else "warning", "ticks": len(plan),
                "seconds": round(time.time() - t0, 2), "max_tracking_error_deg": round(max_err, 2),
                "final_error_deg": round(final_err, 2),
                "final_deg": {n: round(float(final[i]), 1) for i, n in enumerate(JOINTS)}}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Error durante la trayectoria: {exc}. Par desactivado si fue posible."}
    finally:
        if robot is not None:
            try:
                robot.disconnect()  # desactiva el par por defecto
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    mcp.run()
