"""Servidor MCP de puesta en marcha del SO-101 (hub): diagnóstico, configuración y lectura SIN mover el brazo.

Pensado para que Claude prepare el puesto del hackathon con skills:
  - setup_status      : puertos, cámaras, calibraciones, Python/MPS. Solo lectura.
  - write_config      : escribe real/env.local.sh con los puertos, IDs y cámaras que da el usuario.
  - read_positions    : lee las articulaciones del seguidor con la calibración guardada. No mueve el brazo.
  - calibration_plan  : devuelve el comando exacto de calibración para que lo ejecute una persona en
                        su terminal (la calibración de LeRobot pide mover cada articulación a mano).

Lo que NO hace (y está en otros servidores con gates):
  - Ningún movimiento. Eso es so101-real (sim/mcp_server.py) con ALLOW_REAL_MOTION=1.

Arranque (stdio):  ./.venv312/bin/python hub/so101_hub.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "real"
ENV_LOCAL = REAL_DIR / "env.local.sh"
CAL_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

mcp = FastMCP("so101-hub")


# ------------------------------------------------------------------ utilidades
def _serial_ports() -> list[str]:
    return sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*"))


def _probe_cameras(max_index: int = 4) -> dict:
    """Abre cada índice de cámara y dice si entrega imágenes. Un fallo suele ser permiso de cámara."""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return {"error": f"OpenCV no disponible: {exc}"}
    found = {}
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        ok = bool(cap.isOpened() and cap.read()[0])
        cap.release()
        found[str(idx)] = "ok" if ok else "sin imagen"
    if not any(v == "ok" for v in found.values()):
        found["_hint"] = ("Ninguna cámara responde. Revisa el permiso de cámara de la app que lanza el MCP "
                          "(Ajustes del Sistema > Privacidad y seguridad > Cámara).")
    return found


def _calibration_files(robot_id: str) -> list[str]:
    if not CAL_DIR.exists():
        return []
    return [str(p) for p in CAL_DIR.glob(f"**/{robot_id}.json")]


def _read_env_local() -> dict:
    """Lee las variables exportadas en real/env.local.sh (sin ejecutar el shell)."""
    out: dict[str, str] = {}
    if not ENV_LOCAL.exists():
        return out
    for line in ENV_LOCAL.read_text().splitlines():
        m = re.match(r'\s*export\s+(\w+)="?([^"]*)"?', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _python_info() -> dict:
    info = {"python": sys.version.split()[0], "executable": sys.executable}
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"no disponible: {exc}"
    try:
        import lerobot
        info["lerobot"] = getattr(lerobot, "__version__", "instalado")
    except Exception as exc:  # noqa: BLE001
        info["lerobot"] = f"no disponible: {exc}"
    return info


# ------------------------------------------------------------------ herramientas
@mcp.tool()
def setup_status() -> dict:
    """Estado del puesto en solo lectura: puertos USB-serie, cámaras, calibraciones guardadas,
    configuración actual (real/env.local.sh) y entorno Python. No mueve nada ni abre el brazo."""
    cfg = _read_env_local()
    follower_id = cfg.get("SO101_FOLLOWER_ID", os.environ.get("SO101_FOLLOWER_ID", "so101_follower"))
    leader_id = cfg.get("SO101_LEADER_ID", os.environ.get("SO101_LEADER_ID", "so101_leader"))
    ports = _serial_ports()
    problems = []
    if not ports:
        problems.append("No hay puertos USB-serie. Conecta el seguidor (y el líder) con cable de datos.")
    for label, var in [("seguidor", "SO101_FOLLOWER_PORT"), ("líder", "SO101_LEADER_PORT")]:
        port = cfg.get(var, "")
        if not port:
            problems.append(f"Falta {var} en real/env.local.sh (puerto del {label}).")
        elif port not in ports:
            problems.append(f"{var}={port} no existe ahora mismo.")
    if not _calibration_files(follower_id):
        problems.append(f"No hay calibración del seguidor '{follower_id}'. Usa calibration_plan.")
    cams = _probe_cameras()
    return {
        "ports": ports,
        "config": {k: v for k, v in cfg.items()},
        "calibration": {"follower": _calibration_files(follower_id), "leader": _calibration_files(leader_id)},
        "cameras": cams,
        "env": _python_info(),
        "problems": problems,
        "ready_for_read": not any("USB" in p or "FOLLOWER_PORT" in p or "calibración del seguidor" in p for p in problems),
    }


@mcp.tool()
def write_config(follower_port: str = "", leader_port: str = "", follower_id: str = "so101_follower",
                 leader_id: str = "so101_leader", top_cam: int = 0, wrist_cam: int = 1,
                 table_cam: int = -1, vla_device: str = "mps") -> dict:
    """Escribe real/env.local.sh con los valores que dio el usuario. No mueve el brazo.
    Los puertos deben existir en /dev. Tras escribirlo hay que reiniciar los servidores MCP.
    vla_device: 'mps' (Mac, por defecto). Usa 'cuda' solo si el puesto tiene GPU NVIDIA y el usuario lo pide."""
    errors = []
    existing = _serial_ports()
    for label, port in [("follower_port", follower_port), ("leader_port", leader_port)]:
        if port and port not in existing:
            errors.append(f"{label}={port} no existe en /dev ahora mismo.")
    if vla_device not in ("mps", "cuda", "cpu"):
        errors.append("vla_device debe ser mps, cuda o cpu.")
    if errors:
        return {"status": "error", "errors": errors}
    lines = [
        "# Generado por so101_hub.write_config. Puertos y cámaras del puesto.",
        f'export SO101_FOLLOWER_PORT="{follower_port}"',
        f'export SO101_LEADER_PORT="{leader_port}"',
        f'export SO101_FOLLOWER_ID="{follower_id}"',
        f'export SO101_LEADER_ID="{leader_id}"',
        f'export SO101_TOP_CAM="{top_cam}"',
        f'export SO101_WRIST_CAM="{wrist_cam}"',
        f'export SO101_TABLE_CAM="{table_cam}"',
        f'export SO101_VLA_DEVICE="{vla_device}"',
    ]
    ENV_LOCAL.write_text("\n".join(lines) + "\n")
    return {"status": "ok", "file": str(ENV_LOCAL), "note": "Reinicia los servidores MCP para que lean la nueva configuración."}


@mcp.tool()
def read_positions() -> dict:
    """Lee las 6 articulaciones del SEGUIDOR con su calibración guardada. Solo lectura: no envía órdenes.
    Devuelve grados (5 articulaciones) y 0..100 (pinza). Requiere puerto y calibración."""
    cfg = _read_env_local()
    port = cfg.get("SO101_FOLLOWER_PORT") or os.environ.get("SO101_FOLLOWER_PORT", "")
    follower_id = cfg.get("SO101_FOLLOWER_ID") or os.environ.get("SO101_FOLLOWER_ID", "so101_follower")
    if not port:
        return {"status": "error", "message": "Falta SO101_FOLLOWER_PORT. Usa write_config."}
    if port not in _serial_ports():
        return {"status": "error", "message": f"{port} no existe ahora mismo. Revisa el cable de datos."}
    if not _calibration_files(follower_id):
        return {"status": "error", "message": "Sin calibración del seguidor. Usa calibration_plan."}
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"LeRobot no disponible: {exc}"}
    robot = SO101Follower(SO101FollowerConfig(port=port, id=follower_id, use_degrees=True))
    try:
        robot.connect(calibrate=False)
        obs = robot.get_observation()
        joints = {n: round(float(obs[f"{n}.pos"]), 2) for n in JOINTS}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"No se pudo leer el seguidor: {exc}"}
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return {"status": "ok", "units": {"joints": "degrees", "gripper": "0..100"}, "joints": joints,
            "note": "Lectura sin movimiento. Compara con la pose física que tiene el brazo."}


@mcp.tool()
def calibration_plan(role: str = "follower") -> dict:
    """Devuelve el comando de calibración para el seguidor ('follower') o el líder ('leader').
    LeRobot pide mover cada articulación a mano por su rango: debe hacerlo una persona en su terminal,
    no Claude. Antes de calibrar, confirma con la organización si ya hay calibración del puesto."""
    cfg = _read_env_local()
    if role not in ("follower", "leader"):
        return {"status": "error", "message": "role debe ser 'follower' o 'leader'."}
    if role == "follower":
        port = cfg.get("SO101_FOLLOWER_PORT", "")
        rid = cfg.get("SO101_FOLLOWER_ID", "so101_follower")
        cmd = f"./.venv312/bin/lerobot-calibrate --robot.type=so101_follower --robot.port={port} --robot.id={rid}"
    else:
        port = cfg.get("SO101_LEADER_PORT", "")
        rid = cfg.get("SO101_LEADER_ID", "so101_leader")
        cmd = f"./.venv312/bin/lerobot-calibrate --teleop.type=so101_leader --teleop.port={port} --teleop.id={rid}"
    if not port:
        return {"status": "error", "message": f"Falta el puerto del {role}. Usa write_config."}
    return {
        "status": "manual",
        "command": cmd,
        "steps": [
            "Confirma que el brazo está sin alimentar o con el área libre y una persona junto al robot.",
            "Ejecuta el comando en tu terminal (no lo lanza Claude).",
            "Sigue las instrucciones: mueve cada articulación por todo su rango y pulsa Enter.",
            "Al terminar, vuelve a Claude y pide read_positions para comprobar la lectura.",
        ],
        "calibration_dir": str(CAL_DIR),
    }


if __name__ == "__main__":
    mcp.run()
