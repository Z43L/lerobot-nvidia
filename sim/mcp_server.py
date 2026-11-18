"""MCP server para controlar el SO-101 con VLA.

Backend (variable SO101_BACKEND):
  sim  (por defecto) -> simulación MuJoCo en este Mac
  real               -> brazo físico, con real/real_arm.py

Escena en sim (variable SO101_SCENE):
  pick   (por defecto) -> cubo + caja (sim/so101/scene_vla.xml)
  hanoi                -> torre de Hanoi con 3 discos y 3 pivotes (sim/hanoi_env.py)

Claude orquesta: consulta estado, mira cámaras, manda posiciones o lanza una instrucción
a un VLA (`run_skill`). En real, las unidades son normalizadas (-100..100); en sim, radianes.

Con SO101_SCENE=hanoi, Claude resuelve la torre: el VLA sólo se usa para `hanoi_pick_place`
(coger una pieza y moverla); el resto se controla con `move_tcp`, `move_joints` y las cámaras.

Arranque (stdio):  ./.venv312/bin/python sim/mcp_server.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

BACKEND = os.environ.get("SO101_BACKEND", "sim").lower()
SCENE = os.environ.get("SO101_SCENE", "pick").lower()
UNITS = "normalized" if BACKEND == "real" else "rad"

vla = None
if BACKEND == "real":
    sys.path.insert(0, str(ROOT / "real"))
    from real_arm import RealArm  # noqa: E402

    arm = RealArm()
elif SCENE == "hanoi":
    from hanoi_env import DISK_ORDER, DISKS, PEGS, HanoiSim, hanoi_instruction  # noqa: E402
    from vla_policy import CAMS_HANOI, VLARunner  # noqa: E402

    arm = HanoiSim()
    vla = None  # un VLA por color de disco, cargado al primer uso (ver _vla_for)
    _vlas: dict[str, VLARunner] = {}

    def _vla_for(color: str) -> VLARunner:
        """VLA entrenado para ese color. Ruta: variable SO101_HANOI_VLA_<COLOR> (carpeta pretrained_model)."""
        if color not in _vlas:
            path = os.environ.get(f"SO101_HANOI_VLA_{color.upper()}")
            if not path:
                raise RuntimeError(f"Falta SO101_HANOI_VLA_{color.upper()} con el checkpoint del VLA de {color}")
            _vlas[color] = VLARunner(arm, cams=CAMS_HANOI, repo_id=path)
        return _vlas[color]
else:
    from sim_arm import SimArm  # noqa: E402
    from vla_policy import VLARunner  # noqa: E402

    arm = SimArm()
    vla = VLARunner(arm)  # el modelo se carga solo cuando se llama a run_skill

mcp = FastMCP(f"so101-{BACKEND}-{SCENE}" if BACKEND != "real" else "so101-real")


@mcp.tool()
def get_scene() -> dict:
    """Estado del brazo y, en sim, la escena (cubo, o pivotes y discos en Hanoi). Unidades en 'units'."""
    return arm.get_scene_summary()


@mcp.tool()
def move_joints(targets: list[float]) -> dict:
    """Mueve el brazo a 6 valores, en este orden:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    Unidades: radianes en sim; escala normalizada (-100..100, gripper 0..100) en real.
    Los valores se recortan a los límites y el movimiento se hace en tramos suaves.
    """
    return arm.move_joints(targets)


@mcp.tool()
def home() -> dict:
    """Lleva el brazo a la posición de reposo."""
    return arm.home()


@mcp.tool()
def reset_scene() -> dict:
    """Sim: reinicia la escena (en Hanoi, los 3 discos apilados en el pivote A). Real: vuelve a reposo."""
    return arm.reset()


@mcp.tool()
def get_camera_image(camera: str = "top") -> Image:
    """Devuelve la imagen de la cámara: 'table' (vista completa de la mesa), 'top' (cenital) o 'wrist' (muñeca)."""
    path = arm.render(camera)
    return Image(path=str(path))


if SCENE == "hanoi" and BACKEND != "real":
    # Claude resuelve la Hanoi: planifica y decide cada movimiento. El VLA sólo hace
    # "coger una pieza y moverla"; el resto (posición, pinza, cámaras) se controla aquí.

    @mcp.tool()
    def hanoi_state() -> dict:
        """Estado de la torre: discos de abajo a arriba en cada pivote.
        Pivotes: A (amarillo), B (morado), C (naranja). Discos por tamaño, de menor a mayor:
        red, green, blue, cyan, brown. Un disco sólo puede ir encima de uno más grande.
        'held' indica si hay un disco enganchado en la pinza."""
        pegs = arm.get_hanoi_state()
        return {"pegs": pegs, "held": arm.held, "disks": sorted(arm.active, key=DISK_ORDER.index)}

    @mcp.tool()
    def hanoi_reset(n_disks: int = 3) -> dict:
        """Reinicia la torre con n_disks (1..5) apilados en el pivote A, el más grande abajo."""
        arm.reset_hanoi(n=n_disks)
        return hanoi_state()

    @mcp.tool()
    def hanoi_pick_place(disk: str, to_peg: str, max_steps: int = 600) -> dict:
        """Usa el VLA de ese color (uno por color) para coger UN disco y dejarlo en un pivote. 'disk' en {red, green, blue, cyan, brown},
        'to_peg' en {A, B, C}. El disco debe estar encima de su pivote y no puede ir sobre uno más pequeño.
        Devuelve ok=true si el disco queda encima del pivote destino y la pinza está abierta."""
        disk, to_peg = disk.lower(), to_peg.upper()
        if disk not in DISKS or to_peg not in PEGS or disk not in arm.active:
            return {"ok": False, "error": "disco o pivote no válido para la torre actual"}
        pegs = arm.get_hanoi_state()
        src = next((p for p in PEGS if pegs[p][-1:] == [disk]), None)
        if src is None:
            return {"ok": False, "error": f"{disk} no está encima de ningún pivote", "pegs": pegs}
        if src == to_peg:
            return {"ok": True, "note": "ya está en ese pivote", "pegs": pegs}
        if pegs[to_peg] and DISKS[pegs[to_peg][-1]]["size"] < DISKS[disk]["size"]:
            return {"ok": False, "error": f"no se puede poner {disk} sobre {pegs[to_peg][-1]}", "pegs": pegs}

        def done() -> bool:
            return arm.held is None and arm.get_hanoi_state()[to_peg][-1:] == [disk]

        res = _vla_for(disk).run(hanoi_instruction(disk, to_peg), max_steps=max_steps, success=done)
        res["move"] = [disk, src, to_peg]
        res["ok"] = bool(res.get("success"))
        res["pegs"] = arm.get_hanoi_state()
        return res

    @mcp.tool()
    def move_tcp(x: float, y: float, z: float, gripper: float = 1.5) -> dict:
        """Control directo: lleva la punta de la pinza (TCP) a (x, y, z) en metros, con la pinza hacia abajo.
        gripper en radianes: 1.5 abierta, -0.1 cerrada. Al cerrar sobre un disco a <2 cm, se engancha.
        Referencias: base del brazo en (0,0,0); pivotes A (0.19,-0.12), B (0.19,0), C (0.19,0.12); altura de
        transporte z=0.08. Devuelve el error final del TCP."""
        import numpy as np

        q5 = arm.ik(np.array([x, y, z]), q0=arm.data.qpos[:5].copy())
        res = arm.move_joints([*q5.tolist(), gripper])
        arm.grip_update(gripper)
        err = float(np.linalg.norm(arm.tcp() - np.array([x, y, z])))
        return {"tcp": [round(float(v), 4) for v in arm.tcp()], "error_m": round(err, 4),
                "held": arm.held, "reached": res["reached"]}

elif BACKEND != "real":

    @mcp.tool()
    def run_skill(instruction: str, max_steps: int = 600) -> dict:
        """Ejecuta una instrucción con el VLA SmolVLA. Instrucciones válidas, en inglés y exactas, por ejemplo:
        'Pick up a cube and place in the bin'.
        """
        return vla.run(instruction, max_steps=max_steps)

else:

    @mcp.tool()
    def run_skill(instruction: str) -> dict:
        """Ejecuta una instrucción con el VLA en el brazo real (requiere ALLOW_REAL_MOTION=1 y área libre)."""
        return arm.run_skill(instruction)


if __name__ == "__main__":
    if os.environ.get("SO101_VIEWER") == "1" and BACKEND != "real":
        # Visor en vivo: la ventana vive en el hilo principal (lanzar con mjpython);
        # el servidor MCP corre en un hilo de fondo.
        import threading
        import time

        arm.attach_viewer(realtime=True)
        threading.Thread(target=mcp.run, daemon=True).start()
        while arm.viewer.is_running():
            arm.viewer.sync()
            time.sleep(0.03)
    else:
        mcp.run()
