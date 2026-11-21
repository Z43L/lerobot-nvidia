"""MCP server para controlar el SO-101 (simulación o brazo real) con VLA.

Backend (variable SO101_BACKEND):
  sim  (por defecto) -> simulación MuJoCo en este Mac
  real               -> brazo físico (real/real_arm.py). Todo movimiento exige ALLOW_REAL_MOTION=1.

Escena (variable SO101_SCENE):
  pick   (por defecto) -> cubo + caja. Herramienta `run_skill` (VLA de propósito general).
  hanoi                -> torre de Hanoi. Claude orquesta: pide el estado, decide el movimiento y
                          llama a `hanoi_pick_place`, que usa el VLA de ese color (uno por color).
                          En real, cada movimiento se confirma con `hanoi_confirm` tras mirar las cámaras.

Unidades: sim en radianes; real en grados (articulaciones) y 0..100 (pinza), como LeRobot.

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
HANOI = SCENE == "hanoi"
OBJECTS = SCENE == "objects"  # objetos cotidianos con agarre físico (sim/objects_env.py)
REAL = BACKEND == "real"

board = None  # tablero de Hanoi (SimBoard o RealBoard)
vla = None    # VLA de la escena "pick" en sim

class _Unavailable:
    """Sustituto del brazo real cuando no arranca (sin puerto, sin calibración, cámara ausente...).
    El servidor MCP sigue vivo y cada herramienta devuelve el motivo, en vez de 'Connection closed'."""

    def __init__(self, reason: str):
        self.reason = reason

    def __getattr__(self, name):
        def _fail(*args, **kwargs):
            return {"status": "error", "message": f"Brazo real no disponible: {self.reason}"}
        return _fail


if REAL:
    sys.path.insert(0, str(ROOT / "real"))
    from real_arm import RealArm  # noqa: E402

    try:
        arm = RealArm()
        if HANOI:
            from hanoi_board import RealBoard  # noqa: E402

            board = RealBoard(arm)
    except Exception as exc:  # noqa: BLE001
        print(f"[so101-real] el brazo real no arrancó: {exc}", file=sys.stderr, flush=True)
        arm = _Unavailable(str(exc))
        board = _Unavailable(str(exc)) if HANOI else None
else:
    from sim_arm import SimArm  # noqa: E402
    from vla_policy import CAMS_HANOI, VLARunner  # noqa: E402

    if HANOI:
        from hanoi_board import SimBoard  # noqa: E402
        from hanoi_env import HanoiSim  # noqa: E402

        arm = HanoiSim()
        _vlas: dict[str, VLARunner] = {}

        def _vla_for(color: str) -> VLARunner:
            """VLA de ese color. Ruta: SO101_HANOI_VLA_<COLOR> (carpeta pretrained_model)."""
            if color not in _vlas:
                path = os.environ.get(f"SO101_HANOI_VLA_{color.upper()}")
                if not path:
                    raise RuntimeError(f"Falta SO101_HANOI_VLA_{color.upper()} con el checkpoint de {color}")
                _vlas[color] = VLARunner(arm, cams=CAMS_HANOI, repo_id=path)
            return _vlas[color]

        board = SimBoard(arm, _vla_for)
    elif OBJECTS:
        from objects_env import ObjectsSim  # noqa: E402

        arm = ObjectsSim()
    else:
        arm = SimArm()
        vla = VLARunner(arm)  # el modelo se carga solo cuando se llama a run_skill

mcp = FastMCP(f"so101-{BACKEND}-{SCENE}")


# ------------------------------------------------------------ comunes
@mcp.tool()
def get_scene() -> dict:
    """Estado del brazo y, con la escena Hanoi, el tablero. Unidades en 'units' (real) o rad (sim)."""
    if board is not None:
        return {"state": arm.get_state(), "board": board.state()}
    return arm.get_scene_summary()


@mcp.tool()
def move_joints(targets: list[float]) -> dict:
    """Mueve el brazo a 6 valores, en este orden:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    Unidades: radianes en sim; grados en real (gripper 0..100).
    En real requiere ALLOW_REAL_MOTION=1 y se mueve en tramos pequeños.
    """
    return arm.move_joints(targets)


@mcp.tool()
def home() -> dict:
    """Lleva el brazo a la posición de reposo (en real requiere ALLOW_REAL_MOTION=1)."""
    return arm.home()


@mcp.tool()
def reset_scene() -> dict:
    """Sim: reinicia la escena. Real: vuelve a reposo (requiere ALLOW_REAL_MOTION=1)."""
    return arm.reset()


@mcp.tool()
def get_camera_image(camera: str = "top") -> Image:
    """Imagen de cámara: 'top' (cenital), 'wrist' (muñeca) o 'table' (vista de la mesa)."""
    path = arm.render(camera)
    return Image(path=str(path))


# ------------------------------------------------------------ Hanoi
if HANOI:

    @mcp.tool()
    def hanoi_state() -> dict:
        """Tablero: discos de abajo a arriba en cada pivote. Pivotes A (amarillo), B (morado), C (naranja).
        Discos de menor a mayor: red, green, blue, cyan, brown. Un disco sólo va encima de uno más grande.
        En real es el estado lógico; un movimiento sin confirmar aparece en 'pending'."""
        return board.state()

    @mcp.tool()
    def hanoi_reset(n_disks: int = 3) -> dict:
        """Reinicia la torre con n_disks (1..5) apilados en el pivote A, el más grande abajo.
        En real sólo fija el estado lógico: hay que colocar los discos a mano según ese estado."""
        return board.reset(n_disks)

    @mcp.tool()
    def hanoi_pick_place(disk: str, to_peg: str, seconds: float = 20.0) -> dict:
        """Ejecuta un movimiento con el VLA de ese color: coger el disco y dejarlo en to_peg.
        disk en {red, green, blue, cyan, brown}; to_peg en {A, B, C}.
        Sim: devuelve ok al terminar. Real: devuelve imágenes y status 'needs_confirmation';
        revisa las imágenes y llama a hanoi_confirm antes de otro movimiento."""
        if REAL:
            return board.pick_place(disk.lower(), to_peg.upper(), seconds=seconds)
        return board.pick_place(disk.lower(), to_peg.upper())

    @mcp.tool()
    def hanoi_confirm(ok: bool) -> dict:
        """Real: confirma el último movimiento tras comprobar las cámaras (ok=true), o lo rechaza
        (ok=false; el estado no cambia; reintenta o corrige con hanoi_set_state)."""
        return board.confirm(ok)

    @mcp.tool()
    def hanoi_set_state(pegs: dict) -> dict:
        """Real: corrige el estado lógico, p. ej. {"A": ["red"], "B": ["green", "blue"], "C": []}.
        Sólo acepta configuraciones legales."""
        return board.set_state(pegs)

    if not REAL:

        @mcp.tool()
        def move_tcp(x: float, y: float, z: float, gripper: float = 1.5) -> dict:
            """Sim: lleva la punta de la pinza a (x, y, z) en metros, con la pinza hacia abajo.
            gripper en radianes: 1.5 abierta, -0.1 cerrada. Pivotes A (0.19,-0.12), B (0.19,0), C (0.19,0.12)."""
            import numpy as np

            q5 = arm.ik(np.array([x, y, z]), q0=arm.data.qpos[:5].copy())
            res = arm.move_joints([*q5.tolist(), gripper])
            arm.grip_update(gripper)
            err = float(np.linalg.norm(arm.tcp() - np.array([x, y, z])))
            return {"tcp": [round(float(v), 4) for v in arm.tcp()], "error_m": round(err, 4),
                    "held": arm.held, "reached": res["reached"]}

# ------------------------------------------------------------ objetos (sim)
if OBJECTS:

    @mcp.tool()
    def list_objects() -> dict:
        """Objetos de la mesa con su posición en metros (x, y, z) y el PC simulado con su zona de entrega 'pc_side'."""
        from objects_env import PC_POS, PC_SIDE_POS
        return {"objects": arm.objects_summary(), "pc": list(PC_POS), "pc_side": list(PC_SIDE_POS)}

    @mcp.tool()
    def pick(object: str) -> dict:
        """Coge un objeto de la mesa con la pinza y lo levanta. object en {bottle, box, cup}.
        Devuelve held=true si el objeto se sostiene físicamente (sin enganche) tras levantarlo."""
        return arm.pick(object)

    @mcp.tool()
    def place(x: float, y: float, z: float = 0.03) -> dict:
        """Lleva el objeto que se sostiene a (x, y, z) en metros y lo suelta.
        Zona junto al PC: pc_side (0.12, -0.19). Mesa libre: x en 0.10..0.25, y en -0.20..0.20."""
        return arm.place(x, y, z)

    @mcp.tool()
    def move_tcp(x: float, y: float, z: float, gripper: float = 1.5) -> dict:
        """Mueve la punta de la pinza a (x, y, z) en metros, con la pinza hacia abajo.
        gripper: 1.5 abierta, -0.1745 cerrada. Base del brazo en (0, 0, 0)."""
        return arm.move_tcp_cmd(x, y, z, gripper)

    @mcp.tool()
    def reset_objects() -> dict:
        """Vuelve a poner el brazo en reposo y los objetos en su posición inicial."""
        return {"state": arm.reset_objects(), "objects": arm.objects_summary()}


# ------------------------------------------------------------ escena pick
if not HANOI and not OBJECTS:

    @mcp.tool()
    def run_skill(instruction: str, seconds: float = 20.0) -> dict:
        """Ejecuta una instrucción con el VLA. Sim: 'Pick up a cube and place in the bin' (en inglés, exacta).
        Real: ejecuta durante `seconds` (requiere ALLOW_REAL_MOTION=1)."""
        if REAL:
            return arm.run_skill(instruction, seconds=seconds)
        return vla.run(instruction, max_steps=int(seconds * 30))


if __name__ == "__main__":
    if os.environ.get("SO101_VIEWER") == "1" and not REAL:
        # Visor en vivo: la ventana vive en el hilo principal (lanzar con mjpython);
        # el servidor MCP corre en un hilo de fondo.
        import threading
        import time

        arm.attach_viewer(realtime=True)
        arm.guard()  # las herramientas del MCP toman el cerrojo; el visor también
        threading.Thread(target=mcp.run, daemon=True).start()
        while arm.viewer.is_running():
            with arm.lock:
                arm.viewer.sync()
            time.sleep(0.03)
    else:
        mcp.run()
