"""MCP del ajedrez real del SO-101: cámaras + YOLO + reglas + brazo + ACT, orquestados por Claude.

Claude hace el bucle (mira -> decide -> mueve -> verifica). Este servidor solo ofrece las herramientas.

  Cámaras y visión (NO mueven nada)
    chess_cameras, chess_camera, chess_see, chess_square_px, chess_verify_move
  Partida (reglas, NO mueve nada)
    chess_state, chess_new, chess_plan, chess_commit
  Jugada completa (una llamada; dry por defecto)
    chess_move: mapa -> coger -> soltar -> pose final -> verificar con foto
  Brazo (leen, o mueven con el permiso del usuario)
    arm_state, arm_gripper_px, arm_jog, arm_probe, arm_servo, arm_stop
  ACT
    chess_act_step: solo corre con un ACT con objetivo (4 valores origen/destino). Con el ACT actual se niega.

PERMISO DE MOVIMIENTO (cualquier herramienta que mueva el brazo):
  1. ALLOW_REAL_MOTION=1 en el entorno del servidor, y
  2. un permiso con caducidad que SOLO el usuario crea en su terminal:  real/authorize_motion.sh <minutos>
     (escribe real/.motion_ok). Sin él, o caducado, nada se mueve. `arm_stop` lo revoca.
  Siempre: pasos de máx. 2 grados, límites de articulación, y la pinza y el brazo se quedan con torque al parar.

Variables: CHESS_YOLO_CAM (0), CHESS_WRIST_CAM (2), CHESS_BOARD_GEOMETRY (vision/board_camara0.json),
SO101_FOLLOWER_PORT, SO101_FOLLOWER_ID, CHESS_ARM=real|none (none: sin brazo, para probar).

Arranque (stdio):  ./.venv312/bin/python hub/chess_mcp.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import chess
import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP, Image

HUB = Path(__file__).resolve().parent
ROOT = HUB.parent
for sub in ("vision", "sim", "real"):
    sys.path.insert(0, str(ROOT / sub))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import chess_config as CFG  # noqa: E402  (única fuente de cámaras, puertos, rutas y límites)

os.environ.setdefault("CHESS_BOARD_GEOMETRY", str(CFG.BOARD_GEOMETRY))
YOLO_CAM, WRIST_CAM = CFG.YOLO_CAM, CFG.WRIST_CAM
GEOMETRY = CFG.BOARD_GEOMETRY
ARM_MODE = os.environ.get("CHESS_ARM", "real").lower()
MOTION_FILE = CFG.MOTION_FILE
JACOBIAN_FILE = ROOT / "real" / "vision_jacobian.json"
GAME_FILE = str(CFG.GAME_FILE)
FRAMES = ROOT / "vision" / "frames"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LIMITS = {"shoulder_pan": (-100, 100), "shoulder_lift": (-120, 120), "elbow_flex": (-110, 110),
          "wrist_flex": (-115, 115), "wrist_roll": (-180, 180), "gripper": (0, 100)}
MAX_JOG_DEG = 3.0   # por llamada a arm_jog
STEP_DEG = CFG.STEP_DEG  # por paso de control; tope = SO101_MAX_RELATIVE_TARGET (2 por defecto, AGENTS.md)
FILES = "abcdefgh"

mcp = FastMCP("so101-chess")
_cv = None
_robot = None


# ------------------------------------------------------------------ utilidades
def _jpeg(img_bgr: np.ndarray, width: int = 1000) -> Image:
    h, w = img_bgr.shape[:2]
    if w > width:
        img_bgr = cv2.resize(img_bgr, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Image(data=buf.tobytes(), format="jpeg")


def _cam_index(camera: str | int) -> int:
    if isinstance(camera, int):
        return camera
    c = str(camera).lower()
    if c in ("yolo", "mesa", "table", "top"):
        return YOLO_CAM
    if c in ("wrist", "muneca", "muñeca", "gripper"):
        return WRIST_CAM
    return int(c)


def _grab(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(index)
    frame = None
    if cap.isOpened():
        for _ in range(8):  # descarta los primeros fotogramas (exposición)
            ok, f = cap.read()
            if ok:
                frame = f
    cap.release()
    if frame is None:
        raise RuntimeError(f"No se pudo leer la cámara {index}. Revisa permiso de cámara e índice (chess_cameras).")
    return frame


def _vision():
    global _cv
    if _cv is None:
        from chess_vision import ChessVision

        _cv = ChessVision(geometry=GEOMETRY, device=os.environ.get("SO101_VLA_DEVICE", "mps"))
    return _cv


def _corners() -> np.ndarray:
    c = json.loads(GEOMETRY.read_text())["corners_px"]
    return np.array([c["TL"], c["TR"], c["BR"], c["BL"]], np.float32)


def _square_px(sq: str) -> tuple[float, float]:
    """Centro de la casilla en píxeles de la cámara de YOLO (usa las esquinas de board_camara0.json)."""
    f, r = FILES.index(sq[0]), int(sq[1])
    u, v = 8 - r + 0.5, 7 - f + 0.5
    H = cv2.getPerspectiveTransform(np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32), _corners())
    x, y = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), H)[0][0]
    return float(x), float(y)


def _board_center_px() -> tuple[float, float]:
    c = _corners()
    return float(c[:, 0].mean()), float(c[:, 1].mean())


def _game() -> dict:
    p = Path(GAME_FILE)
    if not p.exists():
        raise RuntimeError("No hay partida. Usa chess_new.")
    return json.loads(p.read_text())


def _cli(*args: str) -> str:
    env = dict(os.environ, CHESS_BACKEND="real", CHESS_REAL_EXECUTOR="dry", CHESS_GAME_FILE=GAME_FILE)
    r = subprocess.run([sys.executable, str(ROOT / "sim" / "chess_cli.py"), *args], capture_output=True, text=True,
                       env=env, cwd=ROOT, timeout=120)
    return (r.stdout + r.stderr).strip()


# ------------------------------------------------------------------ permiso de movimiento
motion_permission = CFG.motion_permission  # gate único (ALLOW_REAL_MOTION + permiso con caducidad)


def _arm():
    """Conecta el seguidor SIN cámaras de LeRobot (las cámaras las lee OpenCV a demanda). Torque activo al parar."""
    global _robot
    if ARM_MODE == "none":
        raise RuntimeError("CHESS_ARM=none: sin brazo.")
    if _robot is None:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        port = CFG.require_port()
        robot = SO101Follower(SO101FollowerConfig(
            port=port, id=CFG.FOLLOWER_ID, cameras={}, use_degrees=True,
            max_relative_target=STEP_DEG, disable_torque_on_disconnect=False))
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError("El seguidor no tiene calibración.")
        _robot = robot
    return _robot


def _joints_now() -> dict[str, float]:
    obs = _arm().get_observation()
    return {n: float(obs[f"{n}.pos"]) for n in JOINTS}


def _move_relative(deltas: dict[str, float], hz: float = CFG.HZ) -> dict:
    """Mueve en pasos de STEP_DEG hacia (actual + delta), con límites. Requiere permiso. Devuelve antes/después."""
    ok, why = motion_permission()
    if not ok:
        return {"status": "blocked", "message": why}
    for n, d in deltas.items():
        if n not in JOINTS:
            raise ValueError(f"articulación desconocida: {n}")
        if abs(d) > MAX_JOG_DEG * 2:
            return {"status": "rejected", "message": f"{n}: {d}° supera el máximo por llamada ({MAX_JOG_DEG * 2}°)."}
    robot = _arm()
    before = _joints_now()
    # Solo se mueven las articulaciones pedidas. Las demás se quedan donde están (aunque estén fuera de LIMITS:
    # el reposo plegado del brazo está en pan -103 / lift -137 y recortarlas provocaría un tirón no pedido).
    moved = [n for n, d in deltas.items() if d]
    goal = {}
    for n in moved:
        lo, hi = min(LIMITS[n][0], before[n]), max(LIMITS[n][1], before[n])  # nunca más afuera de donde ya está
        goal[n] = float(np.clip(before[n] + deltas[n], lo, hi))
        if before[n] + deltas[n] != goal[n]:
            return {"status": "rejected", "message": f"{n}: {before[n]:.1f}° {deltas[n]:+.1f}° sale del límite {LIMITS[n]}."}
    for _ in range(80):
        ok, why = motion_permission()
        if not ok:
            return {"status": "stopped", "message": why, "before": before, "after": _joints_now()}
        cur = _joints_now()
        if all(abs(goal[n] - cur[n]) < 0.5 for n in moved):
            break
        act = {f"{n}.pos": cur[n] for n in JOINTS}
        for n in moved:
            act[f"{n}.pos"] = cur[n] + float(np.clip(goal[n] - cur[n], -STEP_DEG, STEP_DEG))
        robot.send_action(act)
        time.sleep(1.0 / hz)
    time.sleep(0.4)
    return {"status": "ok", "before": {k: round(v, 1) for k, v in before.items()},
            "after": {k: round(v, 1) for k, v in _joints_now().items()}}


# ------------------------------------------------------------------ cámaras y visión
@mcp.tool()
def chess_cameras() -> dict:
    """Escanea los índices 0..4 y dice cuáles abren y a qué resolución. No mueve nada.
    Configuradas: cámara de YOLO (mesa) y cámara de la muñeca."""
    out = {}
    for i in range(5):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        out[str(i)] = {"abre": bool(ok), "resolucion": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))] if ok else None}
        cap.release()
    return {"yolo_cam": YOLO_CAM, "wrist_cam": WRIST_CAM, "indices": out}


@mcp.tool()
def chess_camera(camera: str = "yolo", overlay: bool = False) -> Image:
    """Imagen de una cámara para que Claude la mire: 'yolo' (mesa, ve el tablero), 'wrist' (muñeca) o un índice.
    overlay=True (solo mesa): dibuja la rejilla de casillas, las piezas que lee YOLO y la punta del brazo. No mueve nada."""
    idx = _cam_index(camera)
    img = _grab(idx)
    if overlay and idx == YOLO_CAM:
        img = _draw_overlay(img)
    return _jpeg(img)


def _draw_overlay(img: np.ndarray) -> np.ndarray:
    from gripper_locator import find_gripper

    out = img.copy()
    H = cv2.getPerspectiveTransform(np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32), _corners())

    def P(u, v):
        x, y = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), H)[0][0]
        return int(x), int(y)

    for i in range(9):
        cv2.line(out, P(i, 0), P(i, 8), (0, 255, 0), 1)
        cv2.line(out, P(0, i), P(8, i), (0, 255, 0), 1)
    for u in range(8):
        for v in range(8):
            x, y = P(u + .5, v + .5)
            cv2.putText(out, f"{FILES[7 - v]}{8 - u}", (x - 10, y + 4), 0, 0.45, (255, 80, 0), 1)
    for sq, (color, piece, conf) in _vision().read_squares(img).items():
        x, y = _square_px(sq)
        cv2.circle(out, (int(x), int(y)), 14, (0, 0, 255) if color == "b" else (0, 200, 255), 2)
        cv2.putText(out, f"{color}{piece}", (int(x) - 12, int(y) - 16), 0, 0.5, (0, 0, 255), 1)
    g = find_gripper(img, _board_center_px())
    if g:
        cv2.drawMarker(out, (int(g["tip_px"][0]), int(g["tip_px"][1])), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
    return out


@mcp.tool()
def chess_see() -> dict:
    """Foto de la cámara de YOLO -> piezas por casilla (color, tipo, confianza), FEN visto, comparación con la partida
    y posición de la punta del brazo en la imagen. No mueve nada. YOLO lee bien peones y ocupación; los tipos de la fila
    trasera con esta cámara fallan: confirma con la imagen (chess_camera overlay=True) y las reglas."""
    from gripper_locator import find_gripper

    img = _grab(YOLO_CAM)
    cv = _vision()
    seen = cv.read_squares(img)
    out = {"piezas_vistas": len(seen),
           "fen_visto": cv.squares_to_fen(seen),
           "casillas": {sq: {"color": c, "tipo": p, "conf": round(float(cf), 2)} for sq, (c, p, cf) in sorted(seen.items())},
           "brazo_px": find_gripper(img, _board_center_px())}
    try:
        board = chess.Board(_game()["fen"])
        exp = cv.board_to_squares(board)
        diffs = [{"casilla": s, "esperada": exp.get(s), "vista": seen.get(s)} for s in sorted(set(exp) | set(seen))
                 if exp.get(s) != (seen.get(s)[:2] if seen.get(s) else None)]
        out["partida"] = {"fen": board.fen(), "casillas_distintas": len(diffs), "diferencias": diffs}
    except RuntimeError:
        out["partida"] = None
    FRAMES.mkdir(exist_ok=True)
    cv2.imwrite(str(FRAMES / "chess_mcp_see.jpg"), _draw_overlay(img))
    out["imagen_con_rejilla"] = str(FRAMES / "chess_mcp_see.jpg")
    return out


@mcp.tool()
def chess_square_px(square: str) -> dict:
    """Píxel del centro de una casilla en la cámara de YOLO (para guiar el brazo) y quién la ocupa según el último escaneo."""
    sq = square.lower()
    if len(sq) != 2 or sq[0] not in FILES or sq[1] not in "12345678":
        raise ValueError("casilla no válida (a1..h8)")
    x, y = _square_px(sq)
    return {"square": sq, "px": [round(x, 1), round(y, 1)]}


@mcp.tool()
def chess_verify_move(uci: str) -> dict:
    """Tras una jugada física: comprueba con la cámara que el origen está vacío, el destino ocupado por el color correcto,
    y que no sobran piezas. NO registra la jugada (usa chess_commit). Parte de la posición de la partida ANTES de jugar."""
    from chess_vision import verify_move

    cv = _vision()
    board = chess.Board(_game()["fen"])
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        return {"ok": False, "motivo": f"jugada ilegal: {uci}"}
    ok, problems, warnings = verify_move(board, move, cv.read_squares(_grab(YOLO_CAM)))
    return {"ok": ok, "problemas": problems, "avisos": warnings, "nota": "YOLO puede perder piezas tapadas: confirma con chess_camera."}


# ------------------------------------------------------------------ partida
@mcp.tool()
def chess_state() -> str:
    """Tablero, turno, jugadas legales e historial de la partida."""
    return _cli("state")


@mcp.tool()
def chess_new(robot_white: bool = True) -> str:
    """Partida nueva (el robot juega blancas por defecto). Guarda en CHESS_GAME_FILE (por defecto hub/chess_game.json)."""
    return _cli("new", *(["--robot-white"] if robot_white else []))


@mcp.tool()
def chess_plan(uci: str) -> dict:
    """Divide una jugada en pasos coger-y-soltar (captura primero al cementerio, enroque en dos pasos). Coordenadas en METROS,
    marco de la base, de hub/chess_board_real.json si existe (si no, de la sim; `geometria_real` lo indica). No mueve nada."""
    from chess_real import DryRunExecutor, Geometry, RealChessBackend

    board = chess.Board(_game()["fen"])
    move = chess.Move.from_uci(uci)
    backend = RealChessBackend(executor=DryRunExecutor(), geometry=Geometry(), verify=None)
    steps = backend.plan(move, board)
    names = {chess.PAWN: "peón", chess.KNIGHT: "caballo", chess.BISHOP: "alfil", chess.ROOK: "torre", chess.QUEEN: "dama", chess.KING: "rey"}
    return {"uci": uci, "geometria": backend.geometry.name, "geometria_real": backend.geometry.real,
            "pasos": [{"label": s["label"], "pieza": names[s["piece"]], "src_xy": s["src_xy"], "dst_xy": s["dst_xy"],
                       "nota": s.get("note")} for s in steps]}


@mcp.tool()
def chess_move(uci: str, executor: str = "dry") -> dict:
    """UNA llamada por jugada del robot (el "ir a h5" del diagrama): mapa -> coger la pieza -> soltar -> pose final -> foto
    y verificación con la cámara. NO registra la jugada: mira la foto y, si está bien, llama a chess_commit.

    executor: dry (por defecto, no mueve nada) | ik (IK + ACT genérico para agarrar) | ik-scripted (IK + agarre guionizado).
    Los que mueven exigen ALLOW_REAL_MOTION=1 en el servidor Y el permiso con caducidad (real/authorize_motion.sh), y que el
    usuario haya dado el SI para ESTA jugada. Claude nunca crea el permiso."""
    from chess_real import Geometry, RealChessBackend, make_executor, moves_arm

    board = chess.Board(_game()["fen"])
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        return {"status": "refused", "message": f"jugada ilegal: {uci}"}
    if executor.lower() != "dry":  # el gate va ANTES de construir el ejecutor (que ya toca calibración y puerto)
        ok, why = motion_permission()
        if not ok:
            return {"status": "blocked", "message": why}
        global _robot
        if _robot is not None:  # el ejecutor abre su propia conexión al puerto: libera la de las herramientas arm_*
            _robot.disconnect()
            _robot = None
    try:
        ex = make_executor(executor)
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
        return {"status": "error", "message": f"no se pudo preparar el ejecutor '{executor}': {e}"}
    cv = None

    def verify(m, before, after):
        nonlocal cv
        from chess_vision import verify_move

        cv = cv or _vision()
        img = _grab(YOLO_CAM)
        FRAMES.mkdir(exist_ok=True)
        cv2.imwrite(str(FRAMES / "chess_move_after.jpg"), _draw_overlay(img))
        okv, problems, _ = verify_move(before, m, cv.read_squares(img))
        return okv, problems

    try:
        backend = RealChessBackend(executor=ex, geometry=Geometry(require_real=moves_arm(ex)), verify=verify)
        res = backend.execute(move, board)
    except RuntimeError as e:
        return {"status": "error", "message": str(e), "siguiente": "Para. No reenvíes comandos. Revisa el brazo (corte de alimentación a mano si algo va mal)."}
    out = {"status": "ok" if res["board_ok"] else "verification_failed", "uci": uci, "executor": ex.name,
           "pasos": res["steps"], "verificada": res["verified"], "problemas": res["bad_squares"]}
    if res["verified"]:
        out["foto"] = str(FRAMES / "chess_move_after.jpg")
        out["siguiente"] = "chess_commit(uci) si la foto confirma la jugada." if res["board_ok"] else "NO registres: corrige las piezas a mano y vuelve a mirar con chess_see."
    else:
        out["siguiente"] = "Ensayo (dry): no se ha movido nada. Para ejecutar de verdad, repite con executor='ik' tras el SI del usuario."
    return out


@mcp.tool()
def chess_commit(uci: str) -> str:
    """Registra la jugada en la partida (el "registrar" del diagrama). Úsalo SOLO tras verificarla con la cámara. Nunca mueve el brazo."""
    return _cli("commit", uci, "--who", "robot")


# ------------------------------------------------------------------ brazo
@mcp.tool()
def arm_state() -> dict:
    """Lee las 6 articulaciones del seguidor (grados; pinza 0..100) y el estado del permiso. No mueve nada."""
    ok, why = motion_permission()
    return {"joints": {k: round(v, 1) for k, v in _joints_now().items()}, "permiso_movimiento": ok, "detalle": why}


@mcp.tool()
def arm_gripper_px() -> dict:
    """Dónde ve la cámara de YOLO la punta del brazo (píxel) y a qué casilla cae. 2D: no da altura. No mueve nada."""
    from gripper_locator import find_gripper

    img = _grab(YOLO_CAM)
    g = find_gripper(img, _board_center_px())
    if g is None:
        return {"visible": False}
    sq = _vision().square_of(*g["tip_px"])
    return {"visible": True, **g, "casilla_bajo_la_punta": sq}


@mcp.tool()
def arm_jog(joint: str, delta_deg: float) -> dict:
    """Mueve UNA articulación delta_deg grados (máx. 3 por llamada, pasos de 2°). Requiere ALLOW_REAL_MOTION=1 y el permiso
    del usuario (real/authorize_motion.sh). Devuelve las articulaciones antes y después."""
    if abs(delta_deg) > MAX_JOG_DEG:
        return {"status": "rejected", "message": f"máximo {MAX_JOG_DEG}° por llamada"}
    return _move_relative({joint: float(delta_deg)})


CAL_FOLLOWER = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json"
MOVE_STEP_DEG = min(2.0, STEP_DEG)  # arm_move: nunca más de 2° por paso
MOVE_MAX_STEPS = 30      # arm_move: tramo corto; Claude mira la cámara y encadena llamadas
STALL_STEPS = 6          # pasos sin avance (<0.3°) antes de dar el tramo por atascado


def _calibrated_limits() -> dict[str, tuple[float, float]]:
    """Rango real de cada motor en grados (use_degrees): ±(range_max-range_min)/2*360/4095; pinza 0..100."""
    cal = json.loads(CAL_FOLLOWER.read_text())
    lim = {}
    for n in JOINTS[:5]:
        half = (cal[n]["range_max"] - cal[n]["range_min"]) / 2 * 360 / 4095
        lim[n] = (-half, half)
    lim["gripper"] = (0.0, 100.0)
    return lim


@mcp.tool()
def arm_limits() -> dict:
    """Rangos reales de cada motor (de la calibración del seguidor, en grados; pinza 0..100) y la pose actual con el margen
    que le queda a cada articulación hasta su tope. No mueve nada. Úsalo antes de planear con arm_move."""
    lim = _calibrated_limits()
    out = {"limites": {n: [round(a, 1), round(b, 1)] for n, (a, b) in lim.items()}}
    try:
        now = _joints_now()
        out["pose"] = {n: round(v, 1) for n, v in now.items()}
        out["margen_a_tope"] = {n: round(min(now[n] - lim[n][0], lim[n][1] - now[n]), 1) for n in JOINTS}
    except Exception as e:  # noqa: BLE001
        out["pose"] = f"no disponible: {e}"
    out["nota"] = ("Topes prácticos vistos: desde el reposo plegado el codo se atascó en ~42° al ir hacia ~2° "
                   "(carga, colisión o tope sin aclarar).")
    return out


@mcp.tool()
def arm_health() -> dict:
    """Lee de cada motor temperatura (°C), carga (±1000), tensión (0.1 V), límite de par, torque activo y corriente. No mueve
    nada. Úsalo si una articulación se atasca: un servo caliente o sin par explica que no avance."""
    bus = _arm().bus
    out = {}
    for reg in ["Present_Temperature", "Present_Load", "Present_Voltage", "Torque_Limit", "Torque_Enable", "Present_Current"]:
        try:
            out[reg] = {m: int(bus.read(reg, m, normalize=False)) for m in bus.motors}
        except Exception as e:  # noqa: BLE001
            out[reg] = f"no disponible: {str(e)[:80]}"
    return out


@mcp.tool()
def arm_move(joints: dict[str, float], max_steps: int = 20, margin_deg: float = 3.0) -> dict:
    """Lleva las articulaciones indicadas a una pose ABSOLUTA (grados; pinza 0..100) en un tramo corto. Las demás no se tocan.
    Límites que impone el código (no Claude): rango calibrado de cada motor menos `margin_deg` (nunca hacia fuera del rango),
    <= 2° por articulación y paso, <= 30 pasos por llamada, permiso de movimiento comprobado en cada paso (si caduca, para), y
    parada si una articulación pedida no avanza en 6 pasos (atasco: carga, tope o colisión). Devuelve antes, después y lo que
    falta. Encadena llamadas mirando chess_camera entre una y otra. Requiere ALLOW_REAL_MOTION=1 y el permiso del usuario."""
    ok, why = motion_permission()
    if not ok:
        return {"status": "blocked", "message": why}
    unknown = [n for n in joints if n not in JOINTS]
    if unknown:
        return {"status": "rejected", "message": f"articulaciones desconocidas: {unknown}"}
    max_steps = int(min(max(max_steps, 1), MOVE_MAX_STEPS))
    margin_deg = max(float(margin_deg), 1.0)
    lim = _calibrated_limits()
    before = _joints_now()
    goal = {}
    for n, g in joints.items():
        lo, hi = lim[n][0] + margin_deg, lim[n][1] - margin_deg
        if n == "gripper":
            lo, hi = lim[n]
        if not lo <= g <= hi and not (g < lo and g > before[n]) and not (g > hi and g < before[n]):
            return {"status": "rejected", "message": f"{n}: objetivo {g:.1f}° fuera del rango permitido [{lo:.1f}, {hi:.1f}]."}
        goal[n] = float(g)
    robot = _arm()
    hist = []
    status, message = "ok", None
    for k in range(max_steps):
        ok, why = motion_permission()
        if not ok:
            status, message = "stopped", why
            break
        cur = _joints_now()
        d = {n: goal[n] - cur[n] for n in goal}
        if all(abs(v) < 0.7 for v in d.values()):
            break
        act = {f"{n}.pos": cur[n] for n in JOINTS}
        for n, v in d.items():
            act[f"{n}.pos"] = cur[n] + float(np.clip(v, -MOVE_STEP_DEG, MOVE_STEP_DEG))
        robot.send_action(act)
        time.sleep(1 / 30)
        hist.append(cur)
        if len(hist) > STALL_STEPS:
            old = hist[-STALL_STEPS - 1]
            stuck = [n for n in goal if abs(d[n]) > 2.0 and abs(cur[n] - old[n]) < 0.3]
            if stuck:
                status, message = "stalled", f"sin avance en {stuck}: carga, tope o colisión. Paro."
                break
    time.sleep(0.3)
    after = _joints_now()
    out = {"status": status, "before": {k: round(v, 1) for k, v in before.items()},
           "after": {k: round(v, 1) for k, v in after.items()},
           "falta_grados": {n: round(goal[n] - after[n], 1) for n in goal}, "pasos": len(hist)}
    if message:
        out["message"] = message
    return out


@mcp.tool()
def arm_probe(joints: list[str] | None = None, delta_deg: float = 3.0) -> dict:
    """Estima cuántos píxeles se mueve la punta del brazo en la cámara de YOLO por grado de cada articulación (jacobiano de
    imagen, sin calibrar el mundo). Mueve cada articulación +delta_deg y la devuelve. Requiere permiso de movimiento.
    Guarda real/vision_jacobian.json. Solo vale mientras la cámara y el tablero no se muevan."""
    from gripper_locator import find_gripper

    joints = joints or ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    ref = _board_center_px()
    cols, report = {}, {}
    for j in joints:
        p0 = find_gripper(_grab(YOLO_CAM), ref)
        if p0 is None:
            return {"status": "error", "message": f"no veo el brazo en la cámara antes de mover {j}"}
        r1 = _move_relative({j: delta_deg})
        if r1["status"] != "ok":
            return {"status": r1["status"], "message": r1.get("message"), "hecho": report}
        p1 = find_gripper(_grab(YOLO_CAM), ref)
        r2 = _move_relative({j: -delta_deg})
        if p1 is None:
            return {"status": "error", "message": f"perdí el brazo en la imagen al mover {j}", "hecho": report, "vuelta": r2["status"]}
        d = (np.array(p1["tip_px"]) - np.array(p0["tip_px"])) / delta_deg
        cols[j] = d.tolist()
        report[j] = {"px_por_grado": [round(float(d[0]), 2), round(float(d[1]), 2)], "vuelta": r2["status"]}
    JACOBIAN_FILE.write_text(json.dumps({"joints": cols, "delta_deg": delta_deg, "cam": YOLO_CAM,
                                         "geometria": str(GEOMETRY.name)}, indent=2))
    return {"status": "ok", "jacobiano": report, "fichero": str(JACOBIAN_FILE)}


@mcp.tool()
def arm_servo(square: str, execute: bool = False, gain: float = 0.5) -> dict:
    """Propone (o con execute=True ejecuta, con permiso) un paso para acercar la punta del brazo a la casilla `square`, según
    el error en píxeles y el jacobiano de arm_probe. Un paso mueve como mucho 3° por articulación. Repite mirando la imagen.
    No controla la altura (una sola cámara): la altura la gestiona el ACT/el usuario."""
    from gripper_locator import find_gripper

    if not JACOBIAN_FILE.exists():
        return {"status": "error", "message": "falta el jacobiano: ejecuta arm_probe primero."}
    jac = json.loads(JACOBIAN_FILE.read_text())["joints"]
    names = list(jac)
    J = np.array([jac[n] for n in names]).T  # 2 x N
    g = find_gripper(_grab(YOLO_CAM), _board_center_px())
    if g is None:
        return {"status": "error", "message": "no veo el brazo en la cámara."}
    target = np.array(_square_px(square.lower()))
    err = target - np.array(g["tip_px"])
    dq = gain * np.linalg.solve(J.T @ J + 1e-3 * np.eye(len(names)), J.T @ err)
    dq = np.clip(dq, -MAX_JOG_DEG, MAX_JOG_DEG)
    proposal = {n: round(float(v), 2) for n, v in zip(names, dq)}
    out = {"error_px": [round(float(err[0]), 1), round(float(err[1]), 1)], "distancia_px": round(float(np.linalg.norm(err)), 1),
           "tip_px": g["tip_px"], "objetivo_px": [round(float(target[0]), 1), round(float(target[1]), 1)],
           "propuesta_grados": proposal, "ejecutado": False}
    if execute:
        res = _move_relative(proposal)
        out["movimiento"] = res
        out["ejecutado"] = res["status"] == "ok"
    return out


@mcp.tool()
def arm_stop() -> dict:
    """Revoca el permiso de movimiento y no envía más órdenes. NO corta el torque (el brazo se queda donde está).
    Si algo va mal físicamente: corte de alimentación a mano."""
    MOTION_FILE.unlink(missing_ok=True)
    return {"status": "stopped", "message": "Permiso revocado. El usuario debe volver a autorizar para mover."}


# ------------------------------------------------------------------ ACT con objetivo
@mcp.tool()
def chess_act_step(src: str, dst: str, seconds: float = 15.0, checkpoint: str = "") -> dict:
    """Lanza un ACT CON OBJETIVO (4 valores: origen y destino) para coger la pieza en `src` y soltarla en `dst`.
    Solo corre si el checkpoint tiene observation.environment_state de 4 valores y hay permiso de movimiento. El ACT genérico
    de modeloACT (sin objetivo) se rechaza. Antes de arrancar comprueba que la primera acción no se aleje más de 15° de la
    pose actual. Primera prueba: <= 20 s. NO PROBADO con un checkpoint real (no existe todavía)."""
    from deploy_goal import ENV_STATE_KEY, build_obs, load_policy, step_policy
    from goal_utils import parse_goal
    from cam_utils import to_train_size

    ckpt_text = checkpoint or os.environ.get("CHESS_ACT_GOAL_CHECKPOINT", "")
    ckpt = Path(ckpt_text)
    if not ckpt_text or not (ckpt / "config.json").exists():
        return {"status": "refused", "message": "No hay checkpoint de ACT con objetivo. Hay que grabar con real/record_goal.py --pick-place y entrenar (real/train_act.sh)."}
    if seconds > 20:
        return {"status": "refused", "message": "Máximo 20 s en esta herramienta."}
    device = os.environ.get("SO101_VLA_DEVICE", "mps")
    policy, pre, post = load_policy(ckpt, device)
    feats = policy.config.input_features
    if ENV_STATE_KEY not in feats or feats[ENV_STATE_KEY].shape[0] != 4:
        return {"status": "refused", "message": "El checkpoint no es un ACT con objetivo de 4 valores (origen y destino)."}
    ok, why = motion_permission()
    if not ok:
        return {"status": "blocked", "message": why}
    goal = parse_goal(f"{src},{dst}")
    motors = [f"{n}.pos" for n in JOINTS]
    cam_for = {"observation.images.camera1": YOLO_CAM, "observation.images.camera2": WRIST_CAM}
    missing = [k for k in policy.config.image_features if k not in cam_for]
    if missing:
        return {"status": "refused", "message": f"El checkpoint necesita cámaras sin asignar: {missing}"}
    robot = _arm()

    def observe():
        j = _joints_now()
        state = np.array([j[n] for n in JOINTS], np.float32)
        frames = {k: to_train_size(cv2.cvtColor(_grab(cam_for[k]), cv2.COLOR_BGR2RGB)) for k in policy.config.image_features}
        return state, frames

    policy.reset()
    state, frames = observe()
    first = step_policy(policy, pre, post, build_obs(policy, state, goal, frames), device, motors)
    jump = max(abs(first[m] - state[i]) for i, m in enumerate(motors))
    if jump > 15:
        return {"status": "refused", "message": f"La primera acción se aleja {jump:.1f}° de la pose actual (máx. 15°)."}
    policy.reset()
    steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        ok, why = motion_permission()
        if not ok:
            return {"status": "stopped", "message": why, "steps": steps}
        t = time.perf_counter()
        state, frames = observe()
        act = step_policy(policy, pre, post, build_obs(policy, state, goal, frames), device, motors)
        robot.send_action(act)
        steps += 1
        time.sleep(max(0.0, 1 / CFG.HZ - (time.perf_counter() - t)))
    return {"status": "done", "steps": steps, "seconds": round(time.time() - t0, 1), "src": src, "dst": dst,
            "siguiente": "Verifica con chess_verify_move."}


if __name__ == "__main__":
    mcp.run()
