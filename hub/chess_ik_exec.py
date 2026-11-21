"""Ejecutor de jugadas del SO-101 REAL: IK para llegar a la pieza + ventana de ACT para el agarre, y vuelta al origen.

Es el "camino 2": la IK lleva la pinza hasta encima de la pieza y el ACT genérico (SIN objetivo, p. ej. el de 30.000 pasos
de act_tray_a100_100k_ckpts) toma el control durante unos segundos, CONFINADO a una caja de seguridad alrededor de la casilla.
El ACT no sabe qué pieza coger; lo único que se le pide es que baje, cierre la pinza y suba. Eso NO está probado: por eso hay
tambien `--grasp scripted` (IK baja, cierra la pinza hasta que se detiene, sube) para comparar y como plan B.

Secuencia de cada coger-y-soltar (cada jugada termina en el ORIGEN, hub/chess_home.json):
  origen -> encima(src) -> [ACT | bajar+cerrar] -> subir -> encima(dst) -> bajar -> abrir -> subir -> ORIGEN

Movimiento directo (politica del 21-nov). ALLOW_REAL_MOTION=0 lo para todo; CHESS_REQUIRE_PERMISSION=1 vuelve a exigir el
permiso con caducidad. Limites mecanicos siempre: pasos de como mucho 2 grados por articulacion, torque encendido al parar.
Para en el primer fallo (no vuelve solo).

Sin hardware (no mueve nada):
  ./.venv312/bin/python hub/chess_ik_exec.py plan e2e4                 # waypoints, limites, pasos, tiempo
  ./.venv312/bin/python hub/chess_ik_exec.py act-offline --square e2   # que haria el ACT desde encima de la casilla
Con el brazo:
  ./.venv312/bin/python hub/chess_ik_exec.py run e2e4 --grasp act --seconds 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import chess
import cv2
import numpy as np

HUB = Path(__file__).resolve().parent
ROOT = HUB.parent
for sub in ("real", "sim", "hub"):
    sys.path.insert(0, str(ROOT / sub))

import board_touch_calib as b  # noqa: E402
from chess_ik import J4, Kin, steps_between  # noqa: E402
import chess_wrist as WR  # noqa: E402

import chess_config as CFG  # noqa: E402  (única fuente de cámaras, puertos, rutas y límites)

MOTION_FILE = CFG.MOTION_FILE
HOME_FILE = HUB / "chess_home.json"
DEFAULT_ACT = CFG.ACT_GENERIC_CKPT
STEP_DEG = CFG.STEP_DEG   # tope = SO101_MAX_RELATIVE_TARGET (AGENTS.md: empezar con 2 grados/paso)
HZ = CFG.HZ
BIAS_KI = 0.25             # ganancia del sesgo integral (por tick)
BIAS_MAX = 3.0             # grados: sesgo maximo por articulacion
GRIP_OPEN = 40.0            # apertura en el transito alto (ORIGEN). OJO: 40 = ~7,4 cm entre puntas, barre 2 casillas
# Apertura al acercarse a una pieza: lo justo para que entre sin tocarla ni tocar las vecinas (usuario, 21-nov).
# Pinza 0..100 = 140 grados de servo; dedo de 8,2 cm: hueco ~ 2*0.082*sin((v-1.8)*1.4/2 grados). 14 -> ~2,4 cm.
GRIP_APPROACH = {chess.PAWN: 12.0, chess.ROOK: 16.0, chess.KNIGHT: 16.0, chess.BISHOP: 15.0, chess.QUEEN: 17.0, chess.KING: 17.0}
if os.environ.get("CHESS_GRIP_OPEN"):
    GRIP_APPROACH = {k: float(os.environ["CHESS_GRIP_OPEN"]) for k in GRIP_APPROACH}


def gap_cm(v: float) -> float:
    """Hueco aproximado entre las puntas de los dedos (cm) para un valor de pinza."""
    return 2 * 8.2 * float(np.sin(np.radians((v - 1.8) * 1.4 / 2)))
GRIP_CLOSE_STEP = 1.5       # % por tick al cerrar (solo close_until_stall, OBSOLETO: en vacio se para en ~15 por friccion)
GRIP_HOLD_CMD = 0.0         # orden minima de la pinza (cerrada del todo)
GRIP_CLOSE_RATE = float(os.environ.get("CHESS_GRIP_RATE", "1.0"))      # unidades/tick al cerrar (suave)
GRIP_CONTACT_LAG = 3.0      # si la pinza va mas de esto por detras de la orden, hay contacto
GRIP_SQUEEZE = float(os.environ.get("CHESS_GRIP_SQUEEZE", "4"))     # presion: orden = contacto - esto (usuario: cierre suave)
GRIP_EMPTY_MAX = 4.5        # medido 21-nov: cerrada en vacio = 1,8. Por encima de esto hay una pieza entre los dedos
GRASP_OFFSET_FILE = HUB / "chess_grasp_offset.json"   # correccion xy/z del agarre medida con `grasp-probe`
APPROACH_ABOVE = 0.07       # m por encima de la altura de agarre (la punta queda por encima del rey en el transito)
PLACE_EXTRA = 0.004         # m por encima del agarre al soltar
ENVELOPE_MARGIN = 8.0       # grados de holgura de la caja de seguridad del ACT a cada lado del tramo aproximacion-agarre
# Alturas de las piezas (m). MEDIDAS: peon 3,5 cm y rey 5,5 cm (usuario, 2026-11-21). El resto son ESTIMACIONES sin medir.
PIECE_HEIGHT = {chess.PAWN: 0.035, chess.ROOK: 0.040, chess.KNIGHT: 0.043, chess.BISHOP: 0.046, chess.QUEEN: 0.050, chess.KING: 0.055}
GRASP_FRACTION = 0.60       # la punta agarra a este % de la altura de la pieza (por el cuerpo, no por la cabeza)
GRASP_DZ = {p: round(h * GRASP_FRACTION, 4) for p, h in PIECE_HEIGHT.items()}  # m sobre la superficie del tablero
STATE_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


motion_permission = CFG.motion_permission  # gate único


def load_grasp_offset() -> np.ndarray:
    """(dx, dy, dz) en metros que se suman a cada agarre y soltado. Lo escribe `grasp-probe`."""
    if not GRASP_OFFSET_FILE.exists():
        return np.zeros(3)
    d = json.loads(GRASP_OFFSET_FILE.read_text())
    return np.array([d.get("dx_m", 0.0), d.get("dy_m", 0.0), d.get("dz_m", 0.0)], float)


RETRY_OFFSETS_MM = [(0, 0), (0, 3), (0, -3), (3, 0), (-3, 0), (3, 3), (-3, 3), (3, -3), (-3, -3)]  # (sin uso en pick_place)
# 21-nov: los dedos cierran a lo largo de la COLUMNA (uno hacia la fila de delante, otro hacia la de detras). En d2 el dedo
# de atras rozaba el rey de la fila 1 y cerraba en vacio. Si solo una de esas dos casillas esta ocupada, se agarra y se
# suelta desplazado CLEAR_SHIFT hacia la libre. Reintentos: primero a lo largo de la columna, luego laterales.
CLEAR_SHIFT_M = float(os.environ.get("CHESS_CLEAR_SHIFT_MM", "3")) / 1000
RETRY_STEP_M = 0.003


RANK_CORR_FILE = HUB / "chess_rank_correction.json"   # correccion por FILA (adelantar a lo largo de la columna y bajar)


def load_rank_correction() -> dict[int, tuple[float, float]]:
    """{fila: (adelantar_m, bajar_m)} medida con `rank-probe`. Adelantar = a lo largo de la columna hacia la fila 8;
    bajar = la punta mas abajo (dz negativo). Con el brazo estirado (filas lejanas) cede y la punta queda alta/corta."""
    if not RANK_CORR_FILE.exists():
        return {}
    d = json.loads(RANK_CORR_FILE.read_text()).get("por_fila", {})
    return {int(r): (v["adelantar_mm"] / 1000, v["bajar_mm"] / 1000) for r, v in d.items()}


def rank_correction(table: dict[int, tuple[float, float]], rank: int) -> tuple[float, float, str]:
    """Correccion de una fila: la medida, o interpolada entre las dos filas medidas mas cercanas. Fuera del rango medido
    no se extrapola (0, 0)."""
    if rank in table:
        return (*table[rank], "medida")
    lo = [r for r in table if r < rank]
    hi = [r for r in table if r > rank]
    if not lo or not hi:
        return 0.0, 0.0, "sin medir"
    a, b = max(lo), min(hi)
    t = (rank - a) / (b - a)
    return (table[a][0] + t * (table[b][0] - table[a][0]), table[a][1] + t * (table[b][1] - table[a][1]), f"interpolada {a}-{b}")


def save_rank_correction(rank: int, along_m: float, down_m: float, detalle: dict) -> None:
    d = json.loads(RANK_CORR_FILE.read_text()) if RANK_CORR_FILE.exists() else {}
    d.setdefault("por_fila", {})[str(rank)] = {"adelantar_mm": round(along_m * 1000, 1), "bajar_mm": round(down_m * 1000, 1),
                                               "fecha": time.strftime("%Y-%m-%d %H:%M"), **detalle}
    RANK_CORR_FILE.write_text(json.dumps(d, indent=2))


def square_offsets() -> dict[str, list[float]]:
    """Desplazamiento (dx, dy) en metros que ha funcionado en cada casilla (lo aprende pick_place)."""
    if not GRASP_OFFSET_FILE.exists():
        return {}
    return json.loads(GRASP_OFFSET_FILE.read_text()).get("por_casilla", {})


def save_square_offset(sq: str, dxy) -> None:
    d = json.loads(GRASP_OFFSET_FILE.read_text()) if GRASP_OFFSET_FILE.exists() else {}
    d.setdefault("por_casilla", {})[sq] = [round(float(dxy[0]), 4), round(float(dxy[1]), 4)]
    d["fecha"] = time.strftime("%Y-%m-%d %H:%M")
    GRASP_OFFSET_FILE.write_text(json.dumps(d, indent=2))


def load_home() -> np.ndarray | None:
    if not HOME_FILE.exists():
        return None
    p = json.loads(HOME_FILE.read_text()).get("pose_origen_grados")
    return np.array([p[k] for k in STATE_KEYS], float) if p else None


# --------------------------------------------------------------------------- plan (sin hardware)
class Planner:
    def __init__(self, kin: Kin | None = None, offset: np.ndarray | None = None, rank_corr: bool = True):
        self.kin = kin or Kin()
        self.offset = load_grasp_offset() if offset is None else np.asarray(offset, float)
        self.board_bounds = self._board_bounds()
        self.rank_table = load_rank_correction() if rank_corr else {}

    def board_frame(self, xy) -> tuple[str | None, np.ndarray]:
        """Casilla mas cercana (si esta a < 1,5 cm) y vector unitario de su columna (fila 1 -> 8) en xy del robot."""
        sqs = self.kin.squares
        name = min(sqs, key=lambda s: np.hypot(*(np.array(sqs[s]) - np.asarray(xy))))
        if np.hypot(*(np.array(sqs[name]) - np.asarray(xy))) > 0.015:
            return None, np.array([1.0, 0.0])
        u = np.array(sqs[name[0] + "8"]) - np.array(sqs[name[0] + "1"])
        return name, u / np.linalg.norm(u)

    def corrected(self, xy, dz: float) -> tuple[tuple[float, float], float, str]:
        """Aplica la correccion de la fila: adelanta a lo largo de la columna y baja la punta."""
        sq, u = self.board_frame(xy)
        if sq is None or not self.rank_table:
            return (float(xy[0]), float(xy[1])), dz, "sin correccion de fila"
        along, down, why = rank_correction(self.rank_table, int(sq[1]))
        return (float(xy[0] + along * u[0]), float(xy[1] + along * u[1])), dz - down, \
            f"fila {sq[1]} ({why}): +{along * 1000:.1f} mm columna, -{down * 1000:.1f} mm altura"

    @staticmethod
    def _board_bounds() -> tuple[float, float, float]:
        """(x_min, x_max, |y|_max) de la zona del tablero: casillas calibradas + 1 casilla de margen (antes fijo)."""
        board = json.loads((HUB / "chess_board_real.json").read_text())
        pts = list(board["squares"].values())
        m = float(board.get("square_size_m", 0.03))
        xs = [p[0] for p in pts]
        return min(xs) - m, max(xs) + m, max(abs(p[1]) for p in pts) + m

    def waypoints(self, src_xy, dst_xy, piece_type: int, home4: np.ndarray | None) -> list[dict]:
        o = self.offset
        src_xy = (src_xy[0] + o[0], src_xy[1] + o[1])
        dst_xy = (dst_xy[0] + o[0], dst_xy[1] + o[1])
        g = GRASP_DZ[piece_type] + o[2]
        src_xy, g_src, self.last_src_corr = self.corrected(src_xy, g)
        dst_xy, g_dst, self.last_dst_corr = self.corrected(dst_xy, g)
        s_grasp = self.kin.solve_xy(src_xy, g_src)
        s_above = self.kin.solve_xy(src_xy, g_src + APPROACH_ABOVE, k_hint=s_grasp["k"]) if s_grasp["joints"] else None
        d_place = self.kin.solve_xy(dst_xy, g_dst + PLACE_EXTRA)
        d_above = self.kin.solve_xy(dst_xy, g_dst + PLACE_EXTRA + APPROACH_ABOVE, k_hint=d_place["k"]) if d_place["joints"] else None
        wp = []
        op = GRIP_APPROACH[piece_type]

        def add(name, sol, grip):
            wp.append({"name": name, "joints": None if sol is None or sol["joints"] is None else np.array(sol["joints"]),
                       "grip": grip, "tilt": None if sol is None else sol["tilt"], "margin": None if sol is None else sol["margin"]})

        wp.append({"name": "ORIGEN", "joints": None if home4 is None else home4, "grip": GRIP_OPEN, "tilt": None, "margin": None})
        add("encima origen", s_above, op)
        add("agarre origen", s_grasp, op)
        add("subir con pieza", s_above, None)
        add("encima destino", d_above, None)
        add("soltar destino", d_place, None)
        add("subir", d_above, op)
        wp.append({"name": "ORIGEN", "joints": None if home4 is None else home4, "grip": GRIP_OPEN, "tilt": None, "margin": None})
        return wp

    TRANSIT_MIN = PIECE_HEIGHT[chess.KING] + 0.020  # m: punta >= rey (5,5 cm medido) + 2 cm al cruzar por encima de las piezas

    def clearance(self, wp: list[dict]) -> list[str]:
        """Muestrea el camino en el espacio de articulaciones y avisa si la punta baja de TRANSIT_MIN sobre el tablero
        en un tramo que NO es un descenso al agarre o al soltado. No mueve nada."""
        warn = []
        bx0, bx1, by = self.board_bounds
        for a, b2 in zip(wp, wp[1:]):
            if a["joints"] is None or b2["joints"] is None:
                continue
            descent = "agarre" in b2["name"] or "soltar" in b2["name"] or "agarre" in a["name"] or "soltar" in a["name"]
            zmin, where = 9.9, None
            for s in np.linspace(0, 1, 40):
                q = a["joints"] + s * (b2["joints"] - a["joints"])
                tip = b.fk_tip(np.r_[q, 0.0] + self.kin.off)
                over_board = bx0 <= tip[0] <= bx1 and abs(tip[1]) <= by
                if over_board and tip[2] - self.kin.z_contact < zmin:
                    zmin, where = tip[2] - self.kin.z_contact, tip[:2]
            if not descent and where is not None and zmin < self.TRANSIT_MIN:
                warn.append(f"{a['name']} -> {b2['name']}: la punta pasa a {zmin * 100:.1f} cm del tablero sobre x={where[0] * 100:.0f} y={where[1] * 100:.0f} cm (minimo {self.TRANSIT_MIN * 100:.1f})")
        return warn

    def report(self, label: str, wp: list[dict]) -> bool:
        print(f"\n== {label} ==")
        ok, prev, total = True, None, 0
        for w in wp:
            j = w["joints"]
            if j is None:
                print(f"  {w['name']:<18} SIN SOLUCION" + (" (falta hub/chess_home.json)" if w["name"] == "ORIGEN" else ""))
                ok = False
                continue
            n = steps_between(prev, j, STEP_DEG) if prev is not None else 0
            total += n
            flag = ""
            if w["margin"] is not None and w["margin"] < 3:
                flag = f"  [a {w['margin']:.1f} grados de un limite]"
            tilt = "" if w["tilt"] is None else f" inclinacion {w['tilt']:4.0f}"
            print(f"  {w['name']:<18} {np.round(j, 1).tolist()}{tilt} grados  +{n} pasos{flag}")
            prev = j
        print(f"  total aprox {total} pasos = {total / HZ:.0f} s a {STEP_DEG} grados/paso (sin contar ACT/pinza)")
        return ok


# --------------------------------------------------------------------------- brazo real
class Arm:
    def __init__(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        port = CFG.require_port()
        self.robot = SO101Follower(SO101FollowerConfig(
            port=port, id=CFG.FOLLOWER_ID, cameras={}, use_degrees=True,
            max_relative_target={**{k: STEP_DEG for k in STATE_KEYS[:5]}, "gripper": 100.0},  # la pinza sin recorte: si no, no aprieta
            disable_torque_on_disconnect=False))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise RuntimeError("El seguidor no tiene calibracion.")
        # Ganancia del codo: LeRobot la deja en 16 y el codo se queda corto con <=2 grados de adelanto (decidido con el usuario)
        p_elbow = int(os.environ.get("CHESS_P_ELBOW", "28"))
        p_lift = int(os.environ.get("CHESS_P_LIFT", "44"))
        self.robot.bus.write("P_Coefficient", "elbow_flex", p_elbow)
        self.robot.bus.write("P_Coefficient", "shoulder_lift", p_lift)
        print(f"P_Coefficient codo={p_elbow} hombro={p_lift}", flush=True)

    def joints(self) -> np.ndarray:
        o = self.robot.get_observation()
        return np.array([o[f"{k}.pos"] for k in STATE_KEYS], float)

    def send(self, q6) -> None:
        self.robot.send_action({f"{k}.pos": float(v) for k, v in zip(STATE_KEYS, q6)})

    def goto(self, target6, tol: float = 2.5, check_grip: bool = True) -> None:
        """Mueve en pasos de <= STEP_DEG manteniendo la direccion. Para si se pierde el permiso o no converge.

        Con carga (el codo), el servo se queda corto unos grados y nunca llega al objetivo exacto. Por eso se acumula un
        pequeno sesgo por articulacion (accion integral, |sesgo| <= BIAS_MAX) que adelanta la orden hasta que llega. La
        velocidad sigue acotada a STEP_DEG por paso respecto a la posicion actual (ademas lo impone LeRobot)."""
        target = np.asarray(target6, float)
        bias = np.zeros(6)
        n_max = int(steps_between(self.joints(), target, STEP_DEG) * 3 + 60)
        ok_ticks = 0
        for _ in range(n_max):
            ok, why = motion_permission()
            if not ok:
                raise RuntimeError(f"parado: {why}")
            cur = self.joints()
            d = target - cur
            if np.abs(d[:5]).max() < tol and (not check_grip or abs(d[5]) < 4.0):
                ok_ticks += 1
                if ok_ticks >= 3:
                    return
            else:
                ok_ticks = 0
            near = np.abs(d) < STEP_DEG
            bias = np.where(near, np.clip(bias + BIAS_KI * d, -BIAS_MAX, BIAS_MAX), 0.0)
            cmd = (d + bias)[:5]
            big = np.abs(cmd).max()
            if big > STEP_DEG:
                cmd = cmd * (STEP_DEG / big)
            self.send(np.r_[cur[:5] + cmd, target[5]])   # pinza: orden absoluta (apretar = GRIP_HOLD_CMD)
            time.sleep(1 / HZ)
        raise RuntimeError("no converge al waypoint (carga, tope o colision): me paro")

    LIFT_EXTRA = float(os.environ.get("CHESS_LIFT_EXTRA", "6"))

    def lift(self, target6, check_grip: bool = True) -> None:
        """Subir desde abajo por etapas (21-nov: el codo se quedaba 7 grados corto si subia a la vez que el hombro, sin
        carga; subiendo antes el hombro el codo si llega). 1) hombro a objetivo - LIFT_EXTRA (menos = mas alto) con el resto
        quieto; 2) codo y muneca al objetivo; 3) pose exacta."""
        target = np.asarray(target6, float)
        cur = self.joints()
        s1 = cur.copy()
        s1[1] = min(cur[1], target[1]) - self.LIFT_EXTRA
        s1[5] = target[5]
        self.goto(s1, check_grip=check_grip)
        s2 = target.copy()
        s2[1] = s1[1]
        self.goto(s2, check_grip=check_grip)
        self.goto(target, check_grip=check_grip)

    # 0,7 grados ~ 1,8-3,4 mm en la punta (modelo, filas 2-6). 0,3 grados quita ~1-1,5 mm; el limite real lo ponen la zona
    # muerta del servo (1 paso = 0,088 grados) y la holgura. Si no llega, goto_fine devuelve el error que queda (no falla).
    FINE_TOL = float(os.environ.get("CHESS_FINE_TOL", "0.3"))
    FINE_SECONDS = float(os.environ.get("CHESS_FINE_SECONDS", "4"))

    def goto_fine(self, target6, check_grip: bool = True) -> float:
        """Posicionado fino para bajar al agarre o al soltado: primero goto normal y luego afina hasta FINE_TOL grados en
        pan/lift/elbow/wrist_flex (con el sesgo integral). No lanza error si no llega: devuelve el error maximo (grados)."""
        target = np.asarray(target6, float)
        self.goto(target, check_grip=check_grip)
        bias = np.zeros(4)
        for _ in range(int(HZ * self.FINE_SECONDS)):
            cur = self.joints()
            d = target[:4] - cur[:4]
            if np.abs(d).max() < self.FINE_TOL:
                break
            bias = np.clip(bias + 0.15 * d, -4.0, 4.0)
            self.send(np.r_[cur[:4] + np.clip(d + bias, -STEP_DEG, STEP_DEG), target[4], target[5]])
            time.sleep(1 / HZ)
        return float(np.abs(target[:4] - self.joints()[:4]).max())

    _open = GRIP_OPEN

    def grasp(self) -> tuple[float, bool]:
        """Cierre SUAVE: baja la orden de la pinza GRIP_CLOSE_RATE por tick; cuando la pinza deja de seguirla (contacto) fija
        la orden en contacto - GRIP_SQUEEZE (presion ligera) y la guarda en self.hold_cmd para el transporte.
        Sin pieza la pinza sigue la orden hasta ~1,8. Devuelve (valor, hay_pieza)."""
        q = self.joints()
        cmd = float(q[5])
        self._open = cmd
        contact = None
        for _ in range(int(max(cmd, 1.0) / GRIP_CLOSE_RATE) + 30):
            ok, why = motion_permission()
            if not ok:
                raise RuntimeError(f"parado: {why}")
            pos = float(self.joints()[5])
            if contact is None and pos - cmd > GRIP_CONTACT_LAG:
                contact = pos
                break
            cmd = max(GRIP_HOLD_CMD, cmd - GRIP_CLOSE_RATE)
            q[5] = cmd
            self.send(q)
            time.sleep(1 / HZ)
            if cmd <= GRIP_HOLD_CMD and pos <= GRIP_EMPTY_MAX:
                break
        g = float(self.joints()[5]) if contact is None else contact
        held = g > GRIP_EMPTY_MAX
        self.hold_cmd = max(GRIP_HOLD_CMD, g - GRIP_SQUEEZE) if held else float(q[5] if False else self._open)
        q[5] = self.hold_cmd
        for _ in range(10):
            self.send(q)
            time.sleep(1 / HZ)
        return g, held

    def release(self, open_to: float | None = None, settle_s: float = 0.6) -> None:
        """Abre solo hasta open_to (por defecto la apertura con la que se acerco), para no tocar las piezas vecinas."""
        q = self.joints()
        q[5] = self._open if open_to is None else open_to
        t_end = time.time() + settle_s
        while time.time() < t_end:
            self.send(q)
            time.sleep(1 / HZ)

    def close_until_stall(self, q4_roll: np.ndarray) -> float:
        """Cierra la pinza poco a poco hasta que deja de moverse (pieza agarrada o cerrada del todo). Devuelve su valor."""
        g = self.joints()[5]
        stalled, last = 0, g
        for _ in range(int(GRIP_OPEN / GRIP_CLOSE_STEP) + 20):
            ok, why = motion_permission()
            if not ok:
                raise RuntimeError(f"parado: {why}")
            cur = self.joints()
            q = cur.copy()
            q[5] = max(0.0, cur[5] - GRIP_CLOSE_STEP)
            self.send(q)
            time.sleep(1 / HZ)
            now = self.joints()[5]
            stalled = stalled + 1 if abs(now - last) < 0.25 else 0
            last = now
            if stalled >= 5 or now <= 1.0:
                break
        return float(self.joints()[5])


# --------------------------------------------------------------------------- ACT confinado
def trained_gripper_range(checkpoint: Path) -> tuple[float, float]:
    """Rango de la pinza (observation.state) con el que se entrenó el ACT, leído de las estadísticas del checkpoint."""
    from safetensors.torch import load_file

    st = load_file(str(Path(checkpoint) / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    return float(st["observation.state.min"][5]), float(st["observation.state.max"][5])


def act_grip_problem(checkpoint: Path, grip_open: float = GRIP_OPEN, margin: float = 2.0) -> str | None:
    """None si la apertura de aproximación está dentro de lo que el ACT vio al entrenar; si no, el motivo."""
    lo, hi = trained_gripper_range(checkpoint)
    if grip_open > hi + margin:
        return (f"la pinza se acerca abierta a {grip_open:.0f} y el ACT solo vio aperturas de {lo:.1f} a {hi:.1f}: "
                f"arrancaría fuera de su distribución. Usa CHESS_GRIP_OPEN <= {hi:.0f} o el agarre guionizado (ik-scripted).")
    return None


class ACTWindow:
    def __init__(self, checkpoint: Path = DEFAULT_ACT, device: str | None = None):
        from deploy_goal import load_policy

        self.device = device or os.environ.get("SO101_VLA_DEVICE", "mps")
        self.policy, self.pre, self.post = load_policy(Path(checkpoint), self.device)
        self.motors = [f"{k}.pos" for k in STATE_KEYS]
        self.cams = {"front": CFG.YOLO_CAM, "gripper": CFG.WRIST_CAM}

    _caps: dict = {}

    def frames(self) -> dict[str, np.ndarray]:
        """Lee las dos camaras. Las deja ABIERTAS (abrir una camara en macOS tarda ~1-3 s: abrir/cerrar en cada paso dejaba
        al ACT en 1 paso cada 8 s). Se descarta un fotograma del buffer para no usar uno viejo."""
        from deploy_act_generic import to_act_size

        out = {}
        for k, i in self.cams.items():
            c = self._caps.get(i)
            if c is None or not c.isOpened():
                c = self._caps[i] = cv2.VideoCapture(i)
                for _ in range(5):
                    c.read()
            c.grab()
            ok, f = c.read()
            if not ok:
                raise RuntimeError(f"la camara {i} no devuelve imagen")
            out[k] = to_act_size(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return out

    def close(self) -> None:
        for c in self._caps.values():
            c.release()
        self._caps.clear()

    def action(self, state6: np.ndarray, frames: dict) -> np.ndarray:
        from deploy_act_generic import build_obs
        from deploy_goal import step_policy

        act = step_policy(self.policy, self.pre, self.post, build_obs(state6.astype(np.float32), frames), self.device, self.motors)
        return np.array([act[m] for m in self.motors], float)

    @staticmethod
    def envelope(a4: np.ndarray, g4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Caja por articulacion: el tramo aproximacion-agarre + un margen. El ACT no puede salirse de aqui."""
        return np.minimum(a4, g4) - ENVELOPE_MARGIN, np.maximum(a4, g4) + ENVELOPE_MARGIN

    def run(self, arm: Arm, a4, g4, seconds: float) -> dict:
        lo, hi = self.envelope(np.asarray(a4), np.asarray(g4))
        roll = arm.joints()[4]
        self.action(arm.joints(), self.frames())   # calentamiento: la 1a inferencia en mps tarda ~9 s (no cuenta en la ventana)
        self.policy.reset()
        clamped = steps = 0
        t_end = time.time() + seconds
        while time.time() < t_end:
            ok, why = motion_permission()
            if not ok:
                raise RuntimeError(f"parado: {why}")
            t = time.perf_counter()
            state = arm.joints()
            raw = self.action(state, self.frames())
            q = raw.copy()
            q[:4] = np.clip(raw[:4], lo, hi)
            q[4] = roll
            q[5] = float(np.clip(raw[5], 0.0, 100.0))
            clamped += int(np.any(np.abs(q[:4] - raw[:4]) > 1e-6))
            arm.send(q)
            steps += 1
            time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t)))
        return {"steps": steps, "clamped_steps": clamped, "grip_final": float(arm.joints()[5])}


# --------------------------------------------------------------------------- ejecutor (interfaz de sim/chess_real.py)
class IKExecutor:
    """pick_place(label, piece_type, src_xy, dst_xy): lo que RealChessBackend espera de un ejecutor."""

    def __init__(self, grasp: str = "act", seconds: float = 8.0, checkpoint: Path = DEFAULT_ACT):
        self.name = f"ik+{grasp}"
        self.grasp, self.seconds, self.checkpoint = grasp, seconds, checkpoint
        self.planner = Planner()
        self._arm: Arm | None = None
        self._act: ACTWindow | None = None
        self.occupied: set[str] | None = None   # casillas ocupadas (set_board); None = no se sabe, sin desplazamiento
        self.board: chess.Board | None = None
        self._wcam: WR.WristCam | None = None

    def _hover_q(self, xy, piece_type: int, roll: float, op: float) -> np.ndarray | None:
        """Pose con la punta HOVER_M por encima de la cabeza de la pieza en xy (con offset global y correccion de fila)."""
        pl = self.planner
        o = pl.offset
        xy2, z, _ = pl.corrected((xy[0] + o[0], xy[1] + o[1]), PIECE_HEIGHT[piece_type] + WR.HOVER_M + o[2])
        sol = pl.kin.solve_xy(xy2, z)
        return None if sol["joints"] is None else np.array([*sol["joints"], roll, op], float)

    def wrist_ref_key(self, sq: str, piece_type: int) -> str:
        color = "w"
        if self.board is not None and self.board.piece_at(chess.parse_square(sq)) is not None:
            color = "w" if self.board.piece_at(chess.parse_square(sq)).color == chess.WHITE else "b"
        return WR.ref_key(color, chess.piece_symbol(piece_type))

    def wrist_measure(self, arm: "Arm", sq: str, xy, piece_type: int, roll: float, op: float) -> dict:
        """Punto 2: mide con la camara de la muneca donde esta la pieza respecto a los dedos (ver hub/chess_wrist.py).
        Deja el brazo 1 cm por encima de la pieza (desplazado PROBE_MM en lateral). Devuelve la correccion en metros (xy)."""
        key = self.wrist_ref_key(sq, piece_type)
        ref = WR.load_ref(key)
        if ref is None:
            return {"ok": False, "motivo": f"sin referencia {key} (haz `wrist-ref` en una casilla que agarre bien)"}
        u, w = self._file_axes(sq)
        P = WR.PROBE_MM / 1000
        pts, scores = [], []
        for i, off in enumerate([np.zeros(2), P * u, P * w]):
            q = self._hover_q((xy[0] + off[0], xy[1] + off[1]), piece_type, roll, op)
            if q is None:
                return {"ok": False, "motivo": "sin solucion IK 1 cm por encima"}
            err = arm.goto_fine(q)
            time.sleep(0.3)
            img = self._wcam.grab()
            cv2.imwrite(str(HUB / "frames" / "agarre" / f"muneca_{sq}_{i}.jpg"), img)
            p, sc = WR.locate(img, ref)
            pts.append(p)
            scores.append(round(sc, 2))
            if sc < WR.MIN_SCORE:
                return {"ok": False, "motivo": f"la pieza no se reconoce en la foto {i} (coincidencia {sc:.2f} < {WR.MIN_SCORE})",
                        "coincidencias": scores, "clave": key}
        r = WR.solve_correction(pts[0], pts[1], pts[2], ref[1]["pixel"])
        r.update({"clave": key, "coincidencias": scores, "pixel_pieza": np.round(pts[0], 1).tolist(), "pixel_ref": ref[1]["pixel"]})
        if r["ok"]:
            d = r["mm"]
            r["xy_m"] = (np.array(d[0]) * u + np.array(d[1]) * w) / 1000
        return r

    def set_board(self, board: chess.Board) -> None:
        """Ocupacion del tablero ANTES de la jugada. pick_place la actualiza tras cada paso que sale bien."""
        self.occupied = {chess.square_name(sq) for sq in board.piece_map()}
        self.board = board.copy()

    def _square_at(self, xy, tol: float = 0.006) -> str | None:
        sqs = self.planner.kin.squares
        name = min(sqs, key=lambda s: np.hypot(*(np.array(sqs[s]) - np.asarray(xy))))
        return name if np.hypot(*(np.array(sqs[name]) - np.asarray(xy))) < tol else None

    def _file_axes(self, sq: str) -> tuple[np.ndarray, np.ndarray]:
        """Vector unitario a lo largo de la columna (fila 1 -> 8) y el lateral, en xy del robot."""
        sqs = self.planner.kin.squares
        u = np.array(sqs[sq[0] + "8"]) - np.array(sqs[sq[0] + "1"])
        u = u / np.linalg.norm(u)
        return u, np.array([-u[1], u[0]])

    def clear_shift(self, sq: str | None, ignore: tuple = ()) -> tuple[np.ndarray, str]:
        """Desplazamiento (m) hacia la casilla libre de la columna si solo una de las dos vecinas (delante/detras) esta ocupada."""
        if sq is None or self.occupied is None:
            return np.zeros(2), "sin datos del tablero"
        r = int(sq[1])
        back = f"{sq[0]}{r - 1}" if r > 1 else None
        front = f"{sq[0]}{r + 1}" if r < 8 else None
        occ_b = back is not None and back in self.occupied and back not in ignore
        occ_f = front is not None and front in self.occupied and front not in ignore
        if occ_b == occ_f:
            return np.zeros(2), ("delante y detras ocupadas" if occ_b else "delante y detras libres") + ": centro"
        u, _ = self._file_axes(sq)
        sign = 1.0 if occ_b else -1.0
        return sign * CLEAR_SHIFT_M * u, f"{back if occ_b else front} ocupada: {CLEAR_SHIFT_M * 1000:.0f} mm hacia {(front if occ_b else back) or 'el borde'}"

    def _grasp_tries(self, sq: str | None, base: np.ndarray, learned) -> list[tuple[float, float]]:
        """Orden de intentos (21-nov): 1) el desplazamiento que funciono la ultima vez en esa casilla (medido: e2 agarra en
        el centro aunque el rey este detras; d2 solo a +3 mm); 2) hacia la vecina libre; 3) el centro, una vez; 4) mas lejos
        de la vecina ocupada a lo largo de la columna (o +-3 mm si no hay ninguna); 5) laterales +-3 mm."""
        if sq is None:
            u, w = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        else:
            u, w = self._file_axes(sq)
        st = RETRY_STEP_M
        along = float(base @ u)
        cand = [np.asarray(learned, float)] if learned is not None else []
        cand.append(base)
        cand.append(np.zeros(2))
        if abs(along) > 1e-6:            # una vecina ocupada: alejarse mas de ella, nunca acercarse
            cand.append(base + np.sign(along) * st * u)
        else:
            cand += [base + st * u, base - st * u]
        cand += [base + st * w, base - st * w]
        out: list[tuple[float, float]] = []
        for c in cand:
            t = (round(float(c[0]), 4), round(float(c[1]), 4))
            if all(np.hypot(t[0] - o[0], t[1] - o[1]) > 0.0015 for o in out):   # casi iguales (<1,5 mm) = el mismo intento
                out.append(t)
        return out

    def pick_place(self, label, piece_type, src_xy, dst_xy) -> dict:
        ok, why = motion_permission()
        if not ok:
            raise RuntimeError(f"No se mueve el brazo: {why}")
        home6 = load_home()
        if home6 is None:
            raise RuntimeError("Falta hub/chess_home.json (pose de origen): `chess_ik_exec.py set-home` con el brazo colocado.")
        src_sq, dst_sq = self._square_at(src_xy), self._square_at(dst_xy)
        dst_shift, dst_why = self.clear_shift(dst_sq, ignore=(src_sq,))
        dst_xy = (dst_xy[0] + dst_shift[0], dst_xy[1] + dst_shift[1])
        src_shift, src_why = self.clear_shift(src_sq)
        print(f"  [vecinas] origen {src_sq}: {src_why} | destino {dst_sq}: {dst_why}", flush=True)
        wp = self.planner.waypoints(src_xy, dst_xy, piece_type, home6[:4])
        print(f"  [fila] origen: {self.planner.last_src_corr} | destino: {self.planner.last_dst_corr}", flush=True)
        if not self.planner.report(label, wp):
            raise RuntimeError("el plan tiene waypoints sin solucion: no se mueve")
        if any(w["margin"] is not None and w["margin"] < 1.0 for w in wp):
            raise RuntimeError("un waypoint queda a menos de 1 grado de un limite de articulacion: no se mueve")
        first = self._arm is None
        arm = self._arm = self._arm or Arm()
        cur = arm.joints()
        roll = cur[4]
        # El camino ORIGEN -> encima del origen esta validado desde el ORIGEN guardado: el brazo tiene que estar ahi.
        far = np.abs(cur[:4] - home6[:4])
        if far.max() > 8.0:
            raise RuntimeError(f"el brazo NO esta en el origen (diferencias {np.round(far, 1).tolist()} grados, maximo 8): "
                               "colocalo en la pose guardada (hub/chess_home.json) antes de empezar. No se mueve.")
        if first:
            print("Brazo en el origen. Torque encendido: SUELTA el brazo y apartate de su zona. Empiezo en 5 s (Ctrl+C para parar).", flush=True)
            time.sleep(5)

        def q6(w, grip=None):
            g = w["grip"] if grip is None and w["grip"] is not None else (arm.joints()[5] if grip is None else grip)
            return np.array([*w["joints"], roll, g], float)

        op = GRIP_APPROACH[piece_type]                       # apertura justa para esta pieza
        info: dict = {"accion": label, "apertura": op, "hueco_cm": round(gap_cm(op), 1), "offset_m": [round(float(v), 4) for v in self.planner.offset],
                      "vecinas_origen": src_why, "vecinas_destino": dst_why}

        def back_home(motivo: str) -> dict:
            print(f"  [fallo] {motivo}", flush=True)
            arm.release()
            arm.lift(q6(wp[3], op))
            arm.goto(q6(wp[7], op))
            return {**info, "ok": False, "motivo": motivo}

        arm.goto(q6(wp[1], op))                     # encima del origen, pinza abierta
        if self.grasp == "act":
            why_not = act_grip_problem(self.checkpoint, grip_open=op)   # la apertura REAL de esta pieza, no GRIP_OPEN
            if why_not:
                raise RuntimeError(f"No se usa el ACT: {why_not}")
            # Ventana del ACT de 30k confinada a la caja encima->agarre (+-ENVELOPE_MARGIN). Luego la IK termina de bajar.
            self._act = self._act or ACTWindow(self.checkpoint)
            info["act"] = self._act.run(arm, wp[1]["joints"], wp[2]["joints"], self.seconds)
        # Agarre con reintentos: si cierra en vacio, prueba desplazamientos de 3 mm (empieza por el aprendido en esa casilla).
        tries = self._grasp_tries(src_sq, src_shift, square_offsets().get(src_sq) if src_sq else None)
        if WR.MODE != "off" and src_sq:
            self._wcam = self._wcam or WR.WristCam(CFG.WRIST_CAM)
            wm = self.wrist_measure(arm, src_sq, src_xy, piece_type, roll, op)
            info["muneca"] = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in wm.items()}
            if wm["ok"]:
                print(f"  [muneca] pieza a {wm['mm'][0]:+.1f} mm (columna), {wm['mm'][1]:+.1f} mm (lateral) del centro de los dedos"
                      f" | modo {WR.MODE}", flush=True)
                if WR.MODE == "correct":
                    m = (round(float(wm["xy_m"][0]), 4), round(float(wm["xy_m"][1]), 4))
                    tries = [m] + [t for t in tries if np.hypot(t[0] - m[0], t[1] - m[1]) > 0.0015]
            else:
                print(f"  [muneca] sin medida: {wm['motivo']}", flush=True)
        held, intentos = False, []
        for k, (dx, dy) in enumerate(tries[:int(os.environ.get("CHESS_GRASP_TRIES", "6"))]):
            src2 = (src_xy[0] + dx, src_xy[1] + dy)
            w = self.planner.waypoints(src2, dst_xy, piece_type, home6[:4])
            if w[1]["joints"] is None or w[2]["joints"] is None:
                continue
            if k:
                arm.lift(q6(w[1], op))                     # sube y se recoloca encima con el nuevo desplazamiento
            try:
                err = arm.goto_fine(q6(w[2], op))          # bajar (fino)
            except RuntimeError as e:
                if "no converge" not in str(e):
                    raise
                # Algo frena la bajada (la pinza toca la pieza o una vecina): no empujar. Sube y vuelve al ORIGEN.
                info["casilla_origen"] = src_sq
                info["intentos"] = intentos + [{"dxy_mm": [round(dx * 1000, 1), round(dy * 1000, 1)], "choque": True,
                                               "articulaciones": np.round(arm.joints(), 1).tolist()}]
                arm.lift(q6(w[1], op), check_grip=False)
                arm.goto(q6(wp[7], op), check_grip=False)
                return {**info, "ok": False, "motivo": f"choque al bajar a {src_sq} (desplazamiento {dx * 1000:+.0f},{dy * 1000:+.0f} mm): "
                        "subo y vuelvo al origen sin forzar. Mira la foto de la muñeca y recoloca la pieza."}
            grip, held = arm.grasp()                       # cierre suave, detecta contacto
            intentos.append({"dxy_mm": [round(dx * 1000, 1), round(dy * 1000, 1)], "pinza": round(grip, 1), "error_deg": round(err, 2)})
            print(f"  [agarre {k + 1}] desplazamiento {dx * 1000:+.0f},{dy * 1000:+.0f} mm | bajada {err:.2f} grados | "
                  f"pinza {grip:.1f} | {'AGARRADA' if held else 'vacia'}", flush=True)
            if held:
                wp[1], wp[2], wp[3] = w[1], w[2], w[3]
                if src_sq:
                    save_square_offset(src_sq, (dx, dy))
                break
            arm.release()
        info["casilla_origen"], info["intentos"] = src_sq, intentos
        hold = arm.hold_cmd
        info["pinza_cerrada"], info["agarrada"] = round(grip, 1), bool(held)
        if not held:
            return back_home(f"pinza cerrada en vacio ({grip:.1f} <= {GRIP_EMPTY_MAX}): la pieza no estaba entre los dedos. "
                             "Recalibra el agarre con `grasp-probe`.")
        arm.lift(np.array([*wp[3]["joints"], roll, hold]), check_grip=False)   # subir apretando
        grip = float(arm.joints()[5])
        info["pinza_tras_subir"] = round(grip, 1)
        print(f"  [subida] pinza {grip:.1f} ({'sujeta' if grip > GRIP_EMPTY_MAX else 'SE SOLTO'})", flush=True)
        if grip <= GRIP_EMPTY_MAX:
            return back_home(f"la pieza se solto al subir (pinza {grip:.1f})")
        arm.goto(np.array([*wp[4]["joints"], roll, hold]), check_grip=False)
        arm.goto(np.array([*wp[5]["joints"], roll, hold]), check_grip=False)
        arm.goto_fine(np.array([*wp[5]["joints"], roll, hold]), check_grip=False)
        info["pinza_antes_de_soltar"] = round(float(arm.joints()[5]), 1)
        print(f"  [destino] pinza antes de soltar {info['pinza_antes_de_soltar']}", flush=True)
        arm.release()                                      # soltar
        print("  [soltado] subo y vuelvo al origen", flush=True)
        arm.lift(q6(wp[6], op))
        arm.goto(q6(wp[7], op))                     # ORIGEN (punto de retorno)
        if self.occupied is not None:
            self.occupied.discard(src_sq)
            if dst_sq:
                self.occupied.add(dst_sq)
        return {**info, "ok": True}


# --------------------------------------------------------------------------- CLI
def _steps_for(uci: str, fen: str | None):
    from chess_real import Geometry, RealChessBackend

    board = chess.Board(fen) if fen else chess.Board()
    be = RealChessBackend(executor=type("D", (), {"name": "dry", "pick_place": lambda *a, **k: None})(), geometry=Geometry(require_real=True))
    return be.plan(chess.Move.from_uci(uci), board)


def cmd_plan(a) -> int:
    pl = Planner()
    home = load_home()
    if home is None:
        print("AVISO: no hay hub/chess_home.json; ORIGEN sin definir.")
    else:
        print("ORIGEN (hub/chess_home.json):", np.round(home, 1).tolist())
    bad = 0
    for s in _steps_for(a.uci, a.fen):
        wp = pl.waypoints(s["src_xy"], s["dst_xy"], s["piece"], None if home is None else home[:4])
        bad += 0 if pl.report(s["label"], wp) else 1
    print("\nNOTA: alturas de agarre por pieza y apertura de pinza SIN medir con piezas reales (ver constantes del fichero).")
    return 1 if bad else 0


def cmd_act_offline(a) -> int:
    kin = Kin()
    pl = Planner(kin)
    g = GRASP_DZ[chess.PAWN]
    grasp = kin.solve(a.square, g)
    above = kin.solve(a.square, g + APPROACH_ABOVE, k_hint=grasp["k"])
    if grasp["joints"] is None or above["joints"] is None:
        print("la casilla no llega")
        return 1
    win = ACTWindow(Path(a.checkpoint))
    roll = 166.0
    state = np.array([*above["joints"], roll, GRIP_OPEN], float)
    lo, hi = win.envelope(np.array(above["joints"]), np.array(grasp["joints"]))
    act = win.action(state, win.frames())
    print(f"Estado de partida = encima de {a.square} (IK): {np.round(state, 1).tolist()}")
    print(f"Agarre (IK):                              {np.round([*grasp['joints'], roll, 0], 1).tolist()}")
    print(f"Primera accion predicha por el ACT:       {np.round(act, 1).tolist()}")
    print(f"Caja de seguridad (min):                  {np.round(lo, 1).tolist()}  (max): {np.round(hi, 1).tolist()}")
    inside = bool(np.all((act[:4] >= lo) & (act[:4] <= hi)))
    print("la primera accion " + ("cae DENTRO" if inside else "cae FUERA (se recortaria)") + " de la caja")
    print("salto respecto al estado (grados):", np.round(np.abs(act - state), 1).tolist())
    return 0


def cmd_test_grasp(a) -> int:
    """Primer test del agarre guionizado: coge el peon de una casilla y lo DEJA DE NUEVO en la misma (la partida no cambia).
    Sin --go solo planifica y comprueba (no mueve nada). Con --go ejecuta (necesita permiso)."""
    sq = a.square.lower()
    pl = Planner()
    home = load_home()
    print(f"== Preflight (test de agarre guionizado en {sq}, {STEP_DEG} grados/paso) ==")
    problems = []
    if home is None:
        problems.append("falta hub/chess_home.json (coloca el brazo a mano y `set-home`)")
    xy = pl.kin.squares[sq]
    wp = pl.waypoints(xy, xy, chess.PAWN, None if home is None else home[:4])
    ok = pl.report(f"{sq} -> {sq} (peon, vuelve a la misma casilla)", wp)
    if not ok:
        problems.append("hay waypoints sin solucion")
    if any(w["margin"] is not None and w["margin"] < 1.0 for w in wp):
        problems.append("un waypoint queda a menos de 1 grado de un limite")
    if home is not None:
        d = np.abs(home[:4] - wp[1]["joints"]).max() if wp[1]["joints"] is not None else 999
        print(f"  mayor salto de articulacion de ORIGEN a 'encima origen': {d:.0f} grados")
        if d > float(os.environ.get("CHESS_MAX_HOME_JUMP", "90")):  # origen = reposo plegado: CHESS_MAX_HOME_JUMP=150
            problems.append("el origen esta muy lejos de la casilla: define un origen mas cerca (encima del tablero)")
    for w in pl.clearance(wp):
        problems.append("altura de transito: " + w)
    print("\nAvisos/bloqueos:" if problems else "\nSin bloqueos.")
    for p in problems:
        print("  -", p)
    if not a.go:
        print("\n(simulacion: no se ha movido nada. Para ejecutar: --go, con permiso)")
        return 1 if problems else 0
    if problems:
        print("NO se ejecuta: resuelve los bloqueos.")
        return 1
    ex = IKExecutor(grasp="scripted")
    ex.set_board(chess.Board(game_fen()))
    r = ex.pick_place(f"test {sq}->{sq}", chess.PAWN, xy, xy)
    print(json.dumps(r, indent=2, default=float))
    return 0 if r.get("ok") else 3


def cmd_set_home(a) -> int:
    """Guarda la pose ACTUAL del brazo como ORIGEN. Lee los registros (torque sin tocar): el puerto debe estar libre
    (sin el MCP so101-chess conectado). Coloca antes el brazo a mano en una pose segura encima del tablero."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    from chess_ik import CAL_FOLLOWER

    cal = json.loads(CAL_FOLLOWER.read_text())
    port = CFG.require_port()
    bus = FeetechMotorsBus(port=port, motors={n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES if n != "gripper" else MotorNormMode.RANGE_0_100)
                                              for i, n in enumerate(STATE_KEYS)},
                           calibration={n: MotorCalibration(**cal[n]) for n in STATE_KEYS})
    bus.connect(handshake=False)
    print(f"Sujeta el brazo en la pose de origen. Leo en {a.wait:.0f} s...", flush=True)
    time.sleep(a.wait)
    pose = {n: round(float(bus.read("Present_Position", n)), 1) for n in STATE_KEYS}
    bus.disconnect(disable_torque=False)
    print("pose actual:", pose)
    kin = Kin()
    if any(abs(pose[n]) > l - 3 for n, l in zip(J4, kin.limit)) and not a.force:
        print("NO se guarda: alguna articulacion esta a menos de 3 grados de su limite (suele ser el reposo plegado, "
              "en el que el brazo cae al soltarlo). Sujeta el brazo en una pose elevada sobre el tablero; o usa --force.")
        return 1
    HOME_FILE.write_text(json.dumps({"pose_origen_grados": pose, "fecha": time.strftime("%Y-%m-%d %H:%M"),
                                     "nota": "colocada a mano por el usuario"}, indent=2))
    print("guardado", HOME_FILE)
    return 0


def cmd_make_home(a) -> int:
    """Calcula con la IK un ORIGEN seguro: encima del borde del tablero del lado del robot, pinza casi vertical. No mueve nada.
    Con --save lo escribe en hub/chess_home.json (el brazo NO va ahi hasta que lo ordene un `run`)."""
    kin = Kin()
    sol = kin.solve_xy((a.x, a.y), a.dz)
    if sol["joints"] is None:
        print("sin solucion")
        return 1
    cur = load_home()
    roll = float(cur[4]) if cur is not None else 166.6
    pose = dict(zip(STATE_KEYS, [*map(lambda v: round(float(v), 1), sol["joints"]), round(roll, 1), GRIP_OPEN]))
    print(f"ORIGEN propuesto sobre (x={a.x}, y={a.y}, z=contacto+{a.dz}): {pose} | inclinacion {sol['tilt']:.0f} grados, margen {sol['margin']:.1f} grados")
    if a.save:
        HOME_FILE.write_text(json.dumps({"pose_origen_grados": pose, "fecha": time.strftime("%Y-%m-%d %H:%M"),
                                         "nota": "calculado con la IK (hub/chess_ik_exec.py make-home); el brazo no se ha movido ahi"}, indent=2))
        print("guardado", HOME_FILE)
    return 0


def cmd_grasp_probe(a) -> int:
    """Mide la correccion del agarre en una casilla con una pieza: prueba una rejilla de desplazamientos (dx, dy) y alturas
    alrededor del centro calibrado, en cada uno baja, aprieta y mira si la pinza se queda por encima de GRIP_EMPTY_MAX
    (hay pieza). Guarda la media de los desplazamientos que agarran en hub/chess_grasp_offset.json. Empieza y acaba en el
    punto de retorno. La pieza no se levanta (solo se aprieta y se suelta)."""
    sq, s = a.square.lower(), a.step
    home = load_home()
    if home is None:
        print("falta hub/chess_home.json")
        return 1
    base = load_grasp_offset() if a.from_saved else np.zeros(3)
    grid = sorted({(dx, dy) for dx in (-s, 0.0, s) for dy in (-s, 0.0, s)}, key=lambda o: (abs(o[0]) + abs(o[1]), o))
    arm = Arm()
    roll = arm.joints()[4]
    if np.abs(arm.joints()[:4] - home[:4]).max() > 8:
        print("el brazo no esta en el retorno; no se mueve")
        return 1
    results = []
    op = GRIP_APPROACH[chess.PAWN]
    print(f"apertura de acercamiento {op} (~{gap_cm(op):.1f} cm entre puntas)")
    for dz in [float(v) for v in a.dz.split(",")]:
        for dx, dy in grid:
            off = base + np.array([dx, dy, dz])
            pl = Planner(offset=off)
            xy = pl.kin.squares[sq]
            wp = pl.waypoints(xy, xy, chess.PAWN, home[:4])
            if any(w["joints"] is None for w in wp[1:3]):
                print(f"  {np.round(off * 1000, 1).tolist()} mm: sin solucion IK")
                continue
            arm.goto(np.array([*wp[1]["joints"], roll, op]))
            err = arm.goto_fine(np.array([*wp[2]["joints"], roll, op]))
            g, held = arm.grasp()
            arm.release()
            arm.goto(np.array([*wp[1]["joints"], roll, op]))
            results.append({"offset_m": off.round(4).tolist(), "pinza": round(g, 1), "agarra": bool(held), "error_deg": round(err, 2)})
            print(f"  offset {np.round(off * 1000, 1).tolist()} mm -> pinza {g:.1f} {'AGARRA' if held else 'vacia'}  (error pose {err:.1f} grados)", flush=True)
    arm.goto(np.array([*home[:4], roll, op]))
    good = [r for r in results if r["agarra"]]
    if not good:
        print("Ningun desplazamiento agarra. Sube --step o revisa la altura (--dz) y que haya una pieza en la casilla.")
        return 2
    best = np.mean([r["offset_m"] for r in good], axis=0)
    out = {"dx_m": round(float(best[0]), 4), "dy_m": round(float(best[1]), 4), "dz_m": round(float(best[2]), 4),
           "casilla": sq, "fecha": time.strftime("%Y-%m-%d %H:%M"), "aciertos": f"{len(good)}/{len(results)}", "pruebas": results}
    if a.save:
        GRASP_OFFSET_FILE.write_text(json.dumps(out, indent=2))
        print("guardado", GRASP_OFFSET_FILE)
    print("offset del agarre (mm):", np.round(best * 1000, 1).tolist(), f"({len(good)}/{len(results)} agarran)")
    return 0


def cmd_rank_probe(a) -> int:
    """Mide la correccion de UNA fila (punto 1, 21-nov): con un peon en la casilla (por defecto columna e de esa fila),
    prueba una rejilla de (adelantar a lo largo de la columna, bajar), sin aplicar la tabla de filas. En cada punto baja
    con la pinza a la apertura del peon, cierra suave, mira si hay pieza, suelta y sube. Guarda la media de los que agarran
    en hub/chess_rank_correction.json (--save). La pieza no se levanta. Empieza y acaba en el origen."""
    rank = int(a.rank)
    sq = (a.square or f"e{rank}").lower()
    if int(sq[1]) != rank:
        print(f"{sq} no es de la fila {rank}")
        return 1
    home = load_home()
    if home is None:
        print("falta hub/chess_home.json")
        return 1
    alongs = [float(v) / 1000 for v in a.along.split(",")]
    downs = [float(v) / 1000 for v in a.down.split(",")]
    grid = sorted(((al, dn) for al in alongs for dn in downs), key=lambda o: (abs(o[0]) + abs(o[1]), o))
    pl0 = Planner(rank_corr=False)
    xy0 = pl0.kin.squares[sq]
    _, u = pl0.board_frame(xy0)
    op = GRIP_APPROACH[chess.PAWN]
    print(f"rank-probe fila {rank} en {sq} | columna u={np.round(u, 3).tolist()} | apertura {op} (~{gap_cm(op):.1f} cm) | "
          f"{len(grid)} pruebas | bajada fina {Arm.FINE_TOL} grados", flush=True)
    arm = Arm()
    roll = arm.joints()[4]
    if np.abs(arm.joints()[:4] - home[:4]).max() > 8:
        print("el brazo no esta en el origen; no se mueve")
        return 1
    results = []
    for al, dn in grid:
        pl = Planner(offset=pl0.offset + np.array([al * u[0], al * u[1], -dn]), rank_corr=False)
        wp = pl.waypoints(xy0, xy0, chess.PAWN, home[:4])
        if any(w["joints"] is None for w in wp[1:3]):
            print(f"  adelantar {al * 1000:+.0f} mm, bajar {dn * 1000:.0f} mm: sin solucion IK", flush=True)
            continue
        arm.goto(np.array([*wp[1]["joints"], roll, op]))
        err = arm.goto_fine(np.array([*wp[2]["joints"], roll, op]))
        g, held = arm.grasp()
        arm.release()
        arm.lift(np.array([*wp[1]["joints"], roll, op]))
        results.append({"adelantar_mm": round(al * 1000, 1), "bajar_mm": round(dn * 1000, 1), "pinza": round(g, 1),
                        "agarra": bool(held), "error_deg": round(err, 2)})
        print(f"  adelantar {al * 1000:+.0f} mm, bajar {dn * 1000:.0f} mm -> pinza {g:.1f} {'AGARRA' if held else 'vacia'} "
              f"(bajada {err:.2f} grados)", flush=True)
        if a.stop_first and held:
            break
    arm.goto(np.array([*home[:4], roll, op]))
    good = [r for r in results if r["agarra"]]
    if not good:
        print("Ninguna prueba agarra: amplia --along/--down o revisa que el peon este en la casilla.")
        return 2
    al_m = float(np.mean([r["adelantar_mm"] for r in good])) / 1000
    dn_m = float(np.mean([r["bajar_mm"] for r in good])) / 1000
    print(f"correccion fila {rank}: adelantar {al_m * 1000:.1f} mm, bajar {dn_m * 1000:.1f} mm ({len(good)}/{len(results)} agarran)")
    if a.save:
        save_rank_correction(rank, al_m, dn_m, {"casilla": sq, "aciertos": f"{len(good)}/{len(results)}", "pruebas": results})
        print("guardado", RANK_CORR_FILE)
    return 0


def _wrist_session(sq: str):
    """Brazo en el origen -> encima de la casilla. Devuelve (ejecutor, brazo, roll, apertura, waypoints) o None."""
    home = load_home()
    if home is None:
        print("falta hub/chess_home.json")
        return None
    ex = IKExecutor(grasp="scripted")
    ex.set_board(chess.Board(game_fen()))
    xy = ex.planner.kin.squares[sq]
    wp = ex.planner.waypoints(xy, xy, chess.PAWN, home[:4])
    arm = ex._arm = Arm()
    if np.abs(arm.joints()[:4] - home[:4]).max() > 8:
        print("el brazo no esta en el origen; no se mueve")
        return None
    roll, op = arm.joints()[4], GRIP_APPROACH[chess.PAWN]
    arm.goto(np.array([*wp[1]["joints"], roll, op]))
    return ex, arm, roll, op, wp, xy, home


def cmd_wrist_ref(a) -> int:
    """Guarda la referencia de la camara de la muneca: 1 cm por encima de un peon que AGARRA BIEN en esa casilla (usa el
    desplazamiento aprendido de la casilla). Mira despues hub/wrist_ref/<clave>_anotada.jpg: el recuadro debe enmarcar la
    cabeza del peon entre los dedos; si no, repite con --roi x0,y0,x1,y1 (fracciones)."""
    sq = a.square.lower()
    s = _wrist_session(sq)
    if s is None:
        return 1
    ex, arm, roll, op, wp, xy, home = s
    learned = square_offsets().get(sq) or [0.0, 0.0]
    q = ex._hover_q((xy[0] + learned[0], xy[1] + learned[1]), chess.PAWN, roll, op)
    err = arm.goto_fine(q)
    time.sleep(0.5)
    img = WR.WristCam(CFG.WRIST_CAM).grab()
    roi = tuple(float(v) for v in a.roi.split(",")) if a.roi else WR.DEFAULT_ROI
    key = ex.wrist_ref_key(sq, chess.PAWN)
    meta = WR.save_ref(img, key, sq, roi)
    print(f"referencia {key} en {sq} (desplazamiento {np.round(np.array(learned) * 1000, 1).tolist()} mm, bajada {err:.2f} grados): {meta}")
    print("revisa", WR.REF_DIR / f"{key}_anotada.jpg")
    arm.lift(np.array([*wp[1]["joints"], roll, op]))
    arm.goto(np.array([*home[:4], roll, op]))
    return 0


def cmd_wrist_measure(a) -> int:
    """Solo mide (no agarra): error de la pieza respecto a los dedos en una casilla, con la camara de la muneca."""
    sq = a.square.lower()
    s = _wrist_session(sq)
    if s is None:
        return 1
    ex, arm, roll, op, wp, xy, home = s
    ex._wcam = WR.WristCam(CFG.WRIST_CAM)
    r = ex.wrist_measure(arm, sq, xy, chess.PAWN, roll, op)
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}, indent=2))
    arm.lift(np.array([*wp[1]["joints"], roll, op]))
    arm.goto(np.array([*home[:4], roll, op]))
    return 0 if r.get("ok") else 2


def cmd_rank_table(a) -> int:
    """Muestra la correccion que se aplica a cada fila (medida, interpolada o sin medir). No mueve nada."""
    t = load_rank_correction()
    for r in range(1, 9):
        al, dn, why = rank_correction(t, r)
        print(f"  fila {r}: adelantar {al * 1000:+.1f} mm, bajar {dn * 1000:.1f} mm  ({why})")
    return 0


def game_fen() -> str:
    """FEN de la partida guardada (CHESS_GAME_FILE); posicion inicial si no hay partida."""
    p = Path(CFG.GAME_FILE)
    return json.loads(p.read_text()).get("fen", chess.STARTING_FEN) if p.exists() else chess.STARTING_FEN


def cmd_run(a) -> int:
    fen = a.fen or game_fen()
    print(f"Posicion de partida: {fen}", flush=True)
    ex = IKExecutor(grasp=a.grasp, seconds=a.seconds, checkpoint=Path(a.checkpoint))
    ex.set_board(chess.Board(fen))
    for s in _steps_for(a.uci, fen):
        r = ex.pick_place(s["label"], s["piece"], s["src_xy"], s["dst_xy"])
        print(json.dumps(r, indent=2, default=float))
        if not r.get("ok"):
            return 3
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("uci"); p.add_argument("--fen")
    p = sub.add_parser("act-offline"); p.add_argument("--square", default="e2"); p.add_argument("--checkpoint", default=str(DEFAULT_ACT))
    p = sub.add_parser("set-home"); p.add_argument("--wait", type=float, default=5.0); p.add_argument("--force", action="store_true")
    p = sub.add_parser("test-grasp"); p.add_argument("--square", default="e2"); p.add_argument("--go", action="store_true")
    p = sub.add_parser("make-home"); p.add_argument("--x", type=float, default=0.14); p.add_argument("--y", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.12); p.add_argument("--save", action="store_true")
    p = sub.add_parser("grasp-probe"); p.add_argument("--square", default="e2"); p.add_argument("--step", type=float, default=0.008)
    p.add_argument("--dz", default="0.0"); p.add_argument("--save", action="store_true"); p.add_argument("--from-saved", action="store_true")
    p = sub.add_parser("rank-probe"); p.add_argument("--rank", required=True); p.add_argument("--square")
    p.add_argument("--along", default="0,3,-3,6"); p.add_argument("--down", default="0,3")
    p.add_argument("--save", action="store_true"); p.add_argument("--stop-first", action="store_true")
    sub.add_parser("rank-table")
    p = sub.add_parser("wrist-ref"); p.add_argument("--square", default="e2"); p.add_argument("--roi")
    p = sub.add_parser("wrist-measure"); p.add_argument("--square", default="e4")
    p = sub.add_parser("run"); p.add_argument("uci"); p.add_argument("--fen"); p.add_argument("--grasp", choices=["act", "scripted"], default="act")
    p.add_argument("--seconds", type=float, default=8.0); p.add_argument("--checkpoint", default=str(DEFAULT_ACT))
    a = ap.parse_args()
    return {"plan": cmd_plan, "act-offline": cmd_act_offline, "set-home": cmd_set_home, "test-grasp": cmd_test_grasp, "make-home": cmd_make_home, "grasp-probe": cmd_grasp_probe, "rank-probe": cmd_rank_probe, "rank-table": cmd_rank_table, "wrist-ref": cmd_wrist_ref, "wrist-measure": cmd_wrist_measure, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
