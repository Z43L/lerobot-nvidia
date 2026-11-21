"""Ejecutor de jugadas del SO-101 REAL: IK para llegar a la pieza + ventana de ACT para el agarre, y vuelta al origen.

Es el "camino 2": la IK lleva la pinza hasta encima de la pieza y el ACT genérico (SIN objetivo, p. ej. el de 30.000 pasos
de act_tray_a100_100k_ckpts) toma el control durante unos segundos, CONFINADO a una caja de seguridad alrededor de la casilla.
El ACT no sabe qué pieza coger; lo único que se le pide es que baje, cierre la pinza y suba. Eso NO está probado: por eso hay
tambien `--grasp scripted` (IK baja, cierra la pinza hasta que se detiene, sube) para comparar y como plan B.

Secuencia de cada coger-y-soltar (cada jugada termina en el ORIGEN, hub/chess_home.json):
  origen -> encima(src) -> [ACT | bajar+cerrar] -> subir -> encima(dst) -> bajar -> abrir -> subir -> ORIGEN

Seguridad (AGENTS.md): sin ALLOW_REAL_MOTION=1 y sin el permiso con caducidad real/.motion_ok (que crea el USUARIO con
real/authorize_motion.sh) no se mueve nada. Pasos de como mucho 2 grados por articulacion, torque encendido al parar.
Para en el primer fallo (no vuelve solo): corte de alimentacion a mano si algo va mal.

Sin hardware (no mueve nada):
  ./.venv312/bin/python hub/chess_ik_exec.py plan e2e4                 # waypoints, limites, pasos, tiempo
  ./.venv312/bin/python hub/chess_ik_exec.py act-offline --square e2   # que haria el ACT desde encima de la casilla
Con el brazo (solo con SI del usuario):
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

MOTION_FILE = ROOT / "real" / ".motion_ok"
HOME_FILE = HUB / "chess_home.json"
DEFAULT_ACT = ROOT / "act_tray_a100_100k_ckpts" / "checkpoints" / "030000" / "pretrained_model"
STEP_DEG = float(os.environ.get("CHESS_STEP_DEG", "2.0"))   # AGENTS.md: empezar con 2 grados/paso
HZ = 30.0
BIAS_KI = 0.25             # ganancia del sesgo integral (por tick)
BIAS_MAX = 3.0             # grados: sesgo maximo por articulacion
GRIP_OPEN = 40.0            # apertura en el transito alto (ORIGEN). OJO: 40 = ~7,4 cm entre puntas, barre 2 casillas
# Apertura al acercarse a una pieza: lo justo para que entre sin tocarla ni tocar las vecinas (usuario, 21-nov).
# Pinza 0..100 = 140 grados de servo; dedo de 8,2 cm: hueco ~ 2*0.082*sin((v-1.8)*1.4/2 grados). 14 -> ~2,4 cm.
GRIP_APPROACH = {chess.PAWN: 14.0, chess.ROOK: 16.0, chess.KNIGHT: 16.0, chess.BISHOP: 15.0, chess.QUEEN: 17.0, chess.KING: 17.0}
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


def motion_permission() -> tuple[bool, str]:
    if os.environ.get("CHESS_REQUIRE_PERMISSION") != "1":  # por defecto: movimiento directo, sin permiso con caducidad
        return True, "movimiento directo (sin permiso)"
    if os.environ.get("ALLOW_REAL_MOTION") != "1":
        return False, "ALLOW_REAL_MOTION=1 no esta puesto."
    if not MOTION_FILE.exists():
        return False, "Sin permiso del usuario: real/authorize_motion.sh <minutos> (area libre, persona junto al brazo, corte a mano)."
    try:
        if time.time() > float(MOTION_FILE.read_text().strip()):
            return False, "El permiso de movimiento ha caducado."
    except ValueError:
        return False, "real/.motion_ok no es valido."
    return True, "ok"


def load_grasp_offset() -> np.ndarray:
    """(dx, dy, dz) en metros que se suman a cada agarre y soltado. Lo escribe `grasp-probe`."""
    if not GRASP_OFFSET_FILE.exists():
        return np.zeros(3)
    d = json.loads(GRASP_OFFSET_FILE.read_text())
    return np.array([d.get("dx_m", 0.0), d.get("dy_m", 0.0), d.get("dz_m", 0.0)], float)


RETRY_OFFSETS_MM = [(0, 0), (0, 3), (0, -3), (3, 0), (-3, 0), (3, 3), (-3, 3), (3, -3), (-3, -3)]


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
    def __init__(self, kin: Kin | None = None, offset: np.ndarray | None = None):
        self.kin = kin or Kin()
        self.offset = load_grasp_offset() if offset is None else np.asarray(offset, float)

    def waypoints(self, src_xy, dst_xy, piece_type: int, home4: np.ndarray | None) -> list[dict]:
        o = self.offset
        src_xy = (src_xy[0] + o[0], src_xy[1] + o[1])
        dst_xy = (dst_xy[0] + o[0], dst_xy[1] + o[1])
        g = GRASP_DZ[piece_type] + o[2]
        s_grasp = self.kin.solve_xy(src_xy, g)
        s_above = self.kin.solve_xy(src_xy, g + APPROACH_ABOVE, k_hint=s_grasp["k"]) if s_grasp["joints"] else None
        d_place = self.kin.solve_xy(dst_xy, g + PLACE_EXTRA)
        d_above = self.kin.solve_xy(dst_xy, g + PLACE_EXTRA + APPROACH_ABOVE, k_hint=d_place["k"]) if d_place["joints"] else None
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
        for a, b2 in zip(wp, wp[1:]):
            if a["joints"] is None or b2["joints"] is None:
                continue
            descent = "agarre" in b2["name"] or "soltar" in b2["name"] or "agarre" in a["name"] or "soltar" in a["name"]
            zmin, where = 9.9, None
            for s in np.linspace(0, 1, 40):
                q = a["joints"] + s * (b2["joints"] - a["joints"])
                tip = b.fk_tip(np.r_[q, 0.0] + self.kin.off)
                over_board = 0.19 <= tip[0] <= 0.46 and abs(tip[1]) <= 0.14
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

        port = os.environ.get("SO101_FOLLOWER_PORT", "")
        if not port:
            raise RuntimeError("Falta SO101_FOLLOWER_PORT.")
        self.robot = SO101Follower(SO101FollowerConfig(
            port=port, id=os.environ.get("SO101_FOLLOWER_ID", "so101_follower"), cameras={}, use_degrees=True,
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

    FINE_TOL = float(os.environ.get("CHESS_FINE_TOL", "0.7"))

    def goto_fine(self, target6, check_grip: bool = True) -> float:
        """Posicionado fino para bajar al agarre o al soltado: primero goto normal y luego afina hasta FINE_TOL grados en
        pan/lift/elbow/wrist_flex (con el sesgo integral). No lanza error si no llega: devuelve el error maximo (grados)."""
        target = np.asarray(target6, float)
        self.goto(target, check_grip=check_grip)
        bias = np.zeros(4)
        for _ in range(int(HZ * 3)):
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
class ACTWindow:
    def __init__(self, checkpoint: Path = DEFAULT_ACT, device: str | None = None):
        from deploy_goal import load_policy

        self.device = device or os.environ.get("SO101_VLA_DEVICE", "mps")
        self.policy, self.pre, self.post = load_policy(Path(checkpoint), self.device)
        self.motors = [f"{k}.pos" for k in STATE_KEYS]
        self.cams = {"front": int(os.environ.get("CHESS_YOLO_CAM", "0")), "gripper": int(os.environ.get("CHESS_WRIST_CAM", "2"))}

    def frames(self) -> dict[str, np.ndarray]:
        from deploy_act_generic import to_act_size

        out = {}
        for k, i in self.cams.items():
            c = cv2.VideoCapture(i)
            for _ in range(3):
                ok, f = c.read()
            c.release()
            if not ok:
                raise RuntimeError(f"la camara {i} no devuelve imagen")
            out[k] = to_act_size(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return out

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

    def pick_place(self, label, piece_type, src_xy, dst_xy) -> dict:
        ok, why = motion_permission()
        if not ok:
            raise RuntimeError(f"No se mueve el brazo: {why}")
        home6 = load_home()
        if home6 is None:
            raise RuntimeError("Falta hub/chess_home.json (pose de origen): `chess_ik_exec.py set-home` con el brazo colocado.")
        wp = self.planner.waypoints(src_xy, dst_xy, piece_type, home6[:4])
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
            print("Brazo en el origen. Torque encendido: SUELTA el brazo y apartate de su zona. Empiezo en 5 s (Ctrl+C o corte de alimentacion para parar).", flush=True)
            time.sleep(5)

        def q6(w, grip=None):
            g = w["grip"] if grip is None and w["grip"] is not None else (arm.joints()[5] if grip is None else grip)
            return np.array([*w["joints"], roll, g], float)

        op = GRIP_APPROACH[piece_type]                       # apertura justa para esta pieza
        info: dict = {"accion": label, "apertura": op, "hueco_cm": round(gap_cm(op), 1), "offset_m": [round(float(v), 4) for v in self.planner.offset]}

        def back_home(motivo: str) -> dict:
            arm.release()
            arm.goto(q6(wp[3], op))
            arm.goto(q6(wp[7], op))
            return {**info, "ok": False, "motivo": motivo}

        arm.goto(q6(wp[1], op))                     # encima del origen, pinza abierta
        if self.grasp == "act":
            # Ventana del ACT de 30k confinada a la caja encima->agarre (+-ENVELOPE_MARGIN). Luego la IK termina de bajar.
            self._act = self._act or ACTWindow(self.checkpoint)
            info["act"] = self._act.run(arm, wp[1]["joints"], wp[2]["joints"], self.seconds)
        # Agarre con reintentos: si cierra en vacio, prueba desplazamientos de 3 mm (empieza por el aprendido en esa casilla).
        src_sq = min(self.planner.kin.squares, key=lambda s: np.hypot(*(np.array(self.planner.kin.squares[s]) - src_xy)))
        learned = square_offsets().get(src_sq)
        tries = ([tuple(learned)] if learned else []) + [(dx / 1000, dy / 1000) for dx, dy in RETRY_OFFSETS_MM]
        held, intentos = False, []
        for k, (dx, dy) in enumerate(tries[:int(os.environ.get("CHESS_GRASP_TRIES", "6"))]):
            src2 = (src_xy[0] + dx, src_xy[1] + dy)
            w = self.planner.waypoints(src2, dst_xy, piece_type, home6[:4])
            if w[1]["joints"] is None or w[2]["joints"] is None:
                continue
            if k:
                arm.goto(q6(w[1], op))                     # sube y se recoloca encima con el nuevo desplazamiento
            err = arm.goto_fine(q6(w[2], op))              # bajar (fino)
            grip, held = arm.grasp()                       # cierre suave, detecta contacto
            intentos.append({"dxy_mm": [round(dx * 1000, 1), round(dy * 1000, 1)], "pinza": round(grip, 1), "error_deg": round(err, 2)})
            if held:
                wp[1], wp[2], wp[3] = w[1], w[2], w[3]
                save_square_offset(src_sq, (dx, dy))
                break
            arm.release()
        info["casilla_origen"], info["intentos"] = src_sq, intentos
        hold = arm.hold_cmd
        info["pinza_cerrada"], info["agarrada"] = round(grip, 1), bool(held)
        if not held:
            return back_home(f"pinza cerrada en vacio ({grip:.1f} <= {GRIP_EMPTY_MAX}): la pieza no estaba entre los dedos. "
                             "Recalibra el agarre con `grasp-probe`.")
        arm.goto(np.array([*wp[3]["joints"], roll, hold]), check_grip=False)   # subir apretando
        grip = float(arm.joints()[5])
        info["pinza_tras_subir"] = round(grip, 1)
        if grip <= GRIP_EMPTY_MAX:
            return back_home(f"la pieza se solto al subir (pinza {grip:.1f})")
        arm.goto(np.array([*wp[4]["joints"], roll, hold]), check_grip=False)
        arm.goto(np.array([*wp[5]["joints"], roll, hold]), check_grip=False)
        arm.goto_fine(np.array([*wp[5]["joints"], roll, hold]), check_grip=False)
        info["pinza_antes_de_soltar"] = round(float(arm.joints()[5]), 1)
        arm.release()                                      # soltar
        arm.goto(q6(wp[6], op))
        arm.goto(q6(wp[7], op))                     # ORIGEN (punto de retorno)
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
    port = os.environ.get("SO101_FOLLOWER_PORT", "/dev/cu.usbmodem5B610353951")
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


def cmd_run(a) -> int:
    ex = IKExecutor(grasp=a.grasp, seconds=a.seconds, checkpoint=Path(a.checkpoint))
    for s in _steps_for(a.uci, a.fen):
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
    p = sub.add_parser("run"); p.add_argument("uci"); p.add_argument("--fen"); p.add_argument("--grasp", choices=["act", "scripted"], default="act")
    p.add_argument("--seconds", type=float, default=8.0); p.add_argument("--checkpoint", default=str(DEFAULT_ACT))
    a = ap.parse_args()
    return {"plan": cmd_plan, "act-offline": cmd_act_offline, "set-home": cmd_set_home, "test-grasp": cmd_test_grasp, "make-home": cmd_make_home, "grasp-probe": cmd_grasp_probe, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
