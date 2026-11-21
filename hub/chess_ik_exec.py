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

import chess_config as CFG  # noqa: E402  (única fuente de cámaras, puertos, rutas y límites)

MOTION_FILE = CFG.MOTION_FILE
HOME_FILE = HUB / "chess_home.json"
DEFAULT_ACT = CFG.ACT_GENERIC_CKPT
STEP_DEG = CFG.STEP_DEG   # tope = SO101_MAX_RELATIVE_TARGET (AGENTS.md: empezar con 2 grados/paso)
HZ = CFG.HZ
BIAS_KI = 0.25             # ganancia del sesgo integral (por tick)
BIAS_MAX = 3.0             # grados: sesgo maximo por articulacion
GRIP_OPEN = 40.0            # % de apertura de la pinza al acercarse (SIN medir con piezas reales)
GRIP_CLOSE_STEP = 1.5       # % por tick al cerrar
APPROACH_ABOVE = 0.07       # m por encima de la altura de agarre (la punta queda por encima del rey en el transito)
PLACE_EXTRA = 0.004         # m por encima del agarre al soltar
ENVELOPE_MARGIN = 8.0       # grados de holgura de la caja de seguridad del ACT a cada lado del tramo aproximacion-agarre
# Alturas de las piezas (m). MEDIDAS: peon 3,5 cm y rey 5,5 cm (usuario, 2026-11-21). El resto son ESTIMACIONES sin medir.
PIECE_HEIGHT = {chess.PAWN: 0.035, chess.ROOK: 0.040, chess.KNIGHT: 0.043, chess.BISHOP: 0.046, chess.QUEEN: 0.050, chess.KING: 0.055}
GRASP_FRACTION = 0.60       # la punta agarra a este % de la altura de la pieza (por el cuerpo, no por la cabeza)
GRASP_DZ = {p: round(h * GRASP_FRACTION, 4) for p, h in PIECE_HEIGHT.items()}  # m sobre la superficie del tablero
STATE_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


motion_permission = CFG.motion_permission  # gate único


def load_home() -> np.ndarray | None:
    if not HOME_FILE.exists():
        return None
    p = json.loads(HOME_FILE.read_text()).get("pose_origen_grados")
    return np.array([p[k] for k in STATE_KEYS], float) if p else None


# --------------------------------------------------------------------------- plan (sin hardware)
class Planner:
    def __init__(self, kin: Kin | None = None):
        self.kin = kin or Kin()
        self.board_bounds = self._board_bounds()

    @staticmethod
    def _board_bounds() -> tuple[float, float, float]:
        """(x_min, x_max, |y|_max) de la zona del tablero: casillas calibradas + 1 casilla de margen (antes fijo)."""
        board = json.loads((HUB / "chess_board_real.json").read_text())
        pts = list(board["squares"].values())
        m = float(board.get("square_size_m", 0.03))
        xs = [p[0] for p in pts]
        return min(xs) - m, max(xs) + m, max(abs(p[1]) for p in pts) + m

    def waypoints(self, src_xy, dst_xy, piece_type: int, home4: np.ndarray | None) -> list[dict]:
        g = GRASP_DZ[piece_type]
        s_grasp = self.kin.solve_xy(src_xy, g)
        s_above = self.kin.solve_xy(src_xy, g + APPROACH_ABOVE, k_hint=s_grasp["k"]) if s_grasp["joints"] else None
        d_place = self.kin.solve_xy(dst_xy, g + PLACE_EXTRA)
        d_above = self.kin.solve_xy(dst_xy, g + PLACE_EXTRA + APPROACH_ABOVE, k_hint=d_place["k"]) if d_place["joints"] else None
        wp = []

        def add(name, sol, grip):
            wp.append({"name": name, "joints": None if sol is None or sol["joints"] is None else np.array(sol["joints"]),
                       "grip": grip, "tilt": None if sol is None else sol["tilt"], "margin": None if sol is None else sol["margin"]})

        wp.append({"name": "ORIGEN", "joints": None if home4 is None else home4, "grip": GRIP_OPEN, "tilt": None, "margin": None})
        add("encima origen", s_above, GRIP_OPEN)
        add("agarre origen", s_grasp, GRIP_OPEN)
        add("subir con pieza", s_above, None)
        add("encima destino", d_above, None)
        add("soltar destino", d_place, None)
        add("subir", d_above, GRIP_OPEN)
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
            max_relative_target=STEP_DEG, disable_torque_on_disconnect=False))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise RuntimeError("El seguidor no tiene calibracion.")

    def joints(self) -> np.ndarray:
        o = self.robot.get_observation()
        return np.array([o[f"{k}.pos"] for k in STATE_KEYS], float)

    def send(self, q6) -> None:
        self.robot.send_action({f"{k}.pos": float(v) for k, v in zip(STATE_KEYS, q6)})

    def goto(self, target6, tol: float = 1.0) -> None:
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
            if np.abs(d).max() < tol:
                ok_ticks += 1
                if ok_ticks >= 3:
                    return
            else:
                ok_ticks = 0
            near = np.abs(d) < STEP_DEG
            bias = np.where(near, np.clip(bias + BIAS_KI * d, -BIAS_MAX, BIAS_MAX), 0.0)
            cmd = d + bias
            big = np.abs(cmd).max()
            if big > STEP_DEG:
                cmd = cmd * (STEP_DEG / big)
            self.send(cur + cmd)
            time.sleep(1 / HZ)
        raise RuntimeError("no converge al waypoint (carga, tope o colision): me paro")

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
        self.cams = {"front": CFG.YOLO_CAM, "gripper": CFG.WRIST_CAM}

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

        info: dict = {"accion": label}
        arm.goto(q6(wp[1], GRIP_OPEN))                     # encima del origen, pinza abierta
        if self.grasp == "act":
            self._act = self._act or ACTWindow(self.checkpoint)
            info["act"] = self._act.run(arm, wp[1]["joints"], wp[2]["joints"], self.seconds)
        else:
            arm.goto(q6(wp[2], GRIP_OPEN))                 # bajar
            info["grip_cerrada"] = arm.close_until_stall(np.array([roll]))
        grip = arm.joints()[5]
        info["pinza"] = float(grip)
        held = 4.0 < grip < GRIP_OPEN - 5.0
        info["agarrada"] = bool(held)
        arm.goto(np.array([*wp[3]["joints"], roll, grip]))  # subir (con lo que tenga)
        if not held:
            arm.goto(q6(wp[7], GRIP_OPEN))
            return {**info, "ok": False, "motivo": "no hay pieza en la pinza tras el agarre; vuelvo al origen"}
        arm.goto(np.array([*wp[4]["joints"], roll, grip]))
        arm.goto(np.array([*wp[5]["joints"], roll, grip]))
        arm.goto(np.array([*wp[5]["joints"], roll, GRIP_OPEN]))   # soltar
        arm.goto(q6(wp[6], GRIP_OPEN))
        arm.goto(q6(wp[7], GRIP_OPEN))                     # ORIGEN
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
    p = sub.add_parser("run"); p.add_argument("uci"); p.add_argument("--fen"); p.add_argument("--grasp", choices=["act", "scripted"], default="act")
    p.add_argument("--seconds", type=float, default=8.0); p.add_argument("--checkpoint", default=str(DEFAULT_ACT))
    a = ap.parse_args()
    return {"plan": cmd_plan, "act-offline": cmd_act_offline, "set-home": cmd_set_home, "test-grasp": cmd_test_grasp, "make-home": cmd_make_home, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
