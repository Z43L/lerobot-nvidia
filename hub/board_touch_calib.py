"""Calibra el tablero real respecto a la base del SO-101 tocándolo con la punta de la pinza.

Convierte las articulaciones (grados) de 3 casillas de referencia en metros usando la cinemática
directa del modelo de sim (mismo MJCF que el brazo real) y genera hub/chess_board_real.json, con el
mismo formato que hub/chess_board.json. Así square_xy() devuelve el centro real de cada casilla.

Comandos (desde la raíz del proyecto):

  # Cinemática directa: punta de la pinza para 5 articulaciones en grados
  ./.venv312/bin/python hub/board_touch_calib.py fk --deg 0,0,0,0,0

  # Prueba completa en sim: IK a casillas conocidas -> articulaciones -> fit -> comparar
  ./.venv312/bin/python hub/board_touch_calib.py selftest

  # Real (SIN mover el brazo): lee las articulaciones del seguidor y las anota para una casilla.
  # Antes: pon la punta tocando la casilla (lo mueves tú con el líder, con SI y ALLOW_REAL_MOTION=1).
  source real/env.local.sh
  ./.venv312/bin/python hub/board_touch_calib.py read --square a1 --touches hub/touches_real.json

  # Real: calcula el tablero (escribe solo si el error de la comprobación h6 es < 5 mm)
  ./.venv312/bin/python hub/board_touch_calib.py fit --touches hub/touches_real.json --out hub/chess_board_real.json

Convenciones:
  - Marco: base del SO-101 sobre la mesa. x hacia el frente, y a la izquierda, metros.
  - Casilla a1 = índice (0, 0); el centro de la casilla (f, r) está en (f + 0.5, r + 0.5) en unidades de casilla.
  - Puntos de toque: centro de la casilla, a TOUCH_Z de altura (la altura de agarre de sim).
  - a1, h1 y a6 se usan para calibrar; h6 es la comprobación independiente.
  - Alcance (sim): con la pinza vertical a 2 cm, las filas 1-6 llegan; la fila 8 y las esquinas de la 7 NO.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HUB = ROOT / "hub"
SIM_DIR = ROOT / "sim"
SIM_BOARD = HUB / "chess_board.json"
TOUCH_SQUARES = ["a1", "h1", "a6"]  # calibran (la fila 8 queda fuera de alcance con la pinza vertical)
CHECK_SQUARE = "h6"                 # comprobación
TOUCH_Z = 0.02                      # m: altura de contacto (igual que grasp_z en sim)
MAX_CHECK_ERROR_M = 0.005           # 5 mm, como en la skill so101-calibracion-tablero
FRAME = "base del SO-101 sobre la mesa; x hacia el frente, y a la izquierda; metros"

_SIM = None


def _sim():
    """Entorno de sim (una vez) para usar su modelo, su sitio de la punta y su IK."""
    global _SIM
    if _SIM is None:
        sys.path.insert(0, str(SIM_DIR))
        from objects_env import ObjectsSim

        _SIM = ObjectsSim()
    return _SIM


def fk_tip(deg5) -> np.ndarray:
    """Punta de la pinza (m, marco de la base) para 5 articulaciones en grados."""
    sim = _sim()
    sim.data.qpos[:5] = np.radians(np.asarray(deg5, dtype=float)[:5])
    mujoco.mj_forward(sim.model, sim.data)
    return sim.data.site_xpos[sim.tcp_site].copy()


def ik_deg(xyz) -> tuple[np.ndarray, float]:
    """Articulaciones en grados que llevan la punta a xyz, y el error de posición (m)."""
    sim = _sim()
    q5 = sim.ik(np.asarray(xyz, dtype=float))
    err = float(np.linalg.norm(fk_tip(np.degrees(q5)) - np.asarray(xyz)))
    return np.degrees(q5), err


def square_uv(name: str) -> np.ndarray:
    """Centro de la casilla en unidades de casilla (a1 -> (0.5, 0.5))."""
    f = ord(name[0].lower()) - ord("a")
    r = int(name[1]) - 1
    if not (0 <= f < 8 and 0 <= r < 8):
        raise ValueError(f"casilla inválida: {name}")
    return np.array([f + 0.5, r + 0.5])


def fit_board(touches: dict[str, np.ndarray]) -> dict:
    """Ajusta el tablero a partir de las posiciones tocadas (casilla -> xy en metros).

    Modelo afín: xy = [u, v, 1] @ M, con (u, v) en unidades de casilla. Con a1, h1 y a6 es exacto.
    La casilla de comprobación (h6) se usa para medir el error.
    """
    missing = [s for s in TOUCH_SQUARES if s not in touches]
    if missing:
        raise ValueError(f"faltan casillas de calibración: {missing}")
    names = TOUCH_SQUARES
    uv = np.array([square_uv(s) for s in names])
    xy = np.array([touches[s] for s in names])
    X = np.hstack([uv, np.ones((len(uv), 1))])
    M = np.linalg.solve(X, xy)  # 3x2, exacto con 3 puntos no alineados

    ex, ey = M[0], M[1]  # cambio de xy por casilla en cada eje
    size_x, size_y = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    angle = float(np.degrees(np.arccos(np.clip(np.dot(ex, ey) / (size_x * size_y), -1, 1))))
    result = {
        "square_size_m": round((size_x + size_y) / 2, 4),
        "size_mismatch_pct": round(100 * abs(size_x - size_y) / ((size_x + size_y) / 2), 2),
        "orthogonality_deg": round(angle, 2),
        "M": M,
    }
    if CHECK_SQUARE in touches:
        pred = np.array([*square_uv(CHECK_SQUARE), 1.0]) @ M
        result["check_error_m"] = float(np.linalg.norm(pred - touches[CHECK_SQUARE]))
    return result


def board_squares(M: np.ndarray) -> dict[str, list[float]]:
    out = {}
    for f in range(8):
        for r in range(8):
            xy = np.array([f + 0.5, r + 0.5, 1.0]) @ M
            out[f"{chr(ord('a') + f)}{r + 1}"] = [round(float(xy[0]), 4), round(float(xy[1]), 4)]
    return out


def write_board(path: Path, fit: dict, touches: dict[str, np.ndarray], status: str = "real") -> dict:
    M = fit["M"]
    center = np.array([4.0, 4.0, 1.0]) @ M  # esquina común de e4, d4, e5, d5
    data = {
        "frame": FRAME,
        "source": "touch_calibration",
        "status": status,
        "square_size_m": fit["square_size_m"],
        "center_m": [round(float(center[0]), 4), round(float(center[1]), 4)],
        "graveyard_m": [],  # el cementerio real se define aparte (ver sim/chess_real.py)
        "squares": board_squares(M),
        "calibration": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "touch_xy_m": {k: [round(float(v[0]), 4), round(float(v[1]), 4)] for k, v in touches.items()},
            "check_square": CHECK_SQUARE,
            "check_error_mm": round(1000 * fit.get("check_error_m", float("nan")), 2),
            "size_mismatch_pct": fit["size_mismatch_pct"],
            "orthogonality_deg": fit["orthogonality_deg"],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


# ---------------------------------------------------------------- comandos

def load_touches(path: Path) -> dict[str, np.ndarray]:
    """Lee el fichero de toques: [{"square": "a1", "deg": [5 valores]}, ...]. Usa el último de cada casilla."""
    entries = json.loads(path.read_text())
    touches: dict[str, np.ndarray] = {}
    for e in entries:
        touches[e["square"].lower()] = fk_tip(e["deg"])
    return touches


def cmd_fk(args) -> int:
    deg = [float(v) for v in args.deg.split(",")]
    print("punta (m):", np.round(fk_tip(deg), 4).tolist())
    return 0


def cmd_fit(args) -> int:
    touches = load_touches(Path(args.touches))
    fit = fit_board(touches)
    print(f"lado de casilla: {fit['square_size_m'] * 100:.2f} cm  (diferencia x/y {fit['size_mismatch_pct']} %)")
    print(f"ortogonalidad: {fit['orthogonality_deg']}°")
    if "check_error_m" in fit:
        print(f"error en la casilla de comprobación ({CHECK_SQUARE}): {fit['check_error_m'] * 1000:.2f} mm")
    if "check_error_m" not in fit or fit["check_error_m"] > MAX_CHECK_ERROR_M:
        if not args.force:
            print(f"[FAIL] la comprobación no cumple (< {MAX_CHECK_ERROR_M * 1000:.0f} mm). No se escribe. Repite las medidas o usa --force.")
            return 1
        print("[WARN] --force: se escribe aunque la comprobación no cumpla.")
    out = Path(args.out)
    write_board(out, fit, touches, status="real" if not args.sim else "sim_touch")
    print(f"escrito {out}")
    return 0


def cmd_read(args) -> int:
    """Lee las articulaciones del seguidor y las guarda para una casilla. NO mueve el brazo."""
    import os

    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    robot = make_robot_from_config(SO101FollowerConfig(
        port=os.environ.get("SO101_FOLLOWER_PORT", ""),
        id=os.environ.get("SO101_FOLLOWER_ID", "so101_follower"),
        cameras={},
        use_degrees=True,
    ))
    robot.connect()
    try:
        obs = robot.get_observation()
    finally:
        robot.disconnect()
    motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    deg = [round(float(obs[f"{m}.pos"]), 3) for m in motors]
    path = Path(args.touches)
    entries = json.loads(path.read_text()) if path.exists() else []
    entries.append({"square": args.square.lower(), "deg": deg, "time": datetime.now().isoformat(timespec="seconds")})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))
    print(f"{args.square}: {deg} -> guardado en {path}")
    return 0


def cmd_selftest(args) -> int:
    """Sim: casillas del tablero de sim -> IK -> articulaciones -> fit -> comparar con chess_board.json."""
    if not SIM_BOARD.exists():
        print(f"falta {SIM_BOARD}")
        return 1
    truth = json.loads(SIM_BOARD.read_text())["squares"]
    rng = np.random.default_rng(0)
    noise_deg = args.noise_deg
    entries = []
    ik_errors = []
    for name in TOUCH_SQUARES + [CHECK_SQUARE]:
        x, y = truth[name]
        deg, err = ik_deg([x, y, TOUCH_Z])
        ik_errors.append(err)
        deg = deg + rng.normal(0.0, noise_deg, size=5)
        entries.append({"square": name, "deg": deg.round(4).tolist()})
    tmp = ROOT / "hub" / "touches_selftest.json"
    tmp.write_text(json.dumps(entries, indent=2))
    touches = load_touches(tmp)
    fit = fit_board(touches)
    board = write_board(args.out, fit, touches, status="sim_selftest")
    errs = [np.linalg.norm(np.array(board["squares"][s]) - np.array(truth[s])) for s in truth]
    print(f"IK: error máximo de posición al generar toques: {max(ik_errors) * 1000:.2f} mm (ruido {noise_deg}°)")
    print(f"lado recuperado: {fit['square_size_m'] * 100:.2f} cm  (sim: {json.loads(SIM_BOARD.read_text())['square_size_m'] * 100:.2f} cm)")
    print(f"error en la comprobación ({CHECK_SQUARE}): {fit['check_error_m'] * 1000:.2f} mm")
    print(f"error máximo en las 64 casillas frente a chess_board.json: {max(errs) * 1000:.2f} mm")
    tmp.unlink(missing_ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fk", help="punta de la pinza para 5 articulaciones en grados")
    f.add_argument("--deg", required=True, help="pan,lift,elbow,wrist_flex,wrist_roll")
    f.set_defaults(fn=cmd_fk)

    r = sub.add_parser("read", help="leer articulaciones del seguidor para una casilla (no mueve el brazo)")
    r.add_argument("--square", required=True)
    r.add_argument("--touches", default=str(HUB / "touches_real.json"))
    r.set_defaults(fn=cmd_read)

    fi = sub.add_parser("fit", help="calcular el tablero a partir de los toques")
    fi.add_argument("--touches", default=str(HUB / "touches_real.json"))
    fi.add_argument("--out", default=str(HUB / "chess_board_real.json"))
    fi.add_argument("--force", action="store_true", help="escribir aunque la comprobación no cumpla 5 mm")
    fi.add_argument("--sim", action="store_true", help="marcar como sim_touch (pruebas)")
    fi.set_defaults(fn=cmd_fit)

    s = sub.add_parser("selftest", help="prueba completa en sim")
    s.add_argument("--noise-deg", type=float, default=0.0, help="ruido gaussiano por articulación (grados)")
    s.add_argument("--out", type=Path, default=HUB / "chess_board_selftest.json")
    s.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
