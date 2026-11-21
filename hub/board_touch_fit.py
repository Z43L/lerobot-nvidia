"""Calibración robusta del tablero real por contacto: desfases de ángulo + posición del tablero, a la vez.

Por qué existe (board_touch_calib.py fit no basta): con la cinemática del modelo de sim y los grados del seguidor real,
las casillas salen a alturas distintas (hasta 3,5 cm) y el error de comprobación pasa de 12 mm. El cero de LeRobot
(centro del rango) no es el cero del modelo de sim: faltan unos grados de desfase por articulación. Este script
resuelve a la vez 4 desfases (pan, lift, elbow, wrist_flex) y el tablero rígido (origen, giro, lado de casilla y altura
de contacto) por mínimos cuadrados (Levenberg-Marquardt) con N casillas tocadas (mejor >= 7, repartidas).

Uso (desde la raíz; no mueve el brazo):
  ./.venv312/bin/python hub/board_touch_fit.py --touches hub/touches_real.json --out hub/chess_board_real.json

Convención del tablero (marco de la base, metros): rangos hacia +x, archivos hacia +y. a1 = (0.5, 0.5) casillas.
Los xy del fichero están en el marco de la sim tras aplicar los desfases. Para mover el brazo real con la IK de sim:
    grados_reales = grados_IK_sim - joint_offsets_deg   (el fichero guarda joint_offsets_deg)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HUB = Path(__file__).resolve().parent
sys.path.insert(0, str(HUB))
import board_touch_calib as b  # noqa: E402

JOINTS4 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
LO = np.array([-15, -15, -15, -15, -1, -1, -7, 0.025, -0.2])
HI = np.array([15, 15, 15, 15, 1, 1, 7, 0.040, 0.2])
SC = np.array([1, 1, 1, 1, 0.01, 0.01, 0.05, 0.002, 0.01])  # escala de cada parámetro (para el paso numérico)
X0 = np.r_[-3.8, -8.1, -6.2, -2.5, 0.15, -0.09, 0.0, 0.03, 0.02]


def mirror_square(s: str) -> str:
    """Refleja las columnas (a<->h). Las etiquetas impresas del tablero real estan espejadas respecto al ajedrez estandar:
    las lecturas se hicieron con a1 junto al robot, pero las reglas necesitan la dama blanca en d1 (a1 lejos del robot)."""
    return chr(ord("a") + 7 - (ord(s[0]) - 97)) + s[1:]


def uv(s: str) -> np.ndarray:
    """Centro de la casilla en unidades de casilla: (rango -> x, archivo -> y)."""
    return np.array([int(s[1]) - 1 + 0.5, ord(s[0]) - 97 + 0.5])


def board_xy(P, s: str) -> np.ndarray:
    ox, oy, th, sz = P
    c, sn = np.cos(th), np.sin(th)
    return np.array([ox, oy]) + sz * (np.array([[c, -sn], [sn, c]]) @ uv(s))


def residuals(Q, touches: dict[str, np.ndarray], use: list[str]) -> np.ndarray:
    off = np.r_[Q[:4], 0.0]
    r: list[float] = []
    for s in use:
        x = b.fk_tip(touches[s] + off)
        r += list(x[:2] - board_xy(Q[4:8], s)) + [x[2] - Q[8]]
    return np.array(r)


def lm(touches, use, x0):
    x = np.clip(x0, LO, HI)
    lam = 1e-2
    r = residuals(x, touches, use)
    c = r @ r
    for _ in range(200):
        J = np.array([(residuals(np.clip(x + np.eye(9)[i] * SC[i] * 0.02, LO, HI), touches, use) - r) / (SC[i] * 0.02)
                      for i in range(9)]).T
        A, g = J.T @ J, J.T @ r
        ok = False
        for _ in range(12):
            dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            xn = np.clip(x + dx, LO, HI)
            rn = residuals(xn, touches, use)
            cn = rn @ rn
            if cn < c:
                x, r, c, lam, ok = xn, rn, cn, lam / 3, True
                break
            lam *= 4
        if not ok or np.abs(dx / SC).max() < 1e-4:
            break
    return x, c


def solve(touches, use, x0=None):
    if x0 is not None:
        return lm(touches, use, x0)[0]
    best = None
    for th in np.radians([-90, -45, 0, 45, 90, 135, 180, -135]):
        x = X0.copy()
        x[6] = th * 0.15
        x, c = lm(touches, use, x)
        if best is None or c < best[1]:
            best = (x, c)
    return best[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--touches", default=str(HUB / "touches_real.json"))
    ap.add_argument("--out", default=str(HUB / "chess_board_real.json"))
    ap.add_argument("--max-loo-mm", type=float, default=12.0, help="no escribe si el peor error 'una fuera' lo supera")
    ap.add_argument("--user-labels", action="store_true",
                    help="escribe con las etiquetas de las lecturas (a1 junto al robot). Por defecto se reflejan las columnas "
                         "para que sean estandar (dama blanca en d1): ver MIRROR_FILES")
    args = ap.parse_args()

    entries = json.loads(Path(args.touches).read_text())
    touches: dict[str, np.ndarray] = {}
    for e in entries:  # la última lectura de cada casilla
        touches[e["square"].lower()] = np.array(e["deg"], float)
    names = list(touches)
    if len(names) < 6:
        print(f"[FAIL] hacen falta al menos 6 casillas tocadas (hay {len(names)}).")
        return 1
    x = solve(touches, names)
    r = residuals(x, touches, names).reshape(-1, 3)
    rms = float(np.sqrt((r**2).mean()) * 1000)
    loo = {}
    for s in names:
        use = [n for n in names if n != s]
        xx = solve(touches, use, x.copy())
        loo[s] = float(np.linalg.norm(b.fk_tip(touches[s] + np.r_[xx[:4], 0.0])[:2] - board_xy(xx[4:8], s)) * 1000)
    print(f"casillas: {names}")
    print(f"desfases (pan, lift, elbow, wrist_flex): {np.round(x[:4], 2).tolist()} grados | casilla {x[7] * 100:.2f} cm "
          f"| giro {np.degrees(x[6]):.1f} grados | z contacto {x[8] * 100:.1f} cm | rms {rms:.1f} mm")
    print("error 'una fuera' (mm):", {k: round(v, 1) for k, v in loo.items()})
    if max(loo.values()) > args.max_loo_mm:
        print(f"[FAIL] peor error 'una fuera' {max(loo.values()):.1f} mm > {args.max_loo_mm} mm. Mide más casillas o repite las dudosas.")
        return 2

    mir = (lambda s: s) if args.user_labels else mirror_square
    squares = {mir(f"{f}{rk}"): [round(float(board_xy(x[4:8], f"{f}{rk}")[0]), 4), round(float(board_xy(x[4:8], f"{f}{rk}")[1]), 4)]
               for f in "abcdefgh" for rk in range(1, 9)}
    center = board_xy(x[4:8], "d4") + (board_xy(x[4:8], "e5") - board_xy(x[4:8], "d4")) / 2  # esquina común d4,e4,d5,e5
    data = {
        "frame": b.FRAME,
        "source": "touch_calibration_robust",
        "status": "real",
        "labels": "estandar (a1 lejos del robot, dama blanca en d1)" if not args.user_labels else "del usuario (a1 junto al robot)",
        "square_size_m": round(float(x[7]), 4),
        "center_m": [round(float(center[0]), 4), round(float(center[1]), 4)],
        "graveyard_m": [],
        "joint_offsets_deg": {n: round(float(v), 2) for n, v in zip(JOINTS4, x[:4])},
        "contact_z_m": round(float(x[8]), 4),
        "squares": squares,
        "calibration": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "n_touches": len(names),
            "touch_deg": {mir(k): [round(float(v), 3) for v in touches[k]] for k in names},
            "rms_mm": round(rms, 2),
            "leave_one_out_mm": {mir(k): round(v, 1) for k, v in loo.items()},
            "nota": "Los xy estan en el marco de la sim tras aplicar joint_offsets_deg. Para el brazo real: "
                    "grados_reales = grados_IK_sim - joint_offsets_deg. Precision tipica 3-6 mm en el interior, "
                    "hasta ~11 mm en las esquinas (extrapolacion).",
        },
    }
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"escrito {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
