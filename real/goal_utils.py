"""Objetivo de ACT: un punto 'x,y' o una jugada 'origen,destino' en casillas (sin calibrar el tablero real).

Formas aceptadas (separadas por coma o espacio):
  - 'x,y'              -> 2 valores, metros del marco de la sim (como hasta ahora)
  - 'x1,y1,x2,y2'      -> 4 valores: origen y destino en metros
  - 'e2,e4'            -> 2 casillas: se traducen a 4 valores con la geometría de la SIM (hub/chess_board.json)
  - 'e2'               -> 1 casilla: 2 valores

La geometría de la sim solo sirve de ETIQUETA: ACT aprende de las demostraciones reales qué hacer
con ese número. Vale mientras la cámara y el tablero no se muevan entre grabar y desplegar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BOARD_SIM = Path(__file__).resolve().parent.parent / "hub" / "chess_board.json"
_SQ = re.compile(r"^[a-h][1-8]$")


def square_label_xy(name: str) -> tuple[float, float]:
    squares = json.loads(BOARD_SIM.read_text())["squares"]
    return float(squares[name][0]), float(squares[name][1])


def parse_goal(text: str) -> np.ndarray:
    parts = [p for p in re.split(r"[,\s]+", text.strip().lower()) if p]
    if parts and all(_SQ.match(p) for p in parts):
        if len(parts) not in (1, 2):
            raise ValueError("indica una casilla (e2) u origen y destino (e2,e4)")
        return np.array([c for p in parts for c in square_label_xy(p)], dtype=np.float32)
    try:
        vals = [float(p) for p in parts]
    except ValueError as err:
        raise ValueError("objetivo no válido: usa 'x,y', 'x1,y1,x2,y2' o casillas 'e2,e4'") from err
    if len(vals) not in (2, 4):
        raise ValueError("el objetivo debe tener 2 valores (x,y) o 4 (origen y destino), o casillas 'e2,e4'")
    return np.array(vals, dtype=np.float32)


def goal_names(dim: int) -> list[str]:
    return ["target_x", "target_y"] if dim == 2 else ["src_x", "src_y", "dst_x", "dst_y"]
