"""Convierte detecciones de piezas (píxeles de imagen) en estado de tablero (FEN).

Entrada:
  - corners: 4 puntos de píxel de las esquinas exteriores del tablero, en este orden:
    a1 (abajo-izquierda), h1 (abajo-derecha), h8 (arriba-derecha), a8 (arriba-izquierda),
    tal como se ve en la imagen de la cámara.
  - detections: lista de dicts {"label": "white-king", "conf": 0.9, "box": [x1, y1, x2, y2]}.

Salida: FEN de posición (solo la parte de colocación de piezas) y el mapa casilla -> pieza.

La pieza se ubica por el centro inferior de su caja (la base de la pieza), no por el centro,
porque las piezas son altas y la base es la que cae sobre la casilla.
"""

from __future__ import annotations

import numpy as np
import cv2

# Nombres de clase del dataset público chess-full (6 por color).
LABEL_TO_FEN = {
    "white-king": "K", "white-queen": "Q", "white-rook": "R",
    "white-bishop": "B", "white-knight": "N", "white-pawn": "P",
    "black-king": "k", "black-queen": "q", "black-rook": "r",
    "black-bishop": "b", "black-knight": "n", "black-pawn": "p",
}

FILES = "abcdefgh"


def board_homography(corners_px) -> np.ndarray:
    """Matriz que pasa de píxeles de imagen a coordenadas de tablero [0, 8] x [0, 8]."""
    src = np.asarray(corners_px, dtype=np.float32)
    if src.shape != (4, 2):
        raise ValueError("corners debe tener forma (4, 2): a1, h1, h8, a8")
    dst = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def pixel_to_square(H: np.ndarray, x: float, y: float) -> str | None:
    """Devuelve la casilla (p. ej. 'e4') que contiene el píxel (x, y), o None si cae fuera."""
    pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)[0, 0]
    bx, by = pt
    if not (0 <= bx < 8 and 0 <= by < 8):
        return None
    file_idx = int(bx)
    rank_idx = int(by)  # 0 = fila 1 (abajo)
    return f"{FILES[file_idx]}{rank_idx + 1}"


def detections_to_squares(corners_px, detections, min_conf: float = 0.3) -> dict[str, dict]:
    """Asigna cada detección a una casilla. Si hay varias en la misma casilla, gana la de mayor confianza."""
    H = board_homography(corners_px)
    best: dict[str, dict] = {}
    for det in detections:
        if det.get("conf", 0.0) < min_conf:
            continue
        if det["label"] not in LABEL_TO_FEN:
            continue
        x1, y1, x2, y2 = det["box"]
        base_x = (x1 + x2) / 2.0
        base_y = y2  # base de la pieza = borde inferior de la caja
        square = pixel_to_square(H, base_x, base_y)
        if square is None:
            continue
        prev = best.get(square)
        if prev is None or det["conf"] > prev["conf"]:
            best[square] = det
    return best


def squares_to_fen_placement(squares: dict[str, dict]) -> str:
    """Construye la parte de colocación de piezas de un FEN (8 filas, de 8 a 1)."""
    rows = []
    for rank in range(8, 0, -1):
        row = ""
        empty = 0
        for f in FILES:
            det = squares.get(f"{f}{rank}")
            if det is None:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += LABEL_TO_FEN[det["label"]]
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)
