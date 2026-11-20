"""Tests sintéticos de board_mapping (sin cámara ni modelo).

Ejecutar: ./.venv312/bin/python -m pytest vision/test_board_mapping.py  (o python vision/test_board_mapping.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from board_mapping import (  # noqa: E402
    detections_to_squares,
    squares_to_fen_placement,
    pixel_to_square,
    board_homography,
)

# Tablero de 400 px en la imagen, esquina a1 abajo-izquierda, con margen de 100 px.
CORNERS = [(100, 500), (500, 500), (500, 100), (100, 100)]
SQ = 50  # píxeles por casilla


def square_center_px(square: str):
    f = "abcdefgh".index(square[0])
    r = int(square[1]) - 1
    x = 100 + (f + 0.5) * SQ
    y = 500 - (r + 0.5) * SQ
    return x, y


def piece_box_on(square: str, label: str, conf: float = 0.9):
    """Caja de pieza cuya base (borde inferior) cae en el centro de la casilla."""
    cx, cy = square_center_px(square)
    return {"label": label, "conf": conf, "box": [cx - 15, cy - 60, cx + 15, cy]}


def test_pixel_to_square_corners_and_center():
    H = board_homography(CORNERS)
    assert pixel_to_square(H, 325, 325) == "e4"
    assert pixel_to_square(H, 105, 495) == "a1"
    assert pixel_to_square(H, 495, 105) == "h8"
    assert pixel_to_square(H, 50, 50) is None  # fuera del tablero


def test_starting_position_fen():
    start = {
        "a1": "white-rook", "b1": "white-knight", "c1": "white-bishop", "d1": "white-queen",
        "e1": "white-king", "f1": "white-bishop", "g1": "white-knight", "h1": "white-rook",
    }
    for f in "abcdefgh":
        start[f"{f}2"] = "white-pawn"
        start[f"{f}7"] = "black-pawn"
    back = {"a8": "black-rook", "b8": "black-knight", "c8": "black-bishop", "d8": "black-queen",
            "e8": "black-king", "f8": "black-bishop", "g8": "black-knight", "h8": "black-rook"}
    start.update(back)
    dets = [piece_box_on(sq, lab) for sq, lab in start.items()]
    squares = detections_to_squares(CORNERS, dets)
    assert squares_to_fen_placement(squares) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_low_confidence_and_duplicates():
    dets = [
        piece_box_on("e4", "white-pawn", conf=0.4),
        piece_box_on("e4", "black-knight", conf=0.8),  # gana por confianza
        piece_box_on("d4", "white-queen", conf=0.1),  # descartada por confianza mínima
    ]
    squares = detections_to_squares(CORNERS, dets, min_conf=0.3)
    assert list(squares) == ["e4"]
    assert squares["e4"]["label"] == "black-knight"
    assert squares_to_fen_placement(squares) == "8/8/8/8/4n3/8/8/8"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
