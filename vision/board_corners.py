"""Calibra las 4 esquinas del tablero en la cámara cenital (píxeles) para vision/board_mapping.py.

No mueve el brazo. Flujo (desde la raíz del proyecto):

  1. Captura la cenital y marca las esquinas del área de juego (las 8x8 casillas, no el marco):
       ./.venv312/bin/python vision/board_corners.py

     Orden de los clics, según las etiquetas impresas del tablero en la imagen:
       1) a1 = abajo-derecha   2) h1 = abajo-izquierda
       3) h8 = arriba-izquierda   4) a8 = arriba-derecha
     Pulsa ENTER (o espacio) cuando hayas marcado las 4. ESC cancela.

  2. Guarda vision/board_corners.json y vision/frames/board_overlay.png (rejilla de comprobación).
     Revisa el overlay: si una esquina está mal, repite el paso 1.

Índice de cámara: SO101_TOP_CAM (por defecto 0, la cenital según real/config.py).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "hub"))

from board_mapping import board_homography, pixel_to_square, FILES  # noqa: E402
from hand_calibration import click_points  # noqa: E402

FRAMES_DIR = HERE / "frames"
OUT_JSON = HERE / "board_corners.json"
ORDER = ["a1", "h1", "h8", "a8"]


def capture(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir la cámara {index}.")
    ok, frame = False, None
    for _ in range(15):  # calentamiento de exposición
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"La cámara {index} no devuelve imagen.")
    return frame


def square_centers_px(H_inv: np.ndarray) -> dict[str, tuple[int, int]]:
    """Centro en píxeles de cada casilla, usando la inversa de la homografía."""
    centers = {}
    for fi, f in enumerate(FILES):
        for r in range(8):
            pt = cv2.perspectiveTransform(np.array([[[fi + 0.5, r + 0.5]]], dtype=np.float32), H_inv)[0, 0]
            centers[f"{f}{r + 1}"] = (int(pt[0]), int(pt[1]))
    return centers


def draw_overlay(image: np.ndarray, corners: np.ndarray, out: Path) -> None:
    H = board_homography(corners)
    H_inv = np.linalg.inv(H)
    view = image.copy()
    # Contorno del área de juego
    cv2.polylines(view, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    # Rejilla
    for i in range(9):
        p0 = cv2.perspectiveTransform(np.array([[[i, 0]]], dtype=np.float32), H_inv)[0, 0]
        p1 = cv2.perspectiveTransform(np.array([[[i, 8]]], dtype=np.float32), H_inv)[0, 0]
        q0 = cv2.perspectiveTransform(np.array([[[0, i]]], dtype=np.float32), H_inv)[0, 0]
        q1 = cv2.perspectiveTransform(np.array([[[8, i]]], dtype=np.float32), H_inv)[0, 0]
        cv2.line(view, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 255, 0), 1)
        cv2.line(view, tuple(q0.astype(int)), tuple(q1.astype(int)), (0, 255, 0), 1)
    for name, (x, y) in square_centers_px(H_inv).items():
        cv2.putText(view, name, (x - 12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), view)


def self_check(corners: np.ndarray) -> list[str]:
    """Comprueba que los 4 clics forman un cuadrilátero convexo en el orden a1, h1, h8, a8."""
    problems = []
    # Convexo y con el mismo sentido de giro en las 4 esquinas.
    signs = []
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        signs.append(np.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
    if len(set(signs)) != 1 or 0 in signs:
        problems.append("los 4 puntos no forman un cuadrilátero convexo en el orden a1, h1, h8, a8")
    else:
        # El sentido de giro debe coincidir con el de a1 (abajo-dcha) -> h1 (abajo-izq) -> h8 (arriba-izq) -> a8.
        ref = np.array([[1, 1], [0, 1], [0, 0], [1, 0]], float)  # en coordenadas de imagen (y hacia abajo)
        def turn(p):
            return np.sign(sum(p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1] for i in range(4)))
        if turn(corners) != turn(ref):
            problems.append("el orden de los clics va al revés: marca a1 (abajo-dcha), h1, h8, a8 en ese sentido")
    # a1 debe ser la esquina más abajo-derecha de la imagen (supone la cámara cenital y el tablero girado < 45°).
    if len(problems) == 0 and int(np.argmax(corners[:, 0] + corners[:, 1])) != 0:
        problems.append("a1 no es la esquina abajo-derecha: revisa el orden de los clics")
    # Área razonable (un tablero ocupa una parte grande de la imagen).
    area = cv2.contourArea(corners.astype(np.float32))
    if area < 50_000:
        problems.append(f"área del tablero demasiado pequeña ({area:.0f} px²)")
    return problems


def main() -> None:
    index = int(os.environ.get("SO101_TOP_CAM", "0"))
    frame = capture(index)
    FRAMES_DIR.mkdir(exist_ok=True)
    raw = FRAMES_DIR / "board_top.png"
    cv2.imwrite(str(raw), frame)
    print(f"[ OK ] cámara {index}: {frame.shape[1]}x{frame.shape[0]} -> {raw}")

    print("Marca en orden: a1 (abajo-dcha), h1 (abajo-izq), h8 (arriba-izq), a8 (arriba-dcha). ENTER al terminar.")
    pts = click_points(frame, 4, "Esquinas del tablero: a1, h1, h8, a8")
    problems = self_check(pts)
    if problems:
        print("[FAIL] revisa los clics:", "; ".join(problems))
        sys.exit(1)

    overlay = FRAMES_DIR / "board_overlay.png"
    draw_overlay(frame, pts, overlay)
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "camera_index": index,
        "order": ORDER,
        "corners_px": pts.tolist(),
        "image": str(raw.relative_to(ROOT)),
        "status": "pendiente_verificar_overlay",
    }
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(f"[ OK ] esquinas guardadas en {OUT_JSON}")
    print(f"       overlay de comprobación: {overlay}")


if __name__ == "__main__":
    main()
