"""Localiza el brazo naranja del SO-101 en la imagen de la cámara de mesa (cámara 0 de YOLO).

No mueve nada. Devuelve el píxel de la PUNTA (el punto del brazo más cercano al centro del tablero)
y el cuadro del brazo, para que Claude calcule cuánto falta hasta una casilla.
Limitación: con una sola cámara solo se obtiene la posición en la imagen (2D), no la altura.
"""

from __future__ import annotations

import cv2
import numpy as np

# Naranja del SO-101 en OpenCV HSV (H 0..179). Las casillas rojas del tablero tienen H ~170-179/0-4 y menos saturación.
LO = np.array([6, 150, 110], np.uint8)
HI = np.array([22, 255, 255], np.uint8)
MIN_AREA_PX = 400


def orange_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LO, HI)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def find_gripper(img_bgr: np.ndarray, toward_xy: tuple[float, float] | None = None) -> dict | None:
    """Mayor mancha naranja. `toward_xy`: punto de referencia (centro del tablero); la punta es el punto
    de la mancha más cercano a él. Sin referencia, el más a la izquierda."""
    mask = orange_mask(img_bgr)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], default=None)
    if best is None or stats[best, cv2.CC_STAT_AREA] < MIN_AREA_PX:
        return None
    ys, xs = np.nonzero(labels == best)
    pts = np.stack([xs, ys], 1).astype(float)
    if toward_xy is None:
        tip = pts[np.argmin(pts[:, 0])]
    else:
        tip = pts[np.argmin(np.linalg.norm(pts - np.asarray(toward_xy), axis=1))]
    x, y, w, h = (int(stats[best, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return {"tip_px": [round(float(tip[0]), 1), round(float(tip[1]), 1)], "box": [x, y, x + w, y + h],
            "area_px": int(stats[best, cv2.CC_STAT_AREA])}
