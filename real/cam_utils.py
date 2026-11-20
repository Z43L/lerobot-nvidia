"""Resolución de las cámaras del SO-101 real.

La cámara USB de la mesa (SO101_TOP_CAM) no respeta 320x240 en el driver de macOS:
entrega 640x480 aunque se pida 320x240, y LeRobot rechaza la configuración al conectar.
Solución: pedirla a su resolución nativa (640x480) y reducir cada frame a 320x240 por software,
que es la resolución con la que se entrena y despliega ACT (IMG_HW = (240, 320)).

La cámara de la pinza (SO101_WRIST_CAM) sí abre a 320x240 y no se reescala.
"""

from __future__ import annotations

import cv2
import numpy as np

TRAIN_W, TRAIN_H = 320, 240  # ancho x alto de entrenamiento y despliegue
NATIVE_W, NATIVE_H = 640, 480  # resolución que pide el driver para la cámara de la mesa


def capture_size(cam: str) -> tuple[int, int]:
    """Resolución que se pide al driver: la cámara 'top' (mesa) a 640x480, el resto a 320x240."""
    return (NATIVE_W, NATIVE_H) if cam == "top" else (TRAIN_W, TRAIN_H)


def to_train_size(img: np.ndarray) -> np.ndarray:
    """Devuelve el frame a 320x240 (alto 240, ancho 320). No hace nada si ya tiene ese tamaño."""
    if img.shape[0] == TRAIN_H and img.shape[1] == TRAIN_W:
        return img
    return cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
