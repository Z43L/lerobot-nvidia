"""Medida del error de la punta con la camara de la MUNECA (punto 2, 21-nov).

Idea: la camara va fija a la pinza, asi que los dedos salen siempre en el mismo sitio de la imagen. Si el peon esta bien
centrado entre los dedos, sale en un pixel concreto (la REFERENCIA, tomada en un agarre bueno). Antes de agarrar:

1. foto 1 cm por encima de la pieza (p0);
2. foto tras mover la pinza +3 mm a lo largo de la columna (p1) y +3 mm en lateral (p2);
3. con esas tres posiciones del peon en la imagen se obtiene J (pixeles por mm) EN ESA POSE, sin calibrar la camara;
4. correccion d (mm) = J^-1 (p_ref - p0), recortada a MAX_MM.

El peon se busca con comparacion de plantilla (recorte de la referencia) en una ventana alrededor de su sitio. Si la
coincidencia es baja o J esta mal condicionada, no se corrige (devuelve ok=False con el motivo).

Modo (CHESS_WRIST): off | measure (por defecto: mide y lo apunta, NO corrige) | correct (aplica la correccion).
Referencias: hub/wrist_ref/<clave>.png (recorte) + <clave>.json (roi, pixel, casilla) + <clave>_anotada.jpg.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

HUB = Path(__file__).resolve().parent
REF_DIR = HUB / "wrist_ref"
MODE = os.environ.get("CHESS_WRIST", "measure").lower()
HOVER_M = float(os.environ.get("CHESS_WRIST_HOVER_MM", "10")) / 1000   # punta por encima de la cabeza de la pieza
PROBE_MM = float(os.environ.get("CHESS_WRIST_PROBE_MM", "3"))           # movimiento para medir J
MAX_MM = float(os.environ.get("CHESS_WRIST_MAX_MM", "6"))               # correccion maxima
MIN_SCORE = float(os.environ.get("CHESS_WRIST_MIN_SCORE", "0.6"))       # coincidencia minima de la plantilla
SEARCH_PX = int(os.environ.get("CHESS_WRIST_SEARCH_PX", "220"))         # margen de busqueda alrededor de la ROI (px, imagen completa)
SCALE = 0.5                                                              # se trabaja a media resolucion
# ROI por defecto (fracciones de la imagen): zona entre y por encima de las puntas de los dedos, donde sale la cabeza del
# peon 1 cm por encima (fotos hover_e2_*.jpg del 21-nov, camara 1920x1080). Revisar con <clave>_anotada.jpg.
DEFAULT_ROI = tuple(float(v) for v in os.environ.get("CHESS_WRIST_ROI", "0.45,0.30,0.62,0.56").split(","))


def ref_key(color: str, piece_letter: str) -> str:
    return f"{color}{piece_letter.upper()}"


class WristCam:
    _caps: dict = {}

    def __init__(self, index: int):
        self.index = index

    def grab(self) -> np.ndarray:
        c = self._caps.get(self.index)
        if c is None or not c.isOpened():
            c = self._caps[self.index] = cv2.VideoCapture(self.index)
            for _ in range(10):
                c.read()
        for _ in range(4):          # descarta fotogramas viejos del buffer (el brazo se acaba de mover)
            c.grab()
        ok, f = c.read()
        if not ok:
            raise RuntimeError(f"la camara de la muneca ({self.index}) no devuelve imagen")
        return f


def _prep(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (5, 5), 0)


def save_ref(img: np.ndarray, key: str, square: str, roi_frac=DEFAULT_ROI) -> dict:
    """Guarda la referencia: recorte de la ROI y su centro (pixel donde debe salir la pieza bien centrada)."""
    REF_DIR.mkdir(exist_ok=True)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(roi_frac[0] * w), int(roi_frac[1] * h), int(roi_frac[2] * w), int(roi_frac[3] * h)
    crop = img[y0:y1, x0:x1]
    cv2.imwrite(str(REF_DIR / f"{key}.png"), crop)
    meta = {"roi": [x0, y0, x1, y1], "pixel": [(x0 + x1) / 2, (y0 + y1) / 2], "casilla": square, "tam": [w, h],
            "fecha": time.strftime("%Y-%m-%d %H:%M")}
    (REF_DIR / f"{key}.json").write_text(json.dumps(meta, indent=2))
    ann = img.copy()
    cv2.rectangle(ann, (x0, y0), (x1, y1), (0, 0, 255), 4)
    cv2.imwrite(str(REF_DIR / f"{key}_anotada.jpg"), ann)
    return meta


def load_ref(key: str) -> tuple[np.ndarray, dict] | None:
    p, j = REF_DIR / f"{key}.png", REF_DIR / f"{key}.json"
    if not p.exists() or not j.exists():
        return None
    return cv2.imread(str(p)), json.loads(j.read_text())


def locate(img: np.ndarray, ref: tuple[np.ndarray, dict]) -> tuple[np.ndarray, float]:
    """Centro (px, imagen completa) donde mejor encaja la plantilla de la referencia, y la puntuacion (0..1)."""
    tpl, meta = ref
    x0, y0, x1, y1 = meta["roi"]
    h, w = img.shape[:2]
    sx0, sy0 = max(0, x0 - SEARCH_PX), max(0, y0 - SEARCH_PX)
    sx1, sy1 = min(w, x1 + SEARCH_PX), min(h, y1 + SEARCH_PX)
    win, t = _prep(img[sy0:sy1, sx0:sx1]), _prep(tpl)
    if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
        return np.array([np.nan, np.nan]), 0.0
    res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    cx = sx0 + (loc[0] + t.shape[1] / 2) / SCALE
    cy = sy0 + (loc[1] + t.shape[0] / 2) / SCALE
    return np.array([cx, cy]), float(score)


def solve_correction(p0, p1, p2, ref_px, probe_mm: float = PROBE_MM, max_mm: float = MAX_MM) -> dict:
    """p0: pieza en la pose nominal; p1: tras +probe_mm a lo largo de la columna; p2: tras +probe_mm en lateral.
    Devuelve la correccion (a lo largo, lateral) en mm para que la pieza salga en ref_px."""
    J = np.column_stack([(np.asarray(p1) - p0) / probe_mm, (np.asarray(p2) - p0) / probe_mm])   # px por mm
    sv = np.linalg.svd(J, compute_uv=False)
    if sv.min() < float(os.environ.get("CHESS_WRIST_MIN_PXMM", "1.0")) or sv.max() / max(sv.min(), 1e-9) > 8.0:
        return {"ok": False, "motivo": f"J mal condicionada (valores singulares {np.round(sv, 1).tolist()} px/mm)",
                "J": np.round(J, 2).tolist()}
    err_px = np.asarray(ref_px, float) - np.asarray(p0, float)
    d = np.linalg.solve(J, err_px)
    raw = d.copy()
    n = np.linalg.norm(d)
    if n > max_mm:
        d = d * (max_mm / n)
    return {"ok": True, "mm": np.round(d, 2).tolist(), "mm_sin_recortar": np.round(raw, 2).tolist(),
            "error_px": np.round(err_px, 1).tolist(), "J": np.round(J, 2).tolist(), "recortada": bool(n > max_mm)}
