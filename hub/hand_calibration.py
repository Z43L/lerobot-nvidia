"""Calibración de la cámara del Mac (cámara de mesa) respecto a la base del SO-101.

Crea hub/hand_calibration.json con la homografía píxel -> mesa (metros, marco de la base del robot).
Lo usa la skill so101-pasar-lata para localizar la palma de la mano.

Flujo (desde la raíz del proyecto, no mueve ningún brazo):

  1. Captura una foto de la mesa con los 4 marcadores colocados:
       ./.venv312/bin/python hub/hand_calibration.py capture

  2. Marca los 4 puntos en la imagen, en el MISMO orden en que escribas sus coordenadas (x, y en metros):
       ./.venv312/bin/python hub/hand_calibration.py calibrate \
           --xy "0.20,-0.10" --xy "0.20,0.10" --xy "0.30,0.10" --xy "0.30,-0.10"

     Se abre una ventana con la foto. Haz clic en cada marcador en el orden de --xy.
     Opcional: un quinto punto de comprobación con --check "x,y" (se marca después de los 4).

  3. Comprueba:
       ./.venv312/bin/python hub/hand_calibration.py check <u> <v>

Convención de coordenadas (confírmala antes de calibrar):
  - Origen: el centro de la base del SO-101 sobre la mesa.
  - x: hacia el frente del robot (positivo = más lejos de la base).
  - y: lateral (positivo = izquierda del robot visto desde detrás).
  - Unidades: metros.
Los marcadores deben estar sobre la mesa, a la misma altura, y no en línea recta.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = Path(__file__).resolve().parent / "hand_calibration.json"
FRAMES_DIR = Path(__file__).resolve().parent / "frames"
DEFAULT_HAND_Z_M = 0.10  # altura supuesta de la palma; confirmar con el usuario


def _camera_index() -> int:
    idx = int(os.environ.get("SO101_TABLE_CAM", "-1"))
    return idx if idx >= 0 else 0


def capture(index: int | None = None, out: Path | None = None) -> Path:
    """Toma una foto de la cámara de mesa (la webcam del Mac) y la guarda."""
    index = _camera_index() if index is None else index
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir la cámara {index}. Revisa el permiso de cámara de la Terminal.")
    # Unos cuadros de calentamiento para que la exposición se estabilice.
    ok, frame = False, None
    for _ in range(15):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"La cámara {index} no devuelve imagen.")
    FRAMES_DIR.mkdir(exist_ok=True)
    out = out or FRAMES_DIR / "table_calib.png"
    cv2.imwrite(str(out), frame)
    print(f"[ OK ] cámara {index}: {frame.shape[1]}x{frame.shape[0]} -> {out}")
    return out


def parse_xy(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Coordenada mal formada: '{text}'. Usa 'x,y' en metros.")
    return np.array([float(parts[0]), float(parts[1])], dtype=float)


def _check_geometry(xy: np.ndarray) -> None:
    if xy.shape[0] < 4:
        raise ValueError("Hacen falta al menos 4 puntos.")
    # Tres puntos casi alineados dan una homografía inestable.
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            for k in range(j + 1, len(xy)):
                a, b, c = xy[i], xy[j], xy[k]
                area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area2 < 1e-4:  # ~1 cm² de área
                    raise ValueError(f"Los puntos {i+1}, {j+1} y {k+1} están casi alineados. Sepáralos.")


def click_points(image: np.ndarray, n: int, title: str) -> np.ndarray:
    """Pide n clics en la imagen. Devuelve los píxeles (u, v) en el orden de los clics."""
    clicks: list[tuple[int, int]] = []
    win = title
    view = image.copy()

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < n:
            clicks.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = image.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        msg = f"Clic {len(clicks) + 1}/{n}" if len(clicks) < n else "Listo: pulsa ENTER o ESC para cancelar"
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise RuntimeError("Calibración cancelada.")
        if len(clicks) == n and key in (13, 10, 32):  # ENTER o espacio
            break
    cv2.destroyWindow(win)
    return np.array(clicks, dtype=float)


def compute_homography(pix: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Homografía píxel -> mesa y error de reproyección medio (metros) sobre los puntos de calibración."""
    H, _ = cv2.findHomography(pix.astype(np.float32), xy.astype(np.float32), method=0)
    if H is None:
        raise RuntimeError("No se pudo calcular la homografía. Revisa los puntos.")
    err = np.linalg.norm(apply_h(H, pix) - xy, axis=1).mean()
    return H, float(err)


def apply_h(H: np.ndarray, pix: np.ndarray) -> np.ndarray:
    pix = np.atleast_2d(np.asarray(pix, dtype=float))
    homo = np.hstack([pix, np.ones((pix.shape[0], 1))])
    mapped = homo @ H.T
    return mapped[:, :2] / mapped[:, 2:3]


def pixel_to_table(u: float, v: float, calib: dict | None = None) -> tuple[float, float]:
    """Convierte un píxel de la cámara de mesa a (x, y) en metros. Lo usa so101-pasar-lata."""
    calib = calib or json.loads(CALIB_PATH.read_text())
    H = np.array(calib["H"], dtype=float)
    x, y = apply_h(H, [[u, v]])[0]
    return float(x), float(y)


def calibrate(xy_points: list[str], check_xy: str | None, image_path: Path, out: Path = CALIB_PATH) -> dict:
    xy = np.array([parse_xy(s) for s in xy_points], dtype=float)
    _check_geometry(xy)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se puede leer la imagen {image_path}. Ejecuta antes 'capture'.")

    print(f"Marca {len(xy)} marcadores en el orden de --xy:")
    for i, p in enumerate(xy):
        print(f"  {i+1}: x={p[0]:.3f} m, y={p[1]:.3f} m")
    pix = click_points(image, len(xy), "Calibración de mesa: 4 marcadores")
    H, err = compute_homography(pix, xy)

    print(f"Error de reproyección medio sobre los 4 puntos: {err*1000:.1f} mm")

    check = None
    if check_xy:
        cxy = parse_xy(check_xy)
        print(f"Marca ahora el punto de comprobación x={cxy[0]:.3f}, y={cxy[1]:.3f}")
        cpix = click_points(image, 1, "Comprobación")
        est = apply_h(H, cpix)[0]
        check = {"xy_m": cxy.tolist(), "pixel": cpix[0].tolist(), "estimated_xy_m": est.tolist(),
                 "error_mm": float(np.linalg.norm(est - cxy) * 1000)}
        print(f"Comprobación: error {check['error_mm']:.1f} mm (estimado {est[0]:.3f}, {est[1]:.3f})")

    calib = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "camera_index": _camera_index(),
        "image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "frame": "base del SO-101 sobre la mesa; x hacia el frente, y a la izquierda; metros",
        "points_px": pix.tolist(),
        "points_xy_m": xy.tolist(),
        "reprojection_error_mm": round(err * 1000, 2),
        "check": check,
        "H": H.tolist(),
        "hand_z_m": DEFAULT_HAND_Z_M,
        "status": "pendiente_verificar_en_real",
    }
    out.write_text(json.dumps(calib, indent=2))
    print(f"[ OK ] guardado en {out}")
    return calib


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture", help="Foto de la cámara de mesa (sin mover el brazo)")
    cal = sub.add_parser("calibrate", help="Marca 4 puntos y guarda hub/hand_calibration.json")
    cal.add_argument("--xy", action="append", required=True, help="x,y en metros; repite 4 veces")
    cal.add_argument("--check", default=None, help="x,y de un quinto punto para comprobar")
    cal.add_argument("--image", default=str(FRAMES_DIR / "table_calib.png"))
    chk = sub.add_parser("check", help="Convierte un píxel (u v) a x,y en metros")
    chk.add_argument("u", type=float)
    chk.add_argument("v", type=float)
    args = ap.parse_args(argv)

    if args.cmd == "capture":
        capture()
    elif args.cmd == "calibrate":
        if len(args.xy) != 4:
            sys.exit("Hacen falta exactamente 4 --xy.")
        calibrate(args.xy, args.check, Path(args.image))
    elif args.cmd == "check":
        x, y = pixel_to_table(args.u, args.v)
        print(f"pixel ({args.u:.0f}, {args.v:.0f}) -> x={x:.3f} m, y={y:.3f} m")


if __name__ == "__main__":
    main()
