"""Busca las cámaras disponibles y guarda una foto de cada índice en real/frames/.

  ./.venv312/bin/python real/find_cameras.py

Abre las fotos (real/frames/cam_<índice>.png) para ver cuál es la webcam del Mac
y cuál es cada cámara del SO-101. No mueve ningún brazo.
"""

from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parent / "frames"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    found = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            path = OUT / f"cam_{idx}.png"
            cv2.imwrite(str(path), frame)
            found.append(idx)
            print(f"[ OK ] índice {idx}: {frame.shape[1]}x{frame.shape[0]} -> {path}")
        else:
            print(f"[    ] índice {idx}: sin imagen")
    if not found:
        print("\nNo hay cámaras accesibles. Revisa el permiso de cámara de la Terminal (Ajustes del Sistema > Privacidad y seguridad > Cámara).")


if __name__ == "__main__":
    main()
