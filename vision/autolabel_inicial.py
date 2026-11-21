"""Autoetiqueta fotos en vivo del tablero en POSICION INICIAL (verdad conocida) para corregir confusiones de YOLO.

Captura N fotos de la camara, detecta con conf baja, y para cada casilla ocupada en la posicion inicial se queda con la
caja de mas confianza que cae en ella y le pone la clase correcta. Casillas sin caja se cuentan (no se inventan cajas).
Guarda en vision/dataset_camara/images/train y labels/train con prefijo 'live_'. Overlays en vision/frames/autolabel/.

  ./.venv312/bin/python vision/autolabel_inicial.py --camera 0 --n 15
Solo vale con TODAS las piezas en la posicion inicial, camara y tablero quietos (esquinas de board_camara0.json).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chess_vision import LABEL, ChessVision  # noqa: E402

DS = HERE / "dataset_camara"
CLASS_ID = {"black-bishop": 1, "black-king": 2, "black-knight": 3, "black-pawn": 4, "black-queen": 5, "black-rook": 6,
            "white-bishop": 7, "white-king": 8, "white-knight": 9, "white-pawn": 10, "white-queen": 11, "white-rook": 12}
NAME_OF = {v: k for k, v in LABEL.items()}  # (color, tipo) -> nombre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--geometry", default=str(HERE / "board_camara0.json"))
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()

    cv = ChessVision(geometry=a.geometry, conf=a.conf)
    truth = ChessVision.board_to_squares(chess.Board())
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"No abre la camara {a.camera}")
    for _ in range(10):
        cap.read()
    (DS / "images/train").mkdir(parents=True, exist_ok=True)
    (DS / "labels/train").mkdir(parents=True, exist_ok=True)
    ov = HERE / "frames/autolabel"
    ov.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    cobertura = []
    for i in range(a.n):
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        best: dict[str, dict] = {}
        for d in cv.detect(img):
            x1, y1, x2, y2 = d["box"]
            sq = cv.square_of((x1 + x2) / 2, y2 - cv.base_k * (y2 - y1))
            if sq in truth and (sq not in best or d["conf"] > best[sq]["conf"]):
                best[sq] = d
        lines = []
        for sq, d in best.items():
            x1, y1, x2, y2 = d["box"]
            name = NAME_OF[truth[sq]]
            lines.append(f"{CLASS_ID[name]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            cv2.rectangle(img.copy(), (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cobertura.append(len(best))
        base = f"live_{stamp}_{i:02d}"
        cv2.imwrite(str(DS / f"images/train/{base}.jpg"), img)
        (DS / f"labels/train/{base}.txt").write_text("\n".join(lines) + "\n")
        vis = img.copy()
        for sq, d in best.items():
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(vis, f"{sq}{''.join(truth[sq])}", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(ov / f"{base}.jpg"), vis)
        time.sleep(0.4)
    cap.release()
    print(f"fotos={len(cobertura)} cajas por foto (de 32): {cobertura}")
    faltan = sorted(set(truth) - set())  # informativo
    print("overlays en", ov)


if __name__ == "__main__":
    main()
