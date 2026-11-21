"""Visión del ajedrez: foto del tablero -> piezas por casilla -> FEN -> jugada detectada.

  ./.venv312/bin/python vision/chess_vision.py foto.jpg                 # piezas y FEN de una foto
  ./.venv312/bin/python vision/chess_vision.py --camera 1               # captura de la cámara (índice OpenCV)
  ./.venv312/bin/python vision/chess_vision.py evaluar                   # compara con vision/dataset_camara/positions

Como módulo:
    from chess_vision import ChessVision
    cv = ChessVision()
    seen = cv.read_squares(imagen_bgr)          # {'e4': ('w', 'P', 0.93), ...}
    move = cv.detect_move(board_before, seen)   # chess.Move legal que mejor explica lo visto, o None

Geometría: vision/board_camara.json (esquinas del área de juego en la foto + orientación). Si se mueve la cámara o el
tablero hay que volver a medirlas (vision/board_corners.py). Detector: runs/detect/vision/runs/piezas_camara/weights/best.pt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import chess
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_WEIGHTS = Path(os.environ.get("CHESS_YOLO_WEIGHTS", ROOT / "runs/detect/vision/runs/piezas_camara_v2/weights/best.pt"))
DEFAULT_GEOMETRY = Path(os.environ["CHESS_BOARD_GEOMETRY"]) if os.environ.get("CHESS_BOARD_GEOMETRY") else HERE / "board_camara.json"
FILES = "abcdefgh"
LABEL = {
    "white-pawn": ("w", "P"), "white-knight": ("w", "N"), "white-bishop": ("w", "B"), "white-rook": ("w", "R"),
    "white-queen": ("w", "Q"), "white-king": ("w", "K"), "black-pawn": ("b", "P"), "black-knight": ("b", "N"),
    "black-bishop": ("b", "B"), "black-rook": ("b", "R"), "black-queen": ("b", "Q"), "black-king": ("b", "K"),
}
PIECE_TYPE = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN, "K": chess.KING}


class ChessVision:
    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, geometry: Path | str = DEFAULT_GEOMETRY,
                 device: str = "mps", imgsz: int = 640, conf: float = 0.25, base_k: float = 0.15):
        """base_k: la base de la pieza se toma a base_k * alto de caja por encima del borde inferior de la caja."""
        geo = json.loads(Path(geometry).read_text())
        c = geo["corners_px"]
        # Rejilla (u: rango, v: archivo). TL=(0,0), TR=(8,0), BR=(8,8), BL=(0,8); a1 = BR -> u=7.5 es rango 1, v=7.5 es archivo a.
        src = np.array([c["TL"], c["TR"], c["BR"], c["BL"]], np.float32)
        self.H = cv2.getPerspectiveTransform(src, np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32))
        self.weights, self.device, self.imgsz, self.conf, self.base_k = str(weights), device, imgsz, conf, base_k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import os
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def square_of(self, x: float, y: float) -> str | None:
        u, v = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.H)[0][0]
        if not (0 <= u < 8 and 0 <= v < 8):
            return None
        return f"{FILES[7 - int(v)]}{8 - int(u)}"

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        r = self.model.predict(image_bgr, conf=self.conf, iou=0.5, imgsz=self.imgsz, device=self.device,
                               agnostic_nms=True, verbose=False)[0]
        out = []
        for b in r.boxes:
            name = self.model.names[int(b.cls[0])]
            if name not in LABEL:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append({"label": name, "conf": float(b.conf[0]), "box": [x1, y1, x2, y2]})
        return out

    def squares_from_detections(self, dets: list[dict]) -> dict[str, tuple[str, str, float]]:
        """Casilla -> (color, tipo, confianza). Si dos detecciones caen en la misma casilla gana la de más confianza."""
        seen: dict[str, tuple[str, str, float]] = {}
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            sq = self.square_of((x1 + x2) / 2, y2 - self.base_k * (y2 - y1))
            if sq is None:
                continue
            color, kind = LABEL[d["label"]]
            if sq not in seen or d["conf"] > seen[sq][2]:
                seen[sq] = (color, kind, d["conf"])
        return seen

    def read_squares(self, image_bgr: np.ndarray) -> dict[str, tuple[str, str, float]]:
        return self.squares_from_detections(self.detect(image_bgr))

    # ------------------------------------------------------------------ jugada
    @staticmethod
    def board_to_squares(board: chess.Board) -> dict[str, tuple[str, str]]:
        return {chess.square_name(s): ("w" if p.color else "b", p.symbol().upper()) for s, p in board.piece_map().items()}

    @staticmethod
    def diff_score(expected: dict, seen: dict) -> float:
        """Cuanto más alto, mejor encaja lo visto con lo esperado. Pesa la confianza; penaliza piezas que faltan o sobran."""
        score = 0.0
        for sq in set(expected) | set(seen):
            e = expected.get(sq)
            s = seen.get(sq)
            if e and s:
                score += s[2] if (e[0], e[1]) == (s[0], s[1]) else (0.3 * s[2] if e[0] == s[0] else -0.5)
            elif e and not s:
                score -= 0.6   # pieza esperada que no se ve (oclusiones: menos grave que una pieza de más)
            else:
                score -= 1.0   # pieza vista donde no debería haber ninguna
        return score

    def detect_move(self, board: chess.Board, seen: dict, margin: float = 0.5) -> tuple[chess.Move | None, dict]:
        """Jugada legal que mejor explica lo visto. Devuelve (jugada o None, info).

        None si la mejor jugada no supera a la segunda por `margin` ni a 'no ha cambiado nada'.
        """
        candidates = [(None, self.diff_score(self.board_to_squares(board), seen))]
        for m in board.legal_moves:
            b = board.copy(stack=False)
            b.push(m)
            candidates.append((m, self.diff_score(self.board_to_squares(b), seen)))
        candidates.sort(key=lambda t: -t[1])
        best, second = candidates[0], candidates[1]
        info = {"best": best[0].uci() if best[0] else "sin cambios", "score": round(best[1], 2),
                "second": second[0].uci() if second[0] else "sin cambios", "second_score": round(second[1], 2)}
        if best[0] is None or best[1] - second[1] < margin:
            return None, info
        return best[0], info

    @staticmethod
    def squares_to_fen(seen: dict) -> str:
        rows = []
        for rank in range(8, 0, -1):
            row, empty = "", 0
            for f in FILES:
                p = seen.get(f"{f}{rank}")
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p[1] if p[0] == "w" else p[1].lower()
            rows.append(row + (str(empty) if empty else ""))
        return "/".join(rows)


def verify_move(board_before: chess.Board, move: chess.Move, seen: dict) -> tuple[bool, list[str], list[str]]:
    """Comprueba con lo visto que la jugada ya está hecha. Devuelve (ok, problemas, avisos).

    Bloquea por: origen que sigue ocupado, destino vacío o de otro color, tipo equivocado con confianza >= 0,8
    (salvo promoción), casilla capturada que sigue ocupada y piezas de más con confianza >= 0,5.
    Piezas esperadas que no se ven solo avisan (oclusiones). No registra nada.
    """
    after = board_before.copy()
    after.push(move)
    expected = ChessVision.board_to_squares(after)
    name = chess.square_name
    problems: list[str] = []
    warnings: list[str] = []
    vacated = {name(move.from_square)}
    if board_before.is_castling(move):
        rank = chess.square_rank(move.from_square)
        vacated.add(name(chess.square(7 if chess.square_file(move.to_square) == 6 else 0, rank)))
    if board_before.is_en_passant(move):
        vacated.add(name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))))
    for sq in sorted(vacated):
        if sq in seen and sq not in expected:
            problems.append(f"{sq} debería estar vacía y se ve {seen[sq][:2]}")
    dest = name(move.to_square)
    want = expected[dest]
    if dest not in seen:
        problems.append(f"el destino {dest} aparece vacío")
    else:
        color, kind, conf = seen[dest]
        if color != want[0]:
            problems.append(f"el destino {dest} tiene color {color}, se esperaba {want[0]}")
        elif kind != want[1] and conf >= 0.8 and not move.promotion:
            problems.append(f"el destino {dest} tiene {kind}, se esperaba {want[1]} (conf {conf:.2f})")
    for sq, (color, kind, conf) in sorted(seen.items()):
        if sq not in expected and sq not in vacated and conf >= 0.5:
            problems.append(f"pieza de más en {sq}: {color}{kind} (conf {conf:.2f})")
    for sq in sorted(set(expected) - set(seen)):
        warnings.append(f"no se ve la pieza esperada en {sq}")
    return not problems, problems, warnings


def capture(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(index)
    ok, frame = False, None
    if cap.isOpened():  # las primeras imágenes salen negras o sin exposición: descarta unas cuantas
        for _ in range(15):
            got, img = cap.read()
            if got:
                ok, frame = True, img
            time.sleep(0.03)
    cap.release()
    if not ok:
        raise SystemExit(f"No se pudo leer la cámara {index}. Revisa el permiso de cámara de la app y el índice (real/find_cameras.py).")
    return frame


def evaluate(split: str = "val", base_k: float = 0.15, imgsz: int = 640, conf: float = 0.25) -> dict:
    """Compara lo que lee el detector con las posiciones anotadas a mano (vision/dataset_camara/positions)."""
    ds = HERE / "dataset_camara"
    cvn = ChessVision(base_k=base_k, imgsz=imgsz, conf=conf)
    tp = fp = fn = wrong_type = 0
    exact = n = 0
    for img in sorted((ds / "images" / split).glob("*.jpg")):
        truth = json.loads((ds / "positions" / f"{img.stem}.json").read_text())["pieces"]
        seen = cvn.read_squares(cv2.imread(str(img)))
        pred = {sq: v[0] + v[1] for sq, v in seen.items()}
        for sq, p in truth.items():
            if sq in pred:
                tp += pred[sq] == p
                wrong_type += pred[sq] != p
            else:
                fn += 1
        fp += sum(1 for sq in pred if sq not in truth)
        exact += pred == truth
        n += 1
    occ = tp + wrong_type + fn
    return {"split": split, "fotos": n, "piezas": occ, "bien": tp, "tipo_o_color_mal": wrong_type, "no_vistas": fn,
            "de_mas": fp, "recall": round(tp / occ, 3), "precision": round(tp / max(1, tp + wrong_type + fp), 3),
            "fotos_exactas": exact}


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "evaluar":
        for sp in ("val", "train"):
            print(json.dumps(evaluate(sp), ensure_ascii=False))
        return
    if args and args[0] == "--camera":
        img = capture(int(args[1]))
    elif args:
        img = cv2.imread(args[0])
        if img is None:
            raise SystemExit(f"No se puede leer {args[0]}")
    else:
        print(__doc__)
        return
    cvn = ChessVision()
    seen = cvn.read_squares(img)
    for sq in sorted(seen, key=lambda s: (s[1], s[0])):
        c, k, p = seen[sq]
        print(f"{sq}: {c}{k} ({p:.2f})")
    print("FEN (colocación):", cvn.squares_to_fen(seen))


if __name__ == "__main__":
    main()
