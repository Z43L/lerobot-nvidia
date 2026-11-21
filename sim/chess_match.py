"""Partida completa de ajedrez humano vs SO-101: cámara + YOLO + reglas + brazo. Bucle interactivo.

  # Ensayo sin brazo (planifica y no mueve nada), leyendo la cámara 1:
  CHESS_BACKEND=real CHESS_REAL_EXECUTOR=dry ./.venv312/bin/python sim/chess_match.py --camera 1

  # Con fotos en lugar de cámara (pruebas): cada lectura consume la siguiente foto de la lista
  CHESS_BACKEND=real ./.venv312/bin/python sim/chess_match.py --images f3.jpg,f4.jpg,f5.jpg --robot-moves d7d5,g8f6 --auto

Cada vuelta:
  turno del humano  -> espera a que muevas, lee el tablero con la cámara (YOLO) y deduce tu jugada. Si no está claro,
                       te pide la jugada por texto.
  turno del robot   -> elige jugada (o usa --robot-moves), la planifica y la ejecuta con el ejecutor, y comprueba el
                       tablero con la cámara. Si no cuadra, la jugada NO se registra y la partida se detiene.

Seguridad: con un ejecutor que mueva el brazo (cualquiera distinto de dry) exige ALLOW_REAL_MOTION=1 y que escribas SI
antes de CADA jugada del robot. Hoy el único ejecutor disponible es dry (ver sim/chess_real.py).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import chess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chess_cli as C  # noqa: E402


def moving_executor() -> bool:
    return C.backend_name() == "real" and os.environ.get("CHESS_REAL_EXECUTOR", "dry").lower() != "dry"


def gate_real_motion(san: str) -> bool:
    if not moving_executor():
        return True
    if os.environ.get("ALLOW_REAL_MOTION") != "1":
        print("Bloqueado: mover el brazo real necesita ALLOW_REAL_MOTION=1.")
        return False
    print(f"\n*** El robot va a MOVER el brazo real: {san} ***")
    print("Comprueba: área libre, persona junto al brazo, corte de alimentación a mano.")
    return input("Escribe SI para continuar: ").strip() == "SI"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None, help="índice OpenCV de la cámara que ve el tablero")
    ap.add_argument("--images", default=None, help="lista de fotos separadas por comas (pruebas); sustituye a la cámara")
    ap.add_argument("--robot-white", action="store_true")
    ap.add_argument("--new", action="store_true", help="empieza partida nueva (por defecto continúa hub/chess_game.json)")
    ap.add_argument("--robot-moves", default=None, help="jugadas UCI del robot en orden (pruebas); si se acaban, usa el motor")
    ap.add_argument("--auto", action="store_true", help="sin esperar ENTER (pruebas con fotos)")
    ap.add_argument("--max-plies", type=int, default=200)
    args = ap.parse_args()
    if args.camera is None and not args.images:
        ap.error("indica --camera N o --images a.jpg,b.jpg")

    images = [s for s in args.images.split(",")] if args.images else None
    scripted = [m for m in (args.robot_moves or "").split(",") if m]

    def source():
        if images is not None:
            if not images:
                raise SystemExit("Se acabaron las fotos de prueba.")
            return SimpleNamespace(image=images.pop(0), camera=None, verify_image=None, verify_camera=None)
        return SimpleNamespace(image=None, camera=args.camera, verify_image=None, verify_camera=None)

    def verify_source():
        s = source()
        return SimpleNamespace(verify_image=s.image, verify_camera=s.camera, image=None, camera=None)

    if args.new or not C.GAME.exists():
        C.cmd_new(SimpleNamespace(robot_white=args.robot_white))
    print(f"Backend: {C.backend_name()} | ejecutor: {os.environ.get('CHESS_REAL_EXECUTOR', 'dry')}")

    for _ in range(args.max_plies):
        g = C.load_game()
        board = chess.Board(g["fen"])
        print("\n" + board.unicode(borders=True, empty_square="·"))
        if board.is_game_over():
            print("Partida terminada:", board.result(claim_draw=True))
            return 0
        robot_color = chess.WHITE if g["robot_color"] == "white" else chess.BLACK
        if board.turn != robot_color:  # ---------------- humano
            if not args.auto:
                text = input("Mueve tu pieza y pulsa ENTER (o escribe tu jugada, 'q' para salir): ").strip()
                if text.lower() == "q":
                    return 0
                if text:
                    C.cmd_human(SimpleNamespace(move=text))
                    continue
            try:
                C.cmd_human_seen(SimpleNamespace(margin=0.5, **vars(source())))
            except SystemExit as e:
                print(e)
                text = input("No pude leer tu jugada. Escríbela (SAN o UCI) o 'q': ").strip() if not args.auto else "q"
                if text.lower() == "q":
                    return 1
                C.cmd_human(SimpleNamespace(move=text))
        else:  # ---------------------------------------- robot
            if scripted:
                move = C.parse_move(board, scripted.pop(0))
            else:
                move = C.engine_move(board, robot_color)
            san = board.san(move)
            if not gate_real_motion(san):
                print("Cancelado: la jugada del robot no se ejecuta.")
                return 3
            vs = verify_source()
            entry = C._apply(g, board, move, "robot", verify=C.vision_verifier(vs))
            if not entry["board_ok"]:
                print("Partida detenida: el tablero no coincide tras la jugada del robot. Revisa las piezas y reanuda.")
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
