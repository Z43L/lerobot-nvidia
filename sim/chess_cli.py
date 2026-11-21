"""Línea de comandos del ajedrez del SO-101. La usan las skills so101-ajedrez-mover y so101-ajedrez-partida.

Estado de la partida: hub/chess_game.json (FEN, historial y backend). Cada comando reconstruye la
escena desde ese FEN, así que el orden de las llamadas es lo único que importa.

  ./.venv312/bin/python sim/chess_cli.py new [--robot-white]     # nueva partida (el robot juega blancas si se indica)
  ./.venv312/bin/python sim/chess_cli.py state                  # tablero, turno y jugadas legales
  ./.venv312/bin/python sim/chess_cli.py human <jugada>         # jugada del humano (SAN o UCI), la ejecuta el robot en la escena
  ./.venv312/bin/python sim/chess_cli.py robot                  # el robot elige y ejecuta su jugada
  ./.venv312/bin/python sim/chess_cli.py move <uci>             # ejecuta una jugada concreta (ya legal) sin cambiar de turno
  ./.venv312/bin/python sim/chess_cli.py render [fichero.png]   # imagen de la escena (cámara 'table')

Ver la partida: SO101_VIEWER=1 ./.venv312/bin/mjpython sim/chess_cli.py ... (ventana de MuJoCo en vivo).

  ./.venv312/bin/python sim/chess_cli.py see --camera N|--image f.jpg   # qué ve YOLO y en qué difiere de la partida
  ./.venv312/bin/python sim/chess_cli.py human-seen --camera N          # deduce la jugada del humano con la cámara
  ./.venv312/bin/python sim/chess_cli.py commit <uci>                   # registra una jugada YA hecha, sin mover nada
  ./.venv312/bin/python sim/chess_cli.py robot --verify-camera N        # el robot juega y la cámara verifica

Backend: CHESS_BACKEND=sim (por defecto) o real. En real el ejecutor sale de CHESS_REAL_EXECUTOR (dry | ik | ik-scripted
| act, ver sim/chess_real.py). El humano mueve a mano: en real sus jugadas solo se registran, nunca las ejecuta el brazo.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GAME = Path(os.environ.get("CHESS_GAME_FILE", ROOT / "hub" / "chess_game.json"))
VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def load_game() -> dict:
    if not GAME.exists():
        raise SystemExit("No hay partida. Crea una con: chess_cli.py new")
    return json.loads(GAME.read_text())


def save_game(g: dict) -> None:
    GAME.parent.mkdir(exist_ok=True)
    GAME.write_text(json.dumps(g, indent=2, ensure_ascii=False))


def material(board: chess.Board, color: bool) -> int:
    return sum(VALUE[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


def evaluate(board: chess.Board, robot_color: bool) -> int:
    if board.is_checkmate():
        return -100000 if board.turn == robot_color else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return material(board, robot_color)


def engine_move(board: chess.Board, robot_color: bool, depth: int = 2, seed: int | None = None) -> chess.Move:
    """Minimax simple (material) con desempate aleatorio. No es un motor fuerte: es un rival de prueba."""
    rnd = random.Random(seed)

    def search(b: chess.Board, d: int, alpha: float, beta: float) -> float:
        if d == 0 or b.is_game_over():
            return evaluate(b, robot_color)
        if b.turn == robot_color:
            best = -1e9
            for m in b.legal_moves:
                b.push(m)
                best = max(best, search(b, d - 1, alpha, beta))
                b.pop()
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        best = 1e9
        for m in b.legal_moves:
            b.push(m)
            best = min(best, search(b, d - 1, alpha, beta))
            b.pop()
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    scored = []
    for m in board.legal_moves:
        board.push(m)
        scored.append((search(board, depth - 1, -1e9, 1e9), rnd.random(), m))
        board.pop()
    best_score = max(s for s, _, _ in scored)
    options = [m for s, _, m in scored if s == best_score]
    return rnd.choice(options)


def parse_move(board: chess.Board, text: str) -> chess.Move:
    text = text.strip()
    try:
        m = board.parse_san(text)
    except ValueError:
        try:
            m = chess.Move.from_uci(text)
        except ValueError as e:
            raise SystemExit(f"No entiendo la jugada '{text}'. Usa SAN (e4, Nf3) o UCI (e2e4).") from e
    if m not in board.legal_moves:
        raise SystemExit(f"Jugada ilegal en esta posición: {text}. Legales: {', '.join(board.san(x) for x in board.legal_moves)}")
    return m


def backend_name() -> str:
    name = os.environ.get("CHESS_BACKEND", "sim").lower()
    if name not in ("sim", "real"):
        raise SystemExit(f"Backend desconocido: {name}")
    return name


def run_on_backend(board: chess.Board, move: chess.Move, verify=None) -> dict:
    if backend_name() == "real":
        sys.path.insert(0, str(HERE))
        from chess_real import RealChessBackend

        return RealChessBackend(verify=verify).execute(move, board)
    sys.path.insert(0, str(HERE))
    os.environ.setdefault("SO101_GRASP", "attach")
    from chess_sim import ChessSim

    sim = ChessSim()
    if os.environ.get("SO101_VIEWER") == "1":  # ventana de MuJoCo; requiere mjpython en macOS
        sim.attach_viewer(realtime=True)
    return sim.execute(move, board)


def _vision_modules():
    sys.path.insert(0, str(ROOT / "vision"))
    sys.path.insert(0, str(ROOT / "hub"))
    import chess_config

    os.environ.setdefault("CHESS_BOARD_GEOMETRY", str(chess_config.BOARD_GEOMETRY))
    import chess_vision

    return chess_vision


def _read_seen(image=None, camera=None):
    """(chess_vision, imagen BGR, casillas vistas) desde una foto o una cámara OpenCV."""
    import cv2

    cvm = _vision_modules()
    if image:
        img = cv2.imread(str(image))
        if img is None:
            raise SystemExit(f"No puedo leer la imagen {image}")
    elif camera is not None:
        img = cvm.capture(int(camera))
    else:
        raise SystemExit("Indica --camera N o --image foto.jpg")
    vision = cvm.ChessVision()
    return cvm, img, vision.read_squares(img)


def vision_verifier(args):
    """Verificador de la jugada del robot con la cámara (None si no se indicó cámara ni imagen)."""
    image, camera = getattr(args, "verify_image", None), getattr(args, "verify_camera", None)
    if image is None and camera is None:
        return None

    def verify(move, board_before, board_after):
        cvm, _, seen = _read_seen(image, camera)
        ok, problems, warnings = cvm.verify_move(board_before, move, seen)
        for w in warnings:
            print("  aviso:", w)
        return ok, problems

    return verify


def cmd_new(args) -> None:
    board = chess.Board()
    g = {"fen": board.fen(), "history": [], "robot_color": "white" if args.robot_white else "black", "backend": backend_name()}
    save_game(g)
    print("Partida nueva. Turno:", "blancas" if board.turn else "negras", "| el robot juega", g["robot_color"])


def cmd_state(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    print(board.unicode(borders=True, empty_square="·"))
    print("FEN:", board.fen())
    print("Turno:", "blancas" if board.turn else "negras", "| el robot juega", g["robot_color"])
    if board.is_game_over():
        print("Partida terminada:", board.result(claim_draw=True))
    print("Jugadas legales:", ", ".join(board.san(m) for m in board.legal_moves))
    print("Historial:", " ".join(h["san"] for h in g["history"]) or "(vacío)")


def _apply(g: dict, board: chess.Board, move: chess.Move, who: str, verify=None, execute: bool = True) -> dict:
    san = board.san(move)
    if execute:
        res = run_on_backend(board, move, verify=verify)
    else:  # jugada ya hecha a mano (humano) o ya verificada: solo se registra
        after = board.copy()
        after.push(move)
        res = {"uci": move.uci(), "steps": [], "board_ok": True, "bad_squares": [], "fen": after.fen()}
    entry = {"ply": len(g["history"]) + 1, "who": who, "uci": move.uci(), "san": san, "board_ok": res["board_ok"],
             "steps": res["steps"], "bad_squares": res["bad_squares"]}
    if not res["board_ok"]:
        print(f"AVISO: el tablero no quedó bien: {res['bad_squares']}. La jugada no se registra.")
        return entry
    g["fen"] = res["fen"]
    g["history"].append(entry)
    save_game(g)
    print(f"Jugada {entry['ply']} ({who}): {san} [{move.uci()}] {', '.join(res['steps']) or ''}".rstrip())
    return entry


def cmd_human(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    if board.is_game_over():
        raise SystemExit("La partida ya ha terminado.")
    robot_turn = (board.turn == chess.WHITE) == (g["robot_color"] == "white")
    if robot_turn:
        raise SystemExit("Es turno del robot: usa 'robot'.")
    move = parse_move(board, args.move)
    entry = _apply(g, board, move, "humano", execute=backend_name() == "sim")
    if entry["board_ok"]:
        print("Siguiente: robot" if not chess.Board(g["fen"]).is_game_over() else "Partida terminada.")


def cmd_human_seen(args) -> None:
    """Deduce la jugada del humano con la cámara (o una foto) y la registra. No mueve el brazo."""
    g = load_game()
    board = chess.Board(g["fen"])
    if board.is_game_over():
        raise SystemExit("La partida ya ha terminado.")
    robot_turn = (board.turn == chess.WHITE) == (g["robot_color"] == "white")
    if robot_turn:
        raise SystemExit("Es turno del robot: usa 'robot'.")
    cvm, _, seen = _read_seen(getattr(args, "image", None), getattr(args, "camera", None))
    move, info = cvm.ChessVision.detect_move(cvm.ChessVision.__new__(cvm.ChessVision), board, seen,
                                             margin=getattr(args, "margin", 0.5))
    if move is None:
        raise SystemExit(f"No hay una jugada clara ({info}). Mira la imagen y usa 'human <jugada>'.")
    print(f"Veo la jugada {board.san(move)} [{move.uci()}] ({info})")
    _apply(g, board, move, "humano", execute=False)


def cmd_see(args) -> None:
    g = load_game() if GAME.exists() else None
    cvm, _, seen = _read_seen(args.image, args.camera)
    print("Piezas vistas:", len(seen), "| FEN visto:", cvm.ChessVision.squares_to_fen(seen))
    if g:
        exp = cvm.ChessVision.board_to_squares(chess.Board(g["fen"]))
        diffs = [s for s in sorted(set(exp) | set(seen)) if exp.get(s) != (seen[s][:2] if s in seen else None)]
        print("Casillas distintas de la partida:", diffs or "ninguna")


def cmd_commit(args) -> None:
    """Registra una jugada que ya se ejecutó y verificó (el 'registrar' del diagrama). No mueve nada."""
    g = load_game()
    board = chess.Board(g["fen"])
    _apply(g, board, parse_move(board, args.uci), args.who, execute=False)


def cmd_robot(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    if board.is_game_over():
        raise SystemExit("La partida ya ha terminado.")
    robot_color = chess.WHITE if g["robot_color"] == "white" else chess.BLACK
    if board.turn != robot_color:
        raise SystemExit("No es turno del robot.")
    move = engine_move(board, robot_color, seed=args.seed)
    entry = _apply(g, board, move, "robot", verify=vision_verifier(args))
    if not entry["board_ok"]:
        raise SystemExit(2)
    print("Siguiente: humano" if not chess.Board(g["fen"]).is_game_over() else "Partida terminada.")


def cmd_move(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    move = parse_move(board, args.uci)
    _apply(g, board, move, "manual")


def cmd_render(args) -> None:
    sys.path.insert(0, str(HERE))
    os.environ.setdefault("SO101_GRASP", "attach")
    from chess_sim import ChessSim

    g = load_game()
    sim = ChessSim()
    sim.load_board(chess.Board(g["fen"]))
    out = sim.render(camera=args.camera)
    dest = Path(args.file) if args.file else out
    if args.file:
        Path(out).replace(dest) if Path(out) != dest else None
    print("imagen:", dest)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("--robot-white", action="store_true"); p.set_defaults(f=cmd_new)
    sub.add_parser("state").set_defaults(f=cmd_state)
    p = sub.add_parser("human"); p.add_argument("move"); p.set_defaults(f=cmd_human)
    p = sub.add_parser("robot"); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verify-camera", type=int, default=None); p.add_argument("--verify-image", default=None); p.set_defaults(f=cmd_robot)
    p = sub.add_parser("human-seen"); p.add_argument("--camera", type=int, default=None); p.add_argument("--image", default=None)
    p.add_argument("--margin", type=float, default=0.5); p.set_defaults(f=cmd_human_seen)
    p = sub.add_parser("see"); p.add_argument("--camera", type=int, default=None); p.add_argument("--image", default=None); p.set_defaults(f=cmd_see)
    p = sub.add_parser("commit"); p.add_argument("uci"); p.add_argument("--who", default="robot"); p.set_defaults(f=cmd_commit)
    p = sub.add_parser("move"); p.add_argument("uci"); p.set_defaults(f=cmd_move)
    p = sub.add_parser("render"); p.add_argument("file", nargs="?"); p.add_argument("--camera", default="table"); p.set_defaults(f=cmd_render)
    args = ap.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
