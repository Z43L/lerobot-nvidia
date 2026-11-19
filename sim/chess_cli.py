"""Línea de comandos del ajedrez del SO-101. La usan las skills so101-ajedrez-mover y so101-ajedrez-partida.

Estado de la partida: hub/chess_game.json (FEN, historial y backend). Cada comando reconstruye la
escena desde ese FEN, así que el orden de las llamadas es lo único que importa.

  ./.venv312/bin/python sim/chess_cli.py new [--robot-white]     # nueva partida (el robot juega blancas si se indica)
  ./.venv312/bin/python sim/chess_cli.py state                  # tablero, turno y jugadas legales
  ./.venv312/bin/python sim/chess_cli.py human <jugada>         # jugada del humano (SAN o UCI), la ejecuta el robot en la escena
  ./.venv312/bin/python sim/chess_cli.py robot                  # el robot elige y ejecuta su jugada
  ./.venv312/bin/python sim/chess_cli.py move <uci>             # ejecuta una jugada concreta (ya legal) sin cambiar de turno
  ./.venv312/bin/python sim/chess_cli.py render [fichero.png]   # imagen de la escena (cámara 'table')

Backend: CHESS_BACKEND=sim (por defecto). CHESS_BACKEND=real queda reservado para el brazo real y todavía no existe
(ver sim/chess_real.py y la skill so101-ajedrez-mover).
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
GAME = HERE.parent / "hub" / "chess_game.json"
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


def run_on_backend(board: chess.Board, move: chess.Move) -> dict:
    backend = os.environ.get("CHESS_BACKEND", "sim")
    if backend == "real":
        raise SystemExit("CHESS_BACKEND=real todavía no está implementado: ver sim/chess_real.py y la skill so101-ajedrez-mover.")
    if backend != "sim":
        raise SystemExit(f"Backend desconocido: {backend}")
    sys.path.insert(0, str(HERE))
    os.environ.setdefault("SO101_GRASP", "attach")
    from chess_sim import ChessSim

    sim = ChessSim()
    return sim.execute(move, board)


def cmd_new(args) -> None:
    board = chess.Board()
    g = {"fen": board.fen(), "history": [], "robot_color": "white" if args.robot_white else "black", "backend": "sim"}
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


def _apply(g: dict, board: chess.Board, move: chess.Move, who: str) -> dict:
    san = board.san(move)
    res = run_on_backend(board, move)
    entry = {"ply": len(g["history"]) + 1, "who": who, "uci": move.uci(), "san": san, "board_ok": res["board_ok"],
             "steps": res["steps"], "bad_squares": res["bad_squares"]}
    if not res["board_ok"]:
        print(f"AVISO: la escena no quedó bien en {res['bad_squares']}. La jugada no se registra.")
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
    entry = _apply(g, board, move, "humano")
    if entry["board_ok"]:
        print("Siguiente: robot" if not chess.Board(g["fen"]).is_game_over() else "Partida terminada.")


def cmd_robot(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    if board.is_game_over():
        raise SystemExit("La partida ya ha terminado.")
    robot_color = chess.WHITE if g["robot_color"] == "white" else chess.BLACK
    if board.turn != robot_color:
        raise SystemExit("No es turno del robot.")
    move = engine_move(board, robot_color, seed=args.seed)
    _apply(g, board, move, "robot")
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
    p = sub.add_parser("robot"); p.add_argument("--seed", type=int, default=None); p.set_defaults(f=cmd_robot)
    p = sub.add_parser("move"); p.add_argument("uci"); p.set_defaults(f=cmd_move)
    p = sub.add_parser("render"); p.add_argument("file", nargs="?"); p.add_argument("--camera", default="table"); p.set_defaults(f=cmd_render)
    args = ap.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
