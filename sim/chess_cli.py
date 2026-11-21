"""Línea de comandos del ajedrez del SO-101. La usan las skills so101-ajedrez-mover y so101-ajedrez-partida.

Estado de la partida: hub/chess_game.json (FEN, historial y backend). Cada comando reconstruye la
escena desde ese FEN, así que el orden de las llamadas es lo único que importa.

  ./.venv312/bin/python sim/chess_cli.py new [--robot-white]     # nueva partida (el robot juega blancas si se indica)
  ./.venv312/bin/python sim/chess_cli.py state                  # tablero, turno y jugadas legales
  ./.venv312/bin/python sim/chess_cli.py human <jugada>         # jugada del humano (SAN o UCI), la ejecuta el robot en la escena
  ./.venv312/bin/python sim/chess_cli.py robot                  # el robot elige y ejecuta su jugada
  ./.venv312/bin/python sim/chess_cli.py move <uci>             # ejecuta una jugada concreta (ya legal) sin cambiar de turno
  ./.venv312/bin/python sim/chess_cli.py render [fichero.png]   # imagen de la escena (cámara 'table')

Con el brazo real (CHESS_BACKEND=real) la jugada del humano NO se ejecuta (ya está hecha en el tablero físico) y se lee con la cámara:
  ./.venv312/bin/python sim/chess_cli.py see [--image f.jpg | --camera N]          # lo que ve la cámara y la jugada que explica
  ./.venv312/bin/python sim/chess_cli.py human-seen [--image f.jpg | --camera N]   # registra la jugada del humano leída con la cámara
  ./.venv312/bin/python sim/chess_cli.py robot [--verify-image f.jpg | --verify-camera N]   # tras mover, comprueba el tablero con la cámara

Ver la partida: SO101_VIEWER=1 ./.venv312/bin/mjpython sim/chess_cli.py ... (ventana de MuJoCo en vivo).

Backend: CHESS_BACKEND=sim (por defecto). CHESS_BACKEND=real usa sim/chess_real.py con el ejecutor de CHESS_REAL_EXECUTOR
(dry por defecto: planifica y NO mueve el brazo; act: aún solo comprueba el checkpoint).
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
GAME = Path(os.environ.get("CHESS_GAME_FILE", HERE.parent / "hub" / "chess_game.json"))  # CHESS_GAME_FILE: para pruebas
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
    return os.environ.get("CHESS_BACKEND", "sim")


def vision_source(args):
    """Función que devuelve una imagen BGR si se pidió --image/--camera/--verify-*, o None."""
    image = getattr(args, "image", None) or getattr(args, "verify_image", None)
    camera = getattr(args, "camera", None) if getattr(args, "camera", None) is not None else getattr(args, "verify_camera", None)
    if image is None and camera is None:
        return None
    sys.path.insert(0, str(HERE.parent / "vision"))
    import chess_vision

    def read():
        if image is not None:
            import cv2
            img = cv2.imread(image)
            if img is None:
                raise SystemExit(f"No se puede leer la imagen {image}")
            return img
        return chess_vision.capture(int(camera))

    return read


def vision_verifier(args):
    """verify(board_esperado) -> casillas que no cuadran con lo que ve la cámara (None si no hay fuente de visión)."""
    read = vision_source(args)
    if read is None:
        return None
    import chess_vision

    cv = chess_vision.ChessVision()

    def verify(expected: chess.Board, move: chess.Move | None = None) -> list[str]:
        """Casillas que no cuadran tras la jugada. Falla solo por lo que importa:
        origen/destino/capturas de la jugada, y piezas de más (confianza alta). Las piezas que no se ven (oclusiones) solo avisan."""
        seen = cv.read_squares(read())
        exp = cv.board_to_squares(expected)
        touched = set()
        if move is not None:
            before = expected.copy()
            before.pop()
            touched = {chess.square_name(move.from_square), chess.square_name(move.to_square)}
            if before.is_en_passant(move):
                touched.add(chess.square_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))))
            if before.is_castling(move):
                rank = chess.square_rank(move.from_square)
                k = chess.square_file(move.to_square) == 6
                touched |= {chess.square_name(chess.square(7 if k else 0, rank)), chess.square_name(chess.square(5 if k else 3, rank))}
        bad, warn = [], []
        for sq in sorted(set(exp) | set(seen)):
            e, s_ = exp.get(sq), seen.get(sq)
            same = (e is None) == (s_ is None) and (not e or (e[0], e[1]) == (s_[0], s_[1]))
            if same:
                continue
            extra_confident = e is None and s_ is not None and s_[2] >= 0.5
            wrong_piece = e is not None and s_ is not None
            if sq in touched or extra_confident or wrong_piece and s_[2] >= 0.8:
                bad.append(sq)
            else:
                warn.append(sq)
        if warn:
            print(f"  (aviso: no se ve o dudoso en {', '.join(warn)}; no bloquea)")
        return bad

    return verify


def run_on_backend(board: chess.Board, move: chess.Move, verify=None) -> dict:
    backend = backend_name()
    if backend == "real":
        sys.path.insert(0, str(HERE))
        from chess_real import RealChessBackend

        return RealChessBackend(verify=verify).execute(move, board)
    if backend != "sim":
        raise SystemExit(f"Backend desconocido: {backend}")
    sys.path.insert(0, str(HERE))
    os.environ.setdefault("SO101_GRASP", "attach")
    from chess_sim import ChessSim

    sim = ChessSim()
    if os.environ.get("SO101_VIEWER") == "1":  # ventana de MuJoCo; requiere mjpython en macOS
        sim.attach_viewer(realtime=True)
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


def _apply(g: dict, board: chess.Board, move: chess.Move, who: str, execute: bool = True, verify=None) -> dict:
    san = board.san(move)
    if execute:
        res = run_on_backend(board, move, verify)
    else:  # la jugada ya está hecha en el tablero físico: solo se registra
        after = board.copy()
        after.push(move)
        res = {"steps": ["observada con la cámara/indicada por el humano (el robot no la ejecuta)"], "board_ok": True,
               "bad_squares": [], "fen": after.fen()}
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
    entry = _apply(g, board, move, "humano", execute=backend_name() != "real")
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
    _apply(g, board, move, "robot", verify=vision_verifier(args))
    print("Siguiente: humano" if not chess.Board(g["fen"]).is_game_over() else "Partida terminada.")


def cmd_move(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    move = parse_move(board, args.uci)
    _apply(g, board, move, "manual", verify=vision_verifier(args))


def _read_board_from_vision(args):
    read = vision_source(args)
    if read is None:
        raise SystemExit("Indica la fuente: --image foto.jpg o --camera N (índice OpenCV).")
    sys.path.insert(0, str(HERE.parent / "vision"))
    import chess_vision

    cv = chess_vision.ChessVision()
    return cv, cv.read_squares(read())


def _print_seen(cv, board: chess.Board, seen: dict) -> list[str]:
    exp = cv.board_to_squares(board)
    diffs = []
    for sq in sorted(set(exp) | set(seen)):
        e, s_ = exp.get(sq), seen.get(sq)
        if (e is None) != (s_ is None) or (e and s_ and (e[0], e[1]) != (s_[0], s_[1])):
            diffs.append(f"{sq}: esperado {''.join(e) if e else '-'} | visto {''.join(s_[:2]) + f' ({s_[2]:.2f})' if s_ else '-'}")
    print(f"Piezas vistas: {len(seen)} | esperadas: {len(exp)} | casillas distintas: {len(diffs)}")
    for d in diffs:
        print("  ", d)
    return diffs


def cmd_see(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    cv, seen = _read_board_from_vision(args)
    print("FEN visto (colocación):", cv.squares_to_fen(seen))
    print("FEN esperado           :", board.board_fen())
    _print_seen(cv, board, seen)
    move, info = cv.detect_move(board, seen)
    if move:
        print(f"La jugada que mejor explica lo visto: {board.san(move)} [{move.uci()}] (puntos {info['score']} frente a {info['second_score']} de {info['second']})")
    else:
        print(f"Ninguna jugada destaca (mejor: {info['best']} {info['score']}; segunda: {info['second']} {info['second_score']}).")


def cmd_human_seen(args) -> None:
    g = load_game()
    board = chess.Board(g["fen"])
    if board.is_game_over():
        raise SystemExit("La partida ya ha terminado.")
    robot_turn = (board.turn == chess.WHITE) == (g["robot_color"] == "white")
    if robot_turn:
        raise SystemExit("Es turno del robot: usa 'robot'.")
    cv, seen = _read_board_from_vision(args)
    move, info = cv.detect_move(board, seen, margin=args.margin)
    if move is None:
        _print_seen(cv, board, seen)
        raise SystemExit(f"No hay una jugada clara (mejor: {info['best']} {info['score']}; segunda: {info['second']} {info['second_score']}). "
                         "Mira la foto y, si hace falta, usa: chess_cli.py human <jugada>")
    print(f"Jugada vista: {board.san(move)} [{move.uci()}] (puntos {info['score']} frente a {info['second_score']} de {info['second']})")
    entry = _apply(g, board, move, "humano", execute=False)
    if entry["board_ok"]:
        print("Siguiente: robot")


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
    def add_vision(p, verify=False):
        pre = "--verify-" if verify else "--"
        p.add_argument(pre + "image", dest="verify_image" if verify else "image", default=None, help="foto del tablero")
        p.add_argument(pre + "camera", dest="verify_camera" if verify else "camera", type=int, default=None, help="índice de cámara OpenCV")

    p = sub.add_parser("robot"); p.add_argument("--seed", type=int, default=None); add_vision(p, verify=True); p.set_defaults(f=cmd_robot)
    p = sub.add_parser("move"); p.add_argument("uci"); add_vision(p, verify=True); p.set_defaults(f=cmd_move)
    p = sub.add_parser("see"); add_vision(p); p.set_defaults(f=cmd_see)
    p = sub.add_parser("human-seen"); add_vision(p); p.add_argument("--margin", type=float, default=0.5); p.set_defaults(f=cmd_human_seen)
    p = sub.add_parser("render"); p.add_argument("file", nargs="?"); p.add_argument("--camera", default="table"); p.set_defaults(f=cmd_render)
    args = ap.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
