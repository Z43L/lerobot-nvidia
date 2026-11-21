"""Pruebas del ajedrez real SIN brazo: planificación, ejecutor dry, ACT preflight y lectura de jugadas con visión.

  ./.venv312/bin/python sim/test_chess_real.py          (o con pytest si está instalado)
"""
import os
import sys
import tempfile
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "vision"))
os.environ["CHESS_REAL_PLAN_LOG"] = str(Path(tempfile.gettempdir()) / "chess_real_plan_test.json")

from chess_real import ACTExecutor, DryRunExecutor, Geometry, RealChessBackend  # noqa: E402
from chess_vision import ChessVision  # noqa: E402


def backend():
    ex = DryRunExecutor()
    return RealChessBackend(executor=ex, geometry=Geometry(ROOT / "hub" / "chess_board.json")), ex


def test_plan_normal_and_capture():
    b, ex = backend()
    plan = b.plan(chess.Move.from_uci("e2e4"), chess.Board())
    assert len(plan) == 1 and plan[0]["label"] == "e2->e4"
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    plan = b.plan(chess.Move.from_uci("e4d5"), board)
    assert [s["label"] for s in plan] == ["captura en d5", "e4->d5"]  # primero se retira la víctima
    assert plan[0]["dst_xy"] == b.geometry.grave(0, chess.BLACK)  # primera negra capturada -> primer hueco del lado h


def test_plan_castling_en_passant_promotion():
    b, _ = backend()
    plan = b.plan(chess.Move.from_uci("e1g1"), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
    assert len(plan) == 2 and "rey" in plan[0]["label"] and "torre" in plan[1]["label"]
    plan = b.plan(chess.Move.from_uci("e5d6"), chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"))
    assert plan[0]["label"].startswith("captura al paso en d5")
    plan = b.plan(chess.Move.from_uci("a7a8q"), chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"))
    assert "promoción" in plan[-1]["note"]


def test_execute_dry_does_not_move_and_reports():
    b, ex = backend()
    res = b.execute(chess.Move.from_uci("g1f3"), chess.Board())
    assert res["board_ok"] and res["executor"] == "dry" and res["verified"] is False
    assert ex.log and ex.log[0]["ejecutado"] is False


def test_illegal_move_rejected():
    b, _ = backend()
    try:
        b.plan(chess.Move.from_uci("e2e5"), chess.Board())
    except ValueError:
        return
    raise AssertionError("debía rechazar la jugada ilegal")


def test_act_checkpoint_not_usable_yet():
    problems = ACTExecutor().preflight()
    if (ROOT / "checkpoints/act_tray_test200").exists():
        assert any("Git LFS" in p or "environment_state" in p for p in problems), problems
    try:
        ACTExecutor().pick_place("x", chess.PAWN, (0, 0), (0, 0))
    except RuntimeError:
        return
    raise AssertionError("el ejecutor ACT no debe mover nada mientras el checkpoint no sea compatible")


def test_real_executor_needs_calibrated_geometry():
    if (ROOT / "hub" / "chess_board_real.json").exists():
        return
    try:
        Geometry(require_real=True)
    except RuntimeError as e:
        assert "calibración" in str(e)
        return
    raise AssertionError("sin hub/chess_board_real.json el ejecutor real debe negarse")


def test_detect_move_with_noise():
    cv = ChessVision.__new__(ChessVision)  # sin cargar el modelo
    board = chess.Board()
    after = board.copy()
    after.push_san("e4")
    seen = {sq: (c, k, 0.9) for sq, (c, k) in cv.board_to_squares(after).items()}
    assert cv.detect_move(board, seen)[0].uci() == "e2e4"
    seen.pop("a1"); seen.pop("h8")  # piezas ocultas: no debe cambiar la conclusión
    assert cv.detect_move(board, seen)[0].uci() == "e2e4"
    assert cv.detect_move(board, {sq: (c, k, 0.9) for sq, (c, k) in cv.board_to_squares(board).items()})[0] is None  # nada cambió


def test_vision_on_real_photo():
    img = next(iter(sorted((ROOT / "vision/dataset_camara/images/train").glob("*03-19-23*.jpg"))), None)
    if img is None:
        return
    import cv2
    cv = ChessVision()
    seen = cv.read_squares(cv2.imread(str(img)))
    move, _ = cv.detect_move(chess.Board(), seen)
    assert move is not None and move.uci() == "d2d4", move  # foto 3 de Camera/: 1.d4


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK ", t.__name__)
    print(f"{len(tests)} pruebas OK")
