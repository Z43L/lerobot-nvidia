"""Backend REAL del ajedrez del SO-101: planifica una jugada en pasos coger-y-soltar y los pasa a un ejecutor.

Encaja en el diagrama de secuencia así:

    MCP / CLI  --execute(jugada)-->  RealChessBackend
        plan():      captura primero al cementerio, enroque en dos pasos, promoción con ayuda humana
        Geometry:    "mapa: encima" -> centro de cada casilla en metros (hub/chess_board_real.json)
        Ejecutor:    "coge la pieza" / "a f7, soltar" / "pose final"  (dry | ik | ik-scripted | act)
        verify():    "hecho + fotos" -> la cámara comprueba origen, destino y capturas

Ejecutores (CHESS_REAL_EXECUTOR, por defecto dry):
  dry          solo planifica e imprime; no mueve nada.
  ik           IK lleva la pinza encima de la pieza, el ACT genérico baja y agarra, la IK suelta y vuelve al origen
               (hub/chess_ik_exec.py, el "camino 2"). Mueve el brazo.
  ik-scripted  igual, pero el agarre es guionizado (baja, cierra hasta que se detiene, sube). Plan B. Mueve el brazo.
  act          ACT con objetivo (src,dst). Hoy solo hace preflight y se niega a mover.

Un ejecutor implementa:  pick_place(label, piece_type, src_xy, dst_xy) -> dict   (y tiene .name)
Seguridad (AGENTS.md): los ejecutores que mueven exigen ALLOW_REAL_MOTION=1 y el permiso real/.motion_ok; eso se
comprueba dentro del ejecutor, no aquí. Este módulo nunca habla con el brazo por su cuenta.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

import chess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HUB = ROOT / "hub"
BOARD_SIM = HUB / "chess_board.json"
BOARD_REAL = HUB / "chess_board_real.json"
PLAN_LOG = Path(os.environ.get("CHESS_REAL_PLAN_LOG", HUB / "chess_real_plan.json"))

PIECE_ES = {chess.PAWN: "peón", chess.KNIGHT: "caballo", chess.BISHOP: "alfil", chess.ROOK: "torre",
            chess.QUEEN: "dama", chess.KING: "rey"}
GRAVE_SLOTS = 8  # huecos por lado del cementerio

# verify(move, board_before, board_after) -> (ok, problems)
Verifier = Callable[[chess.Move, chess.Board, chess.Board], "tuple[bool, list[str]]"]


# --------------------------------------------------------------------------- geometría ("mapa del tablero")
class Geometry:
    """Centros de casilla y cementerio en metros, en el marco de la base del brazo.

    Por defecto usa la calibración real si existe y, si no, la de la simulación. `require_real=True` (lo que usa un
    ejecutor que mueve el brazo) se niega si no hay calibración real: sin ella el objetivo x,y no es la casilla.
    """

    def __init__(self, path: Path | str | None = None, require_real: bool = False):
        if path is None:
            path = BOARD_REAL if BOARD_REAL.exists() else BOARD_SIM
        self.path = Path(path)
        data = json.loads(self.path.read_text()) if self.path.exists() else None
        self.real = bool(data) and data.get("status") == "real"
        if require_real and not self.real:
            raise RuntimeError("Falta la calibración del tablero real (hub/chess_board_real.json, skill "
                               "so101-calibracion-tablero): sin ella el ejecutor real se niega a mover.")
        if data is None:
            raise RuntimeError(f"No existe la geometría del tablero: {self.path}")
        self.squares: dict[str, list[float]] = data["squares"]
        self.graveyard: list[list[float]] = data["graveyard_m"]
        self.name = self.path.name

    def square(self, name: str) -> tuple[float, float]:
        x, y = self.squares[name]
        return float(x), float(y)

    def grave(self, index: int, color: bool) -> tuple[float, float]:
        """Hueco `index` del cementerio de las piezas de `color` (negras: lado h, blancas: lado a)."""
        if not 0 <= index < GRAVE_SLOTS:
            raise RuntimeError(f"El cementerio de ese color está lleno (hueco {index}).")
        base = 0 if color == chess.BLACK else GRAVE_SLOTS
        x, y = self.graveyard[base + index]
        return float(x), float(y)


# --------------------------------------------------------------------------- ejecutores
class DryRunExecutor:
    name = "dry"

    def __init__(self):
        self.log: list[dict] = []

    def pick_place(self, label, piece_type, src_xy, dst_xy) -> dict:
        entry = {"accion": label, "pieza": PIECE_ES[piece_type], "src_xy": list(src_xy), "dst_xy": list(dst_xy),
                 "ejecutado": False}
        self.log.append(entry)
        print(f"  [dry] {label} ({entry['pieza']}): {tuple(round(v, 3) for v in src_xy)} -> "
              f"{tuple(round(v, 3) for v in dst_xy)}  (no se mueve nada)")
        return entry


class ACTExecutor:
    """ACT con objetivo (4 valores: origen y destino). Hoy NO mueve: solo comprueba si hay un checkpoint utilizable."""

    name = "act"

    def __init__(self, checkpoint: Path | str | None = None):
        sys.path.insert(0, str(HUB))
        import chess_config

        self.checkpoint = Path(checkpoint) if checkpoint else (chess_config.ACT_GOAL_CKPT or ROOT / "checkpoints" / "act_tray_test200")

    def _config_dir(self) -> Path | None:
        for cand in (self.checkpoint, self.checkpoint / "pretrained_model"):
            if (cand / "config.json").exists():
                return cand
        return None

    def preflight(self) -> list[str]:
        """Lista de problemas que impiden usarlo (vacía = utilizable). No mueve nada."""
        cfg_dir = self._config_dir()
        if cfg_dir is None:
            return [f"no hay config.json en {self.checkpoint}: no hay checkpoint"]
        problems = []
        weights = cfg_dir / "model.safetensors"
        if not weights.exists():
            problems.append("falta model.safetensors")
        elif weights.stat().st_size < 1024 and weights.read_bytes().startswith(b"version https://git-lfs"):
            problems.append("model.safetensors es un puntero de Git LFS (git lfs pull)")
        feats = json.loads((cfg_dir / "config.json").read_text()).get("input_features", {})
        env = feats.get("observation.environment_state")
        if env is None:
            problems.append("el checkpoint no tiene observation.environment_state (no recibe objetivo)")
        elif env.get("shape", [0])[0] != 4:
            problems.append("observation.environment_state no tiene 4 valores (origen y destino)")
        return problems

    def pick_place(self, label, piece_type, src_xy, dst_xy) -> dict:
        problems = self.preflight()
        raise RuntimeError("El ejecutor ACT todavía no mueve el brazo. " + ("; ".join(problems) if problems else
                           "Falta la prueba corta en el brazo (real/deploy_goal.py)."))


def make_executor(name: str | None = None):
    """Ejecutor por nombre (CHESS_REAL_EXECUTOR). Los que mueven el brazo se importan solo cuando se piden."""
    name = (name or os.environ.get("CHESS_REAL_EXECUTOR", "dry")).lower()
    if name == "dry":
        return DryRunExecutor()
    if name == "act":
        return ACTExecutor()
    if name in ("ik", "ik-scripted"):
        sys.path.insert(0, str(HUB))
        from chess_ik_exec import IKExecutor  # noqa: E402

        return IKExecutor(grasp="scripted" if name == "ik-scripted" else "act")
    raise ValueError(f"ejecutor desconocido: {name} (dry | ik | ik-scripted | act)")


def moves_arm(executor) -> bool:
    return getattr(executor, "name", "dry") != "dry"


# --------------------------------------------------------------------------- backend
class RealChessBackend:
    """Misma interfaz que ChessSim.execute: devuelve {uci, steps, board_ok, bad_squares, fen, ...}."""

    def __init__(self, executor=None, geometry: Geometry | None = None, verify: Verifier | None = None):
        self.executor = executor or make_executor()
        # Un ejecutor que mueve el brazo exige calibración real; el dry puede usar la de la sim.
        self.geometry = geometry or Geometry(require_real=moves_arm(self.executor))
        self.verify = verify

    # ------------------------------------------------------------------ plan
    def plan(self, move: chess.Move, board: chess.Board) -> list[dict]:
        """Pasos coger-y-soltar de la jugada. No mueve nada. ValueError si la jugada es ilegal."""
        if move not in board.legal_moves:
            raise ValueError(f"jugada ilegal: {move.uci()}")
        g = self.geometry
        name = chess.square_name
        steps: list[dict] = []

        def step(label, piece, src_sq, dst_xy, note=None):
            steps.append({"label": label, "piece": piece, "src_xy": g.square(name(src_sq)), "dst_xy": dst_xy,
                          "note": note})

        victim_sq = None
        if board.is_en_passant(move):
            victim_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            label = f"captura al paso en {name(victim_sq)}"
        elif board.is_capture(move):
            victim_sq = move.to_square
            label = f"captura en {name(victim_sq)}"
        if victim_sq is not None:
            victim = board.piece_at(victim_sq)
            taken = 16 - chess.popcount(board.occupied_co[victim.color])  # capturadas de ese color hasta ahora
            step(label, victim.piece_type, victim_sq, g.grave(taken, victim.color))

        mover = board.piece_at(move.from_square)
        if board.is_castling(move):
            rank = chess.square_rank(move.from_square)
            short = chess.square_file(move.to_square) == 6
            rook_from = chess.square(7 if short else 0, rank)
            rook_to = chess.square(5 if short else 3, rank)
            step(f"rey {name(move.from_square)}->{name(move.to_square)}", chess.KING, move.from_square,
                 g.square(name(move.to_square)))
            step(f"torre {name(rook_from)}->{name(rook_to)}", chess.ROOK, rook_from, g.square(name(rook_to)))
        else:
            note = None
            if move.promotion:
                note = (f"promoción a {chess.piece_name(move.promotion)}: el robot lleva el peón a la casilla; "
                        "la pieza nueva la pone una persona")
            step(f"{name(move.from_square)}->{name(move.to_square)}", mover.piece_type, move.from_square,
                 g.square(name(move.to_square)), note)
        return steps

    # ------------------------------------------------------------------ ejecución
    def execute(self, move: chess.Move, board_before: chess.Board) -> dict:
        steps = self.plan(move, board_before)
        results = [self.executor.pick_place(s["label"], s["piece"], s["src_xy"], s["dst_xy"]) for s in steps]
        after = board_before.copy()
        after.push(move)
        bad: list[str] = []
        verified = False
        if self.verify is not None and moves_arm(self.executor):
            ok, bad = self.verify(move, board_before, after)
            verified = True
        self._write_log(move, steps, results)
        return {"uci": move.uci(), "steps": [s["label"] for s in steps], "board_ok": not bad, "bad_squares": bad,
                "fen": after.fen(), "executor": self.executor.name, "verified": verified, "results": results}

    def _write_log(self, move: chess.Move, steps: list[dict], results: list[dict]) -> None:
        try:
            PLAN_LOG.write_text(json.dumps({
                "uci": move.uci(), "executor": self.executor.name, "geometria": self.geometry.name,
                "geometria_real": self.geometry.real,
                "plan": [{"label": s["label"], "pieza": PIECE_ES[s["piece"]], "src_xy": list(s["src_xy"]),
                          "dst_xy": list(s["dst_xy"])} for s in steps]}, indent=2, ensure_ascii=False))
        except OSError:
            pass
