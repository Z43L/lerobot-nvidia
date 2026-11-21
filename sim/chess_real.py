"""Backend REAL del ajedrez del SO-101: planifica cada jugada como una lista de coger-y-soltar y se la pasa a un ejecutor.

Misma interfaz que ChessSim.execute (la usa sim/chess_cli.py con CHESS_BACKEND=real):
    RealChessBackend(executor).execute(move, board_before) -> {"uci", "steps", "board_ok", "bad_squares", "fen", ...}

Ejecutores (CHESS_REAL_EXECUTOR):
  dry (por defecto)  Solo imprime y guarda el plan (origen -> destino en metros). No toca el brazo. Sirve para probar todo
                     el circuito (visión -> reglas -> plan) sin hardware.
  act                Pasaría cada coger-y-soltar al checkpoint ACT. HOY SOLO COMPRUEBA (`preflight`) y no mueve nada: ver
                     ACTExecutor más abajo para lo que falta.

Geometría: hub/chess_board_real.json (calibración por contacto, skill so101-calibracion-tablero) si existe. Si no, el plan
se hace con hub/chess_board.json (geometría de la SIMULACIÓN, no calibrada) y se avisa. El ejecutor `act` exige la real.

Verificación: si se pasa una función `verify(board_despues, jugada) -> list[str]` (casillas que no cuadran), `board_ok` sale de ella.
Sin ella, `board_ok` es True y `verified` es False (no se ha comprobado con la cámara).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Protocol

import chess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BOARD_REAL = ROOT / "hub" / "chess_board_real.json"
BOARD_SIM = ROOT / "hub" / "chess_board.json"
PLAN_LOG = Path(os.environ.get("CHESS_REAL_PLAN_LOG", ROOT / "hub" / "chess_real_plan.json"))
# ACT con pesos reales (2000 pasos, "pick and place", cámaras front + gripper). checkpoints/act_tray_test200 son punteros de Git LFS.
DEFAULT_CHECKPOINT = ROOT / "modeloACT" / "checkpoints" / "last" / "pretrained_model"
PIECE_NAME = {chess.PAWN: "peón", chess.KNIGHT: "caballo", chess.BISHOP: "alfil", chess.ROOK: "torre",
              chess.QUEEN: "dama", chess.KING: "rey"}


class Geometry:
    """Casillas y cementerio en metros (marco de la base del brazo)."""

    def __init__(self, path: Path | None = None, require_real: bool = False):
        if path is None:
            path = BOARD_REAL if BOARD_REAL.exists() else BOARD_SIM
        if require_real and path != BOARD_REAL:
            raise RuntimeError(
                "Falta hub/chess_board_real.json (calibración del tablero real por contacto). "
                "Usa la skill so101-calibracion-tablero (hub/board_touch_calib.py) antes de mover el brazo."
            )
        data = json.loads(Path(path).read_text())
        self.path = Path(path)
        self.is_real = self.path == BOARD_REAL
        self.status = data.get("status", "")
        self.squares = {k: (float(v[0]), float(v[1])) for k, v in data["squares"].items()}
        self.graveyard = [(float(x), float(y)) for x, y in data.get("graveyard_m", [])]  # solo informativo
        self._build_side_graveyards()

    def _build_side_graveyards(self) -> None:
        """Cementerio = una columna extra de 8 huecos a cada lado del tablero, a la altura de cada fila.

        Lado de la columna a (derecha del robot): piezas BLANCAS capturadas.
        Lado de la columna h (izquierda del robot): piezas NEGRAS capturadas.
        Se deduce de las casillas, así sigue al tablero calibrado (inclinación incluida).
        """
        sq = self.squares
        file_step = [(sq["h1"][i] - sq["a1"][i]) / 7 for i in (0, 1)]  # una casilla hacia la columna h
        self.side_graveyard = {
            chess.WHITE: [(sq[f"a{r}"][0] - file_step[0], sq[f"a{r}"][1] - file_step[1]) for r in range(1, 9)],
            chess.BLACK: [(sq[f"h{r}"][0] + file_step[0], sq[f"h{r}"][1] + file_step[1]) for r in range(1, 9)],
        }

    def xy(self, sq: int) -> tuple[float, float]:
        return self.squares[chess.square_name(sq)]

    def grave(self, index: int, color: bool = chess.WHITE) -> tuple[float, float]:
        """Hueco `index` (0..7) del cementerio de las piezas de `color`. Si está lleno, falla (no pisa piezas)."""
        slots = self.side_graveyard[color]
        if index >= len(slots):
            raise RuntimeError(
                f"Cementerio de {'blancas' if color else 'negras'} lleno ({len(slots)} huecos): "
                "una persona debe vaciarlo antes de seguir."
            )
        return slots[index]


class Executor(Protocol):
    name: str

    def pick_place(self, label: str, piece_type: int, src_xy: tuple[float, float], dst_xy: tuple[float, float]) -> dict: ...


class DryRunExecutor:
    """No mueve nada: anota cada coger-y-soltar. Es el ejecutor por defecto."""

    name = "dry"

    def __init__(self) -> None:
        self.log: list[dict] = []

    def pick_place(self, label, piece_type, src_xy, dst_xy) -> dict:
        item = {"accion": label, "pieza": PIECE_NAME[piece_type], "origen_m": [round(src_xy[0], 4), round(src_xy[1], 4)],
                "destino_m": [round(dst_xy[0], 4), round(dst_xy[1], 4)], "ejecutado": False}
        self.log.append(item)
        print(f"  [dry] {label}: {item['pieza']} {item['origen_m']} -> {item['destino_m']} (NO se mueve el brazo)")
        return {"ok": True, "executed": False}


class ACTExecutor:
    """Pasaría cada coger-y-soltar a un checkpoint ACT. Por ahora solo `preflight`: no mueve el brazo.

    Para que funcione hace falta, como mínimo:
      1. Pesos reales: modeloACT/checkpoints/last/pretrained_model los tiene (los de checkpoints/act_tray_test200 son punteros
         de Git LFS). Ese modelo (2000 pasos, batch 1) falla el test offline de pruebas_act: error de 14,5 grados de media en el
         primer paso, peor que repetir el estado, y salidas fuera de rango.
      2. Un ACT que reciba el objetivo (observation.environment_state con x,y; ver real/deploy_goal.py y real/record_goal.py).
         El de modeloACT solo usa estado + cámaras front/gripper (640x360) y no admite objetivo: no se le puede decir a qué
         casilla ir ni qué pieza coger. Para probarlo tal cual en el brazo: real/deploy_act_generic.py.
      3. Grabar demostraciones de ajedrez con objetivo (real/record_goal.py) y entrenar un ACT con ellas.
    Además, cualquier movimiento real exige ALLOW_REAL_MOTION=1 y el SI explícito del usuario (AGENTS.md).
    """

    name = "act"

    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT, device: str | None = None):
        self.checkpoint = Path(checkpoint)
        self.device = device or os.environ.get("SO101_VLA_DEVICE", "mps")

    def preflight(self) -> list[str]:
        """Lista de problemas que impiden usar el checkpoint. Vacía = compatible."""
        problems: list[str] = []
        if not self.checkpoint.exists():
            return [f"no existe {self.checkpoint}"]
        for f in sorted(self.checkpoint.glob("*.safetensors")):
            head = f.read_bytes()[:64]
            if head.startswith(b"version https://git-lfs"):
                problems.append(f"{f.name} es un puntero de Git LFS ({f.stat().st_size} bytes), no son pesos")
        if not (self.checkpoint / "config.json").exists():
            return problems + [f"falta {self.checkpoint / 'config.json'}"]
        cfg = json.loads((self.checkpoint / "config.json").read_text())
        feats = cfg.get("input_features", {})
        if "observation.environment_state" not in feats:
            problems.append("el checkpoint no usa observation.environment_state: no recibe el objetivo x,y de la casilla")
        cams = sorted(k for k in feats if k.startswith("observation.images."))
        if cams:
            problems.append(f"cámaras que espera: {', '.join(cams)} ({feats[cams[0]]['shape']}); real/deploy_goal.py envía camera1/2/3 a 320x240 (usa real/deploy_act_generic.py)")
        return problems

    def pick_place(self, label, piece_type, src_xy, dst_xy) -> dict:
        problems = self.preflight()
        raise RuntimeError(
            "El ejecutor ACT todavía no puede mover el brazo: "
            + ("; ".join(problems) if problems else "falta definir qué hace cada ejecución del ACT (coger/soltar).")
        )


def make_executor(name: str | None = None):
    name = (name or os.environ.get("CHESS_REAL_EXECUTOR", "dry")).lower()
    if name == "dry":
        return DryRunExecutor()
    if name == "act":
        return ACTExecutor()
    if name in ("ik", "ikact"):  # IK + agarre guionizado (ik) o ventana de ACT confinada (ikact): hub/chess_ik_exec.py
        sys.path.insert(0, str(ROOT / "hub"))
        from chess_ik_exec import IKExecutor

        return IKExecutor(grasp="scripted" if name == "ik" else "act")
    raise SystemExit(f"CHESS_REAL_EXECUTOR desconocido: {name} (usa dry, act, ik o ikact)")


class RealChessBackend:
    def __init__(self, executor=None, geometry: Geometry | None = None,
                 verify: Callable[[chess.Board, chess.Move], list[str]] | None = None):
        self.executor = executor or make_executor()
        self.geometry = geometry or Geometry(require_real=self.executor.name != "dry")
        self.verify = verify
        if not self.geometry.is_real:
            print("AVISO: plan con la geometría de la SIMULACIÓN (hub/chess_board.json), no calibrada en real.")

    @staticmethod
    def captured_count(board: chess.Board, color: bool = chess.WHITE) -> int:
        """Piezas de `color` ya fuera del tablero (siguiente hueco libre de su cementerio), deducido de la posición."""
        return max(0, 16 - sum(1 for p in board.piece_map().values() if p.color == color))

    def plan(self, move: chess.Move, board_before: chess.Board) -> list[dict]:
        """Lista ordenada de coger-y-soltar: [{"label", "piece", "src_xy", "dst_xy", "note"}]. No toca nada."""
        if move not in board_before.legal_moves:
            raise ValueError(f"jugada ilegal: {move.uci()}")
        g, b, steps = self.geometry, board_before, []
        if b.is_en_passant(move):
            cap = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            steps.append({"label": f"captura al paso en {chess.square_name(cap)}", "piece": b.piece_type_at(cap),
                          "src_xy": g.xy(cap), "dst_xy": g.grave(self.captured_count(b, not b.turn), not b.turn)})
        elif b.is_capture(move):
            steps.append({"label": f"captura en {chess.square_name(move.to_square)}", "piece": b.piece_type_at(move.to_square),
                          "src_xy": g.xy(move.to_square), "dst_xy": g.grave(self.captured_count(b, not b.turn), not b.turn)})
        piece = b.piece_type_at(move.from_square)
        if b.is_castling(move):
            kingside = chess.square_file(move.to_square) == 6
            rank = chess.square_rank(move.from_square)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            steps.append({"label": f"enroque: rey {chess.square_name(move.from_square)}->{chess.square_name(move.to_square)}",
                          "piece": piece, "src_xy": g.xy(move.from_square), "dst_xy": g.xy(move.to_square)})
            steps.append({"label": f"enroque: torre {chess.square_name(rook_from)}->{chess.square_name(rook_to)}",
                          "piece": chess.ROOK, "src_xy": g.xy(rook_from), "dst_xy": g.xy(rook_to)})
        else:
            steps.append({"label": f"{chess.square_name(move.from_square)}->{chess.square_name(move.to_square)}",
                          "piece": piece, "src_xy": g.xy(move.from_square), "dst_xy": g.xy(move.to_square)})
        if move.promotion:
            steps[-1]["note"] = (f"promoción a {chess.piece_name(move.promotion)}: el robot lleva el peón y una persona "
                                 f"lo cambia por la pieza nueva (no hay piezas de repuesto en el cementerio)")
        return steps

    def execute(self, move: chess.Move, board_before: chess.Board) -> dict:
        plan = self.plan(move, board_before)
        names = []
        for s in plan:
            self.executor.pick_place(s["label"], s["piece"], s["src_xy"], s["dst_xy"])
            names.append(s["label"] + (f" ({s['note']})" if s.get("note") else ""))
        after = board_before.copy()
        after.push(move)
        bad = self.verify(after, move) if self.verify else []
        PLAN_LOG.write_text(json.dumps({
            "uci": move.uci(), "executor": self.executor.name, "geometria": str(self.geometry.path.name),
            "geometria_real": self.geometry.is_real,
            "plan": [{"label": s["label"], "pieza": PIECE_NAME[s["piece"]], "src_xy": s["src_xy"], "dst_xy": s["dst_xy"]} for s in plan],
        }, indent=2, ensure_ascii=False))
        return {"uci": move.uci(), "steps": names, "board_ok": not bad, "bad_squares": bad, "fen": after.fen(),
                "verified": self.verify is not None, "executor": self.executor.name}
