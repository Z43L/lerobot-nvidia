"""Ajedrez sobre la mesa del SO-101, en simulación (MuJoCo).

Qué hay aquí:
  - Escena: tablero de 24 x 24 cm (casillas de 3 cm) y 32 fichas (cilindros con masa).
  - Reglas: python-chess. El tablero lógico (`self.board`) manda; la escena física lo sigue.
  - Agarre: ENGANCHE TEMPORAL (SO101_GRASP=attach, ver objects_env.py). La pinza no coge por
    fricción; la ficha se engancha al TCP al cerrar la pinza y se suelta al abrirla.
  - Geometría de casillas: hub/chess_board.json si existe (en real lo generará la calibración del
    tablero real); si no, la fórmula de la sim (CENTER / SQ).

Uso desde código (ver sim/chess_cli.py para la línea de comandos que usan las skills):
    from chess_sim import ChessSim
    sim = ChessSim()
    sim.load_board(chess.Board())
    sim.execute(chess.Move.from_uci("e2e4"), board_before)

Limitaciones honestas:
  - La ficha no colisiona con la pinza ni con otras fichas durante el movimiento (solo con el tablero).
  - Promoción: la ficha cambia de tipo en la lógica; la geometría es la de la ficha original.
  - Agarre de ficha tumbada o de pie: no se modela; el enganche es igual para todas.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chess
import mujoco
import numpy as np

import objects_env
from objects_env import ObjectsSim, GRIP_OPEN, GRIP_CLOSED, TRAVEL_Z
from sim_arm import SimArm

HERE = Path(__file__).resolve().parent
SCENE = HERE / "so101" / "scene_chess.xml"
BOARD_JSON = HERE.parent / "hub" / "chess_board.json"

SQ = 0.03                 # lado de la casilla (m)
CENTER = (0.22, 0.0)      # centro del tablero en el marco de la base del robot (m)
HOME_XYZ = (0.10, 0.0, 0.12)

# Radio y semialtura de cada ficha (m)
PIECE_GEOM = {
    chess.PAWN: (0.011, 0.02),
    chess.KNIGHT: (0.013, 0.025),
    chess.BISHOP: (0.013, 0.03),
    chess.ROOK: (0.0135, 0.025),
    chess.QUEEN: (0.015, 0.035),
    chess.KING: (0.016, 0.04),
}
PIECE_SYM = {chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q", chess.KING: "k"}
COLOR_RGBA = {chess.WHITE: "0.95 0.95 0.92 1", chess.BLACK: "0.10 0.10 0.10 1"}

# Cementerio de fichas capturadas: dos filas fuera del tablero (lado izquierdo y derecho del robot)
GRAVE_XY = [(x, y) for y in (-0.17, 0.17) for x in (0.10 + i * 0.03 for i in range(8))]


def _load_board_json() -> dict | None:
    if not BOARD_JSON.exists():
        return None
    try:
        return json.loads(BOARD_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def square_xy(sq: int) -> tuple[float, float]:
    """(x, y) en metros del centro de la casilla. Usa hub/chess_board.json si existe."""
    name = chess.square_name(sq)
    data = _load_board_json()
    if data and name in data.get("squares", {}):
        x, y = data["squares"][name]
        return float(x), float(y)
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return CENTER[0] + (r - 3.5) * SQ, CENTER[1] + (f - 3.5) * SQ


def _piece_bodies() -> list[tuple[str, bool, int]]:
    """Lista fija de fichas: (nombre_del_cuerpo, es_blanca, tipo). 16 por bando."""
    order = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING, chess.BISHOP, chess.KNIGHT, chess.ROOK]
    out = []
    for white in (True, False):
        c = "w" if white else "b"
        for i, t in enumerate(order):
            out.append((f"{c}_{PIECE_SYM[t]}{i}", white, t))
        for i in range(8):
            out.append((f"{c}_p{i}", white, chess.PAWN))
    return out


def build_scene_xml() -> str:
    parts = []
    for sq in chess.SQUARES:
        x, y = square_xy(sq)
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        rgba = "0.85 0.80 0.68 1" if light else "0.35 0.24 0.14 1"
        parts.append(f'<geom name="sq_{chess.square_name(sq)}" type="box" size="{SQ / 2} {SQ / 2} 0.001" '
                     f'pos="{x:.4f} {y:.4f} 0.0005" rgba="{rgba}" contype="0" conaffinity="0"/>')
    # Tablero sólido bajo las casillas: solo colisiona con fichas (contype 2 / conaffinity 4).
    parts.append('<geom name="board" type="box" size="0.14 0.14 0.004" pos="0.20 0.0 -0.004" '
                 'rgba="0.25 0.18 0.1 1" contype="2" conaffinity="4"/>')
    for name, white, t in _piece_bodies():
        r, h = PIECE_GEOM[t]
        parts.append(f'<body name="{name}" pos="0 0 {h}"><freejoint name="{name}_free"/>'
                     f'<geom name="{name}_g" type="cylinder" size="{r} {h}" rgba="{COLOR_RGBA[white]}" '
                     f'mass="0.02" contype="4" conaffinity="2" friction="1 0.005 0.0001"/></body>')
    return f"""<mujoco model="so101_chess_scene">
    <include file="so101_vla.xml" />
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <global offwidth="640" offheight="480" />
    </visual>
    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" directional="true" />
        <geom name="floor" size="1 1 0.05" pos="0 0 0" type="plane" rgba="0.85 0.85 0.85 1" />
        <camera name="top" mode="fixed" pos="0.20 0.00 0.55" xyaxes="0 1 0 -1 0 0" fovy="50" />
        <camera name="table" mode="targetbody" target="base" pos="0.10 -0.80 0.85" fovy="55" />
        {chr(10).join(parts)}
    </worldbody>
</mujoco>
"""


class ChessSim(ObjectsSim):
    """Escena de ajedrez con la misma API de agarre que ObjectsSim (enganche con SO101_GRASP=attach)."""

    def __init__(self, camera_size: tuple[int, int] = (480, 640)):
        self._chess_ready = False
        self.board = chess.Board()
        self.pieces = _piece_bodies()  # (cuerpo, es_blanca, tipo)
        self.sq_piece: dict[int, str] = {}  # casilla -> cuerpo de la ficha
        self.grave_used = 0
        SCENE.write_text(build_scene_xml())
        self.obj_qadr: dict[str, int] = {}
        self.held_obj: str | None = None
        self._pin = None
        self._attach_offset = np.zeros(3)
        SimArm.__init__(self, camera_size=camera_size, scene=SCENE)
        self.tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.jaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        for name, _, _ in self.pieces:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            self.obj_qadr[name] = int(self.model.jnt_qposadr[jid])
        self._obj_geoms = {}
        for name, _, _ in self.pieces:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._obj_geoms[name] = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
        self._obj_contype = self.model.geom_contype.copy()
        self._obj_conaffinity = self.model.geom_conaffinity.copy()
        self.substeps = max(1, int(round(1.0 / objects_env.CONTROL_HZ / self.model.opt.timestep)))
        self.close_local = self._closing_axis_local()
        self._chess_ready = True
        self.load_board(self.board)

    # ----------------------------------------------------------- escena
    def reset(self) -> dict:
        return self.reset_objects()

    def reset_objects(self, layout: dict | None = None) -> dict:
        if not self._chess_ready:  # llamada desde SimArm.__init__
            return SimArm.reset(self)
        return self.load_board(self.board)

    def _place_body(self, name: str, xy: tuple[float, float], t: int) -> None:
        a = self.obj_qadr[name]
        r, h = PIECE_GEOM[t]
        self.data.qpos[a : a + 7] = [xy[0], xy[1], h, 1, 0, 0, 0]
        dof = self.model.jnt_dofadr[self.model.joint(f"{name}_free").id]
        self.data.qvel[dof : dof + 6] = 0.0

    def load_board(self, board: chess.Board) -> dict:
        """Coloca las fichas físicas según `board` (sin física: teletransporte). Reparte cuerpos por color y tipo."""
        self.board = board.copy()
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self.home_qpos
        for gid in sum(self._obj_geoms.values(), []):
            self.model.geom_contype[gid] = self._obj_contype[gid]
            self.model.geom_conaffinity[gid] = self._obj_conaffinity[gid]
        self.held_obj = None
        self._pin = None
        self.grave_used = 0
        # Fichas en el tablero, por color y tipo, en orden de casilla
        want = {}
        for sq, piece in board.piece_map().items():
            want.setdefault((piece.color, piece.piece_type), []).append(sq)
        free = {}
        for name, white, t in self.pieces:
            free.setdefault((white, t), []).append(name)
        # Cuerpos libres de cada color (para promociones: cualquier cuerpo no usado de ese color)
        assigned: dict[int, str] = {}
        leftover: dict[bool, list[str]] = {True: [], False: []}
        for (color, t), squares in want.items():
            pool = free.get((color, t), [])
            for sq in squares:
                if pool:
                    assigned[sq] = pool.pop(0)
                else:
                    leftover[color].append(sq)  # promoción: se usa un cuerpo sobrante de ese color
        for color in (True, False):
            spare = [n for (c, t), names in free.items() if c == color for n in names]
            for sq in leftover[color]:
                if spare:
                    assigned[sq] = spare.pop(0)
        self.sq_piece = dict(assigned)
        for sq, name in assigned.items():
            self._place_body(name, square_xy(sq), board.piece_at(sq).piece_type)
        # Fichas sobrantes (capturadas) al cementerio
        used = set(assigned.values())
        for name, white, t in self.pieces:
            if name not in used:
                gx, gy = GRAVE_XY[self.grave_used % len(GRAVE_XY)]
                self.grave_used += 1
                self._place_body(name, (gx, gy), t)
        for _ in range(60):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return {"fen": board.fen()}

    # ----------------------------------------------------------- utilidades
    def piece_xy(self, name: str) -> np.ndarray:
        return self.obj_pos(name)[:2]

    def check_board(self) -> list[str]:
        """Devuelve las casillas donde la ficha física no está a menos de 1,5 cm de su casilla."""
        bad = []
        for sq, name in self.sq_piece.items():
            x, y = square_xy(sq)
            d = np.linalg.norm(self.piece_xy(name) - np.array([x, y]))
            if d > 0.015:
                bad.append(chess.square_name(sq))
        return bad

    def go_home(self) -> None:
        q5 = self.data.qpos[:5].copy()
        q5 = self.move_tcp_to(HOME_XYZ, None, GRIP_OPEN, 2.0, q5)
        self.hold_grip(q5, GRIP_OPEN, 0.3)

    # ----------------------------------------------------- agarre (enganche)
    def _pick_body(self, name: str, t: int) -> None:
        """Coge la ficha `name` (enganche): aproximación, bajada, cierre y subida."""
        x, y = self.piece_xy(name)
        h = PIECE_GEOM[t][1]
        for gid in self._obj_geoms[name]:
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
        a = self.obj_qadr[name]
        self._pin = (name, self.data.qpos[a : a + 7].copy())
        q5 = self.data.qpos[:5].copy()
        q5 = self.move_tcp_to((x, y, TRAVEL_Z), None, GRIP_OPEN, 2.0, q5)
        q5 = self.move_tcp_to((x, y, h), None, GRIP_OPEN, 1.2, q5)
        self.hold_grip(q5, GRIP_CLOSED, 0.6)
        self._pin = None
        if not self._attach(name):
            raise RuntimeError(f"{name}: la pinza no llega a la ficha (distancia > 3 cm)")
        q5 = self.move_tcp_to((x, y, TRAVEL_Z), None, GRIP_CLOSED, 1.2, q5)
        self.hold_grip(q5, GRIP_CLOSED, 0.3)

    def _place_body_at(self, name: str, xy: tuple[float, float], t: int) -> None:
        """Lleva la ficha enganchada a `xy`, la suelta y sube."""
        h = PIECE_GEOM[t][1]
        q5 = self.data.qpos[:5].copy()
        q5 = self.move_tcp_to((xy[0], xy[1], TRAVEL_Z), None, GRIP_CLOSED, 2.0, q5)
        q5 = self.move_tcp_to((xy[0], xy[1], h + 0.005), None, GRIP_CLOSED, 1.0, q5)
        self.hold_grip(q5, GRIP_OPEN, 0.6)
        self._release()
        q5 = self.move_tcp_to((xy[0], xy[1], TRAVEL_Z), None, GRIP_OPEN, 1.0, q5)
        self.hold_grip(q5, GRIP_OPEN, 0.2)

    def _type_of(self, name: str) -> int:
        for n, _, t in self.pieces:
            if n == name:
                return t
        raise KeyError(name)

    def move_from_to(self, src: int, dst: int) -> bool:
        """Mueve la ficha física de `src` a `dst` (sin reglas: solo mecánica)."""
        name = self.sq_piece.get(src)
        if name is None:
            raise RuntimeError(f"no hay ficha física en {chess.square_name(src)}")
        self._pick_body(name, self._type_of(name))
        self._place_body_at(name, square_xy(dst), self._type_of(name))
        self.sq_piece.pop(src, None)
        self.sq_piece[dst] = name
        return True

    def capture_to_grave(self, sq: int) -> None:
        name = self.sq_piece.pop(sq)
        self._pick_body(name, self._type_of(name))
        gx, gy = GRAVE_XY[self.grave_used % len(GRAVE_XY)]
        self.grave_used += 1
        self._place_body_at(name, (gx, gy), self._type_of(name))

    # ------------------------------------------------------ jugada completa
    def execute(self, move: chess.Move, board_before: chess.Board) -> dict:
        """Ejecuta `move` (legal en `board_before`) en la simulación y actualiza el tablero lógico."""
        if move not in board_before.legal_moves:
            raise ValueError(f"jugada ilegal: {move.uci()}")
        if self.board.fen() != board_before.fen():  # la escena física debe reflejar el tablero previo
            self.load_board(board_before)
        b = board_before.copy()
        steps = []
        if b.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            self.capture_to_grave(cap_sq)
            steps.append(f"captura al paso en {chess.square_name(cap_sq)}")
        elif b.is_capture(move):
            self.capture_to_grave(move.to_square)
            steps.append(f"captura en {chess.square_name(move.to_square)}")
        if b.is_castling(move):
            king_from, king_to = move.from_square, move.to_square
            rook_from = chess.square(7 if chess.square_file(king_to) == 6 else 0, chess.square_rank(king_from))
            rook_to = chess.square(5 if chess.square_file(king_to) == 6 else 3, chess.square_rank(king_from))
            self.move_from_to(king_from, king_to)
            self.move_from_to(rook_from, rook_to)
            steps.append("enroque")
        else:
            self.move_from_to(move.from_square, move.to_square)
        b.push(move)
        if move.promotion:
            name = self.sq_piece[move.to_square]
            steps.append(f"promoción a {chess.piece_name(move.promotion)} (cosmética: el cuerpo es el mismo)")
        self.board = b
        bad = self.check_board()
        return {"uci": move.uci(), "steps": steps, "board_ok": not bad, "bad_squares": bad, "fen": b.fen()}


def write_board_json_from_sim(path: Path = BOARD_JSON) -> Path:
    """Escribe hub/chess_board.json con las casillas de la sim (marco de la base, metros)."""
    squares = {chess.square_name(sq): [round(float(v), 4) for v in square_xy_sim(sq)] for sq in chess.SQUARES}
    data = {
        "frame": "base del SO-101 sobre la mesa; x hacia el frente, y a la izquierda; metros",
        "source": "sim",
        "status": "sim: geometría de la escena, no calibrada en real",
        "square_size_m": SQ,
        "center_m": list(CENTER),
        "graveyard_m": [list(p) for p in GRAVE_XY],
        "squares": squares,
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def square_xy_sim(sq: int) -> tuple[float, float]:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return CENTER[0] + (r - 3.5) * SQ, CENTER[1] + (f - 3.5) * SQ


if __name__ == "__main__":
    os.environ.setdefault("SO101_GRASP", "attach")
    p = write_board_json_from_sim()
    print("escrito", p)
