"""Backend REAL del ajedrez del SO-101. TODAVÍA NO IMPLEMENTADO.

Cuando el brazo real y el tablero real estén en el puesto, aquí va la parte que cambia respecto a la sim:

  1. Calibración del tablero real: 4 esquinas (o marcadores) con la cámara de mesa y su posición en
     el marco de la base del brazo. Genera hub/chess_board.json con status "real" (lo hace
     hub/hand_calibration.py para la mesa y debe ampliarse para las 64 casillas).
  2. Pose de agarre por casilla y altura de cada tipo de pieza: se graban con so101-grabacion-maestra
     (el brazo líder guiado por el humano) y se reproducen con so101-motion (preview + confirmación).
  3. Agarre real: sin enganche. La pinza coge por fricción; hay que medir la anchura y la fuerza.
  4. Cementerio de fichas capturadas: posiciones reales fuera del tablero.

Interfaz que debe cumplir (la misma que usa ChessSim.execute):
    class RealChessBackend:
        def execute(self, move: chess.Move, board_before: chess.Board) -> dict:
            # devuelve {"uci", "steps", "board_ok", "bad_squares", "fen"} igual que la sim
"""


class RealChessBackend:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "El backend real del ajedrez no existe todavía. Ver docstring de sim/chess_real.py "
            "y la skill so101-ajedrez-mover (sección 'Paso a real')."
        )
