---
name: so101-ajedrez-mover
description: Mueve fichas de ajedrez con el SO-101 (sim ahora, real en el futuro) a partir de una jugada UCI/SAN legal. Úsala cuando haga falta mover una ficha de una casilla a otra, capturar, enrocar, promocionar, devolver una ficha capturada al cementerio, o cuando el usuario diga "mueve la ficha", "haz la jugada", "pon la ficha en", "captura en" o "prepara el tablero". La orquestadora so101-ajedrez-partida la usa para cada jugada.
---

# Mover fichas de ajedrez (SO-101)

Esta skill ejecuta **una jugada** sobre el tablero. No decide jugadas ni lleva el turno: eso es `so101-ajedrez-partida`.

Hoy el backend es la **simulación**. Cuando haya brazo y tablero reales, la misma interfaz se cambia por el backend real (ver "Paso a real"). Todo lo que depende del entorno está en `hub/chess_board.json` y `sim/chess_sim.py`, así que las calibraciones se cambian ahí, no en esta skill.

## Comandos

Desde la raíz del proyecto, con `.venv312`:

```bash
./.venv312/bin/python sim/chess_cli.py new [--robot-white]
./.venv312/bin/python sim/chess_cli.py state
./.venv312/bin/python sim/chess_cli.py move <uci>        # ejecuta una jugada legal sin cambiar de turno
./.venv312/bin/python sim/chess_cli.py render [fichero.png]
```

Para la partida completa (humano y robot con turnos) usa `so101-ajedrez-partida`. `move` y `render` sirven para probar esta skill sola.

Variables:
- `CHESS_BACKEND=sim` (por defecto) o `real` (`sim/chess_real.py`: planifica y delega en `CHESS_REAL_EXECUTOR`, `dry` por defecto). Mover el brazo real se hace con `so101-ajedrez-real`, no con esta skill.
- `SO101_GRASP=attach` (el CLI lo pone por defecto): **enganche temporal**. En la sim la pinza no coge por fricción; la ficha se engancha al TCP al cerrar la pinza y se suelta al abrirla. Ver `so101-sim-objects` / `sim/objects_env.py`.

## Sistema de coordenadas

- Marco: base del SO-101 sobre la mesa. x hacia el frente, y a la izquierda, en metros.
- Tablero de referencia (sim): centro en `(0.22, 0.0)`, casilla de 3 cm. Blancas en la parte cercana al robot (filas 1 y 2 con x pequeña).
- Cementerio de fichas capturadas: dos filas fuera del tablero, en `y = ±0.17`, `x` de 0.10 a 0.31 (16 huecos).
- `hub/chess_board.json` guarda las casillas (`squares`) y el cementerio (`graveyard_m`). Lo genera `python sim/chess_sim.py`. Si existe, `square_xy` lo usa en lugar de la fórmula.

## Procedimiento de una jugada (lo que hace `execute`)

1. Valida que la jugada sea legal en el tablero previo; si la escena física no coincide con ese tablero, la reconstruye (`load_board`).
2. **Captura:** coge la ficha capturada y la deja en el siguiente hueco del cementerio. En captura al paso, la ficha capturada está en la casilla de al lado, no en la de destino.
3. **Enroque:** mueve el rey y luego la torre.
4. **Movimiento normal:** coge la ficha en su casilla y la deja en la de destino. Aproximación a altura de viaje 0,085 m, bajada a la altura de la ficha, enganche, subida, traslado, bajada, suelta y subida.
5. **Promoción:** la ficha cambia de tipo en la lógica. En la sim el cuerpo es el mismo (cosmético).
6. Comprueba que cada ficha está a menos de 1,5 cm de su casilla. Si no, devuelve `board_ok: false` y la partida **no** avanza.

Resultado: `{uci, steps, board_ok, bad_squares, fen}`.

## Errores que puedes ver

- `jugada ilegal`: la jugada no está en las legales del FEN actual. No la ejecutes: pide otra al usuario o consulta `state`.
- `la pinza no llega a la ficha (distancia > 3 cm)`: la ficha no está donde el tablero dice (probablemente una ejecución anterior falló). Reconstruye con `load_board` desde el FEN de `hub/chess_game.json`.
- `AVISO: la escena no quedó bien en [...]`: una ficha terminó fuera de su casilla (más de 1,5 cm). No avances; revisa la casilla y repite la jugada desde el mismo FEN.

## Límites conocidos (sim)

- **Error de posición del brazo:** el par de los servos (`forcerange` ±2,94 en `so101/so101_vla.xml`) no sostiene el brazo extendido. Con el tablero a x = 0,22, el error máximo es de 1,3 cm (filas 8). Con el tablero en x = 0,20, la fila 1 se iba a 3,1 cm. Si cambias el tablero, mide el error otra vez (barrido de casillas) antes de jugar.
- **Fichas:** no colisionan con la pinza ni entre sí. La geometría de cada ficha (radio y altura) es aproximada.
- **Agarre:** es el enganche temporal. No se valida que la ficha se sostenga por fricción; no es un agarre real.

## Paso a real

Ya existe el backend (`RealChessBackend`), la calibración por contacto (`hub/chess_board_real.json`) y el ejecutor IK (`hub/chess_ik_exec.py`). Lo que sigue son notas de diseño; el flujo vigente está en `so101-ajedrez-real`. Pendiente de medir con el brazo: agarre por tipo de pieza.

1. **Calibración del tablero real:** la cámara de mesa (`hub/hand_calibration.py`) con marcadores en las esquinas del tablero. Hay que extender el script para las 64 casillas, o usar una homografía del tablero completo, y escribir `hub/chess_board.json` con `"source": "real"` y `"status"` con la fecha de la calibración.
2. **Poses de agarre por tipo de ficha:** el usuario enseña cada pose con el brazo líder y se graban con **`so101-grabacion-maestra`** (graba, convierte a keyframes, valida, practica en la sim y genera la skill `motion-<nombre>`). Una pose por tipo (peón, caballo, alfil, torre, dama, rey) y una altura de agarre por tipo.
3. **Mapeo sim → real:** la skill `so101-grabacion-maestra` corrige las poses de la sim con las del real: los keyframes grabados en real sustituyen a los estimados por IK, y los offsets (`sag`, zona de agarre) se anotan en `hub/chess_board.json`.
4. **Backend real:** `RealChessBackend` en `sim/chess_real.py` (misma interfaz que `ChessSim.execute`). Sin enganche: el agarre es por fricción, y hay que medir anchura y fuerza.
5. **Seguridad:** igual que `so101-real-control`. Preview por movimiento, confirmación explícita del usuario, `SO101_MAX_RELATIVE_TARGET=2` al empezar, área libre y persona junto al brazo.

Al pasar a real, esta skill no se reescribe: se cambian `CHESS_BACKEND`, `hub/chess_board.json` y el backend. Las demás skills (la orquestadora, el CLI) no cambian.
