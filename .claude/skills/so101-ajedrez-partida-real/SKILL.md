---
name: so101-ajedrez-partida-real
description: Cómo se juega una partida real de ajedrez contra el SO-101, turno a turno. Cada turno del robot = ver el tablero con la cámara + YOLO, elegir jugada, coger la pieza, soltarla en su casilla y VOLVER al punto de retorno, verificar con la cámara y registrar. Úsala cuando el usuario diga "empieza la partida", "te toca", "mueve e4", "nueva partida real", "estado de la partida" o "documenta la partida".
---

# Partida real, turno a turno (movimiento directo, sin trámites de seguridad)

Orquesta `so101-ajedrez-camaras-yolo` (ver), `so101-ajedrez-brazo-retorno` (mover y volver) y `so101-ajedrez-partida` (reglas). Todo lo que se hace se anota en `PARTIDA_AJEDREZ_REAL.md`.

## Entorno

```bash
source real/env.local.sh          # puertos y cámaras del puesto (ALLOW_REAL_MOTION=1 ya incluido)
export SO101_MAX_RELATIVE_TARGET=2 CHESS_WRIST_CAM=2
```
Movimiento directo: `motion_permission()` devuelve `True` salvo con `ALLOW_REAL_MOTION=0` (parada, también vía `arm_stop`) o `CHESS_REQUIRE_PERMISSION=1`. Siguen activos los límites mecánicos del código (rango calibrado, ≤2° por paso, parada por atasco).

## Orientación (etiquetas estándar del sistema)

- Robot = blancas, humano = negras. Dama blanca en `d1`. `a1` lejos del robot, `h1` junto al robot.
- Las etiquetas IMPRESAS del tablero real están espejadas: lo impreso `e2` es `d2` para el sistema. Habla siempre en etiquetas del sistema.

## Inicio

1. Brazo en el punto de retorno (`hub/chess_home.json`): `arm_state` debe dar ≈ `[-0,6; -97,4; 97; -98,6; 165,5; 2,4]`.
2. `chess_see` (MCP `so101-chess`) o `./.venv312/bin/python sim/chess_cli.py see --camera 0`. YOLO acierta bien peones y ocupación; la fila trasera falla (dama/rey/alfil): confirma con `chess_camera(overlay=True)`.
3. `CHESS_BACKEND=real ./.venv312/bin/python sim/chess_cli.py new --robot-white` (o `chess_new`).
4. Robot con blancas: la primera jugada es suya (`e2e4`).

## Turno del robot (obligatorio terminar en el punto de retorno)

1. `chess_see` → comprueba que la cámara coincide con la partida (`casillas_distintas` ≈ solo errores de tipo en fila trasera).
2. Elegir jugada: `chess_cli.py robot` (minimax prof. 2) o jugada concreta.
3. `chess_plan <uci>` → pasos coger-y-soltar (captura primero al cementerio, enroque en dos pasos).
4. Ejecutar: `./.venv312/bin/python hub/chess_ik_exec.py run <uci> --grasp scripted` (apertura justa por pieza, cierre suave; detalles en `so101-ajedrez-agarre`)
   Secuencia: ORIGEN → encima origen → agarre → subir → encima destino → soltar → subir → **ORIGEN (retorno)**.
5. `chess_camera(overlay=True)` y `chess_verify_move <uci>`: origen vacío, destino con el color correcto.
6. `chess_commit <uci>` solo si la verificación es correcta.

## Turno del humano

`chess_see` tras su jugada → `chess_cli.py human-seen --camera 0`. Si no hay jugada clara, mira la imagen y usa `human <jugada>`. No adivines.

## Estado y reglas de oro

- Partida en `hub/chess_game.json` (FEN, historial, color). No se edita a mano; para pruebas, `CHESS_GAME_FILE=/otra/ruta.json`.
- No muevas cámara ni tablero ni cambies la luz durante la partida: las esquinas de `vision/board_camara0.json` dejan de valer.
- Si una etapa del brazo falla, el ejecutor se detiene sin reintentar. Vuelve al retorno con `arm_move` por tramos (ver `so101-ajedrez-brazo-retorno`) antes de intentar otra cosa.

## Estado medido (21-nov-2026)

- 1. e4 jugado por el robot de principio a fin (coger → llevar → soltar → retorno), verificado con la cámara 0 y registrado. Turno de negras.
- Retorno desplegado `[-7,9; -17,8; 13,5; 100,0; 165,1; 2,7]`; ganancias P codo 28 / hombro 44 (ver `so101-ajedrez-brazo-retorno`).
- Para ver el tablero desde la cenital, gira `shoulder_pan` a -55 con `arm_move` y vuelve a -7,9: en el retorno el brazo la tapa.
- Sin probar: capturas (cementerio sin calibrar: `graveyard_m` vacío), enroque, otras piezas que no sean peones, el turno del humano con `human-seen` en vivo.
