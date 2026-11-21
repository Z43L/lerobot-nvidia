---
name: so101-ajedrez-real
description: Skill maestra para jugar al ajedrez de verdad contra el SO-101 con cámara, YOLO y el brazo real. Encadena preflight, calibración del tablero, lectura del tablero con YOLO, reglas, ejecución con el ejecutor (dry o ACT) y verificación con la cámara. Úsala cuando el usuario diga "juguemos al ajedrez con el brazo real", "partida real", "lee el tablero con la cámara", "prueba el pipeline de ajedrez" o "prepara el ajedrez para el evento".
---

# Ajedrez real con el SO-101 (cámara + YOLO + reglas + brazo)

Esta skill **coordina**. Las piezas ya existen: `so101-hardware-check`, `so101-calibracion-tablero`, `so101-ajedrez-partida` (reglas y turnos), `so101-ajedrez-mover` (jugada concreta) y `so101-real-control` (seguridad del brazo). Aquí está el orden y lo que cada paso comprueba.

## Reglas que no cambian

- El brazo real se mueve directamente (`ALLOW_REAL_MOTION=1` ya está en el entorno). No se pide `SI` ni se espera confirmación por jugada. `SO101_MAX_RELATIVE_TARGET=2` por defecto.
- Los límites mecánicos del código (rango de articulación, ≤2° por paso, parada por atasco) siguen activos: protegen el hardware.
- No cambies la iluminación ni muevas la cámara o el tablero durante la partida: las esquinas de `vision/board_camara.json` dejan de valer.
- No edites `hub/chess_game.json` a mano. Para pruebas usa `CHESS_GAME_FILE=/ruta/otra.json`.
- Di siempre qué NO está probado (sección final).

## Pipeline, en orden

| # | Etapa | Comando / skill | Qué comprueba |
|---|---|---|---|
| 0 | Preflight del brazo y las cámaras | skill `so101-hardware-check`; `./.venv312/bin/python real/find_cameras.py` | Puertos, calibración, cámaras abren. macOS debe dar permiso de cámara a la app que lanza el shell. |
| 1 | Calibrar el tablero real | skill `so101-calibracion-tablero` -> `hub/chess_board_real.json` | Centros de casilla en metros (marco de la base). Sin este fichero el ejecutor real se niega. |
| 2 | Calibrar la visión | medir las 4 esquinas del área 8x8 en la imagen de la cámara -> `vision/board_camara.json` (TL, TR, BR, BL; a1 = BR con blancas a la derecha) | `chess_cli.py see --camera N` con el tablero en posición inicial debe dar FEN inicial con 0-3 casillas dudosas. |
| 3 | Partida nueva | `CHESS_BACKEND=real ./.venv312/bin/python sim/chess_cli.py new [--robot-white]` | |
| 4 | Turno del humano | `chess_cli.py human-seen --camera N` (o `human <jugada>` si no está claro) | YOLO lee las piezas, python-chess elige la jugada legal que mejor explica lo visto. Si no hay ganadora clara, NO adivines: mira la imagen y pide la jugada. |
| 5 | Turno del robot | `chess_cli.py robot --verify-camera N` | El motor elige, `RealChessBackend` planifica (captura primero al cementerio, enroque en dos pasos, promoción con ayuda humana), el ejecutor lo hace y la cámara verifica origen, destino y capturas. |
| 6 | Bucle completo | `CHESS_BACKEND=real ./.venv312/bin/python sim/chess_match.py --camera N` | Repite 4 y 5, para si el tablero no coincide. |

Ensayo sin brazo: `CHESS_REAL_EXECUTOR=dry` (por defecto). Imprime cada coger-y-soltar con coordenadas y no mueve nada. Ver `hub/chess_real_plan.json`.

## Ejecutores (`CHESS_REAL_EXECUTOR`)

- `dry`: solo planifica. Es el único que funciona hoy.
- `act`: pasaría cada coger-y-soltar a un ACT. **Hoy solo hace `preflight` y se niega a mover.** Medido:
  1. Pesos reales: `modeloACT/checkpoints/last/pretrained_model` (SenseCraft, "data play tray 1", 2000 pasos, tarea "pick and place"). Los de `checkpoints/act_tray_test200` son punteros de Git LFS (134 bytes): inútiles.
  2. Calidad: el test offline de `pruebas_act/` suspende. Error de 14,5 grados de media en el primer paso (repetir el estado da 1-2), 37 en `wrist_roll`, y predicciones fuera del rango entrenado. Carga y predice bien (0,35 s por chunk en mps).
  3. No recibe objetivo: solo `observation.state` y cámaras `front` y `gripper` (640x360). No se le puede decir qué pieza coger ni a qué casilla ir, así que NO puede jugar una jugada concreta. Para eso hace falta un ACT con objetivo (`observation.environment_state`): grabar jugadas con `real/record_goal.py`, entrenar con `real/train_act.sh` y desplegar con `real/deploy_goal.py` (skill `so101-vla-pipeline`).
- **ACT genérico:** `real/deploy_act_generic.py`. `--dry-run` carga los pesos y predice sin enviar nada. En real: límite 2 grados por paso, 15 s, y no arranca si la primera acción se aleja más de 15 grados de la pose actual.

## Orientación del tablero (importante)

- Las etiquetas IMPRESAS del tablero real están espejadas respecto al ajedrez estándar (su `a1` es la esquina junto al robot). El sistema usa **etiquetas estándar**: `a1` = esquina inferior-derecha de la imagen de la cámara 0 (lejos del robot), `h1` = superior-derecha (junto al robot), dama blanca en `d1`. Lo que el tablero imprime como `e2` es `d2` para el sistema.
- Calibración por contacto: `hub/board_touch_fit.py` (ajusta desfases de ángulo y tablero; escribe etiquetas estándar) y `hub/chess_ik_exec.py` (IK + agarre guionizado, `plan`, `test-grasp`, `set-home`). Copias antiguas: `*.bak_etiquetas_usuario`.

## Visión: lo medido

- Detector: `runs/detect/vision/runs/piezas_camara/weights/best.pt` (YOLO11n, 12 clases, entrenado con 69 fotos etiquetadas a mano de `Camera/`).
- Lectura de casillas contra mis anotaciones (17 fotos de validación): recall 0,94 y precisión 0,98 por pieza; 5 de 17 fotos exactas. Los fallos son piezas ocultas o alfil/peón y rey/dama.
- La verificación solo bloquea por origen, destino y capturas de la jugada, piezas de más con confianza >= 0,5 o tipo equivocado con confianza >= 0,8. Piezas que no se ven solo avisan.
- Repetición de 6 jugadas con fotos reales (d4 d5 Nf3 Nf6 Nc3 e6): 6 de 6, y una verificación negativa (foto sin la jugada) se rechaza.
- Re-entrenar con más fotos: `vision/labeler/` (editor y herramientas) y `vision/train_pieces.sh`.

## Si algo falla

- `human-seen` dice "No hay una jugada clara": mira la foto (`see`), pide la jugada al usuario y usa `human <jugada>`.
- La verificación marca casillas tras la jugada del robot: la jugada no se registra. Pide que se corrija la pieza a mano, `see` para confirmar y repite o reanuda.
- Cámara no abre: permiso de cámara de macOS para la app (Claude o Terminal) e índice correcto (`real/find_cameras.py`).
- Brazo que se mueve raro o un golpe: `arm_stop` y revisar. No reenviar comandos en bucle.

## Estado de la prueba (actualizado 21-nov-2026, con brazo)

Probado: planificación, ejecutor `dry`, lectura de jugadas con fotos, verificación, 8 tests de `sim/test_chess_real.py`, cámaras y YOLO en vivo, `arm_state`, plegado al retorno con `arm_move`, preflight IK de e2→e4.
NO completado: ninguna jugada real. El codo no despliega desde el reposo plegado (ver `so101-ajedrez-brazo-retorno`). Checkpoint ACT de 30k: `act_tray_a100_100k_ckpts/checkpoints/030000/pretrained_model` (sin objetivo). Flujo por turnos: `so101-ajedrez-partida-real`. Cámaras/YOLO: `so101-ajedrez-camaras-yolo`. Bitácora: `PARTIDA_AJEDREZ_REAL.md`.

## MCP `so101-chess` (Claude orquesta cámaras + YOLO + reglas + brazo + ACT)

Servidor `hub/chess_mcp.py` (registrado en `.mcp.json`; hay que reiniciar Claude Code para que cargue). **Toda la configuración sale de `hub/chess_config.py`**: entorno > `real/env.local.sh` > defecto. Cámaras: `CHESS_YOLO_CAM`/`SO101_TOP_CAM` y `CHESS_WRIST_CAM`/`SO101_WRIST_CAM`; esquinas en `CHESS_BOARD_GEOMETRY` (por defecto `vision/board_camara0.json`); pesos en `CHESS_YOLO_WEIGHTS`; ACT en `CHESS_ACT_CHECKPOINT`; puerto en `SO101_FOLLOWER_PORT` (sin defecto). `ALLOW_REAL_MOTION` no está en ningún fichero: lo exporta el usuario.

**Camino principal (una llamada por jugada, el diagrama de secuencia): `chess_move(uci, executor)`.** `dry` por defecto. Con `ik` o `ik-scripted` mueve directamente (`ALLOW_REAL_MOTION=1`; el permiso con caducidad solo con `CHESS_REQUIRE_PERMISSION=1`); devuelve foto y verificación, y **no registra**: tras mirar la foto, `chess_commit` (el "registrar", que nunca mueve nada).
- Sin movimiento: `chess_cameras`, `chess_camera(overlay)`, `chess_see`, `chess_square_px`, `chess_verify_move`, `chess_state`, `chess_new`, `chess_plan`, `chess_commit`, `arm_state` (conecta el brazo y mantiene la pose), `arm_gripper_px`.
- Mueven el brazo: `arm_jog`, `arm_move`, `arm_probe`, `arm_servo(execute=True)`, `chess_act_step`. Sin trámite de permiso: se ejecutan directamente. `arm_stop` para el movimiento.
- Flujo de una jugada: `chess_see` -> `chess_plan` -> (permiso) `arm_probe` una vez -> `arm_servo` en bucle mirando `chess_camera` -> `chess_act_step` (solo con ACT con objetivo) -> `chess_verify_move` -> `chess_commit`.
- NO probado con el brazo: `arm_state`, `arm_jog`, `arm_probe`, `arm_servo`, `chess_act_step`. Sin ACT con objetivo, `chess_act_step` se niega.
