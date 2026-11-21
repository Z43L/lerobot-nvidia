---
name: so101-ajedrez-camaras-yolo
description: Cámaras USB conectadas al Mac, qué hace cada una, el modelo YOLO de piezas y el mapeo imagen-casilla del tablero de ajedrez (vision/board_camara0.json). Úsala cuando el usuario diga "qué cámara es cuál", "lee el tablero", "YOLO no ve bien", "recalibra la visión", "mapea el tablero" o falle chess_see / chess_verify_move.
---

# Cámaras, YOLO y mapeo del tablero

## Cámaras (medido 21-nov-2026, `real/find_cameras.py` / `chess_cameras`)

| Índice | Qué es | Resolución | Uso |
|---|---|---|---|
| 0 | Cámara lateral del tablero (robot a la derecha de la imagen) | 1280x960 | **YOLO y verificación** (`CHESS_YOLO_CAM=0`) |
| 1 | Cámara cenital sobre el tablero (se ve el brazo abajo) | 1920x1080 | Vista general, apoyo manual |
| 2 | Cámara de la muñeca (pinza) | 1920x1080 | ACT y agarre (`CHESS_WRIST_CAM=2`) |
| 3 | Webcam del Mac | 1920x1080 | No se usa |

Los índices de macOS pueden cambiar al reconectar USB: reconfirma con `./.venv312/bin/python real/find_cameras.py` o `chess_cameras`. Ojo: `real/env.local.sh` define `SO101_TOP_CAM=1` y `SO101_WRIST_CAM=0` para el pipeline ACT; el MCP de ajedrez usa sus propias variables (`CHESS_YOLO_CAM`, `CHESS_WRIST_CAM`). macOS debe dar permiso de cámara a la app que lanza el shell.

## YOLO

- Detector: `runs/detect/vision/runs/piezas_camara_v2/weights/best.pt` (por defecto en `vision/chess_vision.py`; v1 en `piezas_camara/`). 12 clases `white-/black-` × pawn/knight/bishop/rook/queen/king. Dispositivo `mps`.
- Lectura con el tablero inicial (21-nov): 24 piezas vistas. Peones y ocupación bien; fila trasera mal (dama como peón, rey como alfil, faltan c1, f1, g1, h1, f8, a8). Tomar los tipos de la fila trasera de las reglas, no de YOLO.
- Reentrenar con más fotos: `vision/labeler/` y `vision/train_pieces.sh`.

## Mapeo imagen → casilla

- Geometría: `vision/board_camara0.json` (4 esquinas del área 8x8 en píxeles de la cámara 0: TL=h8, TR=h1, BR=a1, BL=a8). Esquinas actuales: TL (614,308), TR (939,295), BR (1066,563), BL (647,586).
- `chess_square_px(sq)` da el píxel del centro de una casilla (homografía). `chess_camera(overlay=True)` dibuja rejilla, piezas leídas y la punta del brazo; guarda `vision/frames/chess_mcp_see.jpg`.
- Si la rejilla no cae sobre las casillas: mide de nuevo las esquinas (`vision/board_corners.py`) y actualiza el JSON. Cámara o tablero movidos = esquinas inválidas.
- Del píxel al mundo del brazo: no hay calibración cámara→base; el mundo lo da la calibración por contacto (`hub/chess_board_real.json`, skill `so101-calibracion-tablero`).

## Herramientas (MCP `so101-chess`, no mueven nada)

`chess_cameras`, `chess_camera`, `chess_see`, `chess_square_px`, `chess_verify_move`, `arm_gripper_px`.

## Si falla

- Imagen negra/error: permiso de cámara de macOS o índice cambiado.
- `casillas_distintas` alto con el tablero inicial: revisa luz y esquinas antes de culpar a YOLO.
- Verificación falla por una pieza tapada por el brazo: espera a que el brazo esté en el retorno antes de fotografiar.
