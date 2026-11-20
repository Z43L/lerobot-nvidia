# Visión del tablero (piezas de ajedrez)

Objetivo: a partir de una imagen de la cámara cenital, obtener el estado del tablero (FEN).

Estado: **preparado sin hardware ni dataset.** El fine-tune se hace cuando haya puesto y tablero real.

## Piezas

| Fichero | Qué hace | Estado |
|---|---|---|
| `board_mapping.py` | Esquinas del tablero + detecciones → casillas → FEN | Probado con datos sintéticos |
| `test_board_mapping.py` | Tests (posición inicial, confianza, duplicados) | Pasa |
| `data.yaml.example` | Plantilla de dataset YOLO (12 clases) | Pendiente de dataset |
| `train_pieces.sh` | Fine-tune de YOLO en `mps` | Requiere instalar `ultralytics` |

## Convenciones

- Esquinas del tablero en píxeles, en orden: **a1, h1, h8, a8** (como se ven en la imagen).
- La pieza se ubica por el **borde inferior** de su caja (la base), no por el centro.
- Si dos detecciones caen en la misma casilla, gana la de mayor confianza.

## Antes del evento (ya)

1. Elegir el dataset público y **confirmar la licencia en su página**. Las copias de chess-full en Roboflow Universe varían.
2. Instalar `ultralytics` en `.venv312` (cambia el entorno: hacerlo con tu OK).
3. Descargar el dataset en `vision/datasets/` y crear `vision/data.yaml` a partir de la plantilla.
4. Entrenar un primer modelo con el dataset público para validar el pipeline (en `mps`).

## Cuando haya puesto y tablero real

1. Fijar la cámara cenital y tomar una imagen de referencia del tablero para sacar las 4 esquinas.
2. Capturar 150–300 fotos del tablero real con las piezas (distintas luces y posiciones).
3. Etiquetar esas fotos con las mismas 12 clases.
4. Fine-tune sobre el modelo ya entrenado con el dataset público.
5. Validar el FEN contra posiciones conocidas (test de extremo a extremo).

## Pendiente de decidir

- Si la visión va como servidor MCP propio (`so101-vision`) o como script que llama Claude Code.
- Qué cámara usar para el tablero. Ver `real/README.md` para la cámara `table`.
