# Dataset de piezas de ajedrez (fotos de `Camera/`)

86 fotos del tablero real (1280x720, cámara fija) con 12 clases (`black-*`, `white-*`; la clase 0 `bishop` queda sin usar para
mantener los ids de `vision/data.yaml`). Reparto: 69 train / 17 val, semilla fija.

- `images/{train,val}/`, `labels/{train,val}/`: formato YOLO (clase cx cy w h normalizados).
- `positions/*.json`: posición de cada foto (casilla -> pieza y FEN de la colocación).
- `data.yaml`: listo para `vision/train_pieces.sh vision/dataset_camara/data.yaml 50 mps`.
- `review_final/NN.jpg`: recorte de cada foto con las cajas y la casilla de cada pieza.
- `labels_auto_backup/`, `review/`: etiquetas automáticas del primer intento (detector sin revisar). Descartables.

## Cómo se etiquetó

Las posiciones y las clases las leí foto a foto mirando la imagen (no hay detector detrás). Cada foto hereda la posición de la
anterior y se editó con jugadas, capturas y altas en `vision/labeler/annotations.txt`. Luego `vision/labeler/annotate_tools.py apply`
genera las cajas: 1366 se ajustan a la caja de un detector candidato cuando coincide (IoU > 0,5) y 383 son sintéticas, calculadas
desde la casilla con proporciones calibradas (`vision/labeler/annotate_tools.py`, función `synth_box`).

- Esquinas del área de juego medidas en la foto 72 (tablero vacío); `a1` = esquina inferior-derecha de la foto (blancas a la
  derecha). Comprobado con rey/dama de la posición inicial y con las primeras jugadas.
- Las piezas se movieron a mano y no siempre con jugadas legales (hay capturas imposibles y piezas que desaparecen); las
  posiciones son lo que se ve en cada foto, no una partida válida.

## Qué no es perfecto

- Las cajas sintéticas y las ajustadas al detector pueden desviarse unos píxeles. Las piezas altas se inclinan por la perspectiva.
- Fotos con piezas muy apiladas o en el borde entre dos casillas, donde puedo haberme equivocado de casilla o de tipo
  (alfil/peón, rey/dama): 1-2 (posición inicial, se asumió la estándar), 40-41, 50-53, 73-86.
- El reparto train/val es aleatorio entre fotos casi iguales (secuencia de una misma partida): la métrica de validación sale
  optimista. Para medir de verdad, valida con otra sesión de fotos.
- Revisar o corregir: `vision/labeler/editor.html` (carga estas etiquetas; exporta JSON y `apply_corrections.py` lo vuelca).
