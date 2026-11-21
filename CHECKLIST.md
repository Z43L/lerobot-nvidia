# Checklist del hackathon (SO-101 + ACT + visión + Claude)

Sigue el orden. Marca cada casilla al completarla. Si una etapa falla, **para** y resuelvela antes de avanzar.
Las reglas de `AGENTS.md` mandan siempre.

Leyenda: 🔒 mueve el brazo real (movimiento directo, sin confirmación; `ALLOW_REAL_MOTION=0` lo para todo) · 🧠 entrena · ⚠️ bloqueado / no probado

---

## 0. Antes de llegar (ya)

- [ ] Preguntar a la organización:
  - [ ] ¿Hay GPU NVIDIA? (si sí, `cuda` para entrenar; si no, `mps`).
  - [ ] ¿El brazo viene calibrado? ¿Cuál es el **ID de calibración** del seguidor y del líder?
  - [ ] ¿Qué cámaras ponen (modelo, si aceptan 320x240) y dónde va la cámara cenital?
  - [ ] ¿Qué iluminación hay en el puesto y se puede mantener toda la jornada?
  - [ ] ¿Se puede usar la simulación en algún momento? (el FAQ dice "sin simulación de por medio").
  - [ ] ¿Cuál es la forma de entrega (dataset, código, demo) y el tiempo por equipo?
- [ ] Copiar `real/env.sh` a `real/env.local.sh` y dejar preparados los IDs de calibración.
- [ ] Ejecutar `./.venv312/bin/python pipeline/vla_pipeline.py check --backend sim` (sin brazo).
- [ ] Llevar: portátil cargado, cable USB **de datos**, cámaras de repuesto si las hay, cinta métrica, pieza de prueba.

## 1. Llegar al puesto (sin mover el brazo)

- [ ] Conectar líder y seguidor. Anotar qué puerto es cada uno.
- [ ] Rellenar en `real/env.local.sh`:
  - [ ] `SO101_FOLLOWER_PORT`, `SO101_LEADER_PORT` (`lerobot-find-port` si hace falta).
  - [ ] `SO101_FOLLOWER_ID`, `SO101_LEADER_ID` (los de la organización).
  - [ ] `SO101_TOP_CAM`, `SO101_WRIST_CAM`, y `SO101_TABLE_CAM` si hay cámara de mesa (`real/find_cameras.py`, mirar las fotos).
  - [ ] `SO101_MAX_RELATIVE_TARGET=2`.
- [ ] Dar permiso de cámara a la Terminal (Ajustes del Sistema > Privacidad > Cámara).
- [ ] `source real/env.local.sh && ./.venv312/bin/python pipeline/vla_pipeline.py check --backend real` → todo en **OK**.
- [ ] Comprobar la resolución: las cámaras deben abrir a 320x240. Si no, **para**: hay que cambiar la resolución en `real/record_goal.py` y en `real/rollout.sh` antes de grabar.
- [ ] Comprobar en grados: mover una articulación **a mano** (sin 🔒) y leer las posiciones. Deben salir en grados (shoulder_lift puede pasar de 100) y la pinza en 0..100.

## 2. Primera prueba 🔒

- [ ] Ejecutar:
  ```bash
  source real/env.local.sh
  ALLOW_REAL_MOTION=1 ./.venv312/bin/python real/record_goal.py --name test1 --episodes 1
  ```
- [ ] Revisar el episodio: ¿el brazo se mueve suave, sin golpes ni saltos? Si no: `ALLOW_REAL_MOTION=0` (o Ctrl+C) y revisar.
- [ ] `./.venv312/bin/python pipeline/vla_pipeline.py validate --backend real --run test1 --policy act --min-episodes 1` → **OK**.

## 3. Grabación real 🔒

- [ ] Fijar la cámara cenital y la luz. **No cambiar la luz durante la grabación.**
- [ ] Hacer la primera imagen de referencia del tablero (para la homografía de la visión).
- [ ] Grabar 25–30 episodios, con piezas y posiciones **variadas**:
  ```bash
  ALLOW_REAL_MOTION=1 ./.venv312/bin/python real/record_goal.py --name demos --episodes 30
  ```
- [ ] Anotar el objetivo `x,y` (en metros, mismo marco que la visión) de cada episodio. Si el objetivo no sale de la visión, **para** y resuélvelo antes de seguir.
- [ ] Validar: `./.venv312/bin/python pipeline/vla_pipeline.py validate --backend real --run demos --policy act --min-episodes 20` → **OK**.
- [ ] Revisar `pipeline/runs/demos/state.json` y anotar el tiempo de grabación por episodio.

## 4. Entrenamiento 🧠

- [ ] Prueba corta para validar el flujo:
  ```bash
  ./.venv312/bin/python pipeline/vla_pipeline.py train --backend real --run demos --policy act --steps 500 --device mps
  ```
  (usa `cuda` solo si la organización confirma GPU NVIDIA).
- [ ] Anotar segundos por paso en `pipeline/runs/demos/state.json` y calcular el tiempo del entrenamiento completo (referencia medida en sim: ~0,9 s/paso a 240x320 con batch 8).
- [ ] Entrenamiento completo con los pasos que decida el usuario según el tiempo disponible.
- [ ] Guardar la ruta del checkpoint `real/checkpoints/act_demos_<fecha>/checkpoints/last/pretrained_model`.

## 5. Visión y tablero (en paralelo)

### 5.0 Calibración del tablero real ⏳ PENDIENTE (en el puesto, con brazo y tablero montados)

Skill: `so101-calibracion-tablero`. Sin esto, el objetivo `x,y` que pases a ACT no corresponde a la casilla que quieres.

- [ ] Medir el lado de la casilla real (no asumir 3 cm) y anotarlo: ______ m.
- [ ] Comprobar que a1, h1 y a6 llegan a 2 cm de altura (**no usar la fila 8**: no llega con la pinza vertical).
- [ ] Verificar la cinemática de sim frente al brazo real: `hub/board_touch_calib.py fk --deg ...` con el brazo en reposo. Si no coincide: **para**.
- [ ] 🔒 Tocar con el líder a1, h1, a6 y comprobación h6 . Tras cada toque, leer con `hub/board_touch_calib.py read --square <casilla> --touches hub/touches_real.json` (no mueve el brazo).
- [ ] Calcular: `hub/board_touch_calib.py fit --touches hub/touches_real.json --out hub/chess_board_real.json`. Solo escribe si el error de h6 es < 5 mm; si no, repetir las medidas.
- [ ] Probar `square_xy` con 3 casillas al azar.
- [ ] Si se cambia el lado de la casilla, actualizar la sim (`SQ` en `sim/chess_sim.py`) antes de grabar datos de sim.

- [ ] Pesos del detector: `runs/detect/vision/runs/pieces/weights/best.pt` (mover a `vision/` cuando haya un hueco).
- [ ] Probar el detector sobre una foto real del tablero y comprobar que el FEN es correcto.
- [ ] Fijar las 4 esquinas del tablero (a1, h1, h8, a8) en píxeles.

## 6. Despliegue ⚠️

`real/deploy_goal.py` existe y se ha probado solo en `--dry-run` (sin brazo). El bucle con el brazo **no** se ha ejecutado todavía.

- [ ] Comprobación sin brazo: `./.venv312/bin/python real/deploy_goal.py --checkpoint <ruta> --goal 0.19,0.04 --dry-run` → **checkpoint OK**.
- [ ] Prueba corta 🔒 (15–20 s, `SO101_MAX_RELATIVE_TARGET=2`):
  ```bash
  ALLOW_REAL_MOTION=1 ./.venv312/bin/python pipeline/vla_pipeline.py deploy --policy act --checkpoint <ruta> --goal 0.19,0.04 --seconds 20 --run demos
  ```
- [ ] Si la prueba corta falla o el brazo se comporta raro: `ALLOW_REAL_MOTION=0` y **plan B** (§8). No improvises con `lerobot-rollout`: no envía el objetivo.

## 7. Evaluación

- [ ] Probar 10 intentos por casilla (o por posición de objeto) y anotar éxito/fallo.
- [ ] Comparar con la línea base (movimiento sin aprendizaje). Si ACT no mejora, usar la línea base en la demo.

## 8. Plan B (siempre listo)

- [ ] Movimiento grabado para la casilla o el objeto: skill `so101-motion-library` (o `so101-grabacion-maestra`).
- [ ] Cinemática inversa + interpolación hacia la casilla, con pinza abierta/cerrada.
- [ ] La visión y Claude siguen decidiendo la jugada; el movimiento lo hace el plan B.

## 9. Cierre

- [ ] Copiar checkpoints, datasets, `pipeline/runs/` y logs a un disco o a una carpeta aparte.
- [ ] Anotar: episodios grabados, tiempo por episodio, pasos entrenados, tasa de éxito, incidencias.
- [ ] Preparar la entrega que pida la organización (dataset, código, demo).
- [ ] Apagar el brazo y desconectar la alimentación.
