---
name: so101-calibracion-tablero
description: Calibra el tablero de ajedrez real respecto a la base del SO-101 por contacto (la punta de la pinza toca a1, h1 y a6, comprueba h6) con hub/board_touch_calib.py, y genera hub/chess_board_real.json para que el objetivo x,y de ACT sea el centro de la casilla. Úsala cuando el usuario diga "calibra el tablero", "dónde está a1", "mide las casillas", "centro de la casilla", o al montar el tablero real en el puesto, antes de grabar o desplegar ACT con objetivo.
---

# Calibración del tablero real por contacto

**Estado:** el script `hub/board_touch_calib.py` existe y está probado en sim (sin ruido y con ruido de medida). **Pendiente de ejecutar** con el brazo real y el tablero montado.

## Qué resuelve

ACT necesita un objetivo `x,y` en metros, en el marco de la base del SO-101, igual que en sim. La visión (`vision/board_mapping.py`) da la casilla, no los metros. Esta calibración convierte casilla → metros:

- `square_xy(casilla)` devuelve el centro de la casilla en metros desde `hub/chess_board_real.json`.
- El fichero tiene el mismo formato que `hub/chess_board.json` (sim), con `"status": "real"`.
- Sim y real quedan separados, cada uno con su fichero.

## Reglas que no se saltan

1. **Tocar con el líder mueve el brazo real.** Necesita `ALLOW_REAL_MOTION=1` y que el usuario escriba `SI` (igual que cualquier 🔒). Área libre, persona junto al brazo, corte de alimentación a mano.
2. Velocidad: `SO101_MAX_RELATIVE_TARGET=2` (en `real/env.local.sh`).
3. Toca **suave**, a unos milímetros del tablero. No lo golpees.
4. Si la pinza choca con una pieza o se aleja de la casilla, para y vuelve a empezar desde la pose de reposo.
5. **La lectura (`read`) no mueve el brazo**, pero el toque sí. Lee solo cuando la pinza esté quieta.

## Casillas de calibración (importante)

- **Calibran:** a1, h1 y a6. **Comprueba:** h6.
- **No uses la fila 8.** Con la pinza vertical a 2 cm de altura, la fila 8 y las esquinas de la 7 **no llegan** (en sim, error de IK de 5–8 mm). Lo comprobé con la cinemática: las filas 1–6 llegan.
- Si el tablero real está más lejos del robot que en sim, puede que ni la fila 6 llegue. Antes de calibrar, comprueba el alcance (paso 1).

## Pasos

### 1. Medir el tablero y el alcance (sin mover el brazo)

- Lado de una casilla en metros (de borde a borde / 8). **No asumas 3 cm**: la sim usa 3 cm, pero un tablero real suele tener casillas de 4–6 cm.
- Orientación: qué esquina es a1 respecto al robot (normalmente la de abajo-izquierda visto desde el frente del robot).
- Anotar en `CHECKLIST.md` §5.0.
- **Alcance:** a1, h1 y a6 deben estar dentro del alcance de la pinza a 2 cm de altura. Si no llegan, para: mueve el tablero o el robot antes de calibrar.

### 2. Tocar las casillas con el líder

Para cada casilla (a1, h1, a6 y comprobación h6), en este orden:

1. Llevar la punta de la pinza justo encima del centro de la casilla, a unos 5 mm, y bajar hasta tocar suavemente.
2. Con la pinza quieta, leer las articulaciones (no se mueve nada):
   ```bash
   source real/env.local.sh
   ./.venv312/bin/python hub/board_touch_calib.py read --square a1 --touches hub/touches_real.json
   ```
   Repite con `h1`, `a6` y `h6`.
3. Si una lectura sale mal (la punta no estaba en el centro), **repite esa casilla**: `read` añade una entrada nueva y el cálculo usa la última de cada casilla.

**Precisión:** un error de 0,5° por articulación da unos 6 mm en el resultado; con 1° llega a 13 mm. Por eso el toque tiene que ser preciso. Si el resultado no cumple, repite las medidas antes de forzar la escritura.

### 3. Calcular el tablero

```bash
./.venv312/bin/python hub/board_touch_calib.py fit --touches hub/touches_real.json --out hub/chess_board_real.json
```

- Usa la cinemática directa del modelo de sim (`ObjectsSim`), que es el mismo MJCF del SO-101. Da la punta en el marco de la base.
- Muestra el lado de la casilla, la ortogonalidad y el error en h6.
- **Solo escribe si el error de h6 es menor de 5 mm.** Si no, repite las medidas. `--force` existe, pero no lo uses en el puesto sin decírselo al usuario.

### 4. Verificar la cinemática antes de confiar en ella

- Con el brazo en reposo, `fk` debe dar una posición coherente con el brazo real. Si no coincide, **para**: los grados de LeRobot y los del modelo no están alineados y el resultado sería incorrecto.
  ```bash
  ./.venv312/bin/python hub/board_touch_calib.py fk --deg <pan,lift,elbow,wrist_flex,wrist_roll>
  ```
- Prueba sin agarre: lleva la pinza a 3 casillas al azar y comprueba que cae en el centro (con `SI`).

### 5. Usarlo

- `square_xy` lee el fichero y devuelve el centro de la casilla en metros.
- El objetivo de ACT (`--goal x,y`) es ese centro. La visión dice la casilla; el fichero dice dónde está.

## Comandos de referencia

```bash
# Cinemática directa (sin brazo)
./.venv312/bin/python hub/board_touch_calib.py fk --deg 0,0,0,0,0
# Prueba completa en sim (sin ruido y con ruido de medida). Escribe hub/chess_board_selftest.json (no es el real)
./.venv312/bin/python hub/board_touch_calib.py selftest --noise-deg 0.5
```

## Lo que no está probado

- No se ha ejecutado con el brazo real ni con un tablero real.
- `read` lee el seguidor real, pero no se ha probado con hardware.
- La cinemática de sim frente al brazo real no está verificada todavía (paso 4).
- El cementerio real no se calibra aquí: `graveyard_m` queda vacío.
- La cámara de mesa (`hub/hand_calibration.py`) es una alternativa, pero necesita marcadores de posición conocida y no es la opción por defecto.

## Antes del hackathon (en casa)

- Ya hecho: prueba en sim de fk, ajuste y comprobación, con ruido de 0 y 0,5°.
- Pendiente: decidir el lado real de la casilla y actualizar la sim (`SQ` en `sim/chess_sim.py`) si es distinto de 3 cm, antes de grabar datos de sim.
