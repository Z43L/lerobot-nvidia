---
name: so101-ajedrez-partida-real
description: Cómo se juega una partida real de ajedrez contra el SO-101, turno a turno, con el método probado el 21-nov-2026. Turno del humano por cámara ("ya" → chess_see → verificar → registrar), turno del robot con el ejecutor IK (run <uci> --grasp scripted) que vuelve al origen, verificación con la cámara, qué jugadas puede hacer el brazo hoy y qué hacer cuando falla. Úsala cuando el usuario diga "empieza la partida", "juguemos", "partida nueva", "empieza de 0", "ya", "te toca", "mueve e4", "estado de la partida", "termina la partida" o "documenta la partida".
---


# Partida real, turno a turno: método probado (21-nov-2026, noche)

Robot = blancas, humano = negras. Las reglas de ajedrez las lleva `hub/chess_game.json` (MCP `so101-chess`). Los detalles del agarre están en `so101-ajedrez-agarre`, la vuelta al retorno en `so101-ajedrez-brazo-retorno` y la visión en `so101-ajedrez-camaras-yolo`. Todo se anota en `PARTIDA_AJEDREZ_REAL.md`.

## 0. Antes de empezar

```bash
source real/env.local.sh      # puertos, cámaras (YOLO=0, muñeca=2), ALLOW_REAL_MOTION=1
```
- **El puerto del brazo solo admite un proceso a la vez.** El ejecutor (`hub/chess_ik_exec.py`) y las herramientas `arm_*` del MCP se excluyen. Para mover, usa el ejecutor. Del MCP usa solo las herramientas que no tocan el brazo: `chess_see`, `chess_camera`, `chess_verify_move`, `chess_commit`, `chess_new` y `chess_state`.
  Si el puerto está ocupado, `lsof /dev/cu.usbmodem5B610353951` dice quién lo tiene. Si es otra sesión de Claude, **pregunta antes de cerrarla**.
- **No muevas el brazo con `arm_move` desde abajo.** El MCP deja el codo con P=16 y no sube. `Arm` del ejecutor pone P 28 en el codo y 44 en el hombro, y sube por etapas (`Arm.lift`).
- **El brazo debe estar en el origen** (`hub/chess_home.json`, ≈ `[-7,9; -17,8; 13,5; 100; 165; *]`). Si no está a menos de 8°, el ejecutor no arranca.

## 1. Partida nueva

1. `chess_new(robot_white=true)`.
2. `chess_see`. Tienen que coincidir los 16 peones. Que falten piezas altas de la fila trasera (dama, rey, alfiles, torre de h1) es normal: YOLO las pierde.
3. Si un peón no está en su casilla, **pide al usuario que lo coloque a mano**. No lo recojas desde la fila 4 o más lejos (ver §4).
4. El robot abre. **Jugadas probadas:** `e2e4` (agarra en el centro) y `d2d4` (agarra a +3 mm).

## 2. Turno del robot

1. Elige una jugada que el brazo pueda hacer (§4). Si la mejor jugada no se puede, elige otra razonable y dilo.
2. Ejecútala:
   ```bash
   ./.venv312/bin/python hub/chess_ik_exec.py run <uci> --grasp scripted
   ```
   `run` parte de la FEN guardada. Lee las líneas de cada etapa:
   - `[vecinas]`: desplazamiento hacia la casilla libre de la columna.
   - `[fila]`: corrección por fila.
   - `[muneca]`: medida de la cámara de la muñeca.
   - `[agarre n]`: desplazamiento, error de bajada, valor de la pinza y si está AGARRADA o vacía.
   - `[subida]`, `[destino]`, `[soltado]`.
3. **Mira el tablero:** `chess_verify_move <uci>` y `chess_camera(0, overlay=True)`. Los avisos de piezas altas que no se ven no cuentan.
4. Solo si `ok: true`, ejecuta `chess_commit <uci>`.
5. **Si falla** (`ok: false`), el brazo vuelve solo al origen. Mira la cámara: ¿la pieza sigue en su sitio y de pie? No repitas la misma jugada en bucle: cambia de jugada o diagnostica (§5).

## 3. Turno del humano

1. Pídele que **mueva la pieza en el tablero y diga "ya"**. No te fíes de las coordenadas que escriba. Medido el 21-nov: "b1 a f5", "c1 a b3" o "f1 a f4" no eran jugadas legales, y al pedirle e4 puso el peón en **d4**. Las letras impresas del tablero están espejadas respecto al sistema.
2. Con "ya": `chess_see` → busca en `diferencias` qué casilla se ha vaciado y cuál se ha ocupado → `chess_verify_move <uci>` → `chess_commit <uci>`.
3. Si la cámara no ve ningún cambio, o lo que ve no es legal, **no lo registres**. Di lo que ves y pregunta.
4. `chess_commit` guarda todas las jugadas como `who: robot`, también las del humano. Es un fallo conocido y no afecta a las reglas.

## 4. Qué puede hacer el brazo hoy (21-nov)

| Jugada | Estado |
|---|---|
| Peón desde la fila 2 → filas 3 y 4 | ✅ e2e4 (centro), d2d4 (+3 mm), f2 test (+3 mm). Pinza a 12 (~2 cm) |
| Coger desde la fila 4 o más lejos | ❌ El peón queda entre los dedos, pero el dedo móvil lo **empuja**. e4/d4: 0/14 cierres |
| Capturas (pieza comida al cementerio) | Sin probar (`graveyard_m` sin calibrar) |
| Enroque, piezas que no sean peones | Sin probar (apertura 15–17 sin medir) |

Con peón, la pinza marca 4,5–5,3. El umbral `GRIP_EMPTY_MAX` = 4,5 está muy justo: si el agarre sale `vacia` con 4,5, mira la foto.

## 5. Cuando algo falla: diagnóstico

1. **Antes de cualquier sonda (`rank-probe`, `grasp-probe`, `wrist-measure`), mira el tablero.** El 21-nov se probó sobre e4 vacía porque el usuario había recolocado las piezas: medida perdida.
2. Pon la pinza abierta 1 cm por encima o en la pose de agarre, **déjala quieta** y saca fotos con la cámara de la muñeca (2) y la lateral (0).
3. Si con las fotos no basta, pide al usuario que mire y, cuando diga "cierra", cierra despacio (`Arm.grasp`) mientras mira. Así se vio que el dedo empuja el peón.
4. Para volver: `Arm.lift` (primero el hombro, luego el codo) a la pose de encima y después al origen.
5. Herramientas de medida (`so101-ajedrez-agarre`): `rank-probe`, `rank-table`, `wrist-ref`, `wrist-measure`, `test-grasp --go`.

## 6. Terminar la partida

1. Brazo en el origen.
2. Copia `hub/chess_game.json` a `hub/chess_game.final_<fecha>.json`.
3. Anota en `PARTIDA_AJEDREZ_REAL.md` las jugadas, qué falló y lo que queda pendiente.
4. Para empezar de 0: el usuario recoloca las piezas, `chess_new` y luego `chess_see` (§1).

## 7. Orientación y comandos alternativos (sigue valiendo)

- Robot = blancas, dama blanca en `d1`, `a1` lejos del robot y `h1` junto a él. **Las etiquetas impresas del tablero real están espejadas**: lo impreso `e2` es `d2` para el sistema. Habla siempre con las etiquetas del sistema.
- Sin MCP: `./.venv312/bin/python sim/chess_cli.py see --camera 0`, `CHESS_BACKEND=real ./.venv312/bin/python sim/chess_cli.py new --robot-white`, `chess_cli.py robot` (minimax de profundidad 2), `chess_cli.py human-seen --camera 0`.
- Plan sin mover: `chess_plan <uci>` o `./.venv312/bin/python hub/chess_ik_exec.py plan <uci>`.
- No muevas la cámara ni el tablero ni cambies la luz durante la partida: las esquinas de `vision/board_camara0.json` dejan de valer.
- La partida está en `hub/chess_game.json` y no se edita a mano. Para pruebas, `CHESS_GAME_FILE=/otra/ruta.json`.
- Para ver el tablero desde la cenital (cámara 1), gira `shoulder_pan` a -55: en el origen el brazo la tapa.
