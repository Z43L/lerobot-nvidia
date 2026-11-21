---
name: so101-pasar-lata
description: Coge un objeto cilíndrico (lata, botella) de la mesa, lo levanta, lo lleva hacia el frente del robot y, si se pide, lo entrega en una mano localizada con la cámara del Mac. Funciona en simulación y en el SO-101 real: adapta la posición del objeto, la calibración de la mesa y la velocidad al entorno. Úsala cuando el usuario diga "coge la lata", "coge el cilindro", "pasa la lata", "dale la lata a la persona" o "localiza la mano".
---

# Coger y pasar un cilindro (SO-101, sim y real)

La skill es **una sola secuencia** que cambia según el entorno. Antes de moverse, lee los parámetros del entorno (abajo). No hay valores fijos de posición ni de velocidad en el procedimiento.

Secuencia: **reposo → localizar objeto → coger → levantar → llevar al frente → (opcional) localizar la mano → entregar → soltar → reposo.**

No hay mano robótica ni sensor de contacto. La entrega es visual: una persona junto al brazo sujeta la lata cuando se le pide.

## 1. Parámetros del entorno (leer antes de nada)

| Parámetro | Sim | Real | De dónde sale |
|---|---|---|---|
| `backend` | `sim` | `real` | variable `SO101_BACKEND`; si no está, pregunta al usuario |
| Servidor MCP | `so101-sim-objects` | `so101-motion` (preview + confirmación) | según backend |
| Posición del objeto | `list_objects` → `bottle` / `cup` / `box` | cámara de mesa + `hub/hand_calibration.json` | ver paso 2 |
| Base del robot | `(0, 0, 0)` en metros | marco definido en `hub/hand_calibration.json` (`frame`) | sim fija; real del JSON |
| Velocidad | `seconds` de `move_tcp` (más alto = más lento) | tramos de `TRANSIT_STEP` = 4° por 1/30 s, valores de `real/real_arm.py` | ver paso 5 |
| Alcance | `r` ≤ ~0,18 m con la pinza vertical | igual, comprobado con el JSON | ver paso 3 |

**Regla:** nunca uses una posición fija del proyecto. Lee `list_objects` (sim) o la calibración (real) en cada uso.

## 2. Localizar el objeto

- **Sim:** `list_objects`. Usa el objeto cilíndrico (`bottle`, radio 2 cm, alto 10 cm) como lata. Su posición inicial es (0,17; −0,06), pero puede cambiar: lee siempre la actual.
- **Real:** cámara de mesa (webcam del Mac, índice `SO101_TABLE_CAM`). Captura con `hub/hand_calibration.py capture`, localiza el centro de la base del cilindro en píxeles `(u, v)` y convierte con `pixel_to_table(u, v)`. Sin `hub/hand_calibration.json` **no se coge nada en real**.

## 3. Comprobar alcance antes de mover

Con la pinza vertical, la muñeca llega a su tope cerca de radio 0,18 m, y por encima de ~0,09 m de altura el alcance se pierde. Antes de la primera orden:
- `r = sqrt(x² + y²)` del objeto. Si `r` > 0,18 m, no lo intentes con la pinza vertical: dile al usuario que lo acerque o lo cambie de sitio.
- Altura de viaje `z_travel` = 0,085 m (no subas más; el alcance se pierde).
- Altura de agarre `z_grasp` = 0,03 m (ajusta según el objeto: la lata real mide 0,066 m, así que su agarre puede ser más alto).

## 4. Coger (sim o real)

1. **Sim:** `reset_objects` para una escena limpia.
2. Sobre el objeto, a `z_travel`, con la pinza abierta (`GRIP_OPEN` = 1,5 rad en sim).
3. Baja a `z_grasp`.
4. Cierra la pinza (`GRIP_CLOSED` = −0,1745 rad en sim).
5. Levanta a `z_travel` con la pinza cerrada.
6. Comprueba que el objeto va con la pinza:
   - Sim: `pick` devuelve `held: true`. Si `false`, **no sigas**.
   - Real: pide a la persona que confirme que la lata se ha levantado. No hay sensor de agarre.

Si falla: vuelve a reposo, recalcula la posición con el paso 2, y reintenta **una vez**. Si vuelve a fallar, para y dilo al usuario.

## 5. Velocidad

- **Sim:** usa `move_tcp` con `seconds` según la fase: aproximación 2,0 s, descenso 1,2 s, subida 1,5 s. Son valores de la sim, no del real.
- **Real:** el avance lo limita `TRANSIT_STEP` (4°) en `real/real_arm.py`, y `motion_preview` enseña el tiempo estimado. Para una prueba inicial, baja `SO101_MAX_RELATIVE_TARGET` a 2 (ver §7).

Ajusta la velocidad en el entorno, no en el procedimiento: si el brazo va lento en sim, sube `seconds`; si el real va brusco, baja `SO101_MAX_RELATIVE_TARGET`.

## 6. Entrega (opcional, solo si el usuario la pide)

1. Lleva el objeto sujeto a `lata_al_frente` (x mayor, y ≈ 0) a `z_travel`.
2. Localiza la palma con la cámara de mesa y convierte con `pixel_to_table`. Supón la palma a `z_mano` = 0,10 m (de `hub/hand_calibration.json`, `hand_z_m`).
3. Si la palma está fuera de alcance (`r` > 0,18 m), no muevas el brazo.
4. Lleva la lata a 3 cm delante de la palma, sin tocarla.
5. Pide a la persona que sujete la lata por el cuerpo y confirme en el chat.
6. Suelta (abre la pinza).

## 7. Seguridad (real)

- El brazo real se mueve directamente (`ALLOW_REAL_MOTION=1` ya está en el entorno).
- `SO101_MAX_RELATIVE_TARGET=2` por defecto.
- Sin comprobaciones previas.
- Si el brazo se mueve raro, llama a `motion_stop` de inmediato.
- Si el brazo se mueve raro, para.
- Un movimiento se previsualiza y se confirma **por fase**, no todo de una vez.

## 8. Estado de verificación (dilo al usuario)

- **Sim, agarre por fricción: no funciona.** `objects_env.py --grasp-test` devuelve `held: false` con la pinza original (la carcasa y las mordazas bloquean la botella tumbada o de pie). Se probaron almohadillas de agarre y botella tumbada (`bottle` con `lying: True` en `OBJECTS`), sin éxito. Los cambios en la pinza se revirtieron.
- **Sim, enganche temporal (`SO101_GRASP=attach`): funciona.** `pick` engancha el objeto al TCP y `place` lo suelta al abrir la pinza. Para `bottle`, `cup` y `box` devuelve `held: true`. El objeto no colisiona durante la aproximación y no es física real: sirve para validar la lógica de la skill, no el agarre. Sin la variable, la sim usa la física (por defecto).
- **Sim vía MCP:** el servidor `so101-sim-objects` se lanza con el entorno de `.mcp.json`. Para usar el enganche, hay que añadir `SO101_GRASP=attach` a ese bloque `env` y reiniciar el servidor (cambio pendiente de aprobación).
- **Real:** nada probado. Falta `hub/hand_calibration.json` (la cámara de mesa aún no ve los 4 marcadores).
- La lata real no está modelada en la sim; usa `bottle` como sustituto.
- La altura de la palma (`z_mano`) es una suposición hasta confirmarla.
