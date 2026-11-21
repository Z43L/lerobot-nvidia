# Informe de fallos: arquitectura del ajedrez real frente a `origin/main`

Fecha: 2026-11-21. Base analizada: `origin/main` (45ae168). Referencia: los diagramas de arquitectura (secuencia
Persona → Claude → Ajedrez → MCP → ACT → Brazo y vista por capas).

Cada fallo indica la **evidencia** (cómo se comprobó), su **impacto** en el robot y su **estado** en esta rama.
Estados: ✅ corregido y verificado · 🟡 corregido, falta probarlo en el host · ⛔ pendiente.

---

## Resumen

| # | Fallo | Impacto | Estado |
|---|---|---|---|
| 1 | Falta el backend real (`chess_real.py` era un stub de 25 líneas) | Bloqueante: no hay jugada real | ✅ |
| 2 | La CLI no tenía las funciones que usan `chess_match.py` y el MCP | Bloqueante: el bucle de partida no arranca | ✅ |
| 3 | `chess_commit` ("registrar") siempre fallaba | Bloqueante: no se puede registrar una jugada | ✅ |
| 4 | El MCP `so101-chess` no estaba en `.mcp.json` | Claude no podía orquestar | ✅ |
| 5 | El diagrama pide una orden de alto nivel por jugada; el MCP solo tenía herramientas sueltas | Flujo distinto al diseñado | ✅ |
| 6 | El gate de seguridad se comprobaba después de preparar el ejecutor | Seguridad | ✅ |
| 7 | El ACT de 30k (entrenado en CUDA) no cargaba en un Mac | Bloqueante para el ACT en el host | ✅ |
| 8 | Las rutas de los pesos ACT apuntaban a carpetas vacías | Bloqueante: no hay ACT | 🟡 |
| 9 | `chess_plan` devolvía coordenadas de la sim, no metros reales | El brazo iría a la casilla equivocada | ✅ |
| 10 | Tres estrategias de agarre incompatibles, ninguna oficial | Confusión, trabajo perdido | ✅ decidido |
| 11 | Hardcodeos: puertos, cámaras, rutas, límites | Fallos según el puesto | ✅ |
| 12 | `ALLOW_REAL_MOTION=1` fijado en ficheros | Seguridad (contra `AGENTS.md`) | ✅ |
| 13 | Límite de velocidad incoherente (2, 2,6 y 5) | Seguridad | ✅ |
| 14 | Documentación y skills contradictorias | Agentes y personas toman decisiones equivocadas | ✅ |
| 15 | El entorno `.venv312` no existía con ese nombre | Ningún MCP arrancaba | ✅ |
| 16 | Faltan paquetes en `lerobot312` | Sin visión, sin MCP | 🟡 |
| 17 | Agarre sin medir | Puede no coger las piezas | ⛔ |
| 18 | Visión con la cámara del puesto sin validar | Falsos rechazos paran la partida | ⛔ |
| 19 | Nada del movimiento real se ha probado | Riesgo en la demo | ⛔ |

---

## Detalle

### 1. Falta el backend real
- **Evidencia:** en `origin/main`, `sim/chess_real.py` solo tenía `RealChessBackend.__init__` con `raise NotImplementedError`. `sim/test_chess_real.py`, `hub/chess_mcp.py` (`chess_plan`) y `sim/chess_match.py` importan `DryRunExecutor`, `ACTExecutor` y `Geometry`, que no existían. El test fallaba con `ImportError`.
- **Causa:** el commit `06fcab4` subió `chess_mcp.py`, `chess_ik_exec.py`, `chess_match.py`, `test_chess_real.py` y `chess_vision.py`, pero no `chess_real.py` ni `chess_cli.py`. Esas versiones solo existían en otra máquina (`CONTEXTO_PIPELINES_SESION.md` se escribió allí).
- **Corrección:** `sim/chess_real.py` reescrito con planificación (captura al cementerio, al paso, enroque en dos pasos, promoción), `Geometry` (calibración real o de la sim; con `require_real` se niega sin calibración) y ejecutores `dry`, `ik`, `ik-scripted` y `act`.
- **Verificación:** 7 de 8 tests pasan. El 8º necesita `ultralytics` (fallo 16).

### 2. La CLI no tenía lo que usan los demás módulos
- **Evidencia:** `chess_match.py` llama a `C.backend_name`, `C.cmd_human_seen` y `C.vision_verifier`, y `_apply(..., verify=)`. Nada de eso existía. La skill `so101-ajedrez-real` documenta `see`, `human-seen` y `--verify-camera`, que tampoco existían.
- **Corrección:** se añadieron a `sim/chess_cli.py`, junto con `commit`. En backend real, las jugadas del humano solo se registran: el brazo no las ejecuta.

### 3. "Registrar" siempre fallaba
- **Evidencia:** `chess_commit` llama a `chess_cli.py move` con `CHESS_BACKEND=real`, y `run_on_backend` salía con "todavía no está implementado".
- **Corrección:** comando `commit`, que registra sin ejecutar ni mover nada. `chess_commit` lo usa.
- **Verificación:** probado arrancando el MCP: `chess_commit("e2e4")` deja `e4` en el historial.

### 4. El MCP `so101-chess` no estaba registrado
- **Evidencia:** la skill decía "en `.mcp.json`", pero no estaba.
- **Corrección:** registrado, con las mismas rutas del host (`./.venv312/bin/python`).

### 5. Falta la orden de alto nivel del diagrama
- **Evidencia:** el diagrama manda "ir a h5" al MCP y este encadena mapa → ACT → soltar → pose final → fotos. El MCP solo ofrecía `arm_jog`, `arm_servo` y `chess_act_step` (este último exige un ACT con objetivo que no existe).
- **Corrección:** `chess_move(uci, executor)`: una llamada por jugada, que devuelve la foto y la verificación pero **no registra**. Después, `chess_commit`.

### 6. El gate de seguridad iba tarde
- **Evidencia:** con `executor="ik"` y sin permiso, el MCP construía el ejecutor (que lee la calibración y prepara el puerto) **antes** de comprobar el permiso, y devolvía un error en bruto. Se encontró al probar el MCP.
- **Corrección:** el permiso se comprueba primero. La respuesta es `blocked` sin tocar nada.

### 7. El ACT de 30k (CUDA) no cargaba en un Mac
- **Evidencia:** `load_policy` solo cambiaba `cfg.device`, pero `policy_preprocessor.json` también guarda `device_processor: {"device": "cuda"}`. Error: `Requested device 'cuda' but CUDA is not available`.
- **Corrección:** `preprocessor_overrides` y `postprocessor_overrides` con el dispositivo actual, en `real/deploy_goal.py:load_policy` y en `pipeline/offline_eval.py`.
- **Verificación:** los dos modelos cargan y predicen en MPS. Error medio frente al dataset limpio: HF-30k 2,0°, local-25k 3,4°. Es evaluación **dentro de la muestra**: mide el ajuste, no el éxito del agarre. Detalle en `pipeline/MODELOS_ACT.md`.

### 8. Las rutas de los pesos apuntaban a carpetas vacías
- **Evidencia:** `act_tray_a100_100k_ckpts/` y `act_tray_30k_ckpts/` están vacías. `modeloACT/` está en `.gitignore`. El checkpoint de 200 pasos de la otra rama es un puntero de Git LFS (134 bytes).
- **Corrección:** registro de modelos en `hub/chess_config.py` (`ACT_MODELS`), con variables para cada ruta y `CHESS_ACT_CHECKPOINT` para elegir.
- **Pendiente:** copiar los pesos al host con `rsync` (`pipeline/MODELOS_ACT.md`).

### 9. El plan usaba coordenadas de la sim
- **Evidencia:** `chess_plan` documentaba "coordenadas = etiquetas de la geometría de la SIM (no metros reales)". `goal_utils.square_label_xy` lee `hub/chess_board.json` (sim).
- **Corrección:** `Geometry` usa `hub/chess_board_real.json` si existe. `chess_plan` devuelve `geometria_real`.
- **Pendiente:** `real/goal_utils.py` sigue leyendo el mapa de la sim; solo afecta al pipeline de ACT con objetivo.

### 10. Tres estrategias de agarre incompatibles
- **Evidencia:** (a) el pipeline de `AGENTS.md`, ACT con objetivo `x,y` de 2 valores; (b) `chess_act_step`, ACT con objetivo de 4 valores; (c) `chess_ik_exec.py`, IK + ACT genérico. No existe ningún checkpoint con objetivo, y los dos ACT disponibles son genéricos.
- **Decisión:** la vía oficial es la (c), que es la del diagrama. Se expone como ejecutor `ik`, con `ik-scripted` (agarre guionizado) como plan B. El ejecutor `act` (con objetivo) solo hace preflight y se niega a mover.

### 11. Hardcodeos
- **Encontrados:**
  - Puertos `/dev/cu.usbmodem…` en `.mcp.json` y como defecto en `chess_ik_exec.py`.
  - Cámara de muñeca 2 en el ajedrez y 1 en el resto.
  - Ruta `/Users/david/Library/...` en `pruebas_act/act_model.py`.
  - Rutas de pesos fijas.
  - Límites del tablero fijos (`0.19 <= x <= 0.46`) en la comprobación de paso por encima de las piezas.
  - Dos copias de `motion_permission`.
- **Corrección:**
  - Una sola fuente, `hub/chess_config.py`, con prioridad entorno, luego `real/env.local.sh`, luego defecto. Los puertos no tienen defecto.
  - `real/config.py` carga `real/env.local.sh` (lo que escribe `so101-hub write_config`).
  - Los límites del tablero salen de la calibración.

### 12. `ALLOW_REAL_MOTION=1` fijado en ficheros
- **Evidencia:** en `.mcp.json` (`so101-real`, `so101-real-hanoi`) y sin condición en `real/env.sh`. Contradice `AGENTS.md` ("nunca pongas `ALLOW_REAL_MOTION=1` sin ese permiso").
- **Corrección:** eliminado de ambos sitios. El cargador de configuración nunca lo lee de un fichero. Lo exporta el usuario cuando da el SI.
- **Efecto en el host:** hay que exportarlo a mano antes de mover el brazo.

### 13. Límite de velocidad incoherente
- **Evidencia:** defecto 5 en `real/config.py`, `deploy_goal.py` y `record_goal.py`; 2,6 en `chess_mcp.py`; 2 en el resto. `AGENTS.md` pide 2.
- **Corrección:** defecto 2 en todos. El paso del ajedrez nunca supera `SO101_MAX_RELATIVE_TARGET`. Subirlo es una decisión explícita.

### 14. Documentación y skills contradictorias
- **Evidencia:**
  - `so101-ajedrez-mover` y `so101-ajedrez-partida` decían que el backend real no existía; `so101-ajedrez-real` decía que existía y estaba probado.
  - `CONTEXTO_PIPELINES_SESION.md` describe otra máquina.
  - La skill situaba el ACT en `modeloACT/`.
- **Corrección:** las tres skills están alineadas con el código. `AGENTS.md` apunta a la guía del host y al registro de modelos.

### 15. El entorno no se llamaba `.venv312`
- **Evidencia:** los 7 servidores MCP fallaban con `ENOENT ./.venv312/bin/python`. El entorno real es `~/lerobot312` (Python 3.12, lerobot 0.7.0, MPS).
- **Corrección:** enlace simbólico `.venv312 -> ~/lerobot312`, ignorado por git. `pipeline/host_validate.sh` lo crea si falta.

### 16. Faltan paquetes en `lerobot312`
- **Evidencia:** no estaban `mcp`, `chess`, `pyserial`, `mujoco` ni `ultralytics`.
- **Corrección:** instalados `chess`, `pyserial`, `mujoco` y `mcp<2` (el código usa `mcp.server.fastmcp`). Ninguno cambia torch ni numpy.
- **Pendiente:** `ultralytics` actualizaría numpy (2.2.6 → 2.5.5) y opencv. `INSTALL_YOLO=1 pipeline/host_validate.sh` lo instala guardando antes un `pip freeze` para poder volver atrás.

### 17. Agarre sin medir
- `GRIP_OPEN = 40` "sin medir con piezas reales". Solo están medidas las alturas del peón (3,5 cm) y del rey (5,5 cm); el resto son estimaciones.
- Hay que medirlo en el host: peldaños 5c y 6 de la guía.

### 18. Visión sin validar con la cámara del puesto
- Las esquinas de `vision/board_camara0.json` salen de fotos antiguas.
- La skill registra recall 0,94 y 5 de 17 fotos exactas, con fallos entre alfil y peón y entre rey y dama.
- Cada falso rechazo de la verificación detiene la partida.
- Se mide en el host con `CAMERA=0 pipeline/host_validate.sh`.

### 19. Nada del movimiento real está probado
- No se ha ejecutado ningún peldaño 5–8 de `pipeline/GUIA_HOST_AJEDREZ.md`. El Mac de desarrollo no tiene brazo.
- Tampoco el alcance IK de las 64 casillas: necesita la calibración del seguidor, que solo está en el host.

---

## Validación de esta rama

`pipeline/host_validate.sh` en el Mac de desarrollo, que no es el host: **33 PASS, 3 FAIL, 5 SKIP**.
- **FAIL:**
  - Falta `ultralytics` (cuenta dos veces: el paquete y el 8º test).
  - La cámara 0 de este Mac.
- **SKIP:** todo lo que necesita el brazo o su calibración.

Hay que repetirlo en el host. Ningún peldaño con movimiento se ejecuta con algún FAIL abierto.

## Siguientes pasos (en el host)

1. `git checkout` de esta rama y `pipeline/host_validate.sh`.
2. Copiar los pesos ACT con `rsync` y comprobar `--stage models` en el host.
3. Instalar `ultralytics` y validar la visión en vivo (`CAMERA=0`).
4. Peldaños 5–8 de `pipeline/GUIA_HOST_AJEDREZ.md`, uno a uno, con el SI del usuario.
