# Guía de uso: SO-101 + VLA + Claude

Esta guía explica cómo usar el proyecto en el Mac, tanto en simulación como con el brazo real del evento.

## 1. Estructura

| Carpeta / fichero | Qué es |
|---|---|
| `.venv312/` | Entorno Python 3.12 con LeRobot 0.7.0, MuJoCo y el MCP. Úsalo siempre. |
| `sim/` | Simulación MuJoCo del SO-101, VLA en sim y servidor MCP (`sim/mcp_server.py`). |
| `real/` | Scripts para el brazo físico: comprobación, calibración, grabación, entrenamiento y despliegue. |
| `.mcp.json` | Registra dos servidores MCP: `so101-sim` y `so101-real`. |
| `.claude/skills/` | Skills de Claude: `so101-hardware-check`, `so101-sim-control`, `so101-vla-workflow`. |

## 2. Puesta en marcha (una vez)

```bash
cd /Users/david/Desktop/lerobot-nvidia
./.venv312/bin/python -c "import lerobot, mujoco, mcp; print('entorno ok')"
```

Si falla, reinstala con:
```bash
./.venv312/bin/pip install "lerobot[smolvla,dataset,hardware]==0.7.0" mujoco pillow "mcp<2"
```

## 3. Simulación (sin brazo)

Úsala para probar el MCP y el flujo de Claude.

**Desde Claude Code:** abre el proyecto. El servidor `so101-sim` está en `.mcp.json`. Pide, por ejemplo: *"mira la escena del SO-101 y mueve el brazo a reposo"*. Claude usará la skill `so101-sim-control`.

**Vista de la mesa en sim:** la cámara `table` muestra la mesa completa; `top` y `wrist` son las que usa el VLA.

**Manual:**
```bash
./.venv312/bin/python sim/mcp_server.py
```

**Probar el VLA en sim:** en Claude, pide *"lanza el VLA con 'Pick up a cube and place in the bin'"*. Tarda ~1 min en cargar y ~2 min por 600 pasos.

**Resultado esperado hoy:** el VLA se ejecuta de extremo a extremo, pero **no coge el cubo** en sim. Es una limitación conocida (ver §6).

## 4. Brazo real

### 4.1. Antes de llegar al evento
1. Copia `real/env.sh` a `real/env.local.sh` y deja preparado el resto.
2. Confirma que tienes la **calibración** del seguidor y del líder, o pide el fichero al organizador. La página del evento dice que el brazo llega "calibrado", pero hay que comprobarlo.
3. Da permiso de cámara a la Terminal (Ajustes del Sistema > Privacidad > Cámara).
4. Opcional: para ver la mesa completa, usa la webcam del Mac como cámara `table` con `SO101_TABLE_CAM=<índice>` en `real/env.local.sh`. No entra en la política VLA: es para que Claude vea la escena.

### 4.2. En el puesto
```bash
# Rellena los puertos y cámaras en real/env.local.sh y luego:
source real/env.local.sh
./.venv312/bin/python real/check_hardware.py
```
Todo debe salir en `OK`. Si no hay calibración: `real/calibrate.sh`.

### 4.3. Grabar demostraciones
```bash
real/record.sh 1 test_run "Pick up a cube and place in the bin"   # primero 1 episodio
real/record.sh 30 cube_demos "Pick up a cube and place in the bin"
```
El líder lo mueve una persona. Los datos quedan en `real/datasets/<nombre>/`.

### 4.4. Entrenar
```bash
real/train.sh cube_demos 20000 mps     # GPU de Apple (Mac)
real/train.sh cube_demos 20000 cuda    # solo si el equipo tiene GPU NVIDIA
```
Los checkpoints van a `real/checkpoints/`. En Mac, el fine-tune completo es lento.

### 4.5. Ejecutar el VLA en el brazo
```bash
real/rollout.sh real/checkpoints/<carpeta>/checkpoints/last/pretrained_model \
  "Pick up a cube and place in the bin" 60
```
El movimiento es directo (`ALLOW_REAL_MOTION=0` lo para).

### 4.6. Desde Claude con el brazo real
El movimiento es directo:
```bash
SO101_BACKEND=real ALLOW_REAL_MOTION=1 ./.venv312/bin/python sim/mcp_server.py
```
Con `ALLOW_REAL_MOTION=0`, `run_skill` queda bloqueado en real.

## 5. Límites del brazo

- `SO101_MAX_RELATIVE_TARGET` (por defecto 2) limita cuánto se mueve el brazo en cada paso. Bájalo si se mueve de forma brusca.
- Movimiento directo, sin confirmación. `ALLOW_REAL_MOTION=0` lo para todo (y `arm_stop` en el MCP de ajedrez).

## 6. Estado actual (qué funciona y qué no)

| Parte | Estado |
|---|---|
| Simulación del SO-101 y MCP | Funciona |
| SmolVLA de referencia en sim | Se ejecuta, **no completa** la tarea |
| Scripts de hardware, grabación y entrenamiento | Creados, **sin probar con brazo físico** |
| Sintaxis de cámaras en `lerobot-record` / `rollout` | **No verificada**; primero prueba con 1 episodio |
| Calibración del brazo del evento | **Pendiente de comprobar** |

## 7. Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| `Puertos USB-serie detectados: ninguno` | Revisa el cable de datos (no solo de carga) y vuelve a enchufar. |
| `No hay calibración` | `real/calibrate.sh` o pide el fichero al organizador. |
| Cámara `no responde` | Revisa `SO101_TOP_CAM` / `SO101_WRIST_CAM` y el permiso de cámara. |
| `ModuleNotFoundError: datasets` o `pyserial` | Reinstala con los extras `dataset,hardware` (ver §2). |
| MCP no arranca con error de FastMCP | `./.venv312/bin/pip install "mcp<2"`. |
| VLA lento en sim | Normal en CPU (~0,3 s por paso). |

## 8. Checklist del día del hackathon

**Antes de llegar (ya)**
- [ ] Preguntar a la organización: ¿brazo calibrado? ¿hay GPU NVIDIA? ¿qué cámaras ponen? ¿cuál es el **ID de calibración** del seguidor y del líder? (va en `SO101_FOLLOWER_ID` / `SO101_LEADER_ID` de `real/env.local.sh`; si no coincide, LeRobot dice "no hay calibración")
- [ ] Copiar `real/env.sh` a `real/env.local.sh` (puertos y cámaras se rellenan allí).
- [ ] Dar permiso de cámara a la app Claude y a la Terminal.
- [ ] Llevar el proyecto en USB o repo, y el cable USB de datos (no solo de carga).

**Al llegar al puesto**
- [ ] Conectar seguidor y líder. Rellenar puertos en `real/env.local.sh`.
- [ ] `source real/env.local.sh && ./.venv312/bin/python real/check_hardware.py` → todo en OK.
- [ ] Si falta calibración: `real/calibrate.sh` o pedir el fichero a la organización.
- [ ] Confirmar índices de cámara con `real/find_cameras.py` (abrir las fotos).
- [ ] Mover una articulación a mano con `move_joints` (o leer `get_scene`): debe salir en **grados** (rangos de ±100 o más en shoulder_lift), no en -100..100.
- [ ] Comprobar que la calibración del brazo coincide con la del puesto (`real/calibrate.sh` solo si no hay calibración).

**Primer test (no saltarse)**
- [ ] `real/record.sh 1 test_run "Pick up a cube and place in the bin"` → revisar el episodio.
- [ ] Entrenamiento de prueba corto (p. ej. 500 pasos) para validar el flujo: `real/train.sh test_run 500 mps`.

**Datos y entrenamiento**
- [ ] Grabar ~30 episodios de una sola tarea: `real/record.sh 30 cube_demos "..."`.
- [ ] Entrenamiento completo en GPU: `real/train.sh cube_demos 20000 mps`.
- [ ] Guardar la ruta del checkpoint en `real/checkpoints/`.

**Ejecución**
- [ ] Bajar `SO101_MAX_RELATIVE_TARGET` (p. ej. 2–3) para la primera prueba.
- [ ] `real/rollout.sh <checkpoint> "Pick up a cube and place in the bin" 60`.

**Si algo falla**
- [ ] Brazo que se mueve raro: parar con la alimentación y no reenviar comandos.
- [ ] Puerto no encontrado: revisar cable y `lerobot-find-port`.
- [ ] Cámara no responde: revisar índice y permisos; volver a `find_cameras.py`.
