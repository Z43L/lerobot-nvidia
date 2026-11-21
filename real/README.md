# SO-101 real: scripts para el evento

Todo se ejecuta desde la raíz del proyecto con `.venv312`.

## Antes de llegar
1. Copia `real/env.sh` a `real/env.local.sh` y rellénalo cuando sepas los puertos y cámaras del puesto.
2. Pide al organizador la **calibración** del seguidor y del líder (fichero JSON por `id`) o confirma que ya viene guardada en `~/.cache/huggingface/lerobot/calibration`. La página del evento dice que el brazo llega "montado y calibrado", pero hay que comprobarlo.
3. Da permiso de cámara a la Terminal en Ajustes del Sistema > Privacidad > Cámara.

## Orden de uso
```bash
# 1. Comprobar hardware (no mueve el brazo)
./.venv312/bin/python real/check_hardware.py

# 2. Solo si falta calibración
real/calibrate.sh

# 3. Grabar demostraciones con el líder (30 episodios)
real/record.sh 30 cube_demos "Pick up a cube and place in the bin"

# 4. Fine-tune de SmolVLA (GPU de Apple, mps)
real/train.sh cube_demos 20000 mps

# 5. Probar el checkpoint en el brazo real 
real/rollout.sh real/checkpoints/<carpeta>/checkpoints/last/pretrained_model "Pick up a cube and place in the bin" 60
```

## MCP con brazo real
```bash
SO101_BACKEND=real ALLOW_REAL_MOTION=1 ./.venv312/bin/python sim/mcp_server.py
```
Por defecto (`SO101_BACKEND=sim`) el MCP usa la simulación.
En real, `move_joints` y `get_scene` usan grados en las 5 articulaciones y 0..100 en la pinza (igual que LeRobot con `use_degrees=True` y que el checkpoint). En sim, radianes; la conversión está en `sim/vla_policy.py`.

## Límites
- `SO101_MAX_RELATIVE_TARGET` (por defecto 2) limita cuánto puede moverse el brazo en cada paso.
- Movimiento directo, sin confirmación. `ALLOW_REAL_MOTION=0` lo para todo (`SO101_REQUIRE_CONFIRM=1` vuelve a pedir SI en los scripts).

## Estado de verificación
- Verificado sin hardware: `check_hardware.py` detecta que no hay puertos, cámaras ni calibración, y falla sin mover nada.
- **No verificado con hardware real.** Los nombres de flags de `lerobot-record`, `lerobot-train` y `lerobot-rollout` se comprobaron con `--help`, pero la sintaxis del campo `--robot.cameras` (dict) y el arranque real aún no se han probado. Hazlo primero con `record.sh 1 test`.
