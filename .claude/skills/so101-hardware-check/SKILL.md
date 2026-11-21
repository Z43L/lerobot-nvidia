---
name: so101-hardware-check
description: Comprueba el brazo SO-101 real antes de grabar o desplegar (puertos USB, calibración, cámaras y lectura de posiciones). Úsala cuando el usuario diga "revisa el brazo", "comprueba el hardware", "está conectado el SO-101", o antes de cualquier grabación o rollout en el robot real.
---

# Comprobar el SO-101 real

Objetivo: confirmar que el brazo, el líder y las cámaras están listos, sin mover el brazo.

## Pasos

1. Asegúrate de que existe `real/env.local.sh`. Si no, cópialo desde `real/env.sh` y pide al usuario los puertos y los índices de cámara. No inventes puertos.
2. Ejecuta desde la raíz del proyecto:
   ```bash
   source real/env.local.sh && ./.venv312/bin/python real/check_hardware.py
   ```
3. Interpreta la salida:
   - `Puertos USB-serie detectados: ninguno` → el brazo no está enchufado o falta el cable de datos. Pide revisar el USB.
   - `No hay calibración` → ejecuta `real/calibrate.sh` (pide confirmación al usuario, porque mueve articulaciones) o pide al organizador el fichero de calibración.
   - `no responde` en una cámara (incluida la cámara `table` si `SO101_TABLE_CAM` está configurada) → revisa el índice (`SO101_TOP_CAM`, `SO101_WRIST_CAM`) y el permiso de cámara de la Terminal en Ajustes del Sistema.
4. Calibración en uso: indica la ruta y la fecha de los ficheros de `~/.cache/huggingface/lerobot/calibration/` (seguidor y líder). Si la fecha es posterior a la última comprobación, avisa al usuario de que la calibración ha cambiado y vuelve a comprobar posiciones.
5. Si todo está en `OK`, dilo claramente. Si algo falla, di qué falla y qué debe hacer el usuario; no lo arregles cambiando calibraciones por tu cuenta.

## Reglas
- No muevas el brazo en esta skill. `check_hardware.py` solo lee.
- No cambies `real/config.py` para "hacer que funcione". Pide los valores reales al usuario.
- Antes de cualquier movimiento real, confirma que el área está libre.
