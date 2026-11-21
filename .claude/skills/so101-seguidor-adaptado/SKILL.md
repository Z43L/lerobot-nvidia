---
name: so101-seguidor-adaptado
description: Reproduce en el SO-101 REAL (seguidor) una grabación del líder, adaptada a los límites que el seguidor puede cumplir hoy. Úsala cuando el usuario diga "mueve el follower", "reproduce lo grabado en el seguidor", "adapta el movimiento al brazo" o "pásalo al follower".
---

# Seguidor adaptado a una grabación del líder

Lee una grabación del líder (`hub/recordings/<nombre>.json`), recorta cada articulación a lo que el seguidor puede hacer hoy y la ejecuta en el brazo real. Nunca mueve el seguidor sin el sí del usuario.

## Límites actuales (en `hub/follower_adapt.py`, `FOLLOWER_LIMITS`)

- `shoulder_pan`, `shoulder_lift`, `wrist_flex`, `gripper`: rango del líder.
- `elbow_flex`: no bajar de 76° (el seguidor se ha atascado hacia 76–82° en pruebas).
- `wrist_roll`: se mantiene en su posición (HOLD), porque no coincide con el líder (86° de diferencia).

Si cambian estas medidas, actualiza `FOLLOWER_LIMITS` con medidas reales y no con el líder.

## Procedimiento

1. **Estado.** `so101-hub > setup_status` y `read_positions` (solo lectura).
2. **Dry-run.** Sin mover nada:
   ```bash
   SO101_TABLE_CAM=-1 ./.venv312/bin/python hub/follower_adapt.py <nombre>
   ```
   Muestra rangos y el número de waypoints. Enseña al usuario la tabla de rangos y lo que se mantiene en HOLD.
3. **Sin confirmación previa.** Ejecuta directamente.
4. **Ejecución.**
   ```bash
   ALLOW_REAL_MOTION=1 SO101_MAX_RELATIVE_TARGET=2 SO101_TABLE_CAM=-1 ./.venv312/bin/python hub/follower_adapt.py <nombre> --confirm
   ```
5. **Verificación.** Lee posiciones al final y compara con el plan. Si el brazo se mueve raro o se atasca, para (Ctrl+C).

## Reglas

- `SO101_MAX_RELATIVE_TARGET` y `ALLOW_REAL_MOTION` ya están en el entorno.
- `SO101_TABLE_CAM=-1` es necesario mientras el índice 1 esté compartido entre la muñeca y la mesa (error de cámara por frames caducados).
- Lo que no esté verificado en el brazo real se marca como pendiente.
