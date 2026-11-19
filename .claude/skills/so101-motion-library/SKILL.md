---
name: so101-motion-library
description: Graba, valida y reproduce movimientos reutilizables del SO-101 (biblioteca hub/motions). Se practican en la simulación y se pasan al brazo real con los gates de so101-motion. Úsala cuando el usuario diga "graba un movimiento", "practica X en la sim", "repite el movimiento en real", "pásalo al brazo" o "qué movimientos tenemos".
---

# Biblioteca de movimientos del SO-101

Cada movimiento es un fichero `hub/motions/<nombre>.json` con keyframes en **grados** (pinza 0..100), igual para sim y real. `hub/motion_lib.py` lo gestiona.

## Comandos (desde la raíz del proyecto)

```bash
./.venv312/bin/python hub/motion_lib.py list
./.venv312/bin/python hub/motion_lib.py show <nombre>
./.venv312/bin/python hub/motion_lib.py validate <nombre>     # límites (calibración real si existe)
./.venv312/bin/python hub/motion_lib.py sim <nombre>         # reproduce en sim sin ventana
./.venv312/bin/python hub/motion_lib.py new <nombre> "<descripción>"
./.venv312/bin/python hub/motion_lib.py add <nombre> --deg <pan> <lift> <elbow> <wflex> <wroll> <pinza> --note "..."
```

## Grabar en simulación

1. Lleva el brazo a la pose deseada con `move_joints` del servidor `so101-sim-objects`. Va en **radianes**.
2. Lee la pose con `get_scene`.
3. Convierte: articulaciones de radianes a grados (× 180/π). Pinza: `100 × (rad − (−0.1745)) / (1.7453 + 0.1745)`.
4. Añade el keyframe con `motion_lib.py add`. Repite por cada pose.
5. Valida con `motion_lib.py validate` y reproduce con `motion_lib.py sim`.

## Reproducir en simulación con ventana

Los movimientos del MCP se ven en la ventana `objects_env.py --follow`, que el usuario tiene abierta en su terminal. Reproduce keyframe a keyframe con `move_joints` (en radianes) para que se vea.

## Pasar al brazo real

Solo con confirmación explícita del usuario, con el área libre y una persona junto al brazo.

1. `so101-motion > motion_status`. Si `allow_real_motion` es false, el usuario lo activa; tú no.
2. `motion_lib.py validate` con la calibración real. Si falla, no sigas.
3. Para cada keyframe, en orden: `motion_preview` con los grados → enseña el plan → el usuario dice sí → `motion_execute` con el `confirm_token`. Un preview por keyframe.
4. Si algo se mueve raro, `motion_stop` de inmediato.

El primer movimiento en real debe ser pequeño (un solo keyframe cercano a la pose actual) y con `SO101_MAX_RELATIVE_TARGET=2`.

## Reglas

- Los movimientos se graban y se validan antes de tocar el real.
- No inventes keyframes: sácalos de `get_scene` o de la sim.
- La pinza va siempre en 0..100, igual en sim y real.
- Lo que no se ha probado en el real se marca como pendiente.
