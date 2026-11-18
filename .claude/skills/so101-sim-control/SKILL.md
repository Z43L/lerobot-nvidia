---
name: so101-sim-control
description: Controla el SO-101 en la simulación MuJoCo a través del MCP so101-sim (ver escena, mover articulaciones, mirar cámaras y lanzar el VLA). Úsala cuando el usuario quiera probar movimientos, ver la escena o validar el MCP sin el brazo físico.
---

# Controlar el SO-101 en simulación

La simulación siempre está disponible y no mueve hardware real. Usa las herramientas del servidor MCP `so101-sim`.

## Herramientas
- `get_scene`: estado de las 6 articulaciones (radianes) y posición del cubo en metros.
- `move_joints`: 6 valores en radianes, en este orden: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. Se recortan a los límites y se mueven en tramos suaves.
- `home`: vuelve a reposo.
- `reset_scene`: reinicia brazo y cubo.
- `get_camera_image`: imagen de `table` (vista completa de la mesa, para ver la escena), `top` (cenital) o `wrist` (muñeca). Las dos últimas son las que usa el VLA.
- `run_skill`: lanza el VLA SmolVLA con una instrucción exacta, por ejemplo `Pick up a cube and place in the bin`. Tarda ~1 min en cargar el modelo y ~2 min por 600 pasos en CPU.

## Flujo recomendado
1. `get_scene` para ver el estado inicial.
2. `get_camera_image` para mirar la escena antes de decidir.
3. Mueve con `move_joints` en pasos pequeños y vuelve a mirar la cámara.
4. Para probar el VLA, usa `run_skill` y comprueba `success` y `cube_position_m`.

## Limitaciones que hay que decir al usuario
- En sim, el VLA **no** completa la tarea de pick con el checkpoint de referencia. Un resultado negativo en sim no dice nada definitivo sobre el brazo real.
- Las unidades son radianes. No mezcles con las normalizadas (-100..100) del brazo real.
