---
name: so101-mcp-usage
description: Cómo usar los servidores MCP del SO-101 (so101-sim, so101-sim-hanoi, so101-sim-objects, so101-real, so101-real-hanoi) desde Claude, qué herramientas hay, qué variables necesitan y cómo leer sus errores. Úsala cuando el usuario quiera mover el brazo, ver la escena, lanzar el VLA, o cuando un servidor no conecte o devuelva "Brazo real no disponible".
---

# Servidores MCP del SO-101

Los servidores están en `.mcp.json` y todos usan `sim/mcp_server.py` con `SO101_VLA_DEVICE=mps`. Elige el servidor según el destino:

| Servidor | Qué es | Notas |
|---|---|---|
| `so101-sim` | Sim, escena pick (cubo y caja) | Herramienta `run_skill` con el VLA de referencia |
| `so101-sim-hanoi` | Sim, torre de Hanoi | `hanoi_*` y `hanoi_pick_place`; hacen falta checkpoints por color |
| `so101-sim-objects` | Sim, objetos cotidianos y PC | **En desarrollo**: `pick` todavía no levanta objetos. No lo presentes como funcional |
| `so101-real` | Brazo físico | Requiere puerto y calibración; todo movimiento exige `ALLOW_REAL_MOTION=1` |
| `so101-real-hanoi` | Brazo físico, Hanoi | Igual que el real, más cámara de mesa (`SO101_TABLE_CAM`) |

## Herramientas comunes

- `get_scene`: estado del brazo. En real, en grados (articulaciones) y 0..100 (pinza).
- `move_joints(targets)`: 6 valores (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`). Sim en radianes, real en grados.
- `get_camera_image(camera)`: `top`, `wrist` o `table`.
- `home`, `reset_scene`.
- `run_skill(instruction)` (escena pick): VLA en sim o en real.

## Errores y cómo leerlos

- `"Brazo real no disponible: Falta SO101_FOLLOWER_PORT..."`: el servidor real arrancó sin puerto. Pide al usuario que complete `real/env.local.sh` y reinicie el MCP. No inventes puertos.
- `"blocked"` o `"Movimiento real desactivado"`: falta `ALLOW_REAL_MOTION=1`. Pídele confirmación al usuario antes de activarlo.
- `"Connection closed"` en el listado: ya no debería pasar en real, porque el servidor arranca sin hardware. Si pasa, revisa `sim/mcp_server.py` y el log de stderr.

## Reglas de seguridad

- Nunca actives `ALLOW_REAL_MOTION=1` por iniciativa propia.
- Antes de cualquier movimiento real: confirma área libre y persona junto al brazo.
- Empieza con `SO101_MAX_RELATIVE_TARGET=2`.
- Un `run_skill` real corre durante `seconds` sin sensor de éxito: verifica con `get_camera_image` antes y después.

## Comprobación rápida

Pide `get_scene` en el servidor que toque. Si devuelve grados, el backend está bien configurado. Si devuelve radianes en real, algo está mal: para.
