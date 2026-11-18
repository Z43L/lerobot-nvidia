# SO-101 en simulación (MuJoCo)

Entorno de pruebas para el MCP del brazo antes de tener el hardware físico.

## Estructura
- `so101/` — modelo MJCF del SO-101 (The Robot Studio, Apache-2.0), copiado de
  `Simulation/SO101` del repo SO-ARM100.
- `so101/scene_vla.xml` — escena usada por el MCP: mesa, cubo rojo, caja azul y cámaras `top` y `wrist` (nombres del checkpoint).
- `vla_policy.py` — carga SmolVLA (`Harrysunshine/so101-smolvla-real-9task`) bajo demanda y ejecuta la instrucción en bucle sobre la sim.
- `sim_arm.py` — API de simulación: estado, mover articulaciones (con límites y tramos suaves), cámara.
- `mcp_server.py` — servidor MCP (FastMCP) con las herramientas `get_scene`, `move_joints`,
  `home`, `reset_scene`, `get_camera_image` (`top`/`wrist`) y `run_skill` (SmolVLA en la sim).

## Uso
```bash
cd /Users/david/Desktop/lerobot-nvidia
./.venv312/bin/python sim/mcp_server.py   # arranca el servidor por stdio (Python 3.12, con lerobot)
```

Config de cliente MCP (ejemplo):
```json
{
  "mcpServers": {
    "so101-sim": {
      "command": "/Users/david/Desktop/lerobot-nvidia/.venv312/bin/python",
      "args": ["/Users/david/Desktop/lerobot-nvidia/sim/mcp_server.py"]
    }
  }
}
```

## Notas
- Orden de articulaciones: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (rad).
- Las poses que chocan con la mesa o consigo mismo se detienen; `reached: false` lo indica.
- Entorno: `.venv312` (Python 3.12, lerobot[smolvla], mujoco, mcp<2). `.venv` es el antiguo, sin lerobot.
- La primera llamada a `run_skill` tarda ~1 min en cargar el modelo en CPU.
- Estado actual: el VLA se ejecuta de extremo a extremo, pero no completa 'Pick up a cube and place in the bin' en 600 pasos (ver notas del README principal de la conversación).
