# Instrucciones para agentes (SO-101 + VLA)

Proyecto: simulación MuJoCo y brazo real SO-101 controlados por SmolVLA y un servidor MCP, en un Mac (Apple Silicon).

## Dispositivo: usa siempre MPS (GPU de Apple)

- Inferencia y entrenamiento de SmolVLA: **`mps`**. Es la GPU de este Mac y la opción por defecto del proyecto.
- Variable `SO101_VLA_DEVICE=mps` (ya puesta en `.mcp.json`, `real/env.sh` y en el código como valor por defecto).
- `cpu` tarda unos 31 s por chunk de 50 acciones; `mps`, unos 1,4 s. No uses `cpu` salvo que MPS no esté disponible (el código lo avisa y cae a CPU).
- `cuda` solo si el puesto del evento tiene una GPU NVIDIA; pásalo explícitamente, nunca como valor por defecto.
- MLX no está instalado ni integrado: no lo propongas sin un plan de portado.
- Scripts de entrenamiento: `real/train.sh <dataset> <pasos> mps` y `sim/hanoi_train.sh <color> <pasos> mps`. Mantén `PYTORCH_ENABLE_MPS_FALLBACK=1` (ya está en los scripts).

## Entorno

- Python: `./.venv312/bin/python` (Python 3.12, LeRobot 0.7.0, MuJoCo, MCP). No uses el Python del sistema.
- Visor MuJoCo en macOS: `./.venv312/bin/mjpython`.

## Unidades

- Articulaciones del brazo real y del checkpoint: **grados** (LeRobot `use_degrees=True`). Pinza: 0..100.
- Sim: radianes internamente; `sim/vla_policy.py` convierte a grados. No vuelvas a la escala ±100.

## Seguridad del brazo real

- Ningún movimiento real sin `ALLOW_REAL_MOTION=1` y confirmación explícita del usuario.
- Antes de mover el brazo: área libre, persona junto al brazo, corte de alimentación a mano.
- Empieza siempre con `SO101_MAX_RELATIVE_TARGET=2`.

## Git

- No hagas commit ni push sin que el usuario lo pida.
- Mensajes de commit en español, con `Co-Authored-By` si el usuario lo pide.

## Dónde mirar

- `GUIA_SO101.md`: guía de uso y checklist del evento.
- `real/README.md`, `sim/README.md`: detalles de cada parte.
- `.claude/skills/`: flujos paso a paso para Claude (hardware, sim, VLA, MPS, evento).
