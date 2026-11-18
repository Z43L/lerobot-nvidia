---
name: so101-mps-vla
description: Ejecuta y entrena SmolVLA en el Mac usando la GPU de Apple (MPS). Úsala cuando el usuario pida inferencia del VLA, un fine-tune, un benchmark de velocidad, o diga "usa la GPU", "mps", "entrena en el Mac" o "cuánto tarda el VLA".
---

# SmolVLA en MPS (GPU de Apple)

Regla del proyecto: en este Mac se usa **siempre `mps`**. No uses `cpu` salvo que MPS no esté disponible, y no propongas MLX sin un plan de portado (no está instalado).

## Comprobar que MPS funciona

```bash
cd /Users/david/Desktop/lerobot-nvidia
./.venv312/bin/python -c "import torch; print('mps disponible:', torch.backends.mps.is_available())"
```

Si da `False`, el código cae a CPU y lo avisa. Entonces dilo al usuario: el tiempo de inferencia pasa de ~1,4 s a ~31 s por chunk.

## Velocidad de referencia (medida en este Mac, checkpoint de referencia)

| Dispositivo | Chunk de 50 acciones |
|---|---|
| mps | ~1,4 s (0,8–2,0 s) |
| cpu | ~31 s (29–34 s) |

Un chunk son 50 acciones a 30 Hz, es decir 1,67 s de movimiento. Con MPS el brazo se pausa unos 1,4 s por chunk si la inferencia es síncrona.

## Inferencia (sim o real)

- Variable: `SO101_VLA_DEVICE=mps` (ya está en `.mcp.json` y `real/env.sh`).
- Sim: `sim/vla_policy.py` lo lee en `VLARunner`.
- Real: `real/real_vla.py` y `real/real_arm.py` (`run_skill`) lo leen igual.

## Entrenamiento

Con dataset grabado en `real/datasets/<nombre>/`:

```bash
real/train.sh <nombre> 500 mps     # primero una prueba corta para validar el flujo
real/train.sh <nombre> 20000 mps   # entrenamiento completo: decide los pasos antes, según el tiempo medido
```

Hanoi (un VLA por color):

```bash
sim/hanoi_train.sh red 2000 mps
```

Antes de un entrenamiento largo, mide el tiempo por paso con la prueba corta y comunícalo al usuario. Los scripts ya ponen `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Reglas

- No cambies el dispositivo a `cuda` salvo que el usuario lo pida y el equipo tenga NVIDIA (el puesto del evento, por ejemplo).
- Si el usuario pregunta por MLX: explica que no está instalado y que SmolVLA corre en PyTorch.
- Antes de comunicar una cifra de velocidad, di si es sim o real y si es chunk o paso.
