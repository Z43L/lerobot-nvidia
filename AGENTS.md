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

- `CHECKLIST.md`: **el orden exacto del hackathon, paso a paso.** Es la referencia principal durante el evento.
- `GUIA_SO101.md`: guía de uso general (anterior al pipeline ACT).
- `real/README.md`, `sim/README.md`: detalles de cada parte.
- `.claude/skills/`: flujos paso a paso para Claude (hardware, sim, VLA, MPS, evento, pipeline ACT).
- `pipeline/GUIA_HOST_AJEDREZ.md`: lanzar y validar el ajedrez real en el host (`pipeline/host_validate.sh`, peldaños con criterio).
- `pipeline/MODELOS_ACT.md`: los ACT disponibles (HF 30k entrenado en CUDA, local 25k en MPS), cómo cargarlos en el Mac y cómo copiarlos al host.
- Configuración del ajedrez real: solo en `hub/chess_config.py` (entorno > `real/env.local.sh` > defecto). No escribas puertos, cámaras ni rutas de pesos en otros módulos.

## Modo hackathon: qué hacer en cada momento

Durante el evento, sigue `CHECKLIST.md` en orden. Antes de cualquier acción, mira en qué sección estás y qué casillas faltan.

**Al empezar cada sesión del evento:**
1. Lee `CHECKLIST.md` y localiza la primera casilla sin marcar.
2. Ejecuta `./.venv312/bin/python pipeline/vla_pipeline.py status --run <run>` del run que estés usando (o `check` si es el inicio) para saber dónde está el pipeline.
3. Pregunta al usuario en qué sección está y qué ha pasado desde la última vez, antes de tocar nada.

**Pipeline ACT (lo que hay que usar):**
- Política: **ACT** (`--policy act`). SmolVLA queda como alternativa.
- Dispositivo: `mps` por defecto. `cuda` solo si la organización confirma GPU NVIDIA y el usuario lo pide.
- Grabación real: `real/record_goal.py` (guarda el objetivo `x,y` en metros). **No** uses `real/record.sh` para ACT: no guarda el objetivo.
- Entrenamiento: `pipeline/vla_pipeline.py train --policy act`, o `real/train_act.sh`.
- Despliegue con ACT: `real/deploy_goal.py` (o `pipeline/vla_pipeline.py deploy --policy act --goal x,y`). Envía el objetivo en cada paso. Solo probado en `--dry-run`: la primera ejecución con el brazo es una prueba corta de 15–20 s con el `SI` del usuario. No improvises con `lerobot-rollout`, que no envía el objetivo. Si falla, plan B de `CHECKLIST.md` §8.

**Reglas que no cambian en el evento:**
- **Movimiento real:** antes de cualquier 🔒 (grabar con el líder, desplegar, mover un joint), pide el `SI` explícito en el chat y confirma área libre y persona junto al brazo. Nunca pongas `ALLOW_REAL_MOTION=1` sin ese permiso.
- **Velocidad:** `SO101_MAX_RELATIVE_TARGET=2` en la primera prueba. No subas el límite sin acuerdo.
- **Cambios de luz:** no cambies la iluminación durante una grabación. Si cambia, para y graba de nuevo.
- **Datos:** no borres `real/datasets/` ni `pipeline/runs/` sin pedirlo. Son lo que se entrega.
- **Entrenamientos largos:** antes de lanzar más de 2.000 pasos, di cuánto va a tardar (referencia: ~0,9 s/paso a 240x320 con batch 8 en `mps`) y pide confirmación.
- **Git:** no hagas commit ni push durante el evento sin que el usuario lo pida.

**Qué hacer cuando algo falla:**
- Brazo que se mueve raro o un golpe: corte de alimentación y para. No reenvíes comandos.
- `check` falla: resuelve puertos, cámaras o calibración antes de grabar (skill `so101-hardware-check`).
- Cámara que no abre a 320x240: para. Hay que cambiar la resolución en `real/record_goal.py` y `real/rollout.sh`, no ignorar el error.
- Validación con pocos episodios o sin objetivo: no entrenes. Graba más o corrige el grabador.
- Entrenamiento que va más lento de lo previsto: mide s/paso y recalcula con el usuario antes de seguir.
- Si ACT no se puede desplegar: plan B (§8 de `CHECKLIST.md`) y avisar al usuario de que la demo usa línea base.

**Qué informar al usuario en cada paso:**
- Qué sección del checklist estás haciendo y cuántas casillas quedan.
- Tiempo real de cada etapa (de `pipeline/runs/<run>/state.json`).
- Resultados medidos (éxito por casilla, pérdida), no suposiciones.
- Lo que no está probado todavía, en claro.
