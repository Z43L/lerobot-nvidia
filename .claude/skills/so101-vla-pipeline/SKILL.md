---
name: so101-vla-pipeline
description: Guía el pipeline de ACT (y SmolVLA) del SO-101 por etapas con pipeline/vla_pipeline.py: comprobar, grabar con el líder y objetivo, validar el dataset, entrenar en mps, desplegar. Úsala cuando el usuario diga "entrena el ACT", "graba el dataset", "lanza el pipeline", "prepara el hackathon", "valida el dataset", "cuánto tarda el entrenamiento" o quiera el flujo de principio a fin, en sim o en el puesto real.
---

# Pipeline ACT del SO-101 (grabar → validar → entrenar → desplegar)

Claude orquesta; **el humano mueve el líder**. Claude nunca mueve el brazo real por su cuenta.

Desde la raíz del proyecto, con `.venv312`:

```bash
./.venv312/bin/python pipeline/vla_pipeline.py <etapa> [opciones]
```

Cada run deja `pipeline/runs/<run>/state.json` (tiempos) y un log por etapa. `status` lo muestra.

## Qué modelo y qué datos

- **Política:** ACT desde cero (`--policy act`). Es el camino principal.
- **Imágenes:** 240x320, cámaras `camera1` (top), `camera2` (wrist) y, si la hay, `camera3` (table).
- **Objetivo:** `observation.environment_state` = (x, y) del objeto en metros, en el marco de mesa. ACT lo necesita para saber a dónde ir. Lo da la visión y Claude.
- **Sim y real usan el mismo formato.** SmolVLA queda como alternativa (`--policy smolvla`) sin objetivo.

## Reglas que no se saltan

1. **Movimiento real:** `record` (real) y `deploy` mueven el brazo. Exigen `ALLOW_REAL_MOTION=1` y que el usuario escriba `SI`. Pide permiso en el chat antes de ejecutarlos. Nunca pongas `ALLOW_REAL_MOTION=1` sin confirmación del usuario.
2. **Velocidad:** `SO101_MAX_RELATIVE_TARGET=2` en la primera prueba (en `real/env.local.sh`).
3. **Dispositivo:** `mps` por defecto. `cuda` solo si el usuario confirma GPU NVIDIA en el puesto.
4. **Seguridad física:** área libre, persona junto al brazo y corte de alimentación a mano antes de cada grabación o despliegue.
5. Si una etapa falla, lee `pipeline/runs/<run>/<etapa>.log` antes de reintentar.

## Sim (sin brazo): validar el flujo y medir tiempos

```bash
SO101_GRASP=attach ./.venv312/bin/python pipeline/vla_pipeline.py record --backend sim --run sim1 --episodes 30
./.venv312/bin/python pipeline/vla_pipeline.py validate --backend sim --run sim1 --policy act --min-episodes 20
./.venv312/bin/python pipeline/vla_pipeline.py train --backend sim --run sim1 --policy act --steps 2000 --device mps
```

- `SO101_GRASP=attach` es un agarre artificial: sirve para medir tiempos y probar el flujo. **No sirve para evaluar calidad.** Con el modo físico el experto de sim descarta casi todos los intentos.
- Medido: ACT a 240x320, batch 8, `mps` ≈ **0,9 s por paso**. 2.000 pasos ≈ 30 min; 5.000 ≈ 72 min.

## Puesto real (el día del hackathon)

1. **Comprobación (no mueve nada):**
   ```bash
   ./.venv312/bin/python pipeline/vla_pipeline.py check --backend real
   ```
   Si falla, arregla puertos, cámaras o calibración con la skill `so101-hardware-check`.

2. **Prueba de 1 episodio** (mueve el brazo; pide permiso y `SI`). Usa el grabador con objetivo:
   ```bash
   source real/env.local.sh
   ALLOW_REAL_MOTION=1 ./.venv312/bin/python real/record_goal.py --name test1 --episodes 1
   ./.venv312/bin/python pipeline/vla_pipeline.py validate --backend real --run test1 --policy act --min-episodes 1
   ```
   Revisa el episodio con el usuario antes de seguir.

3. **Grabación real** (25–30 episodios de una tarea). El líder lo mueve una persona; antes de cada episodio hay que dar el objetivo `x,y` en metros:
   ```bash
   ALLOW_REAL_MOTION=1 ./.venv312/bin/python real/record_goal.py --name demos --episodes 30
   ./.venv312/bin/python pipeline/vla_pipeline.py validate --backend real --run demos --policy act --min-episodes 20
   ```
   Graba **en el puesto**, con la luz del evento.

4. **Entrenamiento** (no mueve el brazo). Empieza con pocos pasos para validar el flujo y luego el valor que decida el usuario según el plan de tiempos:
   ```bash
   ./.venv312/bin/python pipeline/vla_pipeline.py train --backend real --run demos --policy act --steps 500 --device mps
   ```

5. **Despliegue** (mueve el brazo; pide `SI`). Con ACT envía el objetivo en cada paso con `real/deploy_goal.py`. Primero, sin brazo, comprueba que el checkpoint carga:
   ```bash
   ./.venv312/bin/python real/deploy_goal.py --checkpoint <ruta>/pretrained_model --goal 0.19,0.04 --dry-run
   ```
   Después, en el puesto, 1 prueba corta de 15–20 s:
   ```bash
   ALLOW_REAL_MOTION=1 ./.venv312/bin/python pipeline/vla_pipeline.py deploy --policy act --checkpoint <ruta>/pretrained_model --goal 0.19,0.04 --seconds 20 --run demos
   ```
   Nunca lances `lerobot-rollout` para ACT: no envía el objetivo.

## Qué decir al usuario

- Antes de grabar o desplegar: qué va a pasar, cuántos episodios, y que dé el `SI`.
- Después de entrenar: pasos, tiempo real (de `status`) y ruta del checkpoint.
- Si `validate` falla: cuántos episodios hay y qué falta (por ejemplo, el objetivo). No entrenes con un dataset que no cumple.

## Lo que no está probado

- Ninguna grabación ni movimiento con el brazo físico desde este proyecto.
- El grabador con objetivo (`real/record_goal.py`) solo se ha comprobado sin hardware (features y configuración).
- El despliegue de ACT con objetivo (`real/deploy_goal.py`) solo se ha probado en `--dry-run` con un checkpoint de 2 pasos entrenado en sim. El bucle real con el brazo no se ha ejecutado.
- La calidad del modelo no está medida: los datos de sim no sirven para eso con el agarre actual.
