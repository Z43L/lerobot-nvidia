---
name: so101-vla-workflow
description: Guía el flujo completo con el SO-101 real y un VLA (SmolVLA): grabar demostraciones con el líder, entrenar un fine-tune y desplegar el checkpoint en el brazo. Úsala cuando el usuario quiera grabar datos, entrenar un VLA, o ejecutar un VLA en el brazo físico.
---

# Flujo VLA con el SO-101 real

Orden fijo. No saltes pasos. Antes de cada paso que mueva el brazo, pide confirmación explícita al usuario.

## 1. Comprobar hardware
Usa la skill `so101-hardware-check`. No sigas si hay algún `FAIL` o `WARN` sin resolver.

## 2. Grabar demostraciones
- Confirma con el usuario: nombre del dataset, número de episodios (empieza con 1 de prueba) y la instrucción exacta.
- Prueba primero con 1 episodio:
  ```bash
  real/record.sh 1 test_run "Pick up a cube and place in the bin"
  ```
- Si funciona, graba el resto (por ejemplo 30 episodios). El líder lo maneja el usuario; tú no lo mueves.

## 3. Entrenar
- Un fine-tune de SmolVLA con pocos datos en Mac tarda mucho. Dile al usuario y recomienda GPU NVIDIA.
- Ejecuta solo con su confirmación:
  ```bash
  real/train.sh <nombre_dataset> 20000 mps
  ```
- En Mac, usa siempre `mps` (GPU de Apple). Usa `cuda` solo si el puesto tiene GPU NVIDIA. `cpu` tarda ~20 veces más.

## 4. Desplegar en el brazo real
- Requisitos: área libre, usuario junto al brazo con el corte de alimentación a mano, y `SO101_MAX_RELATIVE_TARGET` bajo (por defecto 5).
- Confirma la instrucción y la duración. Después:
  ```bash
  real/rollout.sh <ruta_del_checkpoint> "Pick up a cube and place in the bin" 60
  ```
- Alternativa desde el MCP: `SO101_BACKEND=real ALLOW_REAL_MOTION=1` (el usuario lo activa, tú no lo pones por tu cuenta).

## Reglas de seguridad
- Nunca pongas `ALLOW_REAL_MOTION=1` tú mismo.
- Nunca subas datos ni checkpoints a Hugging Face sin pedirlo.
- No borres datasets ni checkpoints sin confirmación.
- Si el brazo hace algo inesperado, indica al usuario que lo pare con la alimentación y no sigas enviando comandos.

## Cuando algo falla
- Revisa primero `real/check_hardware.py`.
- Si `record.sh` no arranca, el fallo más probable es la sintaxis de `--robot.cameras`. Revisa la salida y ajusta sin inventar flags: consulta `--help` del comando.
