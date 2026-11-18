---
name: so101-hackathon-day
description: Procedimiento del día del Hackathon LeRobot Bilbao (SO-101 real): configuración del puesto, comprobación de calibración y unidades, primera prueba segura y ejecución del VLA. Úsala cuando el usuario diga que está en el evento, que tiene el brazo en el puesto, o pida el checklist del hackathon.
---

# Día del hackathon (SO-101 real)

Sigue el orden. No saltes pasos. Antes de cualquier movimiento del brazo, confirma con el usuario que el área está libre y que hay una persona junto al robot.

## 1. Antes de tocar el brazo

1. Pide al usuario el **ID de calibración** del seguidor y del líder (lo da la organización). Ponlo en `real/env.local.sh` como `SO101_FOLLOWER_ID` y `SO101_LEADER_ID`.
2. Puertos: `SO101_FOLLOWER_PORT` y `SO101_LEADER_PORT` (`lerobot-find-port` si hace falta).
3. Cámaras: `SO101_TOP_CAM`, `SO101_WRIST_CAM`, y `SO101_TABLE_CAM` si hay webcam de mesa. Confírmalas con `real/find_cameras.py`.
4. Dispositivo: `SO101_VLA_DEVICE=mps` (Mac). Si el puesto tiene GPU NVIDIA, el usuario decide si usar `cuda`.
5. Límite de velocidad: `SO101_MAX_RELATIVE_TARGET=2` para la primera prueba.

## 2. Comprobación de hardware

```bash
source real/env.local.sh && ./.venv312/bin/python real/check_hardware.py
```

Todo debe salir en OK. Si no hay calibración, usa `real/calibrate.sh` solo si la organización no la ha dado.

## 3. Unidades (no saltar)

Mueve una articulación a mano o lee `get_scene`. Los valores deben salir en **grados**: `shoulder_lift` puede pasar de 100. Si ves un rango de -100..100 en las articulaciones, para: el código está mal configurado.

## 4. Primera prueba segura

```bash
real/record.sh 1 test_run "Pick up a cube and place in the bin"
```

Revisa el episodio. Después, entrenamiento corto para validar el flujo:

```bash
real/train.sh test_run 500 mps
```

## 5. Ejecutar el VLA

```bash
real/rollout.sh <ruta_del_checkpoint>/checkpoints/last/pretrained_model "Pick up a cube and place in the bin" 60
```

Antes: `ALLOW_REAL_MOTION=1` solo si el usuario lo confirma. Durante: alguien junto al brazo, corte de alimentación a mano. Un chunk tarda unos 1,4 s en MPS, así que el brazo se pausa entre chunks: no lo interpretes como fallo.

## 6. Si algo va mal

- El brazo se mueve raro: corte de alimentación, no reenvíes comandos.
- Puerto no encontrado: revisa el cable de datos (no solo de carga) y `lerobot-find-port`.
- Cámara no responde: revisa el índice y el permiso de cámara de la Terminal.
- Errores del MCP: consulta la skill `so101-mcp-usage`.

## Lo que no está probado

El brazo real no se ha conectado nunca desde este proyecto. Di al usuario que cada paso de este flujo debe comprobarse en el puesto.
