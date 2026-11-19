---
name: so101-setup
description: Puesta en marcha automatizada del SO-101 con el servidor MCP so101-hub: diagnóstico del puesto, configuración de puertos y cámaras, comprobación de calibración y lectura de posiciones sin mover el brazo. Úsala cuando el usuario diga "prepara el brazo", "configura el puesto", "revisa la calibración", o al llegar al hackathon.
---

# Puesta en marcha del SO-101 (so101-hub)

Claude hace todo lo que se puede automatizar. Lo que exige una persona (calibración, cables, alimentación) se lo pide con instrucciones concretas.

## Herramientas (servidor `so101-hub`)

| Herramienta | Qué hace | Mueve el brazo |
|---|---|---|
| `setup_status` | Puertos, cámaras, calibraciones, config y entorno. Dice qué falta. | No |
| `write_config` | Escribe `real/env.local.sh` con los valores que da el usuario. | No |
| `read_positions` | Lee las 6 articulaciones del seguidor con su calibración. | No |
| `calibration_plan` | Devuelve el comando de calibración para que lo ejecute una persona. | No (lo ejecuta el usuario) |

## Procedimiento

1. **Estado.** Llama a `setup_status`. Lee `problems` y resúmelos al usuario en una lista corta.
2. **Configuración.** Si faltan puertos o cámaras, pide al usuario los valores (puertos de `/dev/cu.usbmodem*`, índices de cámara). Luego llama a `write_config`. Si el usuario no sabe el índice de la cámara, pide que lo compruebe con `real/find_cameras.py`.
3. **Cámaras.** Si `setup_status` dice que no hay cámaras, es casi seguro el permiso: pide al usuario que lo conceda a la app que lanza el MCP (Claude o Terminal) y que reinicie. No es un fallo del brazo.
4. **Calibración.** Si no hay calibración del seguidor (o del líder, si lo usan), llama a `calibration_plan` y pásale al usuario el comando tal cual. Calibrar exige mover cada articulación a mano por su rango: lo hace una persona. Antes, confirma con la organización si el brazo ya viene calibrado y pide el fichero si lo dan.
5. **Verificación.** Tras calibrar, llama a `read_positions`. Pide al usuario que ponga el brazo en una pose conocida (por ejemplo, todo en el centro) y compara. Si los valores no tienen sentido (rangos imposibles, saltos), para y revisa la calibración.
6. **Listo.** Cuando `setup_status` no tenga `problems` y `read_positions` funcione, el puesto está listo para el control. El movimiento se hace con `so101-real` y solo con confirmación explícita (ver `so101-mcp-usage`).

## Reglas

- Ningún movimiento desde esta skill. Si el usuario quiere mover el brazo, pasa a `so101-mcp-usage` y pide confirmación.
- Tras `write_config`, hay que reiniciar los servidores MCP para que lean `env.local.sh`.
- No inventes puertos, IDs de calibración ni índices de cámara.
