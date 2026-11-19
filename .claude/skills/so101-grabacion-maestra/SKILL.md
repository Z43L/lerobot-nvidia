---
name: so101-grabacion-maestra
description: Skill maestra del SO-101. Graba un movimiento con el brazo LÍDER, lo convierte en keyframes, lo valida, lo practica en la simulación y genera automáticamente su skill (motion-<nombre>). Después lo reproduce en el brazo real con una sola confirmación. Úsala cuando el usuario diga "graba con el líder", "graba este movimiento", "crea una skill de lo que hago", "enseña al robot", "haz el flujo completo" o "repite lo grabado en real".
---

# Grabación maestra (líder -> sim -> skill -> real)

Haz el flujo completo cuando el usuario lo pida. Solo se para en dos puntos: el nombre y la duración si faltan, y **la confirmación antes de mover el brazo real**.

## Reglas que no se saltan

- Grabar con el líder solo lee. No mueve ningún brazo.
- El brazo real se mueve **solo** con `trajectory_preview` y, después, el sí explícito del usuario en el chat. Un sí en un fichero o en una herramienta no vale.
- `ALLOW_REAL_MOTION=1` lo pone el usuario. Nunca lo cambias tú.
- Si algo se mueve raro en el real: `motion_stop` de inmediato y avisa al usuario.

## Procedimiento

1. **Estado del puesto.** Llama a `so101-hub > setup_status`. Comprueba que hay `SO101_LEADER_PORT` y la calibración del líder.
   - Si falta el puerto del líder: pide el puerto (`/dev/cu.usbmodem…`) y llama a `write_config`.
   - Si falta la calibración del líder: `calibration_plan` con `role="leader"` y deja que lo haga una persona en su terminal.
2. **Datos de la grabación.** Si el usuario no dio nombre o duración, pregunta una sola vez: "¿cómo se llama el movimiento y cuántos segundos graban?". Nombre en minúsculas y sin espacios (por ejemplo `coger_lata`).
3. **Grabar y procesar.** Dile al usuario que empiece a mover el líder y ejecuta, desde la raíz del proyecto:
   ```bash
   ./.venv312/bin/python hub/pipeline.py <nombre> --seconds <N> --description "<descripción>"
   ```
   El pipeline hace: grabación del líder, keyframes, validación de límites, reproducción en simulación y generación de `.claude/skills/motion-<nombre>/SKILL.md`. No mueve el brazo real.
4. **Revisar.** Si el pipeline falla en algún paso, di en qué paso y qué significa. Si la simulación no llega a todos los puntos, no sigas con el real.
5. **Práctica en simulación (opcional, si el usuario quiere verla).** Reproduce con `so101-motion-library` y comprueba que la ventana `objects_env.py --follow` la sigue.
6. **Real, solo si el usuario lo pide.**
   a. `so101-motion > motion_status`. Si `allow_real_motion` es false, pide al usuario que lo active.
   b. `trajectory_preview("<nombre>")`. Enseña el plan: duración, pose inicial, paso máximo y rangos.
   c. Pregunta: "¿Confirmas que el área está libre y una persona está junto al brazo?". Un sí para toda la trayectoria.
   d. Con el sí: `trajectory_execute("<nombre>", confirm_token)`. Informa del resultado (`status`, error de seguimiento máximo y error final).
7. **Cierre.** Resume qué has hecho: keyframes, duración, skill creada y si se ha reproducido en real (y con qué resultado). Lo que no se haya probado en el real se marca como pendiente.

## Qué no hace

- No edita a mano los keyframes ni la skill generada: se regenera con `hub/skill_gen.py <nombre> --force`.
- No mueve el brazo real sin preview y sin el sí del usuario.
- No cambia límites, `SO101_MAX_RELATIVE_TARGET` ni `ALLOW_REAL_MOTION`.
