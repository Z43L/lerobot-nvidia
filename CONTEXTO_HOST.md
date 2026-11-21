# Contexto para el Claude del host del robot (SO-101, dataset play-tray)

Lee este fichero primero, junto con `AGENTS.md`. Rama: `validacion-host-play-tray`.
Hackathon Physical AI Bilbao (20–22 nov 2026). Equipo de 4. Brazo SO-101 (líder + seguidor) con LeRobot.
Objetivo: política ACT que ejecute los movimientos y Claude orquesta vía MCP (idea final: robot de ajedrez).

## Qué se ha hecho (en el Mac de entrenamiento)
1. Dataset original `Z43L/data-play-tray-1` (60 ep) limpiado → `PIPEFD/data-play-tray-1-clean` (HF, privado, tag v3.0): 59 ep, 35 675 frames, 19,8 min.
   - wrist_roll con signo invertido en ep 32–59 → corregido (×−1). Ep 17 descartado. Recorte de tiempo parado.
   - Mapeo: nuevo = original (ep 0–16); nuevo = original − 1 (ep 18–59).
2. Checker previo (`pipeline/pretrain_check.py`): 20 OK, 0 WARN, 0 FAIL sobre el limpio.
3. Prueba de entrenamiento ACT: 200 pasos, batch 8, `mps`, 3 min 29 s (~1,0–1,1 s/paso). Pérdida 7,27 (paso 40) → 3,49 (paso 200).
   Checkpoint en `checkpoints/act_tray_test200/` (Git LFS: `git lfs install && git lfs pull`).
4. Evaluación offline (`pipeline/offline_eval.py`), ep 0/30/45: error medio ~13–14° vs acción real; peor wrist_flex (28–37°).
   Es línea base de un modelo de 200 pasos: NO es un modelo útil.

## Convenciones que hay que respetar
- Formato LeRobot v3.0, so_follower, 30 fps, tarea "pick and place". Orden de joints:
  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (grados; pinza 0..100, en el dataset ≤ ~33).
- wrist_roll POSITIVO (reposo ≈ +167). Si da negativo en reposo, la calibración del host es distinta: NO ejecutar la política.
  Verificar con `lerobot-teleoperate --display_data=true`.
- Cámaras `front` y `gripper` a 640×360, mismos nombres. `robot.id` = el usado al grabar.
- No modificar el dataset original ni escribir en `Z43L/*` datasets de HF. Cualquier limpieza nueva → dataset nuevo.

## Qué validar en el host (orden)
1. `git lfs install && git pull` en la rama; entorno con `lerobot[training]`.
2. Dataset: `hf download PIPEFD/data-play-tray-1-clean --repo-type dataset --revision v3.0 --local-dir ~/data-play-tray-1-clean`
   (privado; requiere `hf auth login` con acceso). Luego `python pipeline/pretrain_check.py <ruta> --video` → debe dar 20 OK.
3. Calibración y cámaras (sección anterior) SIN mover nada con política.
4. `python pipeline/offline_eval.py checkpoints/act_tray_test200 <ruta_dataset> --device <mps|cuda|cpu>`.
5. Prueba con brazo (solo si el usuario dice `SI` en el chat): área libre, persona junto al corte de alimentación,
   `SO101_MAX_RELATIVE_TARGET=2`, 15–20 s. Este dataset NO tiene objetivo x,y: `real/deploy_goal.py` no aplica;
   se usaría `lerobot-rollout`, vía NO probada. Nunca pongas `ALLOW_REAL_MOTION=1` sin ese permiso.

## Reglas de seguridad
- Brazo que se mueve raro o un golpe: cortar alimentación y parar; no reenviar comandos.
- Seguridad física en código (límites, timeouts, parada), no en prompts.
- No git commit/push sin que el usuario lo pida. No borrar datasets ni checkpoints.

## Pendiente
- Revisar en vídeo ep 50 y 58 (error de seguimiento; antiguos 51 y 59). Confirmar a ojo orientación de pinza en ep 30 vs 31.
- Entrenamiento largo (propuesto 10 000 pasos ≈ 3 h en mps a ~1,1 s/paso; 30 000 ≈ 8,5 h): no lanzado.
- Verificar formato que pide GR00T (v2.1 vs v3.0). PR del limpio a Z43L. Servidor MCP: run_skill, go_home, get_state, stop.
- Los servidores MCP `so101-*` de `.mcp.json` usan `./.venv312/bin/python` (puede no existir en el host): crearlo o ajustar la ruta.
