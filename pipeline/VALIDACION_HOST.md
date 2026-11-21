# Validar en el host del robot (dataset play-tray, ACT)

Checkpoint de prueba: 200 pasos, solo humo (error offline medio ~13°). NO es un modelo útil.
Entrenado con PIPEFD/data-play-tray-1-clean (59 ep, wrist_roll positivo). El checkpoint va por Git LFS en `checkpoints/act_tray_test200/`:
`git lfs install && git pull` (si salen punteros de texto, `git lfs pull`).

1. Comprobar dataset (sin robot):
   `python pipeline/pretrain_check.py <ruta_dataset> --video`
2. Calibración: `lerobot-teleoperate --display_data=true` → wrist_roll debe ser POSITIVO en reposo (~+167).
3. Cámaras `front` y `gripper` a 640x360; mismo `robot.id` que al grabar.
4. Evaluación offline del checkpoint (sin robot):
   `python pipeline/offline_eval.py <checkpoint> <ruta_dataset>`
5. Prueba con brazo: SOLO con SI explícito, área libre, persona en el corte de alimentación,
   `SO101_MAX_RELATIVE_TARGET=2`, 15–20 s. Usa `lerobot-rollout` (este dataset no tiene objetivo x,y,
   así que `real/deploy_goal.py` no aplica). Vía de despliegue sin probar.
