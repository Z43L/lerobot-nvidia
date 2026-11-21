---
name: so101-ajedrez-brazo-retorno
description: Mover una pieza con el SO-101 real y volver siempre al punto de retorno (reposo plegado), con el ejecutor IK de hub/chess_ik_exec.py. Incluye cómo se guarda el retorno, cómo volver por tramos con arm_move y el bloqueo conocido del codo al desplegar. Úsala cuando el usuario diga "mueve una pieza y vuelve", "punto de retorno", "el brazo se atascó", "vuelve a casa" o falle chess_ik_exec.
---

# Brazo: mover una pieza y volver al retorno

## Punto de retorno

`hub/chess_home.json` → `pose_origen_grados` = **reposo plegado real**: `[-0,6; -97,4; 97,0; -98,6; 165,5; 2,4]` (pan, lift, elbow, wrist_flex, wrist_roll, gripper; grados, pinza 0..100). Leído del brazo con `arm_state` el 21-nov-2026. Copia previa: `chess_home.json.bak_antes_retorno` (tenía wrist_flex -44,6, incorrecto).
`set-home` rechaza poses a <3° de un límite (el reposo lo es): escribe el JSON a mano desde `arm_state`.

## Ejecutar una jugada

```bash
source real/env.local.sh
CHESS_MAX_HOME_JUMP=200 ./.venv312/bin/python hub/chess_ik_exec.py plan e2e4              # sin mover
CHESS_MAX_HOME_JUMP=200 ./.venv312/bin/python hub/chess_ik_exec.py test-grasp --square e2 --go   # coge e2 y lo deja en e2
CHESS_MAX_HOME_JUMP=200 ./.venv312/bin/python hub/chess_ik_exec.py run e2e4 --grasp scripted
```
- Exige que el brazo esté a ≤8° del ORIGEN al empezar (`IKExecutor.pick_place`). Espera 5 s en la primera jugada.
- `--grasp scripted` = IK baja, cierra la pinza hasta que se detiene, sube. `--grasp act` = ventana del ACT de 30k confinada a una caja; no probado.
- Se detiene en el primer fallo (`no converge al waypoint ...`) y NO vuelve solo: devuélvelo tú (abajo).
- El MCP `so101-chess` está en `.mcp.json` (hay que reiniciar Claude Code para cargarlo). Sin reiniciar, usa Python: `sys.path.insert(0,"hub"); import chess_mcp as m; m.arm_state()`.

## Volver al retorno por tramos (probado, funciona)

```python
import sys; sys.path.insert(0,"hub"); import chess_mcp as m
for _ in range(6):
    r = m.arm_move({"wrist_flex": -98.6, "shoulder_lift": -96.8, "gripper": 2.4}, max_steps=30)
    if r["status"] != "ok" or all(abs(v) < 1.5 for v in r["falta_grados"].values()): break
```
(`shoulder_lift` -96,8 y no -97,4: `arm_move` deja 3° de margen al tope calibrado.) Plegar funciona a 2°/paso sin atascos.

## Ganancias y tolerancia (resuelto el 21-nov, decisión del usuario)

Los servos con P=16 de LeRobot y ≤2° de adelanto por paso se quedan cortos varios grados (codo 12°, hombro sin moverse). Solución en `hub/chess_ik_exec.py` (`Arm.__init__`): `P_Coefficient` codo 28 (`CHESS_P_ELBOW`) y hombro 44 (`CHESS_P_LIFT`); `goto` con tolerancia 2,5° (pinza 4). Con P hombro 24 y 32 no bastó: subir el hombro desde el alcance máximo (e4) exige 44.
**Retorno actual** = pose desplegada colocada a mano por el usuario: `[-7,9; -17,8; 13,5; 100,0; 165,1; 2,7]` (el reposo plegado está en `chess_home.json.bak_reposo_plegado`; desde ahí el codo no arranca). Ojo: con este retorno el brazo tapa el centro del tablero en la cámara cenital (1); verifica con la cámara 0.
`test-grasp e2 --go` completo OK (agarrada, pinza 15,4, vuelve al retorno). `run e2e4`: coge, sube, lleva y suelta en e4; hubo que terminar a mano la subida y el retorno (ya corregido con P hombro 44). Falta verificar que el peón queda en e4.
