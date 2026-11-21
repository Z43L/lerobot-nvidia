# Bitácora: partida de ajedrez real con el SO-101

Skills: `so101-ajedrez-partida-real` (flujo), `so101-ajedrez-camaras-yolo` (visión), `so101-ajedrez-brazo-retorno` (brazo), `so101-ajedrez-real` (maestra).

## 2026-11-21

**Cambios de política (a petición del usuario):** quitadas las medidas de seguridad de `AGENTS.md` y de las 13 skills que las citaban (SI explícito, área libre, persona junto al brazo, corte a mano, `authorize_motion.sh`). En código, `motion_permission()` de `hub/chess_mcp.py` y `hub/chess_ik_exec.py` devuelve `True` salvo `CHESS_REQUIRE_PERMISSION=1`. Se mantienen los límites mecánicos (rango calibrado, ≤2°/paso, parada por atasco). `so101-chess` añadido a `.mcp.json`.

**Reparación:** `sim/chess_real.py` y `sim/chess_cli.py` estaban en versión stub (faltaba `RealChessBackend`, `DryRunExecutor`, `ACTExecutor`, `Geometry`); restaurados desde `stash@{0}` (rama planB). `sim/test_chess_real.py`: 8 pruebas OK.

**Estado del puesto:** brazo en reposo plegado `[-0,6; -97,4; 96,9; -98,6; 165,5; 2,4]`, motores 32-36 °C, 12,4 V. Cámaras: 0 YOLO (1280x960), 1 cenital, 2 muñeca, 3 webcam Mac. YOLO ve 24 piezas con el tablero inicial (fila trasera mal tipada). Partida nueva sin jugadas, robot blancas.

**Punto de retorno:** `hub/chess_home.json` reescrito con la pose real (antes wrist_flex -44,6).

**Pruebas con el brazo:**
1. `chess_ik_exec.py test-grasp --square e2 --go`: preflight sin bloqueos (236 pasos ≈ 8 s). En real, el codo se queda en 42° (objetivo 2,4°) → `no converge`. Brazo parado con torque.
2. Vuelta al retorno con `goto` también se frena; con `arm_move` por tramos (muñeca, lift, pinza) vuelve sin problema.
3. Codo solo desde 97° a 2°/paso: se para en 95,3° (carga 104, corriente 5). Sano pero sin ganancia de arranque.

**Resultado:** ninguna jugada completada. Bloqueo: despliegue del codo desde el reposo. Pendiente de decisión del usuario (ver skill `so101-ajedrez-brazo-retorno`).

**No probado:** agarre real de piezas, alturas de agarre (solo peón 3,5 cm y rey 5,5 cm medidos), apertura de pinza, ACT 30k en ventana de agarre, verificación tras una jugada real.

### 2026-11-21 (tarde): retorno desplegado
- El usuario colocó el brazo a mano; `hub/chess_home.json` = `[-7,9; -17,8; 13,5; 100,0; 165,1; 2,7]` (el reposo plegado queda en `chess_home.json.bak_reposo_plegado`). Plan e2→e4: 55 pasos ≈ 2 s, salto máximo 14°.
- `test-grasp e2 --go`: para en el primer tramo (ORIGEN → encima de e2). Quedó en `[-2,5; -2,5; 14,2; 97,8; 165,1; 37,8]` frente a objetivo `[1,0; -3,9; 2,4; 95,1; -; 40]`: codo 12° corto (carga 24), pan 3,5° corto, pinza 2,2° corta. Con tolerancia de 1° `goto` no converge.
- Conclusión: el retorno desplegado no basta. Los servos se quedan cortos varios grados con el adelanto de 2°/paso (falta ganancia), sobre todo el codo. Palanca restante: P_Coefficient del codo / adelanto. Pendiente de acuerdo con el usuario.

### 2026-11-21 (tarde, 2): P del servo y primera jugada
- Opción elegida por el usuario: subir P. Codo 28 y hombro 44 (24 y 32 no bastaban), `goto` tol 2,5°. Códigos en `hub/chess_ik_exec.py`.
- `test-grasp e2 --go`: OK completo, vuelve al retorno.
- `run e2e4 --grasp scripted`: pinza cierra a ~15, sube, lleva a e4 y abre. Falló dos veces en pasos intermedios por tolerancia/P y se terminó con scripts; el brazo volvió al retorno.
- Verificación: `chess_verify_move e2e4` = e4 vacío; cámara 0 no ve el peón en e4 ni en e2. Posición del peón NO confirmada. Partida sin registrar (`chess_commit` no ejecutado).

### 2026-11-21 (tarde, 3): depuración del agarre (skill `so101-ajedrez-agarre`)
- El peón no se agarraba. Causas medidas: falso "agarrada" (la pinza se para en vacío en 15,6), presión nula durante el transporte, pinza recortada a 2 por paso por LeRobot y bajada imprecisa (tolerancia de 2,5°). Corregidas en `hub/chess_ik_exec.py`. Cierre suave a petición del usuario.
- ACT 30k: carga en mps (arreglado el `cuda` del preprocesador). Desde encima de e2 no baja: sube y cierra en el aire.
- `grasp-probe` en e2: ventana de ±2 mm; agarra solo en el offset (4; 0; -3) mm, con cierre en 6–7. Pendiente: ver físicamente dónde queda el peón respecto a los dedos.
- Usuario: el hueco entre dedos era demasiado grande. Medido: 40 = ~7,4 cm y 60 = ~10,7 cm entre puntas (más de 2 casillas). Apertura de acercamiento por pieza (peón 14 ≈ 2,4 cm) también al soltar y en el tránsito. Offset anterior descartado. Las piezas blancas quedaron descolocadas por las sondas (foto `hub/frames/agarre/tablero_tras_sondas.jpg`).

### 2026-11-21 (tarde, 4): primera jugada real completada
- Piezas recolocadas por el usuario. `test-grasp e2 --go` OK (contacto 5,5, sujeta 5,1).
- `run e2e4 --grasp scripted`: OK (bajada 0,54°, contacto 5,4, transporte 5,0). Peón centrado en e4, vecinas intactas (foto `hub/frames/agarre/tras_e2e4.jpg`). `chess_verify_move` OK → `chess_commit e2e4`. Partida: 1. e4, turno de negras (humano).

### 2026-11-21 (tarde, 5): b2→b4 y ACT de 30k
- Partida registrada: 1. e4 b6 (b6 leída con `human-seen` desde la cámara 0). El tablero físico tenía la fila 4 vacía: hay que comprobar dónde está el peón de e4.
- d2→d4 y b2→b4 guionizados: el agarre cerró en vacío en todos los intentos (centro y ±3 mm). `grasp-probe b2 --step 0.006` agarró en (0,0) y (0,+6) mm → guardado b2 = (0, +3) mm en `hub/chess_grasp_offset.json`; repetido en `run`, falló otra vez. Hipótesis: cada cierre con éxito desplaza el peón contra el dedo fijo, y la sonda deja la pieza en otro sitio.
- ACT 30k con `--grasp act`: solo 1 paso en la ventana de 8 s. Causas: (1) abría y cerraba las cámaras en cada paso → ahora quedan abiertas (~2,8 Hz); (2) `real/env.local.sh` tenía índices viejos (YOLO=1, muñeca=0) → corregido a YOLO 0, muñeca 2; (3) la 1a inferencia en mps tarda ~9 s (calentamiento pendiente de añadir); (4) `act_grip_problem` comparaba la apertura de tránsito (40) → ahora la de acercamiento (14).
- Predicción del ACT encima de b2: sube el hombro (3,4 → -10/-12): no baja hacia la pieza.
