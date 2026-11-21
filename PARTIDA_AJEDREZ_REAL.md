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

### 2026-11-21 (tarde, 6): verificación de los ACT
- Solo existen los pesos de `act_tray_a100_100k_ckpts` (10k/20k/30k, LFS OK). Las rutas configuradas (`~/act_tray_ckpts`, `outputs/train/...`, `modeloACT`) no existían: `e2e_check --stage models` fallaba.
- `hf-30k-cuda` ahora busca también en el clon local; `deploy_act_generic.py` y `pruebas_act` leen `chess_config`. El verificador compara la apertura de acercamiento por pieza (≤17), no la de tránsito. Resultado: 3 PASS, 0 FAIL.

### 2026-11-21 (noche): apertura del peón 14 → 12
- Pinza a 14 y 1 cm sobre e2 (error de bajada 0,53°). En la foto de la muñeca los dedos cierran a lo largo de la columna e: el de atrás queda junto a la corona del rey de e1. La cámara lateral no sirve para medir el desvío. Fotos: `hub/frames/agarre/hover_e2_*.jpg`.
- `GRIP_APPROACH` del peón pasa a 12 (≈2,0 cm de hueco). En la foto apenas cambia respecto a 14 (`hover_e2_muneca_g12.jpg`).
- `test-grasp e2 --go` con 12: agarre, transporte y soltado hechos; se paró en «subir» tras soltar (codo 7° corto, carga 0, sin calor). El resultado de `agarrada` no llegó a imprimirse: SIN CONFIRMAR. Liberado subiendo primero el hombro y luego el codo solo; vuelta al origen. Piezas blancas de pie (`g12_origen_lateral.jpg`).
- El servidor MCP de ajedrez abre el puerto sin la P alta del codo (16): con él el codo no sube desde abajo. Usar `Arm` del ejecutor (P 28/44).
- Usuario: con 12 el peón SÍ quedó sujeto en el primer intento.
- `Arm.lift` (subidas por etapas: hombro primero con `CHESS_LIFT_EXTRA`=6°, luego codo y muñeca, luego pose exacta) en todas las subidas de `pick_place`, que ahora imprime cada etapa al momento.
- Repetido `test-grasp e2 --go` con 12: OK completo (bajada 0,63°, contacto 5,1, tras subir 4,7, soltado y vuelta al origen sin atascos). Piezas de pie.
- Partida nueva (19:38, la de 1. e4 b6 está en `chess_game.backup_20261121_193101.json`). `run d2d4 --grasp scripted` con apertura 12: d2 cerró en vacío en (0,0), (0,+3) y (0,-3) mm y agarró en (+3,0) mm (pinza 4,9; transporte 4,6). Offset d2 = (+3, 0) mm guardado. `chess_verify_move` OK (peón en d4) → `chess_commit d2d4`. Partida: 1. d4, turno de negras.
- Usuario: los 3 primeros cierres en d2 fallaron porque el dedo de atrás CHOCABA con el rey (fila 1), no por puntería. La bajada no lo detectó (convergió a 0,3–0,56°). El offset de d2 (+3 mm en x, alejándose de la fila 1) evita el choque; no es una corrección de centrado.
- Agarre según las vecinas (`IKExecutor.set_board/clear_shift/_grasp_tries`): si solo una de las casillas de delante/detrás (misma columna) está ocupada, agarra y suelta desplazado 3 mm (`CHESS_CLEAR_SHIFT_MM`) hacia la libre. Reintentos: más lejos de la ocupada a lo largo de la columna, luego laterales ±3 mm. El aprendido se descarta si acerca los dedos a la ocupada. `run` usa la FEN de la partida guardada; `RealChessBackend.execute` pasa el tablero al ejecutor. Sin probar con el brazo.
- Partida nueva (usuario). `run e2e4` con desplazamiento a la vecina libre primero: e2 cerró en vacío en +3, +6 y (+3, ±3) mm; el peón seguía en e2 de pie. Fotos de muñeca a +3 mm y en el centro (`e2_hover_mas3.jpg`, `e2_hover_centro.jpg`): no se distingue la diferencia.
- Conclusión con los datos del día: el mejor desplazamiento depende de cómo quedó cada peón al colocarlo (e2: centro 3/3; d2: solo +3 mm). Orden de intentos nuevo: aprendido de la casilla → hacia la vecina libre → centro → más lejos de la ocupada → laterales. Intentos a menos de 1,5 mm se fusionan.
- `run e2e4` de nuevo: agarra a la primera en el centro (pinza 4,8; tras subir 4,5, justo en el umbral). Verificada y registrada: 1. e4, turno de negras.
- 1... a5 (humano; leída con `chess_see`, verificada y registrada). 2. d4 (robot): d2 agarra a la primera a +3 mm (pinza 5,0; tras subir 4,7). Verificada y registrada. Partida: 1. e4 a5 2. d4, turno de negras.
- Nota: `chess_commit` etiqueta la jugada del humano como "(robot)" en su mensaje; revisar el campo `who` del historial.
- 3. e5 (robot) NO ejecutada: e4 cerró en vacío 5 veces (centro, ±3 mm columna y lateral). Pinza abierta en la pose de agarre de e4: en la muñeca el peón parece entre los dedos; en la lateral el dedo delantero queda sobre la bisagra del tablero (filas 4/5). Fila 4 = brazo casi estirado (muñeca a 2,8° del tope, hombro 50°, carga 192): sospecha de que la punta queda más alta o más lejos de lo calculado. SIN MEDIR. Fotos `e4_agarre_*.jpg`.
- Partida TERMINADA por el usuario: 1. e4 a5 2. d4 f5 (turno de blancas). Copia: `hub/chess_game.final_20261121_202208.json`. El historial marca las 4 jugadas como `who: robot` (a5 y f5 fueron del humano): fallo de `chess_commit`.
- Balance del agarre de peones a 12: fila 2 → fila 4 OK (e2 centro, d2 +3 mm). Fila 4 no agarra. Pendiente: medir el agarre en la fila 4 con el brazo quieto en la pose y la persona mirando; umbral de pieza 4,5 demasiado justo (sujeta con 4,5–4,7).

### 2026-11-21 (noche, 2): correcciones en coordenadas del tablero (rama `agarre-coordenadas-tablero`)
- Punto 1: tabla por fila `hub/chess_rank_correction.json` (adelantar a lo largo de la columna y bajar), aplicada por `Planner` al origen y al destino por separado; filas sin medir entre dos medidas se interpolan, fuera no se extrapola. Medir: `chess_ik_exec.py rank-probe --rank 4 [--square e4] [--along=0,3,-3,6] [--down=0,3] [--save]`. Ver: `rank-table`.
- Punto 3: bajada fina `CHESS_FINE_TOL` 0,7 → 0,3 grados, con hasta `CHESS_FINE_SECONDS`=4 s. Si no llega, sigue y devuelve el error.
- Punto 2 (cámara de la muñeca): sin implementar.
- Nada de esto está probado con el brazo.
- Punto 2 implementado: `hub/chess_wrist.py` + `wrist-ref` / `wrist-measure`; en `pick_place` mide antes de los intentos (`CHESS_WRIST=measure` por defecto, `correct` para aplicar). Probado sin brazo: la plantilla recupera desplazamientos sintéticos exactos y la resolución de J da la corrección esperada. Entre las fotos reales de e2 a +3 mm y centro el peón solo se movió ~7 px (≈2,5 px/mm, o la pinza no se movió los 3 mm con tolerancia 0,7°). Sin referencia guardada todavía.
