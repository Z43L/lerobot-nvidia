# Guía del host: lanzar y validar el ajedrez real del SO-101

Para quien opere en el Mac del puesto (el host, con el brazo y las cámaras), sea una persona o Claude Code.
Las reglas de `AGENTS.md` mandan siempre. 🔒 = mueve el brazo real (movimiento directo; `ALLOW_REAL_MOTION=0` o `arm_stop` lo paran).

## Arquitectura (la de los diagramas)

```
Persona ─ Claude + skills ─ motor de ajedrez (sim/chess_cli.py)
                │
                ▼
     MCP so101-chess (hub/chess_mcp.py)          ← parada: ALLOW_REAL_MOTION=0 / arm_stop
        chess_move(uci, executor)                 ← UNA llamada por jugada
          ├─ mapa del tablero   hub/chess_board_real.json (calibración por contacto con el líder)
          ├─ RealChessBackend   sim/chess_real.py (captura→cementerio, enroque, promoción)
          ├─ ejecutor           dry | ik-scripted | ik (IK encima → ACT agarra → IK suelta → origen)
          └─ verificación       vision/chess_vision.py verify_move (foto tras la jugada)
        chess_commit(uci)                         ← "registrar": solo tras ver la foto. Nunca mueve.
```

Configuración: una sola fuente, `hub/chess_config.py`. Prioridad: entorno, luego `real/env.local.sh`, luego defecto.
Los puertos no tienen defecto.

## 1. Preparar el host (una vez)

```bash
cd <repo>
git fetch && git checkout <rama-con-el-refactor>     # debe existir hub/chess_config.py
ln -s ~/lerobot312 .venv312                         # .mcp.json usa ./.venv312
```

- **`real/env.local.sh`**: puertos, IDs y cámaras. Lo escribe `so101-hub write_config` o se crea a mano:
  `SO101_FOLLOWER_PORT`, `SO101_LEADER_PORT`, `SO101_FOLLOWER_ID`, `SO101_LEADER_ID`, `SO101_TOP_CAM`, `SO101_WRIST_CAM`,
  `SO101_MAX_RELATIVE_TARGET="2"`.
- **Calibración del seguidor**: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<ID>.json`. Sin ella no hay IK.
- **Modelos ACT**: van en el repo con Git LFS (`models/act/`). Hace falta `git-lfs` (`brew install git-lfs`); `host_validate.sh` hace el `git lfs pull`.
- **Permiso de cámara de macOS** para la app que lanza el shell (Terminal o VS Code).
- Reinicia Claude Code para que cargue el MCP `so101-chess`.

## 2. Validación automática (no mueve nada)

```bash
pipeline/host_validate.sh                      # todo sin cámara de tablero en vivo
CAMERA=0 pipeline/host_validate.sh             # con las piezas en posición inicial delante de la cámara 0
INSTALL_YOLO=1 pipeline/host_validate.sh       # si falta ultralytics (guarda pip freeze para volver atrás)
```

El script crea `.venv312` si falta, instala lo básico (`chess`, `pyserial`, `mcp<2`), descarga los modelos (`git lfs pull`) y ejecuta
`pipeline/e2e_check.py`. El informe queda en `pipeline/runs/e2e_<fecha>/`. Sale con código 1 si hay algún FAIL.

| Peldaño | Qué comprueba | Criterio de PASS |
|---|---|---|
| 0 Entorno | paquetes, MPS, 8 tests, límite de paso, nada activa `ALLOW_REAL_MOTION` | todo importable; 8 OK; ≤ 2°/paso |
| 1 Hardware | puertos USB, puerto configurado, calibración del seguidor, cámaras | existen; cámaras dan imagen |
| 2 Visión | pesos YOLO, esquinas, fotos etiquetadas, `verify_move`, tablero en vivo | recall y precisión ≥ 0,90; en vivo ≤ 3 casillas dudosas |
| 3 Mapa | calibración real, RMS, validación cruzada, origen, alcance IK de las 64 casillas | RMS ≤ 5 mm; peor toque ≤ 12 mm; 64/64 con margen ≥ 1° |
| models | cada ACT carga en el dispositivo del host, predice y error offline | carga; salida finita; ≤ 5° de media |
| 4 Ensayo | partida dry por CLI, plan en metros reales, sim, MCP (dry, ilegal, gate, registro) | todo verde; el gate devuelve `blocked` |

**Un SKIP no es un PASS.** En el host no debería quedar ningún SKIP en los peldaños 1 y 3.
**No se pasa a la sección 3 con algún FAIL.**

## 3. Peldaños con movimiento 🔒 (uno a uno)

Movimiento directo, sin confirmación. Si algo va raro: `arm_stop` en el MCP (bloquea todo hasta `arm_resume`) o
`ALLOW_REAL_MOTION=0` en el entorno, y no reenvíes órdenes en bucle. Límites siempre activos: ≤ 2°/paso, rango
calibrado y parada por atasco.

| # | Comando | PASS si |
|---|---|---|
| 5a | `./.venv312/bin/python hub/chess_ik_exec.py set-home` (brazo colocado a mano en el origen) | guarda `hub/chess_home.json` |
| 5b | `./.venv312/bin/python hub/chess_ik_exec.py test-grasp --square e2` (sin `--go`: solo planifica) | waypoints con solución |
| 5c | 🔒 `... test-grasp --square e2 --go` (peón en e2) | coge el peón, lo deja en e2 y vuelve al origen, sin golpes |
| 6 | 🔒 MCP `chess_move("e2e4", executor="ik-scripted")` → mirar la foto → `chess_commit("e2e4")` | `status: ok`, foto correcta, registrada |
| 7 | 🔒 MCP `chess_move(..., executor="ik")` (agarre con ACT, `CHESS_ACT_CHECKPOINT` elegido) | igual que el 6 |
| 8 | 🔒 `CHESS_BACKEND=real CHESS_REAL_EXECUTOR=ik-scripted ./.venv312/bin/python sim/chess_match.py --camera 0` | varias jugadas sin que la verificación pare |

Anota en cada intento: éxito o fallo, la casilla, el tiempo y el motivo. Lo pide `CHECKLIST.md` §7.

## 4. Si algo falla

- **El brazo se mueve raro o hay un golpe:** `arm_stop` y revisa. No reenvíes comandos en bucle.
- **`chess_move` devuelve `verification_failed`:** no registres. Corrige las piezas a mano, `chess_see` y repite.
- **`no hay pieza en la pinza`:** el agarre falla. Revisa `GRIP_OPEN` (40, sin medir) y las alturas en `hub/chess_ik_exec.py`. Solo el peón (3,5 cm) y el rey (5,5 cm) están medidos.
- **El ACT empeora el agarre:** quédate con `ik-scripted` (plan B, `CHECKLIST.md` §8).

## No probado todavía

Cualquier movimiento del brazo, la cámara del puesto en vivo, el alcance IK de las 64 casillas (necesita la
calibración del host), el agarre de cada tipo de pieza y el ACT dentro del bucle de la jugada.
