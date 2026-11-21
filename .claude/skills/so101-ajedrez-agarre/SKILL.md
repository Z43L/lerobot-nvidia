---
name: so101-ajedrez-agarre
description: Diagnóstico y ajuste del agarre de piezas de ajedrez con el SO-101 real (pinza, cierre suave, detección de pieza, precisión de la bajada, offset del agarre con grasp-probe y papel del ACT de 30k). Úsala cuando el usuario diga "no agarra bien", "se cae la pieza", "aprieta mucho/poco", "calibra el agarre", "usa el ACT para coger" o falle pick_place.
---

# Agarre de piezas: cómo funciona y cómo depurarlo

Código: `hub/chess_ik_exec.py` (`Arm.grasp`, `Arm.goto_fine`, `IKExecutor.pick_place`, comando `grasp-probe`). Bitácora: `PARTIDA_AJEDREZ_REAL.md`.

## Secuencia de un coger-y-soltar

ORIGEN → encima (IK, +7 cm) → [ventana ACT opcional] → **bajada fina** (`goto_fine`, ≤0,7°) → **cierre suave** (`grasp`) → ¿hay pieza? → subir manteniendo la presión → encima destino → bajada fina → abrir → subir → ORIGEN.
Si no hay pieza tras cerrar, o se suelta al subir, abre, sube y vuelve al ORIGEN con `ok: false` y el motivo.

## Fallos encontrados el 21-nov (y corregidos)

| # | Fallo | Medida | Corrección |
|---|---|---|---|
| 1 | `close_until_stall` daba "agarrada" siempre | En vacío se paraba en 15,6 (fricción, P=16), igual que "con peón" | `grasp()`: la orden baja `CHESS_GRIP_RATE` (1/tick); hay contacto cuando la pinza va >3 por detrás de la orden. Vacía llega a **1,6–1,8**; hay pieza si > `GRIP_EMPTY_MAX` = 4,5 |
| 2 | No apretaba durante el transporte | La orden de pinza = valor leído → presión 0; carga pasaba de 76 a -24 al subir | Orden mantenida = contacto − `CHESS_GRIP_SQUEEZE` (4) durante todo el transporte |
| 3 | La pinza también estaba recortada a 2 por paso | `max_relative_target=2` de LeRobot se aplicaba a la pinza | `max_relative_target` por motor: 2° articulaciones, 100 pinza; `goto` manda la pinza en absoluto |
| 4 | Apretaba demasiado (usuario) | orden 0 sin recorte = fuerza máxima | cierre suave + presión de 4 unidades sobre el contacto |
| 5 | Bajada imprecisa | `goto` con tolerancia 2,5° ≈ 1 cm en la punta | `goto_fine` hasta `CHESS_FINE_TOL`=0,7° con sesgo integral (medido 0,5–0,7°) |
| 6 | El ACT no cargaba | preprocesador con `device: cuda` (A100) | `real/deploy_goal.load_policy` fuerza el dispositivo (`preprocessor_overrides`) |

## Apertura de la pinza (CLAVE, 21-nov)

La pinza 0..100 recorre 140° de servo con un dedo de 8,2 cm: hueco entre puntas ≈ `2·8,2·sin((v−1,8)·0,7°)` cm (`gap_cm(v)`).

| v | 8 | 12 | 14 | 16 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| hueco (cm) | 1,2 | 2,0 | 2,4 | 2,8 | 3,6 | 7,4 | 10,7 |

A 40/60 el dedo móvil barría el peón y las vecinas (casilla = 3 cm): por eso los agarres salían erráticos y las piezas se movían.
Regla (usuario): acercarse con los dedos abiertos **solo lo justo** para la pieza y soltar abriendo lo mismo, nunca más. `GRIP_APPROACH` por pieza: peón 14, alfil 15, torre/caballo 16, dama/rey 17 (`CHESS_GRIP_OPEN` lo fuerza para todas). En el ORIGEN y durante el tránsito la pinza también queda a esa apertura. El ancho real de cada pieza está SIN medir: ajústalo si roza.

## Estado: FUNCIONA (21-nov, apertura justa, sin offset)

- `test-grasp e2 --go`: bajada 0,47°, contacto 5,5, sujeta tras subir (5,1), lo deja en e2 y vuelve al retorno.
- `run e2e4 --grasp scripted`: bajada 0,54°, contacto 5,4, transporte 5,0, peón centrado en e4, vecinas intactas; `chess_verify_move` OK y registrado con `chess_commit`.
- Valores de pinza con peón ≈ 5–5,5 (vacía 1,6–1,8): el umbral de 4,5 tiene poco margen. Si falla `agarrada`, mira la foto antes de tocar el umbral.
- Las sondas con la pinza muy abierta (40–60) descolocaron piezas: no repitas `grasp-probe` con aperturas grandes.
- Si un agarre falla con la apertura justa: recoloca la pieza y prueba `grasp-probe --step 0.003` (desplazamiento ±3 mm).

```bash
source real/env.local.sh
./.venv312/bin/python hub/chess_ik_exec.py grasp-probe --square e2 --step 0.004 --dz=-0.003 [--from-saved] [--save]
./.venv312/bin/python hub/chess_ik_exec.py test-grasp --square e2 --go
```
Variables: `CHESS_GRIP_OPEN` (fuerza la apertura de acercamiento), `CHESS_GRIP_RATE`, `CHESS_GRIP_SQUEEZE`, `CHESS_FINE_TOL`, `CHESS_P_ELBOW` (28), `CHESS_P_LIFT` (44).

## ACT de 30k (`act_tray_a100_100k_ckpts/checkpoints/030000/pretrained_model`)

- Dataset `PIPEFD/data-play-tray-1-clean`, de otro brazo y otra calibración. En sus datos: `wrist_roll` mediano 80° (el nuestro 165°), codo mediano -77°, pinza siempre ≤33.
- Sin objetivo: no sabe qué pieza coger. Entradas: estado + `front` + `gripper` a 640x360.
- Predicción medida encima de e2 (sin enviar): **no baja**. Sube el hombro (-17 a -21°) y cierra la pinza en el aire (32 → 14–18), tanto con `front` = cámara 0 como con la 1.
- Uso en el ejecutor: `--grasp act` lanza el ACT confinado a la caja encima→agarre (±8°); después la IK termina la bajada y el cierre suave decide. El ACT, por tanto, no aporta precisión al agarre. Para eso hace falta un ACT con objetivo grabado en este tablero (`real/record_goal.py`).

## Cómo depurar un agarre

1. Brazo en el retorno. `test-grasp --square X --go` y leer `pinza_cerrada`, `agarrada`, `error_bajada_deg`.
2. Si `agarrada` es false: `grasp-probe` en esa casilla, con la pieza recolocada.
3. Mirar en directo: pose de agarre con la pinza abierta y fotos de las cámaras 0 (lateral) y 2 (muñeca). La cenital (1) queda tapada por el brazo en el retorno: gira `shoulder_pan` a -55 con `arm_move` para ver el tablero.
