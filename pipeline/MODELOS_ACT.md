# Modelos ACT del proyecto

Los dos son **ACT genéricos sin objetivo**, con la misma arquitectura:
- Entrada: `observation.state` (6 articulaciones en grados) y cámaras `front` y `gripper` a 640×360.
- Salida: chunk de 100 acciones.

No saben qué pieza coger. En el ajedrez se usan solo para el agarre (ejecutor `ik`), con la IK llevando la pinza encima
de la casilla y una caja de seguridad alrededor.

| Nombre (`chess_config.ACT_MODELS`) | Ruta por defecto | Entrenado | Dataset |
|---|---|---|---|
| `hf-30k-cuda` | `~/act_tray_ckpts/checkpoints/030000/pretrained_model` (`ACT_HF_30K`) | Hugging Face, **CUDA**, checkpoint de 30k de una ejecución de 100k, batch 16 | `PIPEFD/data-play-tray-1-clean` (59 ep) |
| `local-25k-mps` | `outputs/train/act_tray_clean_25000/checkpoints/025000/pretrained_model` (`ACT_LOCAL_25K`) | local, **MPS**, 25k pasos, batch 8 (`pipeline/train_play_tray.sh`), pérdida final ≈ 0,12 | el mismo |

Elegir otro: `CHESS_ACT_CHECKPOINT=<ruta>`. Sin esa variable se usa el primero que exista, en el orden de la tabla.

## CUDA en un Mac

Un checkpoint entrenado en CUDA guarda `"device": "cuda"` en `config.json` **y** en `policy_preprocessor.json`
(`device_processor`). Cambiar solo `cfg.device` no basta: falla con `Requested device 'cuda' but CUDA is not available`.
Arreglado en `real/deploy_goal.py:load_policy` y en `pipeline/offline_eval.py`, con
`preprocessor_overrides={"device_processor": {"device": <dispositivo>}}`. Cualquier otro código que cargue un ACT debe
usar `load_policy`.

## Medido (2026-11-21, MPS, Mac de entrenamiento)

Error medio de la primera acción frente al dataset limpio, episodios 0, 10, 30, 45 y 58, uno de cada 25 fotogramas:

| Modelo | Error medio | Peor articulación | Tiempo por chunk (MPS) |
|---|---|---|---|
| `hf-30k-cuda` | **2,0°** | `wrist_flex` ≈ 3–5° | 0,04 s |
| `local-25k-mps` | 3,4° | `shoulder_lift` y `wrist_flex` ≈ 5–7° | 0,04–0,07 s |
| local 10k / 5k | 4,1° / 4,8° | `wrist_flex` hasta 9–12° | — |

**Cuidado:** es evaluación dentro de la muestra (los dos se entrenaron con esos episodios). Mide el ajuste, no la
generalización ni el éxito del agarre. El criterio real es el peldaño 7 de `pipeline/GUIA_HOST_AJEDREZ.md`.
Repetir la medida: `./.venv312/bin/python pipeline/e2e_check.py --stage models`.

## Copiarlos al host (los pesos no van en git)

```bash
rsync -avh --progress ~/act_tray_ckpts/ USUARIO@HOST:~/act_tray_ckpts/
rsync -avh --progress outputs/train/act_tray_clean_25000/checkpoints/025000/ \
      USUARIO@HOST:<repo>/outputs/train/act_tray_clean_25000/checkpoints/025000/
```

Unos 200 MB por checkpoint. Para la evaluación offline en el host hace falta también el dataset en
`~/data-play-tray-1-clean` (o `ACT_EVAL_DATASET`).
