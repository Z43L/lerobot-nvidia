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

## Verificación de pesos (2026-11-21, Mac de desarrollo)

| Ruta | Estado |
|---|---|
| `act_tray_a100_100k_ckpts/checkpoints/{010000,020000,030000}/pretrained_model` | **Pesos reales** (207 MB cada uno, `git lfs fsck` OK). Clon de `huggingface.co/PIPEFD/act_tray_a100_100k_ckpts`: en HF solo hay 10k, 20k y 30k (el entrenamiento estaba configurado a 100k). Los tres cargan y predicen en mps. |
| `~/act_tray_ckpts/...` | No existe en este Mac (es la ruta del host). `hf-30k-cuda` cae al clon de arriba. |
| `outputs/train/act_tray_clean_25000/...` (`local-25k-mps`) | No existe en este Mac. |
| `modeloACT/` | No existe (zip ignorado en git). `real/deploy_act_generic.py` y `pruebas_act/act_model.py` ya no lo usan: leen `chess_config.ACT_GENERIC_CKPT`. |
| `checkpoints/act_tray_test200` (ACT con objetivo) | No existe. No hay ningún ACT con objetivo. |

**Submódulo (21-nov):** `act_tray_a100_100k_ckpts` es un submódulo de git que apunta a `https://huggingface.co/PIPEFD/act_tray_a100_100k_ckpts` (commit `a1ca91e`, rama `main`). Se eliminó `act_tray_30k_ckpts`, que era una referencia duplicada al mismo repo y commit. Requiere `git-lfs`.

```bash
# Solo los pesos del 30k (~200 MB), sin estados del optimizador (394 MB por checkpoint) ni 10k/20k:
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init act_tray_a100_100k_ckpts
git -C act_tray_a100_100k_ckpts lfs pull --include "checkpoints/030000/pretrained_model/*"
# Todo (~1,8 GB): git submodule update --init  (con git-lfs instalado)
```
Comprobar: `./.venv312/bin/python pipeline/e2e_check.py --stage models` (3 PASS el 21-nov).

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

## Qué aprendieron (agarre)

- Dataset: 59 episodios de *pick and place* de objetos de una bandeja. Todos cierran la pinza; en 34 de 59 se ve que la
  pinza queda bloqueada por el objeto.
- Pinza en el entrenamiento: estado 1,6–32,3 y orden máxima 32,7 (apertura típica 19–23). **Nunca más de ~32.**
- Por eso el ejecutor `ik` exige `CHESS_GRIP_OPEN` <= máximo entrenado + 2 para usar el ACT. Con 40 (el valor por
  defecto, pensado para el agarre guionizado) se niega.
- No vieron piezas de ajedrez. Que agarren piezas está por demostrar en el host (peldaño 7).

## Copiarlos al host (los pesos no van en git)

```bash
rsync -avh --progress ~/act_tray_ckpts/ USUARIO@HOST:~/act_tray_ckpts/
rsync -avh --progress outputs/train/act_tray_clean_25000/checkpoints/025000/ \
      USUARIO@HOST:<repo>/outputs/train/act_tray_clean_25000/checkpoints/025000/
```

Unos 200 MB por checkpoint. Para la evaluación offline en el host hace falta también el dataset en
`~/data-play-tray-1-clean` (o `ACT_EVAL_DATASET`).
