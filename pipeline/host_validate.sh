#!/usr/bin/env bash
# Validación automatizada del ajedrez real desde el HOST. NO mueve el brazo y NO activa ALLOW_REAL_MOTION.
#   pipeline/host_validate.sh                 # comprueba y ejecuta los peldaños sin movimiento; guarda el informe
#   CAMERA=0 pipeline/host_validate.sh        # además lee el tablero en vivo con esa cámara (posición inicial puesta)
#   INSTALL_YOLO=1 pipeline/host_validate.sh  # instala ultralytics (con copia del entorno para poder volver atrás)
# Variables: VENV (por defecto el enlace .venv312 del repo o ~/lerobot312), ACT_EVAL_DATASET, SO101_VLA_DEVICE.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="pipeline/runs/e2e_${STAMP}"
mkdir -p "$OUT"

say() { printf '\n== %s\n' "$*"; }
VENV="${VENV:-}"
if [ -z "$VENV" ]; then
  if [ -x ".venv312/bin/python" ]; then VENV=".venv312"; elif [ -x "$HOME/lerobot312/bin/python" ]; then VENV="$HOME/lerobot312"; fi
fi
[ -n "$VENV" ] && [ -x "$VENV/bin/python" ] || { echo "No encuentro el entorno. Crea el enlace:  ln -s ~/lerobot312 .venv312   (o VENV=ruta)"; exit 2; }
if [ ! -e .venv312 ]; then ln -s "$(cd "$VENV" && pwd)" .venv312 && echo "Creado enlace .venv312 -> $VENV (las rutas de .mcp.json lo necesitan)"; fi
PY="$ROOT/.venv312/bin/python"

say "1. Estado del repo"
git rev-parse --abbrev-ref HEAD; git log --oneline -1
[ -f hub/chess_config.py ] || { echo "FALTA hub/chess_config.py: este checkout no tiene el refactor. Haz git pull de la rama correcta."; exit 2; }

say "2. Configuración del puesto (real/env.local.sh)"
if [ -f real/env.local.sh ]; then sed -E 's/^(export [A-Z0-9_]*(PORT|ID)=).*/\1.../' real/env.local.sh | head -12
else echo "NO existe real/env.local.sh. Créalo con so101-hub write_config (puertos, IDs, cámaras)."; fi
grep -qE "^[[:space:]]*export[[:space:]]+ALLOW_REAL_MOTION" real/env.local.sh 2>/dev/null && echo "AVISO: env.local.sh fija ALLOW_REAL_MOTION; se ignora, pero quítalo."

say "3. Paquetes"
MISSING=$("$PY" - <<'P'
import importlib.util
miss=[m for m in ("torch","lerobot","cv2","chess","mcp","serial","ultralytics") if importlib.util.find_spec(m) is None]
print(" ".join(miss))
P
)
echo "Faltan: ${MISSING:-ninguno}"
BASIC=""; for m in $MISSING; do case $m in chess) BASIC="$BASIC chess";; serial) BASIC="$BASIC pyserial";; mcp) BASIC="$BASIC mcp<2";; esac; done
if [ -n "$BASIC" ]; then echo "Instalando (no tocan torch ni numpy):$BASIC"; "$PY" -m pip install -q $BASIC || echo "FALLÓ pip install$BASIC"; fi
if echo "$MISSING" | grep -qw ultralytics; then
  if [ -n "${INSTALL_YOLO:-}" ]; then
    "$PY" -m pip freeze > "$OUT/pip_antes.txt"
    echo "Copia del entorno en $OUT/pip_antes.txt. Instalando ultralytics (puede cambiar numpy/opencv)..."
    "$PY" -m pip install -q ultralytics
    if ! "$PY" -c "import torch, lerobot, cv2, numpy" 2>/dev/null; then
      echo "ROMPIÓ el entorno. Volver atrás:  $PY -m pip install -r $OUT/pip_antes.txt --force-reinstall --no-deps"; exit 3
    fi
  else
    echo "ultralytics falta: sin él no hay visión (YOLO). Instálalo con:  INSTALL_YOLO=1 pipeline/host_validate.sh"
  fi
fi

say "4. Modelos ACT (los pesos NO están en git)"
"$PY" - <<'P'
import sys; sys.path.insert(0,"hub")
import chess_config as C
for n,p in C.ACT_MODELS.items():
    ok=(p/"model.safetensors").exists() and (p/"model.safetensors").stat().st_size>1_000_000
    print(("OK    " if ok else "FALTA "), n, p)
P
echo "Si faltan, desde el Mac donde se entrenaron:  rsync -avh --progress ~/act_tray_ckpts/ USUARIO@HOST:~/act_tray_ckpts/ ;  rsync -avh --progress outputs/train/act_tray_clean_25000/ USUARIO@HOST:<repo>/outputs/train/act_tray_clean_25000/"

say "5. Verificación sin movimiento (peldaños 0-4 y models)"
ARGS=(--json "$OUT/e2e.json"); [ -n "${CAMERA:-}" ] && ARGS+=(--camera "$CAMERA")
env -u ALLOW_REAL_MOTION PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" pipeline/e2e_check.py "${ARGS[@]}" | tee "$OUT/e2e.txt"
RC=${PIPESTATUS[0]}

say "Resultado"
echo "Informe: $OUT/e2e.txt  y  $OUT/e2e.json"
if [ "$RC" -ne 0 ]; then echo "HAY FALLOS (ver arriba). No pases a los peldaños que mueven el brazo."; exit 1; fi
echo "Sin fallos. Los SKIP no están comprobados: revísalos."
echo "Siguiente (mueve el brazo; requiere el SI del usuario, ver pipeline/GUIA_HOST_AJEDREZ.md, sección 'Peldaños con movimiento')."
