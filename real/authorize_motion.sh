#!/usr/bin/env bash
# Permiso de movimiento del brazo real para el MCP de ajedrez (hub/chess_mcp.py). LO EJECUTA EL USUARIO.
#   ! real/authorize_motion.sh 10     -> permite mover el brazo 10 minutos (por defecto 5)
#   ! real/authorize_motion.sh 0      -> revoca el permiso
# Úsalo solo con: área libre, persona junto al brazo y corte de alimentación a mano.
set -euo pipefail
cd "$(dirname "$0")"
MIN="${1:-5}"
if [ "$MIN" = "0" ]; then rm -f .motion_ok; echo "Permiso de movimiento revocado."; exit 0; fi
echo $(( $(date +%s) + MIN * 60 )) > .motion_ok
echo "Movimiento del brazo permitido durante $MIN min (hasta $(date -v+${MIN}M +%H:%M:%S)). Revoca con: real/authorize_motion.sh 0"
