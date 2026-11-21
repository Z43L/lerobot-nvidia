"""Reproduce en el SEGUIDOR una grabación del LÍDER, adaptada a lo que el seguidor puede hacer hoy.

Por defecto NO mueve nada (dry-run): calcula el plan y lo muestra.
Para mover el brazo hacen falta las dos cosas:
  ALLOW_REAL_MOTION=1 y --confirm (el usuario ha dado su sí explícito en el chat).

  ./.venv312/bin/python hub/follower_adapt.py mapa_lider                       # dry-run
  ALLOW_REAL_MOTION=1 ./.venv312/bin/python hub/follower_adapt.py mapa_lider --confirm

Adaptación:
- Rango de cada articulación = rango del líder en la grabación, recortado por los límites
  VERIFICADOS del seguidor (FOLLOWER_LIMITS). Lo que no se ha verificado no se mueve.
- Articulaciones bloqueadas o sin verificar se mantienen en su posición actual (HOLD).
- Cada waypoint se envía con SO101_MAX_RELATIVE_TARGET (2 grados por ciclo, como manda AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "real"))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REC_DIR = ROOT / "hub" / "recordings"

# Límites verificados del seguidor (grados). Se ajustan con medidas reales, no con el líder.
#  - elbow_flex: el seguidor se atasca hacia 76-82 (medido). No bajar de 76 hasta revisarlo.
#  - wrist_roll: no coincide con el líder (86° de diferencia). Se mantiene (HOLD).
FOLLOWER_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-120.0, 120.0),
    "elbow_flex": (76.0, 119.4),
    "wrist_flex": (-115.0, 115.0),
    "wrist_roll": None,   # HOLD: no verificada
    "gripper": (0.0, 100.0),
}
MAX_WAYPOINTS = 60  # submuestreo: como mucho ~60 puntos por grabación


def load_recording(name: str) -> np.ndarray:
    p = REC_DIR / f"{name}.json"
    d = json.loads(p.read_text())
    return np.asarray(d["deg"], dtype=float)


def plan(name: str, current: dict) -> dict:
    rec = load_recording(name)
    idx = np.linspace(0, len(rec) - 1, min(MAX_WAYPOINTS, len(rec))).astype(int)
    rec = rec[idx]
    out_waypoints = []
    notes = {}
    for j, joint in enumerate(JOINTS):
        lim = FOLLOWER_LIMITS[joint]
        if lim is None:
            notes[joint] = "HOLD (no verificada)"
            continue
        lo, hi = lim
        lo_rec, hi_rec = rec[:, j].min(), rec[:, j].max()
        notes[joint] = f"líder {lo_rec:.1f}..{hi_rec:.1f} -> seguidor {max(lo, lo_rec):.1f}..{min(hi, hi_rec):.1f}"
    for row in rec:
        wp = []
        for j, joint in enumerate(JOINTS):
            if FOLLOWER_LIMITS[joint] is None:
                wp.append(current[joint])          # HOLD
            else:
                lo, hi = FOLLOWER_LIMITS[joint]
                wp.append(float(np.clip(row[j], lo, hi)))
        out_waypoints.append(wp)
    return {"name": name, "waypoints": out_waypoints, "notes": notes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--confirm", action="store_true", help="mueve el brazo real (requiere ALLOW_REAL_MOTION=1)")
    args = ap.parse_args()

    from real_arm import RealArm

    arm = RealArm()
    try:
        cur = arm.get_state()["joints"]
        p = plan(args.name, cur)
        print("Estado actual:", cur)
        for k, v in p["notes"].items():
            print(f"  {k:14s} {v}")
        print(f"Waypoints: {len(p['waypoints'])}")
        if not args.confirm:
            print("DRY-RUN: no se ha movido nada. Para mover: ALLOW_REAL_MOTION=1 ... --confirm")
            return
        if os.environ.get("ALLOW_REAL_MOTION") == "0":
            print("Bloqueado: ALLOW_REAL_MOTION=0 (parada activada).")
            return
        for i, wp in enumerate(p["waypoints"]):
            r = arm.move_joints(wp, max_chunks=400)
            print(f"wp {i + 1}/{len(p['waypoints'])} alcanzado={r['reached']}", flush=True)
        print("final:", arm.get_state()["joints"])
    finally:
        arm.close()


if __name__ == "__main__":
    main()
