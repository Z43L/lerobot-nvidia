"""Biblioteca de movimientos reutilizables del SO-101: se graban en simulación y se reproducen en real.

Formato (hub/motions/<nombre>.json):
  {
    "name": "saludo",
    "description": "...",
    "units": "degrees",                 # articulaciones en grados; pinza 0..100 (igual que el real)
    "keyframes": [
      {"deg": [pan, lift, elbow, wrist_flex, wrist_roll, gripper], "note": "..."},
      ...
    ]
  }

El mismo fichero sirve para los dos brazos:
  - sim:  grados -> radianes; pinza 0..100 -> rango de la pinza de la simulación.
  - real: se envían los grados y 0..100 tal cual, siempre con so101-motion (preview + confirmación).

Uso (desde la raíz del proyecto):
  ./.venv312/bin/python hub/motion_lib.py list
  ./.venv312/bin/python hub/motion_lib.py show saludo
  ./.venv312/bin/python hub/motion_lib.py validate saludo       # límites (calibración real si existe)
  ./.venv312/bin/python hub/motion_lib.py sim saludo            # reproduce en la simulación (sin ventana)
  ./.venv312/bin/python hub/motion_lib.py new nombre "descripción"
  ./.venv312/bin/python hub/motion_lib.py add nombre --deg 0 -40 80 50 0 90 --note "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MOTIONS_DIR = Path(__file__).resolve().parent / "motions"
sys.path.insert(0, str(ROOT / "sim"))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Límites de seguridad en grados, igual que hub/so101_motion.py
LIMITS_DEG = {
    "shoulder_pan": (-100.0, 100.0), "shoulder_lift": (-120.0, 120.0), "elbow_flex": (-110.0, 110.0),
    "wrist_flex": (-115.0, 115.0), "wrist_roll": (-180.0, 180.0), "gripper": (0.0, 100.0),
}
# Rango de la pinza en la simulación (radianes): 0 = cerrada, 100 = abierta
SIM_GRIPPER = (-0.1745, 1.7453)


def path_of(name: str) -> Path:
    return MOTIONS_DIR / f"{name}.json"


def load(name: str) -> dict:
    p = path_of(name)
    if not p.exists():
        raise FileNotFoundError(f"No existe el movimiento '{name}' en {MOTIONS_DIR}")
    return json.loads(p.read_text())


def save(motion: dict) -> Path:
    MOTIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = path_of(motion["name"])
    p.write_text(json.dumps(motion, indent=2, ensure_ascii=False) + "\n")
    return p


def list_motions() -> list[str]:
    return sorted(p.stem for p in MOTIONS_DIR.glob("*.json")) if MOTIONS_DIR.exists() else []


def validate(motion: dict, limits_deg: dict | None = None) -> list[str]:
    """Devuelve la lista de problemas (vacía = válido). limits_deg puede venir de la calibración real."""
    problems = []
    if motion.get("units") != "degrees":
        problems.append("units debe ser 'degrees'.")
    if not motion.get("keyframes"):
        problems.append("No hay keyframes.")
    lim = dict(LIMITS_DEG)
    if limits_deg:
        lim.update({k: tuple(v) for k, v in limits_deg.items() if k in lim})
    for i, kf in enumerate(motion.get("keyframes", [])):
        deg = kf.get("deg", [])
        if len(deg) != 6:
            problems.append(f"keyframe {i}: hacen falta 6 valores.")
            continue
        for j, (name, v) in enumerate(zip(JOINTS, deg)):
            lo, hi = lim[name]
            if not (lo <= v <= hi):
                problems.append(f"keyframe {i}: {name}={v} fuera de [{lo}, {hi}].")
    return problems


def sim_radians(deg6: list[float]) -> np.ndarray:
    """Keyframe en grados/0..100 -> objetivo de la simulación (radianes; pinza en su rango)."""
    q = np.radians(np.asarray(deg6[:5], dtype=float))
    g = float(np.clip(deg6[5], 0, 100)) / 100.0
    grip = SIM_GRIPPER[0] + g * (SIM_GRIPPER[1] - SIM_GRIPPER[0])
    return np.concatenate([q, [grip]])


def play_sim(motion: dict, arm) -> dict:
    """Reproduce el movimiento en un SimArm/ObjectsSim ya creado. Respeta la velocidad del real."""
    results = []
    for i, kf in enumerate(motion["keyframes"]):
        r = arm.move_joints(sim_radians(kf["deg"]).tolist())
        results.append({"keyframe": i, "reached": r["reached"], "seconds": r["seconds"]})
    return {"name": motion["name"], "steps": results,
            "all_reached": all(s["reached"] for s in results),
            "total_seconds": round(sum(s["seconds"] for s in results), 2)}


def real_targets(motion: dict) -> list[list[float]]:
    """Objetivos en grados/0..100 listos para so101-motion (un preview + confirmación por keyframe)."""
    return [[round(float(v), 2) for v in kf["deg"]] for kf in motion["keyframes"]]


def _calibration_limits() -> dict | None:
    try:
        from sim_arm import load_real_calibration, STS3215_RESOLUTION
    except Exception:  # noqa: BLE001
        return None
    cal = load_real_calibration()
    if not cal:
        return None
    out = {}
    for name in JOINTS[:5]:
        c = cal.get(name)
        if c:
            mn, mx = float(c["range_min"]), float(c["range_max"])
            mid = (mn + mx) / 2
            out[name] = ((mn - mid) * 360 / (STS3215_RESOLUTION - 1), (mx - mid) * 360 / (STS3215_RESOLUTION - 1))
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for c in ("show", "validate", "sim"):
        sp = sub.add_parser(c)
        sp.add_argument("name")
    sp = sub.add_parser("new")
    sp.add_argument("name")
    sp.add_argument("description", nargs="?", default="")
    sp = sub.add_parser("add")
    sp.add_argument("name")
    sp.add_argument("--deg", nargs=6, type=float, required=True)
    sp.add_argument("--note", default="")
    args = ap.parse_args()

    if args.cmd == "list":
        print("\n".join(list_motions()) or "(no hay movimientos)")
        return 0
    if args.cmd == "new":
        save({"name": args.name, "description": args.description, "units": "degrees", "keyframes": []})
        print(f"creado {path_of(args.name)}")
        return 0
    if args.cmd == "add":
        m = load(args.name) if path_of(args.name).exists() else {"name": args.name, "description": "",
                                                                 "units": "degrees", "keyframes": []}
        m["keyframes"].append({"deg": [round(v, 2) for v in args.deg], "note": args.note})
        save(m)
        print(f"keyframe {len(m['keyframes']) - 1} añadido a {args.name}")
        return 0

    m = load(args.name)
    if args.cmd == "show":
        print(json.dumps(m, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "validate":
        problems = validate(m, _calibration_limits())
        cal = "con calibración real" if _calibration_limits() else "con límites por defecto (sin calibración)"
        print(f"{args.name}: {'OK' if not problems else 'PROBLEMAS'} ({cal})")
        for p in problems:
            print(" -", p)
        return 0 if not problems else 1
    if args.cmd == "sim":
        from sim_arm import SimArm
        problems = validate(m, _calibration_limits())
        if problems:
            print("No se reproduce: " + "; ".join(problems))
            return 1
        report = play_sim(m, SimArm())
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["all_reached"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
