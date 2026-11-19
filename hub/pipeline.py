"""Flujo automático de una trayectoria grabada con el líder (todo menos mover el brazo real).

  ./.venv312/bin/python hub/pipeline.py <nombre> --seconds 8 [--description "..."]
  ./.venv312/bin/python hub/pipeline.py <nombre> --demo          # sin hardware (prueba)

Pasos: 1) grabar del líder (solo lectura)  2) convertir en keyframes  3) validar límites
       4) reproducir en simulación  5) generar la skill .claude/skills/motion-<nombre>/
El brazo real NO se mueve aquí. Para moverlo: so101-motion > trajectory_preview y, con el sí del
usuario, trajectory_execute.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent
ROOT = HUB.parent
PY = ROOT / ".venv312" / "bin" / "python"
sys.path.insert(0, str(HUB))

import motion_lib  # noqa: E402
import trajectory as T  # noqa: E402


def run(args: list[str]) -> int:
    print("$ " + " ".join(args), flush=True)
    return subprocess.call(args, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--description", default="")
    ap.add_argument("--demo", action="store_true", help="grabación sintética, sin hardware")
    args = ap.parse_args()
    name = args.name

    if args.demo:
        if run([str(PY), str(HUB / "teleop_record.py"), "demo", name]) != 0:
            return 1
    else:
        if run([str(PY), str(HUB / "teleop_record.py"), "record", name, "--seconds", str(args.seconds)]) != 0:
            print("No se ha podido grabar. Revisa el líder con so101-hub > setup_status.")
            return 1
        if run([str(PY), str(HUB / "teleop_record.py"), "process", name,
                "--description", args.description]) != 0:
            return 1

    problems = motion_lib.validate(motion_lib.load(name), motion_lib._calibration_limits())
    if problems:
        print("Validación fallida:", *problems, sep="\n - ")
        return 1
    print("Validación: OK")

    from sim_arm import SimArm
    report = motion_lib.play_sim(motion_lib.load(name), SimArm())
    print("Simulación:", json.dumps({k: report[k] for k in ("all_reached", "total_seconds")}, ensure_ascii=False))
    if not report["all_reached"]:
        print("La simulación no ha alcanzado todos los puntos. No generes la skill todavía.")
        return 1

    if run([str(PY), str(HUB / "skill_gen.py"), name, "--force"]) != 0:
        return 1

    summary = T.playback_summary(motion_lib.load(name))
    print(f"\nListo. Duración en real: {summary['seconds']} s, paso máx {summary['max_step_deg_per_tick']}°.")
    print(f"Siguiente: so101-motion > trajectory_preview(\"{name}\") y pedir el sí del usuario.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
