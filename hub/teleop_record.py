"""Graba movimientos con el brazo LÍDER y los convierte en movimientos reutilizables.

Solo LEE el líder: no envía órdenes a ningún brazo, así que grabar no mueve nada.

  ./.venv312/bin/python hub/teleop_record.py record <nombre> --seconds 8
      Graba desde SO101_LEADER_PORT (configurado con so101-hub > write_config) y guarda la grabación
      en hub/recordings/<nombre>.json. Ctrl+C termina antes.

  ./.venv312/bin/python hub/teleop_record.py process <nombre> [--description "..."]
      Simplifica la grabación en keyframes y guarda el movimiento en hub/motions/<nombre>.json.
      Después, genera la skill con hub/skill_gen.py.

  ./.venv312/bin/python hub/teleop_record.py demo <nombre>
      Grabación sintética de prueba (sin hardware): imita un saludo para probar la tubería.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HUB = Path(__file__).resolve().parent
sys.path.insert(0, str(HUB))

import motion_lib  # noqa: E402
import trajectory as T  # noqa: E402


def cmd_record(name: str, seconds: float) -> int:
    port = os.environ.get("SO101_LEADER_PORT", "")
    leader_id = os.environ.get("SO101_LEADER_ID", "so101_leader")
    if not port:
        print("Falta SO101_LEADER_PORT. Configúralo con so101-hub > write_config.")
        return 1
    if not Path(port).exists() and port not in __import__("glob").glob("/dev/cu.usb*"):
        print(f"El puerto del líder no existe ahora mismo: {port}")
        return 1
    print(f"Grabando {seconds}s del LÍDER en {port}. Solo se lee, no se mueve ningún brazo.")
    t, deg = T.record_from_leader(seconds, port, leader_id)
    if len(t) < 2:
        print("No se han leído muestras suficientes.")
        return 1
    p = T.save_recording(name, t, deg)
    print(f"Grabación guardada: {p} ({len(t)} muestras, {round(t[-1], 1)} s)")
    print(f"Siguiente paso: teleop_record.py process {name}")
    return 0


def cmd_process(name: str, description: str) -> int:
    rec = T.load_recording(name)
    kfs = T.simplify(rec["t"], rec["deg"])
    motion = T.to_motion(name, kfs, description or f"Grabado desde el líder: {name}")
    problems = motion_lib.validate(motion)
    if problems:
        print("Problemas de límites en la grabación:")
        for p in problems:
            print(" -", p)
        return 1
    p = motion_lib.save(motion)
    summary = T.playback_summary(motion)
    print(f"Movimiento guardado: {p}")
    print(f"Keyframes: {summary['keyframes']} | duración de reproducción: {summary['seconds']} s | "
          f"paso máx: {summary['max_step_deg_per_tick']}°/paso")
    return 0


def cmd_demo(name: str) -> int:
    t = np.arange(0, 6, 1 / 30)
    pan = 8 * np.sin(2 * np.pi * 0.8 * t) * (t > 2)
    wr = 30 * np.sin(2 * np.pi * 0.8 * t) * (t > 2)
    grip = np.where(t < 2, 86, np.where((t // 0.6) % 2 == 0, 19, 86))
    deg = np.stack([pan, 0 * t - 34, 0 * t + 51, 0 * t + 34, wr, grip], axis=1).tolist()
    T.save_recording(name, t.tolist(), deg)
    print(f"Grabación de prueba: {T.RECORDINGS_DIR / (name + '.json')}")
    return cmd_process(name, "Demostración sintética (sin hardware)")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("name")
    r.add_argument("--seconds", type=float, default=8.0)
    p = sub.add_parser("process")
    p.add_argument("name")
    p.add_argument("--description", default="")
    d = sub.add_parser("demo")
    d.add_argument("name")
    args = ap.parse_args()
    if args.cmd == "record":
        return cmd_record(args.name, args.seconds)
    if args.cmd == "process":
        return cmd_process(args.name, args.description)
    return cmd_demo(args.name)


if __name__ == "__main__":
    sys.exit(main())
