"""Interfaz JSON entre Claude (o un script) y el checkpoint ACT de modeloACT/.

Solo inferencia offline: no conecta con el robot ni envía comandos.

Comandos:
  predict --state "pan,lift,elbow,wrist_flex,wrist_roll,gripper" --front img.png --gripper img2.png
      Devuelve JSON con el chunk de 100 acciones en grados y una comprobación de rango.
  test
      Ejecuta pruebas_act/test_act.py y muestra el resumen.

Uso: ./.venv312/bin/python pruebas_act/act_cli.py predict --state "-18,-104,97,-100,-150,12" --front f.png --gripper g.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from act_model import JOINTS  # noqa: E402


def load_rgb(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(json.dumps({"ok": False, "error": f"no se puede leer la imagen {path}"}))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cmd_predict(a) -> int:
    from act_model import ACTOffline
    state = [float(x) for x in a.state.split(",")]
    if len(state) != 6:
        print(json.dumps({"ok": False, "error": "state debe tener 6 valores en grados"}))
        return 2
    imgs = {"front": load_rgb(a.front), "gripper": load_rgb(a.gripper)}
    m = ACTOffline()
    chunk = m.predict_chunk(state, imgs)
    # rango de entrenamiento (fichero generado por test_act.py); si no existe, no se comprueba
    rng_file = HERE / "results" / "latest.json"
    out = {"ok": True, "device": m.device, "joints": JOINTS,
           "chunk_shape": list(chunk.shape),
           "first_action_deg": dict(zip(JOINTS, np.round(chunk[0], 2).tolist())),
           "last_action_deg": dict(zip(JOINTS, np.round(chunk[-1], 2).tolist()))}
    if rng_file.exists():
        r = json.loads(rng_file.read_text())["tests"]["3_output_range"]
        lo = np.array([r["train_action_min"][j] for j in JOINTS])
        hi = np.array([r["train_action_max"][j] for j in JOINTS])
        outside = (chunk < lo) | (chunk > hi)
        out["fuera_de_rango_entrenamiento"] = {j: int(outside[:, i].sum()) for i, j in enumerate(JOINTS)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_test(a) -> int:
    r = subprocess.run([sys.executable, str(HERE / "test_act.py")], capture_output=True, text=True)
    report = json.loads((HERE / "results" / "latest.json").read_text())
    summary = {k: v["pass"] for k, v in report["tests"].items()}
    summary["all_pass"] = report["all_pass"]
    print(json.dumps(summary, indent=2))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--state", required=True)
    p.add_argument("--front", required=True)
    p.add_argument("--gripper", required=True)
    p.set_defaults(fn=cmd_predict)
    t = sub.add_parser("test")
    t.set_defaults(fn=cmd_test)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
