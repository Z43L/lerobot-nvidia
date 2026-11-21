"""Pipeline por etapas: comprobar -> grabar -> validar -> entrenar -> desplegar.

Cada ejecución tiene un nombre (run) y deja en pipeline/runs/<run>/:
  - state.json: estado y tiempos de cada etapa (sirve para planificar el hackathon).
  - <etapa>.log: salida completa de la etapa.

Backends:
  real  -> datasets en real/datasets/<run>, grabación con real/record.sh (MUEVE el brazo).
  sim   -> datasets en sim/datasets/<run>, grabación con sim/objects_record.py (sin brazo).

Políticas (train):
  act      -> ACT desde cero con real/train_act.sh. Requiere observation.environment_state (objetivo).
  smolvla  -> fine-tune de SmolVLA con real/train.sh.

Uso (desde la raíz del proyecto, con .venv312):
  ./.venv312/bin/python pipeline/vla_pipeline.py check --backend sim
  ./.venv312/bin/python pipeline/vla_pipeline.py record --backend sim --run sim1 --episodes 30
  ./.venv312/bin/python pipeline/vla_pipeline.py validate --backend sim --run sim1 --policy act
  ./.venv312/bin/python pipeline/vla_pipeline.py train --backend sim --run sim1 --policy act --steps 2000
  ./.venv312/bin/python pipeline/vla_pipeline.py record --run demos --episodes 30   # real
  ./.venv312/bin/python pipeline/vla_pipeline.py status --run demos

Seguridad: "record" y "deploy" con backend real mueven el brazo. Piden confirmación escrita (SI)
y exigen ALLOW_REAL_MOTION=1. Ninguna etapa mueve el brazo sin eso.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "pipeline" / "runs"
PY = ROOT / ".venv312" / "bin" / "python"
DATASETS = {"real": ROOT / "real" / "datasets", "sim": ROOT / "sim" / "datasets"}
MIN_FREE_GB = 10
ENV_STATE_KEY = "observation.environment_state"


# ---------- utilidades de estado y log ----------

def run_dir(name: str) -> Path:
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_state(name: str) -> dict:
    f = RUNS / name / "state.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"run": name, "stages": {}}


def save_state(name: str, state: dict) -> None:
    (run_dir(name) / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))


def run_stage(name: str, stage: str, cmd: list[str], env: dict | None = None) -> int:
    """Ejecuta un comando como etapa: log completo, tiempos y estado guardados."""
    state = load_state(name)
    log_path = run_dir(name) / f"{stage}.log"
    start = time.time()
    print(f"[{stage}] {' '.join(cmd)}")
    print(f"[{stage}] log: {log_path}")
    with open(log_path, "a") as log:
        log.write(f"\n=== {datetime.now().isoformat()} ===\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})},
                              stdout=log, stderr=subprocess.STDOUT)
    seconds = round(time.time() - start, 1)
    state["stages"][stage] = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "seconds": seconds,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    save_state(name, state)
    print(f"[{stage}] {'OK' if proc.returncode == 0 else 'FALLO'} en {seconds} s")
    return proc.returncode


def confirm_motion(action: str) -> bool:
    """Movimiento directo; con SO101_REQUIRE_CONFIRM=1 pide confirmación escrita."""
    if os.environ.get("SO101_REQUIRE_CONFIRM") != "1":  # movimiento directo (política del 21-nov); SO101_REQUIRE_CONFIRM=1 pide SI
        return True
    print(f"\n*** {action} mueve el brazo REAL. ***")
    answer = input("Escribe SI para continuar: ").strip()
    return answer == "SI"


def motion_gate(action: str) -> bool:
    if os.environ.get("ALLOW_REAL_MOTION") == "0":
        print(f"Bloqueado: {action} está parado con ALLOW_REAL_MOTION=0.")
        return False
    return confirm_motion(action)


def env_with_real() -> dict:
    """Carga real/env.sh y env.local.sh como variables de entorno para los scripts."""
    cmd = "source real/env.sh; [ -f real/env.local.sh ] && source real/env.local.sh; /usr/bin/env"
    out = subprocess.run(["bash", "-c", cmd], cwd=ROOT, capture_output=True, text=True)
    env = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


# ---------- etapas ----------

def stage_check(args) -> int:
    """Comprobaciones previas que no mueven nada."""
    problems = []
    if not PY.exists():
        problems.append(f"falta {PY}")
    else:
        r = subprocess.run([str(PY), "-c", "import torch,lerobot,cv2;print(torch.backends.mps.is_available())"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            problems.append("no importan torch/lerobot/cv2 en .venv312")
        elif r.stdout.strip() != "True" and args.device == "mps":
            problems.append("MPS no disponible (usa --device cpu solo para pruebas)")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < MIN_FREE_GB:
        problems.append(f"poco espacio en disco: {free_gb:.1f} GB libres")
    if args.backend == "real":
        if not (ROOT / "real" / "env.local.sh").exists():
            problems.append("falta real/env.local.sh (copia real/env.sh y rellena puertos)")
        else:
            rc = subprocess.run([str(PY), "real/check_hardware.py"], cwd=ROOT, env=env_with_real()).returncode
            if rc != 0:
                problems.append("check_hardware.py ha fallado (revisa puertos, cámaras y calibración)")
    for p in problems:
        print(f"[FAIL] {p}")
    if not problems:
        print("[ OK ] comprobaciones previas correctas")
    return 1 if problems else 0


def stage_record(args) -> int:
    if args.backend == "sim":
        out = DATASETS["sim"] / args.run
        cmd = [str(PY), "sim/objects_record.py", "--episodes", str(args.episodes),
               "--res", args.res, "--out", str(out)]
        # Sin SO101_GRASP=attach el experto de sim descarta casi todos los intentos (ver sim/README.md).
        return run_stage(args.run, "record", cmd)
    # Grabador real con objetivo por episodio. Pide datos por teclado y su propia confirmación SI,
    # así que se ejecuta con la terminal (sin redirigir la salida a un log).
    if os.environ.get("ALLOW_REAL_MOTION") == "0":
        print("Bloqueado: grabar con el líder está parado con ALLOW_REAL_MOTION=0.")
        return 3
    cmd = [str(PY), "real/record_goal.py", "--name", args.run, "--episodes", str(args.episodes)]
    state = load_state(args.run)
    start = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env_with_real()}).returncode
    state["stages"]["record"] = {"status": "ok" if rc == 0 else "failed", "returncode": rc,
                                 "seconds": round(time.time() - start, 1),
                                 "finished": datetime.now().isoformat(timespec="seconds")}
    save_state(args.run, state)
    return rc


def stage_validate(args) -> int:
    info_file = DATASETS[args.backend] / args.run / "meta" / "info.json"
    if not info_file.exists():
        print(f"[FAIL] no existe {info_file}")
        return 1
    info = json.loads(info_file.read_text())
    episodes = info.get("total_episodes", 0)
    frames = info.get("total_frames", 0)
    fps = info.get("fps", 0)
    features = info.get("features", {})
    cameras = [k for k in features if k.startswith("observation.images.")]
    shapes = {k: features[k].get("shape") for k in cameras}
    print(f"episodios={episodes} frames={frames} fps={fps} cámaras={shapes}")
    problems = []
    if episodes < args.min_episodes:
        problems.append(f"solo {episodes} episodios (mínimo {args.min_episodes})")
    if not cameras:
        problems.append("el dataset no tiene cámaras")
    if args.policy == "act" and ENV_STATE_KEY not in features:
        problems.append(f"ACT necesita {ENV_STATE_KEY} (objetivo); el dataset no lo tiene")
    if fps and frames:
        seconds = frames / fps
        print(f"duración total ~{seconds/60:.1f} min, media ~{seconds/max(episodes,1):.1f} s/episodio")
    for p in problems:
        print(f"[FAIL] {p}")
    state = load_state(args.run)
    state["stages"]["validate"] = {"status": "ok" if not problems else "failed",
                                   "episodes": episodes, "frames": frames, "fps": fps,
                                   "finished": datetime.now().isoformat(timespec="seconds")}
    save_state(args.run, state)
    return 1 if problems else 0


def stage_train(args) -> int:
    root = DATASETS[args.backend] / args.run
    if args.policy == "act":
        cmd = ["bash", "real/train_act.sh", args.run, str(args.steps), args.device, str(root)]
    else:
        cmd = ["bash", "real/train.sh", args.run, str(args.steps), args.device]
        if args.base:
            cmd.append(args.base)
    return run_stage(args.run, "train", cmd)


def stage_deploy(args) -> int:
    if args.backend != "real":
        print("deploy solo está implementado para el brazo real.")
        return 2
    if args.policy == "act":
        # ACT con objetivo: real/deploy_goal.py puede pedir SI por consola (SO101_REQUIRE_CONFIRM=1): sin redirigir la salida.
        if not args.goal:
            print("[FAIL] ACT necesita --goal 'x,y' (metros) para el objetivo de este episodio")
            return 2
        if os.environ.get("ALLOW_REAL_MOTION") == "0":
            print("Bloqueado: desplegar en el brazo está parado con ALLOW_REAL_MOTION=0.")
            return 3
        cmd = [str(PY), "real/deploy_goal.py", "--checkpoint", args.checkpoint,
               "--goal", args.goal, "--seconds", str(args.seconds)]
        name = args.run or "deploy"
        state = load_state(name)
        start = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env_with_real()}).returncode
        state["stages"]["deploy"] = {"status": "ok" if rc == 0 else "failed", "returncode": rc,
                                     "seconds": round(time.time() - start, 1),
                                     "finished": datetime.now().isoformat(timespec="seconds")}
        save_state(name, state)
        return rc
    if not motion_gate("deploy (VLA en el brazo)"):
        return 3
    cmd = ["bash", "real/rollout.sh", args.checkpoint, args.task, str(args.seconds)]
    return run_stage(args.run or "deploy", "deploy", cmd, env=env_with_real())


def stage_status(args) -> int:
    state = load_state(args.run)
    if not state["stages"]:
        print(f"Sin etapas registradas para {args.run}")
        return 0
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("check", help="comprobaciones previas (no mueve el brazo)")
    c.add_argument("--backend", choices=["real", "sim"], default="real")
    c.add_argument("--device", default="mps")
    c.set_defaults(fn=stage_check)

    r = sub.add_parser("record", help="grabar (real: MUEVE el brazo; sim: sin brazo)")
    r.add_argument("--run", required=True)
    r.add_argument("--episodes", type=int, default=30)
    r.add_argument("--task", default="Pick up a cube and place in the bin")
    r.add_argument("--backend", choices=["real", "sim"], default="real")
    r.add_argument("--res", default="240x320", help="alto x ancho (sim)")
    r.set_defaults(fn=stage_record)

    v = sub.add_parser("validate", help="comprobar el dataset grabado")
    v.add_argument("--run", required=True)
    v.add_argument("--backend", choices=["real", "sim"], default="real")
    v.add_argument("--policy", choices=["act", "smolvla"], default="act")
    v.add_argument("--min-episodes", type=int, default=20)
    v.set_defaults(fn=stage_validate)

    t = sub.add_parser("train", help="entrenar la política")
    t.add_argument("--run", required=True)
    t.add_argument("--backend", choices=["real", "sim"], default="real")
    t.add_argument("--policy", choices=["act", "smolvla"], default="act")
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--device", default="mps")
    t.add_argument("--base", default="", help="solo smolvla: checkpoint base")
    t.set_defaults(fn=stage_train)

    d = sub.add_parser("deploy", help="ejecutar checkpoint en el brazo real (MUEVE el brazo)")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--task", default="Pick up a cube and place in the bin")
    d.add_argument("--seconds", type=int, default=60)
    d.add_argument("--run", default="")
    d.add_argument("--policy", choices=["act", "smolvla"], default="smolvla")
    d.add_argument("--goal", default="", help="ACT: objetivo 'x,y' en metros para este episodio")
    d.add_argument("--backend", choices=["real", "sim"], default="real")
    d.set_defaults(fn=stage_deploy)

    s = sub.add_parser("status", help="ver tiempos y estado de un run")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=stage_status)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
