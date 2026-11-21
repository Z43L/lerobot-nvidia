"""Ejecuta el ACT de modeloACT/ ("pick and place", 2 cámaras front + gripper, SIN objetivo) en el SO-101 real.

Este ACT no recibe a qué casilla ir: hace lo que aprendió con la bandeja. Sirve para una PRIMERA PRUEBA REAL del ACT y para
ver cómo se mueve el brazo. NO sirve para jugar una jugada concreta de ajedrez (para eso hace falta un ACT con objetivo:
real/record_goal.py + real/deploy_goal.py).

Seguridad (AGENTS.md):
  - Mueve el brazo REAL: exige ALLOW_REAL_MOTION=1 y que escribas SI. Área libre, persona junto al brazo, corte a mano.
  - SO101_MAX_RELATIVE_TARGET=2 por defecto (grados por paso).
  - Guarda de arranque: si la primera acción predicha se aleja más de --max-first-jump grados de la posición actual en
    alguna articulación, NO arranca. (El test offline del checkpoint dio un error de 14,5 grados de media en el primer paso y
    37 en wrist_roll: sin esta guarda el brazo daría un tirón al empezar.)
  - Ctrl+C para en cualquier momento; el límite de tiempo (--seconds) es corto por defecto.

Sin hardware (carga los pesos reales y predice con imágenes de ejemplo, no envía nada):
  ./.venv312/bin/python real/deploy_act_generic.py --dry-run

Con el brazo (desde la raíz, tras `source real/env.local.sh`):
  ALLOW_REAL_MOTION=1 SO101_MAX_RELATIVE_TARGET=2 ./.venv312/bin/python real/deploy_act_generic.py --seconds 15
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from deploy_goal import load_policy, step_policy  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "modeloACT" / "checkpoints" / "last" / "pretrained_model"
FPS = 30
IMG_H, IMG_W = 360, 640
STATE_KEY = "observation.state"
CAM_KEYS = {"front": "observation.images.front", "gripper": "observation.images.gripper"}


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v.lstrip("-").isdigit() else default


def to_act_size(img: np.ndarray) -> np.ndarray:
    """RGB HxWx3 uint8 a 640x360 (el tamaño con el que se entrenó)."""
    if img.shape[0] == IMG_H and img.shape[1] == IMG_W:
        return img
    return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def build_obs(state: np.ndarray, frames: dict[str, np.ndarray]) -> dict:
    obs = {STATE_KEY: state.astype(np.float32)}
    for cam, key in CAM_KEYS.items():
        obs[key] = to_act_size(frames[cam])
    return obs


def check_checkpoint(policy) -> list[str]:
    problems = []
    feats = policy.config.input_features
    for key in (STATE_KEY, *CAM_KEYS.values()):
        if key not in feats:
            problems.append(f"el checkpoint no usa {key}")
    if "observation.environment_state" in feats:
        problems.append("este checkpoint usa objetivo: usa real/deploy_goal.py")
    return problems


def dry_run(checkpoint: Path, device: str) -> int:
    policy, pre, post = load_policy(checkpoint, device)
    problems = check_checkpoint(policy)
    for p in problems:
        print(f"[FAIL] {p}")
    if problems:
        return 1
    motors = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    ex = ROOT / "pruebas_act" / "results"
    frames = {}
    for cam in CAM_KEYS:
        f = ex / f"ejemplo_{cam}.png"
        img = cv2.imread(str(f)) if f.exists() else None
        frames[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.random.randint(0, 255, (IMG_H, IMG_W, 3), np.uint8)
    state = np.array([-8.09, -15.38, 22.51, 79.82, 166.68, 14.15], dtype=np.float32)
    policy.reset()
    act = step_policy(policy, pre, post, build_obs(state, frames), device, motors)
    a0 = np.array([act[m] for m in motors])
    print(f"checkpoint OK ({checkpoint.name}); primera acción predicha: {np.round(a0, 1).tolist()}")
    print(f"salto máximo frente al estado de ejemplo: {np.abs(a0 - state).max():.1f} grados (no se envía nada)")
    return 0


def confirm_motion() -> bool:
    print("\n*** El ACT va a MOVER el brazo REAL. ***")
    print("Comprueba: área libre, persona junto al brazo, corte de alimentación a mano.")
    return input("Escribe SI para continuar: ").strip() == "SI"


def real_run(checkpoint: Path, seconds: float, device: str, max_first_jump: float) -> int:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    policy, pre, post = load_policy(checkpoint, device)
    problems = check_checkpoint(policy)
    for p in problems:
        print(f"[FAIL] {p}")
    if problems:
        return 1
    cam_idx = {"front": env_int("SO101_TOP_CAM", 1), "gripper": env_int("SO101_WRIST_CAM", 0)}
    w, h = env_int("SO101_ACT_CAM_W", 1280), env_int("SO101_ACT_CAM_H", 720)  # 16:9 como el entrenamiento (640x360)
    cams = {c: OpenCVCameraConfig(index_or_path=cam_idx[c], width=w, height=h, fps=FPS) for c in CAM_KEYS}
    robot = make_robot_from_config(SO101FollowerConfig(
        port=os.environ.get("SO101_FOLLOWER_PORT", ""),
        id=os.environ.get("SO101_FOLLOWER_ID", "so101_follower"),
        cameras=cams,
        use_degrees=True,
        max_relative_target=float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2")),
    ))
    motors = [k for k in robot.action_features if k.endswith(".pos")]
    if os.environ.get("ALLOW_REAL_MOTION") != "1":
        print("Bloqueado: desplegar en el brazo necesita ALLOW_REAL_MOTION=1.")
        return 3
    if not confirm_motion():
        print("Cancelado.")
        return 3

    robot.connect()
    try:
        policy.reset()
        obs = robot.get_observation()
        state = np.array([obs[m] for m in motors], dtype=np.float32)
        act = step_policy(policy, pre, post, build_obs(state, {c: obs[c] for c in CAM_KEYS}), device, motors)
        jump = np.abs(np.array([act[m] for m in motors]) - state)
        print(f"salto de la primera acción por articulación (grados): {np.round(jump, 1).tolist()}")
        if jump.max() > max_first_jump:
            print(f"ABORTADO sin mover: salto máximo {jump.max():.1f} > {max_first_jump}. El modelo no es fiable desde esta pose.")
            return 4
        policy.reset()
        for t in range(int(seconds * FPS)):
            t0 = time.perf_counter()
            obs = robot.get_observation()
            state = np.array([obs[m] for m in motors], dtype=np.float32)
            act = step_policy(policy, pre, post, build_obs(state, {c: obs[c] for c in CAM_KEYS}), device, motors)
            robot.send_action(act)
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
            if t % FPS == 0:
                print(f"t={t / FPS:.1f}s", flush=True)
    except KeyboardInterrupt:
        print("\nParado por el usuario. No se envían más comandos.")
    finally:
        robot.disconnect()
    print("Despliegue terminado.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--max-first-jump", type=float, default=15.0, help="grados; si la primera acción se aleja más, no arranca")
    ap.add_argument("--device", default=os.environ.get("SO101_VLA_DEVICE", "mps"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run(args.checkpoint, args.device)
    return real_run(args.checkpoint, args.seconds, args.device, args.max_first_jump)


if __name__ == "__main__":
    sys.exit(main())
