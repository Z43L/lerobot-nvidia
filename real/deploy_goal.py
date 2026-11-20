"""Ejecuta un checkpoint de ACT con OBJETIVO en el SO-101 real.

ACT recibe en cada paso: estado de las articulaciones, las cámaras y el objetivo (x, y) en metros
(observation.environment_state). Es el mismo formato con el que se grabó en real/record_goal.py y
en sim/objects_record.py. El objetivo se fija al inicio del episodio (lo da la visión + Claude).

Mueve el brazo: exige ALLOW_REAL_MOTION=1 y que el usuario escriba SI. Empieza siempre con
SO101_MAX_RELATIVE_TARGET=2 (en real/env.local.sh).

Uso (con el brazo conectado, desde la raíz del proyecto):
  source real/env.local.sh
  ALLOW_REAL_MOTION=1 ./.venv312/bin/python real/deploy_goal.py \
      --checkpoint real/checkpoints/act_demos_<fecha>/checkpoints/last/pretrained_model \
      --goal 0.19,0.04 --seconds 30

Sin hardware (carga el checkpoint, hace una pasada con datos falsos y sale; no mueve nada):
  ./.venv312/bin/python real/deploy_goal.py --checkpoint <ruta> --goal 0.19,0.04 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cam_utils import TRAIN_H, TRAIN_W, capture_size, to_train_size  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 30
ENV_STATE_KEY = "observation.environment_state"
STATE_KEY = "observation.state"
# Mismo mapeo que en record_goal.py: el checkpoint se entrenó con camera1/2/3.
CAM_TO_KEY = {"top": "observation.images.camera1", "wrist": "observation.images.camera2",
              "table": "observation.images.camera3"}


def parse_goal(text: str) -> np.ndarray:
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 2:
        raise ValueError("el objetivo debe ser 'x,y' en metros, p. ej. 0.19,0.04")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)


def load_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(str(checkpoint), config=cfg)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(checkpoint))
    return policy, pre, post


def check_features(policy, motors: list[str] | None) -> list[str]:
    """Comprueba que el checkpoint espera exactamente lo que vamos a enviar. Devuelve problemas."""
    problems = []
    if ENV_STATE_KEY not in policy.config.input_features:
        problems.append(f"el checkpoint no usa {ENV_STATE_KEY}: no es un ACT con objetivo")
    if STATE_KEY not in policy.config.input_features:
        problems.append(f"el checkpoint no usa {STATE_KEY}")
    if motors is not None and len(motors) != policy.config.output_features["action"].shape[0]:
        problems.append("el número de articulaciones no coincide con la acción del checkpoint")
    return problems


def build_obs(policy, state: np.ndarray, goal: np.ndarray, frames: dict[str, np.ndarray]) -> dict:
    """Observación para el preprocesador. frames: clave del checkpoint -> imagen HxWx3 uint8."""
    obs = {STATE_KEY: state.astype(np.float32), ENV_STATE_KEY: goal.astype(np.float32)}
    for key in policy.config.image_features:
        obs[key] = frames[key]
    return obs


def step_policy(policy, pre, post, obs_np: dict, device: str, motors: list[str]) -> dict:
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference

    obs = prepare_observation_for_inference(dict(obs_np), torch.device(device), task=None, robot_type="so101_follower")
    batch = pre(obs)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = post(action)
    return make_robot_action(action, {"action": {"names": motors}})


def dry_run(checkpoint: Path, goal: np.ndarray, device: str) -> int:
    policy, pre, post = load_policy(checkpoint, device)
    motors = [f"joint_{i}.pos" for i in range(policy.config.output_features["action"].shape[0])]
    problems = check_features(policy, motors)
    for p in problems:
        print(f"[FAIL] {p}")
    if problems:
        return 1
    print(f"checkpoint OK: {policy.config.type}, imágenes {list(policy.config.image_features)}")
    h, w = policy.config.image_features[next(iter(policy.config.image_features))].shape[1:]
    frames = {k: np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for k in policy.config.image_features}
    state = np.zeros(policy.config.input_features[STATE_KEY].shape[0], dtype=np.float32)
    policy.reset()
    act = step_policy(policy, pre, post, build_obs(policy, state, goal, frames), device, motors)
    print("acción de prueba (datos falsos, no se envía nada):", {k: round(v, 2) for k, v in list(act.items())[:3]}, "...")
    return 0


def confirm_motion() -> bool:
    print("\n*** El VLA/ACT va a MOVER el brazo REAL. ***")
    print("Comprueba: área libre, persona junto al brazo, corte de alimentación a mano.")
    return input("Escribe SI para continuar: ").strip() == "SI"


def real_run(checkpoint: Path, goal: np.ndarray, seconds: float, device: str) -> int:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    def env_int(name: str, default: int) -> int:
        v = os.environ.get(name, "").strip()
        return int(v) if v.lstrip("-").isdigit() else default

    policy, pre, post = load_policy(checkpoint, device)
    cam_idx = {"top": env_int("SO101_TOP_CAM", 0), "wrist": env_int("SO101_WRIST_CAM", 1),
               "table": env_int("SO101_TABLE_CAM", -1)}
    needed = {CAM_TO_KEY[c]: c for c in CAM_TO_KEY if cam_idx[c] >= 0}
    missing = [k for k in policy.config.image_features if k not in needed]
    if missing:
        print(f"[FAIL] el checkpoint necesita {missing} y no hay cámara asignada (revisa SO101_TABLE_CAM)")
        return 1

    cams = {c: OpenCVCameraConfig(index_or_path=cam_idx[c], width=capture_size(c)[0], height=capture_size(c)[1], fps=FPS)
            for c in CAM_TO_KEY if cam_idx[c] >= 0 and CAM_TO_KEY[c] in policy.config.image_features}
    robot = make_robot_from_config(SO101FollowerConfig(
        port=os.environ.get("SO101_FOLLOWER_PORT", ""),
        id=os.environ.get("SO101_FOLLOWER_ID", "so101_follower"),
        cameras=cams,
        use_degrees=True,
        max_relative_target=float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "5")),
    ))
    motors = [k for k in robot.action_features if k.endswith(".pos")]
    problems = check_features(policy, motors)
    for p in problems:
        print(f"[FAIL] {p}")
    if problems:
        return 1
    if os.environ.get("ALLOW_REAL_MOTION") != "1":
        print("Bloqueado: desplegar en el brazo necesita ALLOW_REAL_MOTION=1.")
        return 3
    if not confirm_motion():
        print("Cancelado.")
        return 3

    robot.connect()
    steps = int(seconds * FPS)
    try:
        policy.reset()
        for t in range(steps):
            t0 = time.perf_counter()
            obs = robot.get_observation()
            state = np.array([obs[m] for m in motors], dtype=np.float32)
            frames = {CAM_TO_KEY[c]: to_train_size(obs[c]) for c in cams}
            act = step_policy(policy, pre, post, build_obs(policy, state, goal, frames), device, motors)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--goal", required=True, help="objetivo 'x,y' en metros")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--device", default=os.environ.get("SO101_VLA_DEVICE", "mps"))
    ap.add_argument("--dry-run", action="store_true", help="carga el checkpoint y prueba un paso sin hardware")
    args = ap.parse_args()

    goal = parse_goal(args.goal)
    if args.dry_run:
        return dry_run(args.checkpoint, goal, args.device)
    return real_run(args.checkpoint, goal, args.seconds, args.device)


if __name__ == "__main__":
    sys.exit(main())
