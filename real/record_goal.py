"""Graba demostraciones reales con el líder y un OBJETIVO por episodio (para ACT con objetivo).

Igual que real/record.sh, pero además guarda observation.environment_state = (x, y) del objetivo
en metros, en el mismo marco de mesa que sim/objects_record.py. Así el dataset real y el de sim
tienen el mismo formato. Las imágenes van a 320x240 (ver SO101_* en real/env.sh).

Cada episodio:
  1. Pide el objetivo "x,y" (metros). Lo da la visión + Claude (o la persona).
  2. Pulsa Enter para empezar; graba SO101_EPISODE_S segundos (por defecto 20) a 30 Hz.
  3. Al terminar, pide Enter para el siguiente (tiempo para reponer la pieza).

Uso (con el brazo conectado, desde la raíz del proyecto):
  source real/env.local.sh
  ALLOW_REAL_MOTION=1 ./.venv312/bin/python real/record_goal.py --episodes 30 --name demos_goal

Sin hardware (solo comprueba features y configuración, no conecta nada):
  ./.venv312/bin/python real/record_goal.py --dry-run --name test
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cam_utils import TRAIN_H, TRAIN_W, capture_size, to_train_size  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FPS = 30
IMG_HW = (240, 320)  # alto x ancho: igual que las cámaras en real/record.sh y sim
CAM_TO_KEY = {"top": "camera1", "wrist": "camera2", "table": "camera3"}  # mismos nombres que en sim
ENV_STATE_KEY = "observation.environment_state"


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return int(v) if v.strip().lstrip("-").isdigit() else default


def build_robot_config():
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    cams = {
        "top": OpenCVCameraConfig(index_or_path=env_int("SO101_TOP_CAM", 0), width=capture_size("top")[0],
                                  height=capture_size("top")[1], fps=FPS),
        "wrist": OpenCVCameraConfig(index_or_path=env_int("SO101_WRIST_CAM", 1), width=TRAIN_W, height=TRAIN_H, fps=FPS),
    }
    table = env_int("SO101_TABLE_CAM", -1)
    if table >= 0:
        cams["table"] = OpenCVCameraConfig(index_or_path=table, width=capture_size("table")[0],
                                           height=capture_size("table")[1], fps=FPS)
    return SO101FollowerConfig(
        port=os.environ.get("SO101_FOLLOWER_PORT", ""),
        id=os.environ.get("SO101_FOLLOWER_ID", "so101_follower"),
        cameras=cams,
        use_degrees=True,
        max_relative_target=float(os.environ.get("SO101_MAX_RELATIVE_TARGET", "2")),
    )


def build_leader_config():
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    return SO101LeaderConfig(
        port=os.environ.get("SO101_LEADER_PORT", ""),
        id=os.environ.get("SO101_LEADER_ID", "so101_leader"),
        use_degrees=True,
    )


def motor_names(action_features: dict) -> list[str]:
    return [k for k in action_features if k.endswith(".pos")]


def make_features(motors: list[str], cams: list[str]) -> dict:
    h, w = IMG_HW
    img = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": motors},
        "observation.environment_state": {"dtype": "float32", "shape": (2,), "names": ["target_x", "target_y"]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": motors},
    }
    for cam in cams:
        features[f"observation.images.{CAM_TO_KEY[cam]}"] = dict(img)
    return features


def parse_goal(text: str) -> np.ndarray:
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 2:
        raise ValueError("el objetivo debe ser 'x,y' en metros, p. ej. 0.19,0.04")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)


def confirm_motion() -> bool:
    print("\n*** La grabación con el líder MUEVE el brazo REAL. ***")
    print("Comprueba: área libre, persona junto al brazo, corte de alimentación a mano.")
    return input("Escribe SI para continuar: ").strip() == "SI"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--task", default="Pick up the object and place it in the bin")
    ap.add_argument("--episode-s", type=float, default=float(os.environ.get("SO101_EPISODE_S", "20")))
    ap.add_argument("--dry-run", action="store_true", help="no conecta el hardware; solo muestra features")
    args = ap.parse_args()

    out_root = ROOT / "real" / "datasets" / args.name
    motors = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    cams = ["top", "wrist"] + (["table"] if env_int("SO101_TABLE_CAM", -1) >= 0 else [])

    if args.dry_run:
        feats = make_features(motors, cams)
        for k, v in feats.items():
            print(f"{k}: {v['shape']}")
        print(f"dataset -> {out_root} (dry run: no se crea nada)")
        return 0

    if os.environ.get("ALLOW_REAL_MOTION") != "1":
        print("Bloqueado: grabar con el líder necesita ALLOW_REAL_MOTION=1.")
        return 3
    if not confirm_motion():
        print("Cancelado.")
        return 3

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    robot = make_robot_from_config(build_robot_config())
    teleop = make_teleoperator_from_config(build_leader_config())
    motors = motor_names(robot.action_features)
    ds = LeRobotDataset.create(
        repo_id=f"local/{args.name}",
        fps=FPS,
        root=out_root,
        robot_type="so101_follower",
        features=make_features(motors, cams),
        use_videos=True,
    )
    try:
        robot.connect()
        teleop.connect()
        for ep in range(args.episodes):
            goal = None
            while goal is None:
                try:
                    goal = parse_goal(input(f"[{ep + 1}/{args.episodes}] objetivo x,y (m): "))
                except ValueError as err:
                    print(err)
            input("Enter para empezar a grabar este episodio...")
            n = int(args.episode_s * FPS)
            for _ in range(n):
                t0 = time.perf_counter()
                obs = robot.get_observation()
                act = teleop.get_action()
                robot.send_action(act)
                frame = {
                    "observation.state": np.array([obs[m] for m in motors], dtype=np.float32),
                    ENV_STATE_KEY: goal,
                    "action": np.array([act[m] for m in motors], dtype=np.float32),
                    "task": args.task,
                }
                for cam in cams:
                    frame[f"observation.images.{CAM_TO_KEY[cam]}"] = to_train_size(obs[cam])
                ds.add_frame(frame)
                time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
            ds.save_episode()
            print(f"episodio {ep + 1} guardado ({n} frames, objetivo {goal.tolist()})")
            input("Repón la pieza y pulsa Enter para el siguiente episodio (Ctrl+C para terminar)...")
    except KeyboardInterrupt:
        print("\nGrabación interrumpida: se guardan los episodios terminados.")
    finally:
        ds.finalize()
        robot.disconnect()
        teleop.disconnect()
    print(f"dataset en {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
