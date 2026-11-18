"""Graba demostraciones de pick-and-place de discos de Hanoi en formato LeRobot.

Cada episodio: una configuración legal aleatoria de los 3 discos en los pivotes, y un
movimiento legal (disco superior de un pivote a otro). El experto (hanoi_expert.py)
ejecuta el movimiento y se guardan, en cada paso de control (30 Hz):
  observation.state  (6: 5 articulaciones en grados, pinza 0..100; como LeRobot use_degrees=True)
  action             (6, comandos normalizados)
  observation.images.camera1 (top), camera2 (wrist), camera3 (table)   (480x640)
  Los nombres coinciden con el checkpoint SO-101 (camera1/2/3), así no hace falta renombrar.
  task               (instrucción en inglés)

Uso:  ./.venv312/bin/python sim/hanoi_record.py --episodes 150
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

from hanoi_env import DISK_ORDER, DISKS, PEG_COLOR_NAME, PEGS, HanoiSim
from hanoi_expert import HanoiExpert

ROOT = Path(__file__).resolve().parent
REPO_ID = "local/hanoi_pick_place"
DATASET_DIR = ROOT / "datasets" / "hanoi_pick_place"
FPS = 30
IMG_HW = (480, 640)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TEMPLATES = [
    "Pick up the {color} disk and place it on the {peg} peg",
    "Move the {color} disk to the {peg} peg",
]
CAMS = ["top", "wrist", "table"]  # renders de MuJoCo
CAM_KEYS = [f"observation.images.camera{i}" for i in (1, 2, 3)]  # nombres del checkpoint


def instruction(color: str, dst: str, rng: np.random.Generator) -> str:
    tpl = TEMPLATES[rng.integers(len(TEMPLATES))]
    return tpl.format(color=color, peg=PEG_COLOR_NAME[dst])


def random_move(rng: np.random.Generator, n: int, only: str | None = None):
    """Configuración legal aleatoria de n discos + un movimiento legal (color, origen, destino).
    Con `only`, el disco movido es siempre ese color (se reintenta hasta que sea movible)."""
    while True:
        config, color, src, dst = _random_move(rng, n)
        if only is None or color == only:
            return config, color, src, dst


def _random_move(rng: np.random.Generator, n: int):
    pegs = list(PEGS)
    config = {p: [] for p in pegs}
    for color in reversed(DISK_ORDER[:n]):  # apilar de grande a pequeño: siempre legal
        config[pegs[rng.integers(len(pegs))]].append(color)
    options = []  # movimientos legales (origen, destino)
    for src in pegs:
        if not config[src]:
            continue
        color = config[src][-1]
        for dst in pegs:
            if dst != src and (not config[dst] or DISKS[config[dst][-1]]["size"] > DISKS[color]["size"]):
                options.append((src, dst))
    src, dst = options[rng.integers(len(options))]
    color = config[src][-1]
    return config, color, src, dst


def make_features() -> dict:
    names = JOINTS
    img = {"dtype": "video", "shape": (*IMG_HW, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.camera1": img,
        "observation.images.camera2": dict(img),
        "observation.images.camera3": dict(img),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--disks", type=int, nargs="+", default=[3, 4, 5], help="números de discos a usar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--only", choices=list(DISK_ORDER), default=None,
                    help="grabar sólo movimientos de este color (un VLA por color)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.only and args.out == DATASET_DIR:
        args.out = ROOT / "datasets" / f"hanoi_{args.only}"
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sim = HanoiSim(camera_size=IMG_HW)
    renderer = mujoco.Renderer(sim.model, *IMG_HW)
    expert = HanoiExpert(sim, rng=rng, jitter=0.0)

    ds = LeRobotDataset.create(
        repo_id=f"local/{args.out.name}",
        fps=FPS,
        root=args.out,
        robot_type="so101_follower",
        features=make_features(),
        use_videos=True,
    )

    def render(cam: str) -> np.ndarray:
        renderer.update_scene(sim.data, camera=cam)
        return renderer.render().copy()

    saved = skipped = 0
    while saved < args.episodes:
        n = int(rng.choice(args.disks))
        config, color, src, dst = random_move(rng, n, args.only)
        sim.reset_hanoi(config)
        buf: list[dict] = []

        def on_step(state: np.ndarray, action: np.ndarray) -> None:
            buf.append({"state": state, "action": action, "cams": [render(c) for c in CAMS]})

        expert.on_step = on_step
        text = instruction(color, dst, rng)
        try:
            expert.move_disk(color, src, dst)
        except (RuntimeError, ValueError) as err:
            skipped += 1
            print(f"[skip] {err}", flush=True)
            sim.reset_hanoi(n=3)
            continue
        if not sim.get_hanoi_state()[dst] or sim.get_hanoi_state()[dst][-1] != color:
            skipped += 1
            print(f"[skip] {color} no llegó a {dst}", flush=True)
            sim.reset_hanoi(n=3)
            continue

        for f in buf:
            frame = {"observation.state": f["state"], "action": f["action"], "task": text}
            frame.update({key: img for key, img in zip(CAM_KEYS, f["cams"])})
            ds.add_frame(frame)
        ds.save_episode()
        saved += 1
        print(f"episodio {saved}/{args.episodes}: {len(buf)} frames | {text}", flush=True)

    ds.finalize()
    print(f"listo: {saved} episodios, {skipped} descartados -> {args.out}")


if __name__ == "__main__":
    main()
