"""Graba demostraciones de pick-and-place de objetos en sim (formato LeRobot).

Cada episodio: un objeto (bottle, box o cup) en una posición aleatoria de la mesa, el experto
(ObjectsSim.pick) lo coge y ObjectsSim.place lo deja en la zona fija "bin". Se guardan, en cada
paso de control (30 Hz):
  observation.state  (6: 5 articulaciones en grados, pinza 0..100)
  action             (6, misma escala)
  observation.images.camera1 (top), camera2 (wrist), camera3 (table)
  task               (instrucción en inglés)

El destino es fijo a propósito: la instrucción no dice dónde está la pieza, así que la
variación está en el origen (posición, objeto, orientación) y el destino es siempre la zona bin.

Uso:  ./.venv312/bin/python sim/objects_record.py --episodes 30 --out datasets/objects_pick_bin
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np

from objects_env import OBJECTS, PC_SIDE_POS, ObjectsSim

ROOT = Path(__file__).resolve().parent
REPO_ID = "local/objects_pick_bin"
DATASET_DIR = ROOT / "datasets" / "objects_pick_bin"
FPS = 30
IMG_HW = (240, 320)  # alto x ancho; igual que la resolución de las cámaras del puesto real
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMS = ["top", "wrist", "table"]
CAM_KEYS = [f"observation.images.camera{i}" for i in (1, 2, 3)]
GRIP_IDX = 5


def rad_to_norm(sim, q: np.ndarray) -> np.ndarray:
    """Radianes de la sim -> unidades de LeRobot: grados en las articulaciones, 0..100 en la pinza.
    Igual que VLARunner.rad_to_norm (sim/vla_policy.py)."""
    lo, hi = sim.model.jnt_range[:6, 0], sim.model.jnt_range[:6, 1]
    frac = (q - lo) / (hi - lo)
    out = np.degrees(q).astype(np.float64)
    out[GRIP_IDX] = np.clip(frac[GRIP_IDX], 0.0, 1.0) * 100.0
    return out


BIN_XY = PC_SIDE_POS  # zona de entrega fija
BIN_RADIUS = 0.05     # éxito si el objeto acaba a menos de 5 cm del destino
LABEL = {"bottle": "bottle", "box": "box", "cup": "cup"}
TEMPLATES = [
    "Pick up the {label} and place it in the bin",
    "Grab the {label} and put it in the bin",
]

# Zona de origen aleatoria: dentro del alcance del brazo y lejos de la zona bin.
X_RANGE = (0.13, 0.22)
Y_RANGE = (-0.12, 0.12)
Y_PARKED = 0.35  # los objetos que no se usan se aparcan fuera del alcance


def make_features(hw: tuple[int, int]) -> dict:
    img = {"dtype": "video", "shape": (*hw, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        # Objetivo numérico (xy del objeto en la mesa, lo que daría la visión). ACT lo recibe como estado del entorno.
        "observation.environment_state": {"dtype": "float32", "shape": (2,), "names": ["target_x", "target_y"]},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "observation.images.camera1": img,
        "observation.images.camera2": dict(img),
        "observation.images.camera3": dict(img),
    }


def random_layout(rng: np.random.Generator, name: str) -> dict:
    x = rng.uniform(*X_RANGE)
    y = rng.uniform(*Y_RANGE)
    yaw = rng.uniform(-np.pi / 4, np.pi / 4)
    layout = {n: (0.0, Y_PARKED + 0.1 * i, 0.0) for i, n in enumerate(OBJECTS) if n != name}
    layout[name] = (x, y, yaw)
    return layout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--res", default="240x320", help="alto x ancho de las cámaras")
    args = ap.parse_args()
    hw = tuple(int(v) for v in args.res.lower().split("x"))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sim = ObjectsSim(camera_size=hw)
    renderer = mujoco.Renderer(sim.model, *hw)

    ds = LeRobotDataset.create(
        repo_id=f"local/{args.out.name}",
        fps=FPS,
        root=args.out,
        robot_type="so101_follower",
        features=make_features(hw),
        use_videos=True,
    )

    def render(cam: str) -> np.ndarray:
        renderer.update_scene(sim.data, camera=cam)
        return renderer.render().copy()

    saved = skipped = 0
    t0 = time.time()
    while saved < args.episodes:
        name = list(OBJECTS)[rng.integers(len(OBJECTS))]
        sim.reset_objects(random_layout(rng, name))
        goal = np.asarray(sim.obj_pos(name)[:2], dtype=np.float32)  # objetivo = posición inicial del objeto
        buf: list[dict] = []

        original_tick = sim._tick

        def recording_tick(q5, grip, _orig=original_tick):
            ctrl = np.concatenate([np.asarray(q5, dtype=float), [float(grip)]])
            buf.append({
                "state": rad_to_norm(sim, sim.data.qpos[:6].copy()).astype(np.float32),
                "action": rad_to_norm(sim, ctrl).astype(np.float32),
                "cams": [render(c) for c in CAMS],
            })
            _orig(q5, grip)

        sim._tick = recording_tick
        try:
            res = sim.pick(name)
            ok = bool(res.get("ok"))
            if ok:
                sim.place(BIN_XY[0], BIN_XY[1], 0.03)
                final = sim.obj_pos(name)[:2]
                ok = float(np.linalg.norm(final - np.array(BIN_XY))) < BIN_RADIUS
        except (RuntimeError, ValueError) as err:
            ok = False
            print(f"[skip] {err}", flush=True)
        finally:
            sim._tick = original_tick

        if not ok:
            skipped += 1
            print(f"[skip] {name}: no se completó pick+place", flush=True)
            continue

        text = TEMPLATES[rng.integers(len(TEMPLATES))].format(label=LABEL[name])
        for f in buf:
            frame = {"observation.state": f["state"], "observation.environment_state": goal,
                     "action": f["action"], "task": text}
            frame.update({key: img for key, img in zip(CAM_KEYS, f["cams"])})
            ds.add_frame(frame)
        ds.save_episode()
        saved += 1
        print(f"episodio {saved}/{args.episodes}: {len(buf)} frames ({len(buf)/FPS:.1f} s) | {text} | "
              f"{time.time()-t0:.0f} s transcurridos", flush=True)

    ds.finalize()
    print(f"listo: {saved} episodios, {skipped} descartados, {time.time()-t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
