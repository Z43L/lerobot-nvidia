#!/usr/bin/env python
"""Comprobación previa al entrenamiento de un dataset LeRobot v3.0 (solo lectura).

Uso:
  ~/lerobot312/bin/python pipeline/pretrain_check.py ROOT [--video] [--device mps]

Sale con código 1 si hay algún FAIL. WARN no bloquea, pero hay que revisarlo.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CAMS = {"front": (360, 640), "gripper": (360, 640)}
res = {"OK": 0, "WARN": 0, "FAIL": 0}


def rep(level, msg):
    res[level] += 1
    print(f"[{level:4}] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--video", action="store_true", help="decodifica un frame por vídeo y un batch")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-jump", type=float, default=50.0, help="grados/frame")
    ap.add_argument("--track-warn", type=float, default=12.0, help="error medio |action-state| (grados)")
    a = ap.parse_args()
    root = Path(a.root).expanduser()

    # 1. Metadatos
    info = json.loads((root / "meta/info.json").read_text())
    rep("OK" if info.get("codebase_version") == "v3.0" else "FAIL", f"versión {info.get('codebase_version')}")
    rep("OK" if info.get("fps") == 30 else "FAIL", f"fps {info.get('fps')}")
    feats = info["features"]
    for k in ("action", "observation.state"):
        names = feats.get(k, {}).get("names")
        rep("OK" if names == JOINTS else "FAIL", f"{k}: orden de articulaciones")
    for cam, (h, w) in CAMS.items():
        f = feats.get(f"observation.images.{cam}")
        ok = f and tuple(f["shape"][:2]) == (h, w)
        rep("OK" if ok else "FAIL", f"cámara {cam} {f['shape'] if f else 'AUSENTE'} (esperado {h}x{w})")

    # 2. Datos
    files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))
    if not files:
        rep("FAIL", "no hay parquet en data/")
        sys.exit(1)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    act = np.stack(df["action"].to_numpy())
    st = np.stack(df["observation.state"].to_numpy())
    ep = df["episode_index"].to_numpy()
    rep("OK" if len(set(ep)) == info["total_episodes"] else "FAIL",
        f"episodios {len(set(ep))} (info: {info['total_episodes']}), frames {len(df)} (info: {info['total_frames']})")
    rep("OK" if len(df) == info["total_frames"] else "FAIL", "frames coinciden con info.json")
    exp = list(range(info["total_episodes"]))
    rep("OK" if sorted(set(ep)) == exp else "FAIL", "índices de episodio contiguos 0..N-1")
    rep("OK" if not (np.isnan(act).any() or np.isnan(st).any()) else "FAIL", "sin NaN")

    # 3. Convención wrist_roll
    wr = act[:, 4]
    neg = (wr < 0).mean()
    rep("OK" if neg == 0 else "FAIL", f"wrist_roll action [{wr.min():.1f}, {wr.max():.1f}], negativos {neg:.1%} (debe ser positivo)")
    wrs = st[:, 4]
    rep("OK" if (wrs < 0).mean() == 0 else "FAIL", f"wrist_roll state [{wrs.min():.1f}, {wrs.max():.1f}]")
    g = act[:, 5]
    rep("OK" if g.max() <= 100 and g.min() >= 0 else "WARN", f"pinza [{g.min():.1f}, {g.max():.1f}] (dataset previo: ≤ ~33)")

    # 4. Saltos y error de seguimiento por episodio
    bad_jump, bad_track = [], []
    for e in sorted(set(ep)):
        m = ep == e
        if np.abs(np.diff(act[m], axis=0)).max() > a.max_jump:
            bad_jump.append(int(e))
        te = np.abs(act[m] - st[m]).mean()
        if te > a.track_warn:
            bad_track.append((int(e), round(float(te), 1)))
    rep("OK" if not bad_jump else "FAIL", f"saltos >{a.max_jump}°/frame: {bad_jump or 'ninguno'}")
    rep("OK" if not bad_track else "WARN", f"error medio |action-state| >{a.track_warn}°: {bad_track or 'ninguno'}")

    # 5. Duraciones
    dur = pd.Series(ep).value_counts().sort_index().to_numpy() / info["fps"]
    rep("OK" if dur.min() > 5 else "WARN", f"duración ep: media {dur.mean():.1f}s, min {dur.min():.1f}s, max {dur.max():.1f}s")

    # 6. Vídeo y batch (opcional, más lento)
    if a.video:
        try:
            import torch
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset("local/check", root=root)
            item = ds[0]
            for cam in CAMS:
                t = item[f"observation.images.{cam}"]
                rep("OK" if tuple(t.shape[-2:]) == CAMS[cam] else "FAIL", f"frame {cam} decodificado {tuple(t.shape)}")
            last = ds[len(ds) - 1]
            rep("OK", "último frame decodificado")
            dev = a.device
            if dev == "mps" and not torch.backends.mps.is_available():
                rep("WARN", "MPS no disponible")
            else:
                torch.zeros(1, device=dev)
                rep("OK", f"dispositivo {dev} usable")
        except Exception as ex:
            rep("FAIL", f"lectura de vídeo/dataset: {type(ex).__name__}: {ex}")

    print(f"\nResumen: {res['OK']} OK, {res['WARN']} WARN, {res['FAIL']} FAIL")
    sys.exit(1 if res["FAIL"] else 0)


if __name__ == "__main__":
    main()
