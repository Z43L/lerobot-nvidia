#!/usr/bin/env python
"""Evaluación offline de un checkpoint ACT: error en grados frente a la acción real (sin robot)."""
import argparse
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("root")
ap.add_argument("--episodes", default="0,30,45")
ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--device", default="mps")
a = ap.parse_args()

eps = [int(x) for x in a.episodes.split(",")]
ds = LeRobotDataset("local/eval", root=a.root, episodes=eps)
pol = ACTPolicy.from_pretrained(a.ckpt).to(a.device).eval()
ov = {"device_processor": {"device": a.device}}  # un checkpoint de CUDA lleva "cuda" fijado en los procesadores
pre, post = make_pre_post_processors(pol.config, a.ckpt, preprocessor_overrides=ov, postprocessor_overrides=ov)
names = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
for e in eps:
    idx = [i for i in range(len(ds)) if int(ds.hf_dataset[i]["episode_index"]) == e][:: a.stride]
    errs = []
    for i in idx:
        it = ds[i]
        batch = {k: v.unsqueeze(0) for k, v in it.items() if k.startswith("observation")}
        pol.reset()
        with torch.no_grad():
            out = post(pol.select_action(pre(batch)))
        errs.append(np.abs(out.squeeze().cpu().numpy() - it["action"].numpy()))
    m = np.mean(errs, axis=0)
    print(f"ep {e}: error medio {m.mean():.1f}°  " + " ".join(f"{n}={v:.1f}" for n, v in zip(names, m)))
