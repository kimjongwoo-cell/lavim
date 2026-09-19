#!/usr/bin/env python3
"""CSIS 가 케이스당 얼마나 더 드는가 (커널 2048^2 + greedy B^2N)."""
import sys, glob, time, torch
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.csis_select import csis_keep_mask
from memory.qpris_select import qpris_keep_mask

f = sorted(glob.glob("/home/users/whddn12316/wsi_latent_0915_decode_hj/"
                     "sender_relay_exp/runs/qpris/smoke_dump/qpris_*.pt"))[0]
D = torch.load(f, map_location="cpu", weights_only=False)
z, counts, parents = D["z"].double(), D["token_counts"], D["parents"]
devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devs:
    zz = z.to(dev)
    t = time.time(); qpris_keep_mask(zz, None, counts, parents, 512, q_mode="none")
    if dev == "cuda":
        torch.cuda.synchronize()
    a = time.time() - t
    t = time.time(); csis_keep_mask(zz, counts, parents, 512,
                                    grid_hws=D["grid_hws"], boxes=D["boxes"])
    if dev == "cuda":
        torch.cuda.synchronize()
    b = time.time() - t
    print(f"{dev:5s}  PCRIS v2 {a:6.2f}s   CSIS {b:6.2f}s   차이 {b - a:+.2f}s")
