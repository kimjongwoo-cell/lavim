#!/usr/bin/env python3
"""null 보정(v2)이 provenance 조건의 변별력을 바꾸는가 — v1(보정 없음)과 나란히."""
import sys, glob, torch
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.qpris_select import qpris_keep_mask

DUMP = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris/smoke_dump"
B = 512


def ov(a, b):
    return (a & b).sum().item() / max(a.sum().item(), 1)


print(f"{'case':5s} {'nullcal':9s} {'true vs wrong':>14s} {'true vs none':>13s}")
print("-" * 45)
for i, f in enumerate(sorted(glob.glob(f"{DUMP}/qpris_*.pt"))):
    D = torch.load(f, map_location="cpu", weights_only=False)
    z, counts, parents = D["z"].double(), D["token_counts"], D["parents"]
    for nc in (True, False):
        k = {c: qpris_keep_mask(z, None, counts, parents, B, cond=c, context="parent",
                                q_mode="none", nullcal=nc).bool()
             for c in ("true", "none", "wrong")}
        tag = "v2 (켬)" if nc else "v1 (끔)"
        print(f"{i:<5d} {tag:9s} {ov(k['true'], k['wrong']):14.3f} {ov(k['true'], k['none']):13.3f}")
print("\n무작위 대조 ≈ 0.25. 1.000이면 그 조건이 선택을 전혀 안 바꾼 것.")
