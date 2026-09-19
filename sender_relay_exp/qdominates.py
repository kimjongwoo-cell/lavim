#!/usr/bin/env python3
"""q 가중이 켜지면 선택을 무엇이 결정하는가 — 부피(log-det)인가 q 순위인가."""
import sys, glob, torch
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.qpris_select import qpris_keep_mask, null_calibrated_residuals, question_basis, question_relevance

DUMP = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris/smoke_dump"
B = 512


def ov(a, b):
    return (a & b).sum().item() / max(a.sum().item(), 1)


def topb(score):
    m = torch.zeros(score.shape[0], dtype=torch.bool)
    m[score.argsort(descending=True)[:B]] = True
    return m


for f in sorted(glob.glob(f"{DUMP}/qpris_*.pt")):
    D = torch.load(f, map_location="cpu", weights_only=False)
    z, counts, parents, Q = D["z"].double(), D["token_counts"], D["parents"], D["q"].double()
    print(f"=== {f.split('/')[-1]}  q_arm={D['q_arm']}  q_text={D['q_text'][:48]!r}")

    R, _ = null_calibrated_residuals(z, counts, parents, cond="true",
                                     context="parent", nullcal=True)
    qi = question_relevance(R, question_basis(Q))

    k_true = qpris_keep_mask(z, Q, counts, parents, B, cond="true", context="parent",
                             q_mode="true", nullcal=True).bool()
    k_none = qpris_keep_mask(z, None, counts, parents, B, cond="true", context="parent",
                             q_mode="none", nullcal=True).bool()

    print(f"    q=true 선택 vs  q 단독 top-B            : {ov(k_true, topb(qi)):.3f}")
    print(f"    q=true 선택 vs  trace 순서 q*||r~||^2   : {ov(k_true, topb(qi * R.norm(dim=1) ** 2)):.3f}")
    print(f"    q=true 선택 vs  ||r~|| 단독 top-B       : {ov(k_true, topb(R.norm(dim=1))):.3f}")
    print(f"    q=true 선택 vs  q=none 선택(부피만)     : {ov(k_true, k_none):.3f}")
    torch.manual_seed(0)
    rnd = torch.zeros(R.shape[0], dtype=torch.bool)
    rnd[torch.randperm(R.shape[0])[:B]] = True
    print(f"    [대조] 무작위                            : {ov(k_true, rnd):.3f}")
    print()
