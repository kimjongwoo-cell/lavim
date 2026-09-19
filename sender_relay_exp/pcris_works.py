#!/usr/bin/env python3
"""PCRIS v2의 두 부품이 실제로 선택을 결정하는가 — 오프라인 반증.

부품 1: provenance-conditioned residual  r~_i = (I - P_pi(o)) h_i / nu_i
   반증 = cond를 true / none / wrong(틀린 부모)로 바꿔 선택 겹침을 본다.
   wrong과 겹침이 1에 가까우면 provenance 조건이 선택을 안 바꾼 것이다.

부품 2: global log-det greedy  argmax log det(I + sum r~ r~^T)
   반증 = 같은 r~ 로 단순 top-B(norm 순) 를 골라 겹침을 본다.
   겹침이 1에 가까우면 log-det의 다양성 항이 하는 일이 없다.
"""
import sys, glob, torch
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.qpris_select import qpris_keep_mask, null_calibrated_residuals

DUMP = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris/smoke_dump"
B = 512


def ov(a, b):
    return (a & b).sum().item() / max(a.sum().item(), 1)


for f in sorted(glob.glob(f"{DUMP}/qpris_*.pt")):
    D = torch.load(f, map_location="cpu", weights_only=False)
    z, counts, parents = D["z"].double(), D["token_counts"], D["parents"]
    print(f"=== {f.split('/')[-1]}  tokens={sum(counts)}  parents={parents}")

    keeps = {}
    for cond in ("true", "none", "wrong"):
        k, info = qpris_keep_mask(z, None, counts, parents, B, cond=cond,
                                  context="parent", q_mode="none",
                                  nullcal=True, return_info=True)
        keeps[cond] = k.bool()
        if cond == "true":
            print(f"    ranks(부모 span 차원)={info['ranks']}  F={info['F']}  "
                  f"gain_first={info['gain_first']} gain_last={info['gain_last']}")

    # 부품 2: 같은 잔차로 단순 top-B (log-det 없이 norm 순)
    R, _ = null_calibrated_residuals(z, counts, parents, cond="true",
                                     context="parent", nullcal=True)
    topb = torch.zeros(R.shape[0], dtype=torch.bool)
    topb[R.norm(dim=1).argsort(descending=True)[:B]] = True

    torch.manual_seed(0)
    rnd = torch.zeros(R.shape[0], dtype=torch.bool)
    rnd[torch.randperm(R.shape[0])[:B]] = True

    print(f"    [부품1] cond=true vs wrong(틀린 부모) 겹침 : {ov(keeps['true'], keeps['wrong']):.3f}")
    print(f"    [부품1] cond=true vs none(조건 없음) 겹침  : {ov(keeps['true'], keeps['none']):.3f}")
    print(f"    [부품2] log-det vs 단순 top-B(norm) 겹침   : {ov(keeps['true'], topb):.3f}")
    print(f"    [대조]  log-det vs 무작위 겹침             : {ov(keeps['true'], rnd):.3f}")
    print()
print("겹침 1.000 = 그 부품이 선택을 전혀 안 바꿨다는 뜻.  무작위 대조 ≈ B/N = 0.25.")
