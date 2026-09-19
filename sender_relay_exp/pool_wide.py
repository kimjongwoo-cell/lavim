#!/usr/bin/env python3
"""P_o^- 을 좁게(현행) vs 넓게(coarse 관찰 전부) 잡으면 뭐가 달라지나.

현행  : cand = {실제로 누군가의 부모인 관찰}          → 우리 트리에서 {0,1}, alts 1개
넓게  : cand = {모든 coarse(root) 관찰}               → {0,1,2}, alts 2개
스펙  : "같은 case 안의 alternative coarse parents" — 둘 다로 읽힌다.
"""
import sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.pcris_select import _basis, logdet_greedy
from memory.pcsi_select import _sanitize, wrong_parents

DUMP = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris/smoke_dump"
B, EPS = 512, 1e-12


def offsets(counts):
    o, s = [], 0
    for c in counts:
        o.append(s); s += c
    o.append(s)
    return o


def residuals(z, counts, parents, cond="true", pool="narrow"):
    n_obs = len(counts)
    base = _sanitize(parents, n_obs)
    par = base if cond == "true" else ((-1,) * n_obs if cond == "none" else wrong_parents(base))
    offs = offsets([int(c) for c in counts])
    H = F.normalize(z.double(), dim=-1)
    R, nu = H.clone(), torch.ones(H.shape[0], dtype=torch.float64)
    bases = {}

    def span(o):
        if o not in bases:
            bases[o] = _basis(H[offs[o]:offs[o + 1]], None)
        return bases[o]

    if pool == "narrow":                       # 현행: 실제로 부모인 관찰만
        cand = sorted({int(p) for p in par if p >= 0})
    else:                                      # 넓게: 모든 coarse(root) 관찰
        cand = sorted([o for o in range(n_obs) if base[o] < 0])

    for o in range(n_obs):
        p = par[o]
        if p < 0:
            continue
        U = span(p)
        Ho = H[offs[o]:offs[o + 1]]
        R[offs[o]:offs[o + 1]] = Ho - (Ho @ U) @ U.T
        alts = [a for a in cand if a != p]
        if alts:
            acc = torch.zeros(Ho.shape[0], dtype=torch.float64)
            for a in alts:
                Ua = span(a)
                acc = acc + (1.0 - ((Ho @ Ua) ** 2).sum(dim=1)).clamp(min=0.0)
            nu[offs[o]:offs[o + 1]] = torch.sqrt(acc / len(alts) + EPS)
    return R / nu[:, None], nu, len([a for a in cand if a != 0])


def keep(R):
    m, _, _ = logdet_greedy(R, B)
    return m.bool()


def ov(a, b):
    return (a & b).sum().item() / max(a.sum().item(), 1)


for f in sorted(glob.glob(f"{DUMP}/qpris_*.pt")):
    D = torch.load(f, map_location="cpu", weights_only=False)
    z, counts, parents = D["z"], D["token_counts"], D["parents"]
    print(f"=== {f.split('/')[-1]}  parents={parents}")
    out = {}
    for pool in ("narrow", "wide"):
        Rt, nu, nalt = residuals(z, counts, parents, "true", pool)
        Rw, _, _ = residuals(z, counts, parents, "wrong", pool)
        out[pool] = (keep(Rt), keep(Rw))
        child = nu[nu != 1.0]
        print(f"    {pool:6s} alts/child={nalt}  nu 중앙값={child.median().item():.4f} "
              f"(min {child.min().item():.4f} / max {child.max().item():.4f})")
    print(f"    좁게 vs 넓게 — 선택 겹침            : {ov(out['narrow'][0], out['wide'][0]):.3f}")
    print(f"    provenance 변별력 true vs wrong  좁게: {ov(*out['narrow']):.3f}"
          f"   넓게: {ov(*out['wide']):.3f}")
    print()
print("변별력은 낮을수록 좋다(부모를 틀리게 주면 선택이 많이 바뀌어야 함). 무작위 ≈ 0.25.")
