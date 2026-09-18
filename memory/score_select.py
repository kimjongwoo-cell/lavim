"""SCoRe (Xu et al., CVPR 2026) Algorithm 1 as a matched-budget visual-KV selection arm.

VLMAS_WSI_CONSOL_MODE=score  (VLMAS_WSI_CONSOL_SC=1 for the question->vision salience;
without it the weights are uniform and this degenerates to plain k-center).
  greedy weighted k-center with cosine distance on the visual states:
      seed  s = argmax_i w_i
      step  i = argmax_i  d_min(i) * w_i^alpha ,  d_min(i) = min_{j in S} (1 - cos(x_i, x_j))
  alpha = VLMAS_SCORE_ALPHA (paper best 0.8). Ties resolve to the lower index.

This is the literature selection baseline for the matched-budget accuracy table (the
project had only ever run it offline on selection statistics: cluster coverage 0.982 vs
top-k 0.919 on GTEx-10). The computation is done on CPU in float32 so the greedy order is
reproducible across devices.
"""
from __future__ import annotations

import torch


def score_keep_mask(embeddings, relevance, token_counts, budget, alpha: float = 0.8,
                    return_info: bool = False):
    """Bool keep-mask of length sum(token_counts) selecting `budget` tokens."""
    counts = [int(c) for c in token_counts]
    n_tok = sum(counts)
    budget = max(0, min(int(budget), n_tok))
    x = embeddings.detach().float().reshape(n_tok, -1).cpu()
    if relevance is None:
        w = torch.ones(n_tok)
        rel_kind = "uniform"
    else:
        w = relevance.detach().float().reshape(-1).cpu()
        if int(w.numel()) != n_tok:
            raise ValueError(f"relevance has {int(w.numel())} entries but token_counts sum to {n_tok}")
        rel_kind = "q2v"
    keep = torch.zeros(n_tok, dtype=torch.bool)
    if budget == 0:
        return (keep, {"n": n_tok, "kept": 0, "rel": rel_kind}) if return_info else keep
    if budget == n_tok:
        keep[:] = True
        return (keep, {"n": n_tok, "kept": n_tok, "rel": rel_kind}) if return_info else keep

    xn = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    wa = torch.clamp(w, min=1e-12).pow(float(alpha))
    first = int(torch.argmax(w))                       # ties -> lower index
    chosen = [first]
    dmin = 1.0 - xn @ xn[first]
    for _ in range(budget - 1):
        comp = dmin * wa
        comp[torch.tensor(chosen, dtype=torch.long)] = float("-inf")
        nxt = int(torch.argmax(comp))
        chosen.append(nxt)
        dmin = torch.minimum(dmin, 1.0 - xn @ xn[nxt])
    keep[torch.tensor(chosen, dtype=torch.long)] = True
    if not return_info:
        return keep
    per_crop, base = [], 0
    for c in counts:
        per_crop.append(int(keep[base:base + c].sum()))
        base += c
    info = {"n": n_tok, "kept": int(keep.sum()), "rel": rel_kind, "alpha": float(alpha),
            "seed_idx": first, "per_crop": per_crop,
            "radius": round(float(dmin.max()), 5)}
    return keep, info


__all__ = ["score_keep_mask"]
