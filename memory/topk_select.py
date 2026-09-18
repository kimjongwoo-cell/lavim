"""Score-based visual KV pruning: plain top-k by question->vision salience.

VLMAS_WSI_CONSOL_MODE=topk (needs VLMAS_WSI_CONSOL_SC=1 for the salience; without it the
relevance is uniform and the mask degenerates to the first B tokens).
  keep = the B visual tokens with the largest relevance, ties -> lower index.
  VLMAS_TOPK_PER_CROP=1 keeps a proportional share inside every crop instead of a single
  global ranking (stratified top-k), so no crop can be emptied.
This is the "score pruning" reference arm for the matched-budget accuracy comparison; it
uses no coverage, no provenance and no representation similarity.
"""
from __future__ import annotations

import torch


def salience_topk_keep_mask(relevance, token_counts, budget, per_crop: bool = False):
    counts = [int(c) for c in token_counts]
    n_tok = sum(counts)
    budget = max(0, min(int(budget), n_tok))
    if relevance is None:
        rel = torch.zeros(n_tok)
    else:
        rel = relevance.detach().float().reshape(-1).cpu()
        if int(rel.numel()) != n_tok:
            raise ValueError(f"relevance has {int(rel.numel())} entries but token_counts sum to {n_tok}")
    keep = torch.zeros(n_tok, dtype=torch.bool)
    if budget == n_tok:
        keep[:] = True
        return keep
    order = lambda v, k: torch.argsort(-v, stable=True)[:k]     # ties -> lower index
    if not per_crop:
        keep[order(rel, budget)] = True
        return keep
    offs, base = [0], 0
    for c in counts:
        base += c
        offs.append(base)
    share = [budget * c // n_tok for c in counts]
    left = budget - sum(share)
    for i in sorted(range(len(counts)), key=lambda i: (-counts[i], i))[:left]:
        share[i] += 1
    for o, (a, b) in enumerate(zip(offs[:-1], offs[1:])):
        k = min(share[o], b - a)
        if k:
            keep[a + order(rel[a:b], k)] = True
    return keep
