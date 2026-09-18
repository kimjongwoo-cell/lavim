"""WSI Hierarchy-Conditioned Visual Compression (C1, user spec Sec. 3.1,
2026-09-07 evening revision).

Select R* = argmin_{|R|<=B} sum_i || z_i - Proj_{S_i(R)} z_i ||^2 where
S_i(R) = span{ z_j : j in R ∩ N_WSI(i) } and N_WSI = same-crop ∪ directional
ancestors (level-0 center containment). No scores, no weights — the objective
is pure reconstruction distortion; the only knob is the budget B.

Algorithm: deterministic greedy distortion reduction with incremental
Gram-Schmidt residuals (group-OMP style). Every token keeps its current
residual res_i (distance to its OWN valid retained subspace). Selecting j
updates each i in its influence column: res_i -= (res_i . r_j)(r_j) with
r_j = normalized residual of j. Within a shared-subspace group (same crop)
this is EXACT incremental projection; across scales (ancestor edges) it is a
first-order update of the joint subspace. Selected tokens have res = 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def subspace_keep_mask(
    features: torch.Tensor,        # [N, D] frozen visual states z_i
    valid: torch.Tensor,           # [N, N] bool, (i, j): j may represent i
    budget: int,
) -> torch.Tensor:
    """Boolean keep mask [N] minimizing total valid-subspace reconstruction
    distortion under |R| <= B (greedy)."""
    X = features.float()
    N = X.shape[0]
    device = X.device
    if budget >= N:
        return torch.ones(N, dtype=torch.bool, device=device)
    res = X.clone()                              # res_i w.r.t. S_i(R)
    keep = torch.zeros(N, dtype=torch.bool, device=device)
    validf = valid.to(device=device, dtype=X.dtype)
    for _ in range(budget):
        norms = res.norm(dim=1).clamp(min=1e-8)
        rhat = res / norms.unsqueeze(1)          # candidate directions
        # gain_j = sum_i valid[i,j] * (res_i . rhat_j)^2
        M = res @ rhat.T                          # [i, j]
        gains = (M * M * validf).sum(dim=0)
        gains[keep] = -1.0
        gains = torch.where(norms < 1e-6, torch.full_like(gains, -1.0), gains)
        j = int(torch.argmax(gains))
        if gains[j] <= 0:
            break
        keep[j] = True
        r_j = rhat[j]
        coef = (res @ r_j) * validf[:, j]        # only tokens j may represent
        res = res - coef.unsqueeze(1) * r_j.unsqueeze(0)
        res[j] = 0.0
    return keep


@torch.no_grad()
def hierarchy_preserving_keep_mask(
    features: torch.Tensor,        # [N, D] frozen visual states z_i
    group_ids: torch.Tensor,       # [N] crop id per token
    budget: int,
) -> torch.Tensor:
    """Hierarchy-PRESERVING compression (user spec Sec. 3.1, 0907 final rev.).

    Reconstruction is strictly WITHIN-crop (no cross-scale surrogates:
    S_p(R) spans only same-crop retained tokens). Every crop keeps >=1 token:
    initialization picks, per crop, the token whose 1-D subspace minimizes
    that crop's distortion; the remaining budget is one global greedy on
    total distortion reduction. Only knob: budget B."""
    X = features.float()
    N = X.shape[0]
    device = X.device
    gid = group_ids.to(device)
    groups = torch.unique(gid)
    if budget >= N:
        return torch.ones(N, dtype=torch.bool, device=device)
    budget = max(budget, int(groups.numel()))   # the floor needs one per crop
    same = gid.unsqueeze(0) == gid.unsqueeze(1)  # [i, j] valid mask
    samef = same.to(X.dtype)
    res = X.clone()
    keep = torch.zeros(N, dtype=torch.bool, device=device)

    def select(j: int) -> None:
        r_j = res[j] / res[j].norm().clamp(min=1e-8)
        coef = (res @ r_j) * samef[:, j]
        res.sub_(coef.unsqueeze(1) * r_j.unsqueeze(0))
        res[j] = 0.0
        keep[j] = True

    # per-crop init: argmin_j D_p({j}) == argmax_j sum_i (z_i . z_j-hat)^2
    for g in groups.tolist():
        idx = (gid == g).nonzero(as_tuple=True)[0]
        Z = res[idx]
        Zh = Z / Z.norm(dim=1, keepdim=True).clamp(min=1e-8)
        gains = (Z @ Zh.T).pow(2).sum(dim=0)
        select(int(idx[int(torch.argmax(gains))]))

    while int(keep.sum()) < budget:
        norms = res.norm(dim=1).clamp(min=1e-8)
        rhat = res / norms.unsqueeze(1)
        M = res @ rhat.T
        gains = (M * M * samef).sum(dim=0)
        gains[keep] = -1.0
        gains = torch.where(norms < 1e-6, torch.full_like(gains, -1.0), gains)
        j = int(torch.argmax(gains))
        if gains[j] <= 0:
            break
        select(j)
    return keep


@torch.no_grad()
def total_distortion(features: torch.Tensor, valid: torch.Tensor,
                     keep: torch.Tensor) -> float:
    """Exact objective value for a keep set (per-token least squares onto its
    valid retained span) — used by tests / audits, not by selection."""
    X = features.float()
    total = 0.0
    for i in range(X.shape[0]):
        cols = (valid[i] & keep).nonzero(as_tuple=True)[0]
        if cols.numel() == 0:
            total += float(X[i].pow(2).sum())
            continue
        B = X[cols].T                            # [D, k]
        sol = torch.linalg.lstsq(B, X[i].unsqueeze(1)).solution
        total += float((X[i] - (B @ sol).squeeze(1)).pow(2).sum())
    return total
