"""Pathology-Structured Visual Memory Consolidation (Method Sec. 3.3).

Task-weighted facility-location selection of ORIGINAL visual K/V columns on
the WSI-valid representation graph, replacing independent top-k pruning.
Training-free: uses frozen group representations z_c, the question-conditioned
relevance s_c the pipeline already captures, and navigation's level-0
coordinates. Env-gated at the call site; this module is pure functions.

Definitions (paper numbering):
  3.3.1  kappa(c,r) = (1 + cos(z_c, z_r)) / 2           (non-negative similarity)
  3.3.2  N(c) = {r : same image (same ROI & scale)} ∪ A(c)
         A(c) = coarser tokens whose level-0 footprint contains c's center
         (directional: coarse may represent fine, never the reverse)
  3.3.3  C(c,R) = max_{r in R ∩ N(c)} kappa(c,r);  F(R) = sum_c s_c C(c,R)
         greedy argmax with |R| <= B  ((1-1/e) guarantee)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def token_footprints(
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    grid_hws: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly split each crop's level-0 box over its (h, w) merge-group grid.

    Returns (rects [G,4] float (x0,y0,x1,y1), centers [G,2]). Images with no
    box get NaN rects (they then join no ancestor relation).
    """
    rects, centers = [], []
    for box, (grid_h, grid_w) in zip(boxes, grid_hws):
        count = grid_h * grid_w
        if box is None:
            nan = torch.full((count, 4), float("nan"))
            rects.append(nan)
            centers.append(torch.full((count, 2), float("nan")))
            continue
        x, y, width, height = (float(v) for v in box)
        cell_w, cell_h = width / grid_w, height / grid_h
        rows = torch.arange(grid_h).repeat_interleave(grid_w).float()
        cols = torch.arange(grid_w).repeat(grid_h).float()
        x0 = x + cols * cell_w
        y0 = y + rows * cell_h
        rects.append(torch.stack([x0, y0, x0 + cell_w, y0 + cell_h], dim=1))
        centers.append(torch.stack([x0 + cell_w / 2, y0 + cell_h / 2], dim=1))
    return torch.cat(rects), torch.cat(centers)


def apply_parent_shuffle(
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    magnifications: tuple[int, ...],
    seed: int = 0,
) -> tuple[tuple[int, int, int, int] | None, ...]:
    """Shuffled-parent control (Sec. 3.3 causal ablation).

    Each fine image's level-0 box is remapped, preserving its relative
    position, into a DIFFERENT coarse parent's frame (cyclic derangement over
    coarse images). Edge structure of the ancestor relation is statistically
    preserved while the physical cross-scale correspondence is destroyed —
    the desired gate is True hierarchy > Shuffled hierarchy.
    """
    import random as _random
    coarse = [
        idx for idx, m in enumerate(magnifications)
        if boxes[idx] is not None
        and any(
            m2 > m for m2, b2 in zip(magnifications, boxes) if b2 is not None
        )
    ]
    if len(coarse) < 2:
        return boxes
    order = sorted(coarse)
    _random.Random(seed).shuffle(order)
    # cyclic shift of the shuffled order => derangement over coarse images
    mapping = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}

    def parent_of(idx):
        bx, by, bw, bh = boxes[idx]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        for j in coarse:
            if j == idx or magnifications[j] >= magnifications[idx]:
                continue
            jx, jy, jw, jh = boxes[j]
            if jx <= cx < jx + jw and jy <= cy < jy + jh:
                return j
        return None

    out = list(boxes)
    for idx, box in enumerate(boxes):
        if box is None:
            continue
        parent = parent_of(idx)
        if parent is None or parent not in mapping:
            continue
        px, py, pw, ph = boxes[parent]
        qx, qy, qw, qh = boxes[mapping[parent]]
        x, y, w, h = box
        sx, sy = qw / max(pw, 1e-6), qh / max(ph, 1e-6)
        out[idx] = (
            int(qx + (x - px) * sx),
            int(qy + (y - py) * sy),
            max(1, int(w * sx)),
            max(1, int(h * sy)),
        )
    return tuple(out)


def valid_representative_mask(
    image_ids: torch.Tensor,
    magnifications: torch.Tensor,
    rects: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """[G,G] bool: entry (c, r) True iff r may represent c (Sec. 3.3.2)."""
    same_image = image_ids.unsqueeze(0) == image_ids.unsqueeze(1)  # (c, r)
    coarser = magnifications.unsqueeze(1) > magnifications.unsqueeze(0)  # m_c > m_r
    cx = centers[:, 0].unsqueeze(1)
    cy = centers[:, 1].unsqueeze(1)
    inside = (
        (rects[:, 0].unsqueeze(0) <= cx)
        & (cx < rects[:, 2].unsqueeze(0))
        & (rects[:, 1].unsqueeze(0) <= cy)
        & (cy < rects[:, 3].unsqueeze(0))
    )
    ancestor = coarser & inside
    return same_image | ancestor


@torch.no_grad()
def wsi_greedy_selection(
    features: torch.Tensor,
    relevance: torch.Tensor,
    valid: torch.Tensor,
    budget: int,
    kappa_shift: bool = True,
) -> torch.Tensor:
    """Greedy maximization of F(R) = sum_c s_c * max_{r in R∩N(c)} kappa(c,r).

    features [G,D], relevance [G] (>=0), valid [G,G] bool (c reads r), budget B.
    Returns keep mask [G] bool with exactly min(B, G) True entries.
    Fully vectorized: one [G,G] gain matrix update per greedy step.
    """
    total = features.shape[0]
    budget = min(budget, total)
    z = F.normalize(features.float(), dim=-1)
    # kappa_shift=True: spec Sec.3.3.1 (1+cos)/2 — but orthogonal tokens then
    # "cover" each other at 0.5, letting junk clusters absorb budget.
    # kappa_shift=False: max(0, cos) — orthogonal content contributes nothing.
    if kappa_shift:
        kappa = (1.0 + z @ z.T) / 2.0                  # [G,G] in [0,1]
    else:
        kappa = (z @ z.T).clamp(min=0.0)
    kappa = torch.where(valid, kappa, torch.zeros_like(kappa))
    weights = relevance.float().clamp(min=0.0).unsqueeze(1)  # [G,1]
    weighted = kappa * weights                          # row c, col r
    cover = torch.zeros(total, dtype=torch.float32, device=features.device)
    keep = torch.zeros(total, dtype=torch.bool, device=features.device)
    for _ in range(budget):
        gain = (weighted - (cover * relevance.float().clamp(min=0.0)).unsqueeze(1)).clamp(min=0.0)
        gain = torch.where(valid, gain, torch.zeros_like(gain))
        totals = gain.sum(dim=0)                        # [G] marginal gains
        totals[keep] = -1.0
        best = int(totals.argmax())
        keep[best] = True
        cover = torch.maximum(cover, kappa[:, best])
    return keep


@torch.no_grad()
def consolidation_keep_mask(
    features: torch.Tensor,
    relevance: torch.Tensor | None,
    token_counts: tuple[int, ...],
    magnifications: tuple[int, ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    grid_hws: tuple[tuple[int, int], ...],
    budget: int,
    kappa_shift: bool = True,
) -> torch.Tensor:
    """End-to-end Sec. 3.3 selection over one Reasoner append's visual groups."""
    device = features.device
    image_ids = torch.repeat_interleave(
        torch.arange(len(token_counts), device=device),
        torch.tensor(token_counts, dtype=torch.long, device=device),
    )
    mags = torch.repeat_interleave(
        torch.tensor(magnifications, dtype=torch.long, device=device),
        torch.tensor(token_counts, dtype=torch.long, device=device),
    )
    rects, centers = token_footprints(boxes, grid_hws)
    rects, centers = rects.to(device), centers.to(device)
    valid = valid_representative_mask(image_ids, mags, rects, centers)
    if relevance is None:
        relevance = torch.ones(features.shape[0], device=device)
    return wsi_greedy_selection(features, relevance, valid, budget, kappa_shift)
