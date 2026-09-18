"""Pathology-structured visual memory consolidation (post-correction of the
prefill pruning selection; env-gated experiment, off = no-op).

Two budget-preserving components operate on the merge-group keep mask that
``hierarchy_prefill_keep_mask`` produced, BEFORE the deep vision blocks run.
Both only revive/drop ORIGINAL model-made groups (never synthesize states),
and both swap strictly within one image, so per-image counts — and therefore
the total KV budget, the per-image quota, and the hierarchy guarantees — are
unchanged by construction.

Component 1 — Morphology-Coverage Rescue (within-scale redundancy)
    Facility-location coverage of the image's full evidence:
        F(R) = sum_c max_{r in R} cos(z_c, z_r)
    Greedy swap R' = R - {r} + {d} accepted only while the coverage gain is
    positive. Unlike a novelty criterion, typical morphology that represents
    many tokens is naturally retained (the GTEx failure mode of novelty
    scoring), while a redundant near-duplicate of an already-kept group is
    the one traded away.

Component 2 — Cross-Scale Evidence Consolidation (cross-magnification
    redundancy)
    For a high-mag image with a spatial parent (parent_indices), each child
    group's parent-redundancy is max cos to ANY parent group. Retained child
    groups that merely repeat coarse content (highest redundancy) are swapped
    for dropped child groups carrying detail invisible at the parent scale
    (lowest redundancy), while redundancy(r) - redundancy(d) > margin.
    The redundancy is used ONLY ordinally within one child image — never as a
    score multiplier — so a systematic magnification offset between scales
    cancels instead of masquerading as "new diagnostic evidence".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _image_slices(token_counts: tuple[int, ...]) -> list[tuple[int, int]]:
    slices, offset = [], 0
    for count in token_counts:
        slices.append((offset, offset + count))
        offset += count
    return slices


@torch.no_grad()
def morphology_coverage_rescue(
    features: torch.Tensor,
    keep: torch.Tensor,
    token_counts: tuple[int, ...],
    *,
    gain_eps: float | None = None,
    max_swaps_per_image: int | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Greedy budget-preserving facility-location swaps, per image.

    features: [G, D] group features (one per merge-group), keep: [G] bool.
    Returns (new keep mask, total swaps, total coverage gain). Per-image kept
    counts are provably unchanged (one drop + one add per swap, same image).

    gain_eps=None uses an image-size-adaptive floor max(0.05, 2e-3*n): a swap
    must recover a genuinely uncovered morphology (gain roughly the size of
    the newly covered cluster), not micro-recenter within an already-covered
    one (per-row gains at encoder noise level, which sum with n).
    """
    keep = keep.clone()
    total_swaps, total_gain = 0, 0.0
    normalized = F.normalize(features.float(), dim=-1)
    for start, stop in _image_slices(token_counts):
        z = normalized[start:stop]
        local_keep = keep[start:stop].clone()
        n = z.shape[0]
        kept = torch.nonzero(local_keep, as_tuple=True)[0]
        dropped = torch.nonzero(~local_keep, as_tuple=True)[0]
        if kept.numel() < 2 or dropped.numel() == 0:
            continue
        similarity = z @ z.T  # [n, n]
        eps = gain_eps if gain_eps is not None else max(0.05, 2e-3 * n)
        budget = int(kept.numel())
        cap = budget if max_swaps_per_image is None else min(budget, max_swaps_per_image)
        for _ in range(cap):
            kept = torch.nonzero(local_keep, as_tuple=True)[0]
            dropped = torch.nonzero(~local_keep, as_tuple=True)[0]
            sim_kept = similarity[:, kept]  # [n, |R|]
            top2 = sim_kept.topk(min(2, kept.numel()), dim=-1).values
            best = top2[:, 0]
            second = top2[:, 1] if top2.shape[1] > 1 else torch.full_like(best, -1.0)
            base_cover = best.sum()
            argbest = sim_kept.argmax(dim=-1)  # column into kept
            best_gain, best_pair = 0.0, None
            for r_col, r_idx in enumerate(kept.tolist()):
                # coverage with r removed: rows whose best was r fall to 2nd best
                affected = argbest == r_col
                cover_without_r = torch.where(affected, second, best)
                # add each candidate d: rows take max(cover_without_r, sim[:, d])
                cand = similarity[:, dropped]  # [n, |D|]
                covered = torch.maximum(cover_without_r.unsqueeze(1), cand)
                gains = covered.sum(dim=0) - base_cover  # [|D|]
                g, d_col = float(gains.max()), int(gains.argmax())
                if g > best_gain:
                    best_gain, best_pair = g, (r_idx, int(dropped[d_col]))
            if best_pair is None or best_gain <= eps:
                break
            r_idx, d_idx = best_pair
            local_keep[r_idx] = False
            local_keep[d_idx] = True
            total_swaps += 1
            total_gain += best_gain
        keep[start:stop] = local_keep
    return keep, total_swaps, total_gain


@torch.no_grad()
def cross_scale_consolidate(
    features: torch.Tensor,
    keep: torch.Tensor,
    token_counts: tuple[int, ...],
    parent_indices: tuple[int, ...],
    *,
    margin: float = 0.05,
    max_swaps_per_image: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Swap parent-redundant retained high-mag groups for parent-novel dropped
    ones, within each child image. Images without a distinct valid parent
    (roots / self-parents) are untouched. Returns (new keep mask, swaps)."""
    keep = keep.clone()
    total_swaps = 0
    normalized = F.normalize(features.float(), dim=-1)
    slices = _image_slices(token_counts)
    for image_index, (start, stop) in enumerate(slices):
        parent = parent_indices[image_index]
        if not (0 <= parent < len(token_counts)) or parent == image_index:
            continue
        p_start, p_stop = slices[parent]
        parent_z = normalized[p_start:p_stop]
        if parent_z.shape[0] == 0:
            continue
        child_z = normalized[start:stop]
        redundancy = (child_z @ parent_z.T).max(dim=-1).values  # [n_child]
        local_keep = keep[start:stop].clone()
        kept = torch.nonzero(local_keep, as_tuple=True)[0]
        dropped = torch.nonzero(~local_keep, as_tuple=True)[0]
        if kept.numel() == 0 or dropped.numel() == 0:
            continue
        # most parent-redundant retained vs most parent-novel dropped
        kept_order = kept[torch.argsort(redundancy[kept], descending=True, stable=True)]
        dropped_order = dropped[torch.argsort(redundancy[dropped], stable=True)]
        pairs = min(kept_order.numel(), dropped_order.numel())
        if max_swaps_per_image is not None:
            pairs = min(pairs, max_swaps_per_image)
        for pair in range(pairs):
            r_idx = int(kept_order[pair])
            d_idx = int(dropped_order[pair])
            if float(redundancy[r_idx] - redundancy[d_idx]) <= margin:
                break
            local_keep[r_idx] = False
            local_keep[d_idx] = True
            total_swaps += 1
        keep[start:stop] = local_keep
    return keep, total_swaps
