"""Role-Sensitive Visual Memory Handoff (RSVMH) — Reasoner→Answerer support reassembly.

Notion C1/C2 Method Evolution "Role-Sensitive Visual Memory Handoff (RSVMH)" (C2 candidate).
Sensitivity = closed-form support-deletion response Ŝ_g at the last Reasoner latent row
(memory/rss_diag.py ReasonerProbe, layer floor(3L/4)). Handoff = terminal restage of the SAME
retained visual K/V (VLMAS_KV_RESTAGE=1, no park) with the physical supports (one retained crop
each) placed in ascending order of the chosen score, so the most sensitive support lands nearest
the Answerer prompt. Support-internal token order and K/V pairing are kept; only keys are
re-rotated (restage), values untouched; token budget unchanged; no duplication.

    VLMAS_KV_RESTAGE=1 VLMAS_KV_RESTAGE_ORDER=<mode> [VLMAS_KV_RESTAGE_MROPE=1]
      sensitivity      ascending Ŝ  -> highest Ŝ at recency           (RSVMH)
      sensitivity_rev  descending Ŝ -> highest Ŝ farthest              (order-direction control)
      mass             ascending native attention mass of the support  (mass-estimator baseline)
      size             ascending retained token count                  (size baseline)
      page_shuffle     fixed pseudo-random page order per case         (order-content control)
      state_backed     ascending SBSH S_g = ||d zbar_R / d u_g||^2 (memory/sbsh.py) -> largest S at recency
      state_backed_rev descending S_g (direction control)
    VLMAS_KV_RESTAGE_ORDER=original (or unset) = plain full restage, unchanged.

With VLMAS_KV_RESTAGE_MROPE=1 each support keeps its native (t,h,w) geometry as a rigid block and
blocks are laid out sequentially in the new order (memory/paging.page_anchor_positions); otherwise
the default 1-D restage positions follow the permuted token order.
Off (no RSVMH order) = byte-identical to the existing restage path.
"""
from __future__ import annotations

import random

import torch

ORDERS = ("sensitivity", "sensitivity_rev", "mass", "size", "page_shuffle", "state_backed", "state_backed_rev")


def enabled_order(mode: str | None) -> bool:
    return (mode or "").strip() in ORDERS


def page_scores(proxy: dict, mode: str) -> list[float] | None:
    if mode in ("sensitivity", "sensitivity_rev"):
        values = proxy.get("s_hat")
    elif mode == "mass":
        values = proxy.get("mass_sum_heads")
    elif mode == "size":
        values = proxy.get("size")
    else:
        return None
    if values is None:
        return None
    values = [float(v) for v in values]
    return [-v for v in values] if mode == "sensitivity_rev" else values


def page_order(scores: list[float] | None, n_pages: int, mode: str, seed: int = 0) -> list[int]:
    """Page indices from farthest to nearest the consumer (ascending score; ties keep original order)."""
    if mode == "page_shuffle":
        order = list(range(n_pages))
        random.Random(seed).shuffle(order)
        return order
    assert scores is not None and len(scores) == n_pages
    return sorted(range(n_pages), key=lambda i: (scores[i], i))


def token_permutation(order: list[int], counts: list[int]) -> tuple[list[int], list[int]]:
    """Contiguous pages (original order, sizes `counts`) -> token permutation for `order`."""
    starts, offset = [], 0
    for c in counts:
        starts.append(offset)
        offset += int(c)
    perm: list[int] = []
    for p in order:
        perm.extend(range(starts[p], starts[p] + int(counts[p])))
    return perm, [int(counts[p]) for p in order]


@torch.no_grad()
def plan(bb, cols: torch.Tensor, mode: str, case_seed: int = 0):
    """Return (perm LongTensor, page_counts_in_new_order, note) or (None, None, note) to fall back.

    `cols` are the absolute visual columns the terminal restage is about to move (native order).
    The Reasoner probe must have recorded the same column set, in the same order.
    """
    ctx = getattr(bb, "_rss_ctx", None)
    if ctx is None or not getattr(ctx, "done", False) or ctx.proxy is None:
        return None, None, f"no reasoner proxy ({None if ctx is None else ctx.note})"
    vis = ctx.vis.to(cols.device)
    if int(vis.numel()) != int(cols.numel()) or not bool(torch.equal(vis, cols.long())):
        return None, None, f"column mismatch (probe {int(vis.numel())} vs restage {int(cols.numel())})"
    ids = ctx.image_ids.to(cols.device).long()
    if int(ids.numel()) and bool((ids[1:] < ids[:-1]).any()):
        return None, None, "support ids not contiguous"
    proxy = ctx.proxy
    supports = [int(s) for s in proxy["supports"]]
    counts = [int((ids == s).sum()) for s in supports]
    if sum(counts) != int(cols.numel()) or counts != [int(c) for c in proxy["size"]]:
        return None, None, "support sizes mismatch"
    if mode in ("state_backed", "state_backed_rev"):
        sb = getattr(ctx, "sbsh", None) or {}
        if sb.get("S") is None or [int(s) for s in sb.get("supports", [])] != supports:
            return None, None, f"no SBSH scores ({sb.get('skip') or sb.get('error') or 'missing / support mismatch'})"
        scores = [float(v) for v in sb["S"]]
        if mode == "state_backed_rev":
            scores = [-v for v in scores]
    else:
        scores = page_scores(proxy, mode)
    if mode != "page_shuffle" and scores is None:
        return None, None, f"no scores for mode {mode}"
    order = page_order(scores, len(supports), mode, seed=case_seed)
    perm, new_counts = token_permutation(order, counts)
    note = (f"order={mode} pages(far->near)={[supports[i] for i in order]} "
            f"mag={[proxy['mag'][i] for i in order]} "
            f"s_hat={[round(proxy['s_hat'][i], 3) for i in order]}")
    if mode in ("state_backed", "state_backed_rev"):
        note += f" S={[float(f'{abs(scores[i]):.3g}') for i in order]}"
    return torch.tensor(perm, device=cols.device, dtype=torch.long), new_counts, note


__all__ = ["ORDERS", "enabled_order", "page_scores", "page_order", "token_permutation", "plan"]
