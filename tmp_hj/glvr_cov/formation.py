"""Grounding-formation variants for the Order 88 Phase A diagnostic (F0 / F1 / F2).

F0  VLMAS_RNLCR_S1_STEPS=0     no formation; the native latents are replayed as they are.
F1  (default)                  memory/rnlcr.py's shared prototypes: every latent is pulled
                               toward the same P_Q (mean of the top-k visual states) and away
                               from the same N_Q (mean of the bottom-k).
F2  VLMAS_GLVR_FORM=f2         each latent gets its OWN disjoint visual support: the top
                               k*m states are dealt round-robin into m blocks, so latent k is
                               pulled toward the mean of block k and away from its own negative
                               block. Same step count, learning rate and loss shape as F1 --
                               only the targets differ, which is what Gate A tests.

Nothing here is a canonical method; it exists to tell whether the shared prototype is what
collapses the trajectory (u_{V,1} ~= ... ~= u_{V,m}).
"""
from __future__ import annotations

import os

import torch


def enabled() -> bool:
    return (os.environ.get("VLMAS_GLVR_FORM", "").strip().lower() or "f1") == "f2"


def per_latent_targets(score: torch.Tensor, V: torch.Tensor, m: int, k: int):
    """score [n] relevance, V [n, D] visual states -> (P [m, D], N [m, D], top idx, bot idx).

    The top k*m states by relevance are dealt round-robin so that block j holds ranks
    j, j+m, j+2m, ...: every latent sees the same relevance range but disjoint states, and
    no state is shared between two latents. Negatives mirror this from the bottom.
    """
    n = int(score.numel())
    m = max(1, int(m))
    k = max(1, min(int(k), n // (2 * m) if n >= 2 * m else 1))
    order = torch.argsort(score.float(), descending=True, stable=True)
    top = order[: k * m]
    bot = order.flip(0)[: k * m]
    Vf = V.float()
    P = torch.stack([Vf[top[j::m]].mean(0) for j in range(m)])
    N = torch.stack([Vf[bot[j::m]].mean(0) for j in range(m)])
    return P, N, top, bot


def ground_loss_per_latent(Z: torch.Tensor, P: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """-(1/m) sum_k cos(z_k, P_k) + (1/m) sum_k cos(z_k, N_k); F1's loss with per-row targets."""
    zc = torch.nn.functional.cosine_similarity
    return -zc(Z, P, dim=-1).mean() + zc(Z, N, dim=-1).mean()


def ground_stage_per_latent(Z0: torch.Tensor, P: torch.Tensor, N: torch.Tensor,
                            steps: int, lr_rel: float):
    """Same optimiser, step count and relative learning rate as memory.rnlcr.ground_stage."""
    with torch.enable_grad():
        Z = Z0.detach().float().clone().requires_grad_(True)
        rms = float(Z0.float().pow(2).mean().sqrt())
        opt = torch.optim.Adam([Z], lr=lr_rel * rms)
        losses = []
        for _ in range(int(steps)):
            opt.zero_grad()
            loss = ground_loss_per_latent(Z, P, N)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        losses.append(float(ground_loss_per_latent(Z, P, N)))
    return Z.detach(), losses


def support_overlap(idx: torch.Tensor, m: int) -> float:
    """Mean pairwise Jaccard overlap of the m round-robin blocks (0 by construction for F2)."""
    blocks = [set(idx[j::m].tolist()) for j in range(m)]
    pairs = [(a, b) for i, a in enumerate(blocks) for b in blocks[i + 1:]]
    if not pairs:
        return 0.0
    return sum(len(a & b) / max(1, len(a | b)) for a, b in pairs) / len(pairs)
