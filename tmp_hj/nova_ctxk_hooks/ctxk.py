"""09-19 C1 kernel-side neighbourhood information (this session). memory/ files are NOT modified.

Two arms, both acting ONLY on the kernel that the existing support-log-det greedy already uses
(memory/secld_select.py: k_ij = q_i^T q_j masked to the same crop). Budget, greedy, support rule,
diagnostics stay exactly as they are.

  VLMAS_CTXK=rbf   L' = L (*) S_spatial,  S_spatial_ij = exp(-||p_i - p_j||^2 / (2 sigma^2)) on the crop
                   token grid (p = integer token coordinates). PSD by the Schur product theorem
                   (RBF is PSD, L is a Gram matrix), so the Cholesky recursion stays valid; the diagonal
                   is unchanged because S_spatial_ii = 1. Same construction as
                   LDDP (CVPR'17, arXiv:1704.03533) / DPP-NMS (BMVC'20, arXiv:2008.11451) Lemma 1,
                   with IoU replaced by the grid RBF.
  VLMAS_CTXK=ctx   two-scale similarity without touching the greedy at all:
                   c_i = normalize(GaussianPool_sigma(z)_i) inside the crop,
                   u_i = ||q_i|| * [sqrt(1-w) z_i ; sqrt(w) c_i]  ->  u_i^T u_j = ||q_i|| ||q_j||
                   [(1-w) cos(z_i,z_j) + w cos(c_i,c_j)]. Row norms (hence the diagonal) are preserved.

env: VLMAS_CTXK=off|rbf|ctx, VLMAS_CTXK_SIGMA (default 2), VLMAS_CTXK_W (default 0.6, ctx only).
Log line per selection: [CTXK] mode= sigma= w= n= crops= dim= sec=
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

STORE: dict = {"grids": None}


def cfg() -> tuple[str, float, float]:
    return (os.environ.get("VLMAS_CTXK", "off").strip().lower(),
            float(os.environ.get("VLMAS_CTXK_SIGMA", "2")),
            float(os.environ.get("VLMAS_CTXK_W", "0.6")))


def set_grids(grid_thw, merge: int = 2) -> None:
    """Token-grid shape per crop, in the same order as the selection rows."""
    thw = [tuple(int(v) for v in row) for row in torch.as_tensor(grid_thw).tolist()]
    STORE["grids"] = [(h // merge, w // merge) for t, h, w in thw for _ in range(t)]


def _coords(grids, device) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = [], []
    for gh, gw in grids:
        r = torch.arange(gh, device=device).repeat_interleave(gw).float()
        c = torch.arange(gw, device=device).repeat(gh).float()
        rows.append(r); cols.append(c)
    return torch.cat(rows), torch.cat(cols)


@torch.no_grad()
def two_scale(q: torch.Tensor, grids, sigma: float, w: float) -> torch.Tensor:
    """u_i = ||q_i|| [sqrt(1-w) z_i ; sqrt(w) c_i], c = in-crop Gaussian pool of z."""
    n = q.norm(dim=-1, keepdim=True)
    z = q / n.clamp_min(1e-12)
    rad = max(1, int(round(3 * sigma)))
    k1 = torch.arange(-rad, rad + 1, device=q.device, dtype=torch.float32)
    k1 = torch.exp(-(k1 ** 2) / (2 * sigma * sigma)); k1 = k1 / k1.sum()
    out, off = [], 0
    for gh, gw in grids:
        m = gh * gw
        g = z[off:off + m].view(gh, gw, -1).permute(2, 0, 1).unsqueeze(0)          # [1, d, gh, gw]
        g = F.conv2d(F.pad(g, (rad, rad, 0, 0), mode="replicate"), k1.view(1, 1, 1, -1).expand(g.shape[1], 1, 1, -1),
                     groups=g.shape[1])
        g = F.conv2d(F.pad(g, (0, 0, rad, rad), mode="replicate"), k1.view(1, 1, -1, 1).expand(g.shape[1], 1, -1, 1),
                     groups=g.shape[1])
        out.append(F.normalize(g.squeeze(0).permute(1, 2, 0).reshape(m, -1), dim=-1))
        off += m
    c = torch.cat(out)
    return n * torch.cat([(1.0 - w) ** 0.5 * z, w ** 0.5 * c], dim=-1)


def patch(secld):
    """Wrap secld_select.support_logdet_greedy (used by both secld_select.select and nova_select's
    logdet branch) and secld_select.evidence (only to record the crop grid shapes)."""
    mode, sigma, w = cfg()
    if mode not in ("rbf", "ctx"):
        return
    orig_ev = secld.evidence

    def evidence(pixel_values, grid_thw, *a, **k):
        set_grids(grid_thw, merge=int(k.get("merge", 2)))
        return orig_ev(pixel_values, grid_thw, *a, **k)

    secld.evidence = evidence
    orig_greedy = secld.support_logdet_greedy

    def support_logdet_greedy(q, support, budget, **kw):
        t0 = time.time()
        grids = STORE.get("grids")
        if grids is None or sum(gh * gw for gh, gw in grids) != int(q.shape[0]):
            raise RuntimeError(f"CTXK: grids {grids} do not match q {tuple(q.shape)}")
        if mode == "ctx":
            qq = two_scale(q, grids, sigma, w)
            print(f"[CTXK] mode=ctx sigma={sigma:g} w={w:g} n={int(q.shape[0])} crops={len(grids)} "
                  f"dim={int(qq.shape[1])} sec={time.time() - t0:.3f}", flush=True)
            return orig_greedy(qq, support, budget, **kw)
        rows, cols = _coords(grids, q.device)
        s2 = 2.0 * sigma * sigma
        sp = torch.exp(-((rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2) / s2)
        print(f"[CTXK] mode=rbf sigma={sigma:g} n={int(q.shape[0])} crops={len(grids)} "
              f"offdiag_mean={float(sp.mean()):.4f} sec={time.time() - t0:.3f}", flush=True)
        return _greedy_hadamard(q, sp, support, budget, **kw)

    secld.support_logdet_greedy = support_logdet_greedy
    print(f"[CTXK] patched memory.secld_select.support_logdet_greedy (mode={mode} sigma={sigma:g} w={w:g})",
          flush=True)


@torch.no_grad()
def _greedy_hadamard(q: torch.Tensor, sp: torch.Tensor, support: torch.Tensor, budget: int,
                     first_per_support: int = 0):
    """Copy of memory/secld_select.support_logdet_greedy with ONE change: the kernel column is
    multiplied by the spatial factor, k_j = (q q_j^T) (*) sp[:, j] (*) same-crop mask. The dual
    diagonal 1 + |q_i|^2 is unchanged because sp_ii = 1, so gains stay comparable to the base arm."""
    import math
    q = q.double() if q.device.type == "cpu" else q.float()
    sp = sp.to(q.dtype)
    N = int(q.shape[0])
    budget = max(0, min(int(budget), N))
    support = support.to(q.device)
    d2 = 1.0 + (q * q).sum(dim=1)
    C = torch.zeros(N, budget, dtype=q.dtype, device=q.device)
    chosen = torch.zeros(N, dtype=torch.bool, device=q.device)
    order, gains, d2s = [], [], []
    n_sup = int(support.max()) + 1 if N else 0
    forced = []
    if first_per_support > 0:
        for _ in range(first_per_support):
            forced.extend(range(n_sup))
    t = 0

    def take(j: int):
        nonlocal d2, t
        dj2 = float(d2[j])
        order.append(j)
        gains.append(math.log(max(dj2, 1e-300)))
        d2s.append(dj2)
        chosen[j] = True
        if t == budget - 1:
            t += 1
            return
        dj = math.sqrt(max(dj2, 1e-300))
        kj = (q @ q[j]) * sp[:, j] * (support == support[j]).to(q.dtype)
        e = (kj - C[:, :t] @ C[j, :t]) / dj
        e[chosen] = 0.0
        C[:, t] = e
        d2 = d2 - e * e
        t += 1

    for g in forced:
        if t >= budget:
            break
        score = d2.masked_fill(chosen | (support != g), float("-inf"))
        if not torch.isfinite(score.max()):
            continue
        take(int(torch.argmax(score)))
    while t < budget:
        score = d2.masked_fill(chosen, float("-inf"))
        take(int(torch.argmax(score)))
    return torch.tensor(order, dtype=torch.long), gains, d2s
