"""C2 #14 Early-Observed Visual Variation Coverage (EOVC). Notion 3ddd771b-c9e2-812f.

Opt-in; off = no call, byte-identical. Replaces BOTH the Pruning-B score stack
(novelty + 0.25*texture, per-crop min-max, min-8-per-image quota, rescue, cross-scale) and the
#14.6 residual-salience demand with ONE representation and ONE set objective.

Page §1-§5, on the early-observation hidden states (after observe_blocks vision blocks):

  u_hat_{i,r} = u_{i,r} / ||u_{i,r}||                 direction only, per raw patch (r = 1..4)
  mu_g        = mean over all 4*|V_g| patch states of support g
  Y_i         = [u_hat_{i,1} - mu_g, ..., u_hat_{i,4} - mu_g]        in R^{d x 4}   <- the ONLY repr
  ||Y_i||_F^2 = 4||ubar_i - mu_g||^2 + sum_r ||u_hat_{i,r} - ubar_i||^2   (exact, no coefficient)

  U_g(S_g)    = orth([Y_j]_{j in S_g})
  D_g(S_g)    = sum_{i in V_g} || (I - U_g U_g^T) Y_i ||_F^2       unexplained support variation
  S*          = argmin sum_g D_g(S cap V_g)   s.t. |S| <= B  and  |S cap V_g| >= 1 for all g

  first pick of each support   = argmax_j [D_g(empty) - D_g({j})]     (same objective, no anchor)
  then global greedy on        Delta(j|S) = D_g(S_g) - D_g(S_g u {j})

Gain without materialising the projector: with A_g the d x 4|V_g| matrix of all Y_i and
R_j = (I - U U^T) Y_j, the variation explained by adding span(R_j) is

  Delta(j|S) = tr( pinv(Yr_j^T Yr_j) (Yr_j^T Ar Ar^T Yr_j) )   on the current residuals

so each candidate needs two 4x4 matrices; all candidates of a support come from one matmul.

No salience weight, no 0.25 coefficient, no per-crop min-max, no quota beyond one token per
physical support, no rescue, no cross-scale swap, no tissue/white threshold, no supervised probe.

Env
  VLMAS_EOVC=1            enable
  VLMAS_EOVC_TOL=1e-6     relative tolerance for subspace rank and for degenerate candidates
"""
from __future__ import annotations

import os

import torch


def enabled() -> bool:
    return os.environ.get("VLMAS_EOVC", "").strip() == "1"


def tol() -> float:
    try:
        return float(os.environ.get("VLMAS_EOVC_TOL", "1e-6"))
    except ValueError:
        return 1e-6


# --------------------------------------------------------------------------- pure


def support_matrices(grouped: torch.Tensor) -> torch.Tensor:
    """[n, 4, d] early-observation patch states of ONE support -> Y [n, d, 4] (page §1).

    Rows are direction-normalised first, then centred on the support mean over all 4n states.
    """
    u = grouped if grouped.dtype in (torch.float32, torch.float64) else grouped.to(torch.float32)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    mu = u.reshape(-1, u.shape[-1]).mean(dim=0)              # [d]
    return (u - mu).transpose(1, 2).contiguous()             # [n, d, 4]


def variation(Y: torch.Tensor) -> torch.Tensor:
    """||Y_i||_F^2 per group."""
    return Y.pow(2).sum(dim=(-2, -1))


def decompose(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The page §2 split: (between-group 4||ubar-mu||^2, within-group sum_r ||u-ubar||^2)."""
    ubar = Y.mean(dim=-1, keepdim=True)                      # [n, d, 1]
    between = 4.0 * ubar.squeeze(-1).pow(2).sum(dim=-1)
    within = (Y - ubar).pow(2).sum(dim=(-2, -1))
    return between, within


def unexplained(Y: torch.Tensor, U: torch.Tensor) -> float:
    """D_g for an explicit orthonormal basis U [d, k] (reference form, used by the tests)."""
    if U.numel() == 0 or U.shape[1] == 0:
        return float(variation(Y).sum())
    R = Y - torch.einsum("dk,nke->nde", U, torch.einsum("dk,nde->nke", U, Y))
    return float(R.pow(2).sum())


def orth(M: torch.Tensor, rtol: float | None = None) -> torch.Tensor:
    """Orthonormal basis of the column space of M [d, m] by SVD rank."""
    if M.numel() == 0:
        return M.new_zeros(M.shape[0], 0)
    Mf = M if M.dtype in (torch.float32, torch.float64) else M.to(torch.float32)
    U, s, _ = torch.linalg.svd(Mf, full_matrices=False)
    if s.numel() == 0:
        return M.new_zeros(M.shape[0], 0)
    r = int((s > (rtol if rtol is not None else tol()) * float(s[0])).sum()) if float(s[0]) > 0 else 0
    return U[:, :r]


def _gains_res(Yr: torch.Tensor, rtol: float) -> torch.Tensor:
    """Delta(j|S) for every group of one support from the CURRENT RESIDUALS Yr [n, d, 4].

    With Yr_i = (I - U U^T) Y_i, adding group j contributes span(Yr_j), which is already
    orthogonal to U, so the gain is tr(pinv(Yr_j^T Yr_j) (Yr_j^T Ar Ar^T Yr_j)) with Ar the
    d x 4n matrix of all current residuals. Nothing here needs U itself.
    """
    n, d, q = Yr.shape
    G = torch.einsum("nde,ndf->nef", Yr, Yr)                     # [n, 4, 4]
    Ar = Yr.transpose(0, 1).reshape(d, n * q)                    # [d, 4n] blockwise
    Z = (Ar.transpose(0, 1) @ Ar).reshape(-1, n, q).transpose(0, 1)   # [n, 4n, 4]
    H = torch.einsum("nme,nmf->nef", Z, Z)                       # [n, 4, 4]
    Gi = torch.linalg.pinv(G, rtol=rtol, hermitian=True)
    out = torch.einsum("nef,nfe->n", Gi, H).clamp_min(0.0)
    rn = G.diagonal(dim1=-2, dim2=-1).sum(-1)                    # ||Yr_j||_F^2
    return torch.where(rn > rtol * rn.max().clamp_min(1e-30), out, torch.zeros_like(out))


def select(grouped: torch.Tensor, token_counts, budget: int, rtol: float | None = None) -> dict:
    """Page §4-§5 greedy. grouped [N, 4, d] over all supports, token_counts per support.

    The per-support residuals Yr are updated in place after every pick, so D_g is read off
    directly and no orthonormal basis has to be stored or re-orthogonalised.
    """
    rtol = tol() if rtol is None else rtol
    counts = [int(c) for c in token_counts]
    offs = torch.tensor([0] + counts, device=grouped.device).cumsum(0)
    N = int(offs[-1])
    assert grouped.shape[0] == N, (grouped.shape, N)
    B = max(0, min(int(budget), N))
    G = len(counts)

    Yr, gains, tot = [], [], []
    for g in range(G):
        Y = support_matrices(grouped[offs[g]:offs[g + 1]])
        Yr.append(Y.clone())
        tot.append(float(variation(Y).sum()))
        gains.append(_gains_res(Yr[g], rtol))
    D0 = float(sum(tot))
    keep = torch.zeros(N, dtype=torch.bool, device=grouped.device)
    picks, got, ties = [], [], 0

    def take(g: int, jl: int):
        got.append(float(gains[g][jl]))
        picks.append((g, jl, got[-1]))
        Q = orth(Yr[g][jl], rtol)                       # d x r, r <= 4, already orthogonal to U
        if Q.shape[1]:
            Yr[g] = Yr[g] - torch.einsum("dk,nke->nde", Q,
                                         torch.einsum("dk,nde->nke", Q, Yr[g]))
        keep[int(offs[g]) + jl] = True
        gains[g] = _gains_res(Yr[g], rtol)
        gains[g][keep[int(offs[g]):int(offs[g + 1])]] = -1.0

    for g in range(min(G, B)):                          # §4 feasibility: one per support
        take(g, int(torch.argmax(gains[g])))
    n_first = len(got)
    saturated_at = None
    while int(keep.sum()) < B:
        best_g, best_j, best_v = -1, -1, -1.0
        for g in range(G):
            if gains[g].numel() == 0:
                continue
            v, j = torch.max(gains[g], dim=0)
            if float(v) > best_v:
                best_g, best_j, best_v = g, int(j), float(v)
        if best_g < 0:
            break
        if best_v <= rtol * D0 / max(N, 1):             # §12 falsifier 2: residual saturation
            if saturated_at is None:
                saturated_at = int(keep.sum())
            ties += 1
            rest = (~keep).nonzero(as_tuple=True)[0]
            if rest.numel() == 0:
                break
            v_all = torch.cat([variation(Yr[g]) for g in range(G)])
            j = int(rest[int(torch.argmax(v_all[rest]))])
            g = int((offs[1:] <= j).sum())
            take(g, j - int(offs[g]))
            continue
        take(best_g, best_j)
    D_end = float(sum(float(variation(Yr[g]).sum()) for g in range(G)))
    per = [int(keep[int(offs[g]):int(offs[g + 1])].sum()) for g in range(G)]
    expl = D0 - D_end
    return {"keep": keep, "per_support": per, "picks": picks, "D0": D0, "D_end": D_end,
            "explained_frac": expl / D0 if D0 > 0 else float("nan"),
            "saturated_at": saturated_at,
            "first_gain_share": (sum(got[:n_first]) / expl) if expl > 0 else float("nan"),
            "ties": ties, "budget": B, "n": N}


def summary(info: dict) -> str:
    per = info["per_support"]
    return (f"budget={info['budget']}/{info['n']} per_support={per} "
            f"sd={torch.tensor(per, dtype=torch.float32).std(unbiased=False):.1f} "
            f"explained={info['explained_frac']:.4f} "
            f"first_share={info['first_gain_share']:.3f} "
            f"saturated_at={info['saturated_at']} ties={info['ties']}")


__all__ = ["enabled", "tol", "support_matrices", "variation", "decompose", "unexplained",
           "orth", "select", "summary"]
