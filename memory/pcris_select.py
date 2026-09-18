"""Provenance-Conditioned Residual Information Selection (PCRIS) -- C1 candidate.

Notion: C1 / C2 Method Evolution -> "Provenance-Conditioned Residual Information Selection",
section 8 (Korean paper-facing draft) with sections 3-5, implemented as written:

  h_hat_i = h_i / ||h_i||  (C1-boundary visual state = Reasoner-boundary LLM-input embedding)
  U_o     = orthonormal basis of span{h_hat_p : p in V_pi(o)}  (SVD; context="ancestors" ->
            every ancestor observation along the acquisition chain)
  r_i     = (I - U_o U_o^T) h_hat_i  for a fine observation,  r_i = h_hat_i for a root
  F(S)    = log det(I + sum_{i in S} r_i r_i^T),  |S| <= B  (c_i = 1 -> cardinality)
  greedy  j* = argmax_{j not in S} F(S + j) - F(S); one global budget, no scale/crop quota,
          no question signal.

Not fixed by the spec (implementation choices, reported in info):
  * numerical rank of the parent span: singular values above
    sigma_max * max(n, d) * eps(float32)  (numpy matrix_rank default at input precision);
  * greedy by incremental Cholesky on L = I + R R^T (F(S) = log det L_SS, the marginal gain of
    j is the log Schur complement), float64; ties -> lowest index.
Controls (section 9): cond="none" -> r_i = h_hat_i for every token (plain log-det, no
provenance); cond="wrong" -> each fine observation residualized against a different parent of
the same hierarchy level (memory.pcsi_select.wrong_parents).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from memory.pcsi_select import _sanitize, wrong_parents

_EPS32 = float(torch.finfo(torch.float32).eps)


def _ancestors(par, o: int) -> list[int]:
    chain, seen = [], {o}
    p = par[o]
    while p >= 0 and p not in seen:
        chain.append(p)
        seen.add(p)
        p = par[p]
    return chain


def _basis(states: torch.Tensor, tol: float | None = None) -> torch.Tensor:
    """Orthonormal basis [d, rank] of the row space of states [n, d] (float64)."""
    if states.shape[0] == 0:
        return states.new_zeros(states.shape[1], 0)
    _u, s, vh = torch.linalg.svd(states, full_matrices=False)
    if tol is None:
        tol = float(s.max()) * max(states.shape) * _EPS32
    rank = int((s > tol).sum())
    return vh[:rank].T


def provenance_residuals(z, token_counts, parents, cond: str = "true",
                         context: str = "parent", tol: float | None = None):
    """r_i for every token (float64 [N, d]) and info {parents (effective), ranks per obs}."""
    counts = [int(c) for c in token_counts]
    n_obs, n_tok = len(counts), sum(counts)
    if int(z.shape[0]) != n_tok:
        raise ValueError(f"z has {int(z.shape[0])} rows but token_counts sum to {n_tok}")
    if context not in ("parent", "ancestors"):
        raise ValueError(f"VLMAS_PCRIS_CONTEXT must be parent|ancestors, got {context!r}")
    base = _sanitize(parents, n_obs)
    if cond == "true":
        par = base
    elif cond == "none":
        par = (-1,) * n_obs
    elif cond == "wrong":
        par = wrong_parents(base)
    else:
        raise ValueError(f"VLMAS_PCRIS_COND must be true|none|wrong, got {cond!r}")
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    H = F.normalize(z.double(), dim=-1)
    R = H.clone()
    ranks = [0] * n_obs
    for o in range(n_obs):
        if par[o] < 0:
            continue
        ctx = [par[o]] if context == "parent" else _ancestors(par, o)
        U = _basis(torch.cat([H[offs[c]:offs[c + 1]] for c in ctx]), tol)
        ranks[o] = int(U.shape[1])
        Ho = H[offs[o]:offs[o + 1]]
        R[offs[o]:offs[o + 1]] = Ho - (Ho @ U) @ U.T
    return R, {"parents": par, "ranks": ranks}


def logdet_value(R, keep) -> float:
    """F(S) = log det(I + sum_{i in S} r_i r_i^T) = log det(I_k + R_S R_S^T)."""
    Rs = R.double()[keep.to(R.device).bool()]
    if Rs.shape[0] == 0:
        return 0.0
    return float(torch.logdet(torch.eye(Rs.shape[0], dtype=torch.float64, device=Rs.device) + Rs @ Rs.T))


def logdet_greedy(R, budget: int):
    """Greedy argmax marginal gain of F; returns (keep bool [N], order, per-step gains)."""
    R = R.double()
    n_tok = int(R.shape[0])
    budget = max(0, min(int(budget), n_tok))
    keep = torch.zeros(n_tok, dtype=torch.bool, device=R.device)
    order: list[int] = []
    gains: list[float] = []
    if budget == 0:
        return keep, order, gains
    d2 = 1.0 + (R * R).sum(dim=1)                       # diag of L = I + R R^T
    C = R.new_zeros(budget, n_tok)                      # incremental Cholesky rows
    for t in range(budget):
        g = torch.log(d2.clamp(min=1.0)).masked_fill(keep, float("-inf"))
        m = float(g.max())
        j = int((g >= m).nonzero()[0])
        keep[j] = True
        order.append(j)
        gains.append(m)
        if t == budget - 1:
            break
        e = R @ R[j]
        e[j] += 1.0                                     # row j of L
        e = (e - C[:t].T @ C[:t, j]) / torch.sqrt(d2[j])
        C[t] = e
        d2 = d2 - e * e
    return keep, order, gains


def pcris_keep_mask(z, token_counts, parents, budget, magnifications=None,
                    cond: str = "true", context: str = "parent", return_info: bool = False,
                    tol: float | None = None):
    R, rinfo = provenance_residuals(z, token_counts, parents, cond=cond, context=context, tol=tol)
    keep, order, gains = logdet_greedy(R, budget)
    keep = keep.to(z.device)
    if not return_info:
        return keep
    counts = [int(c) for c in token_counts]
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    obs_of = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).tolist()
    par = rinfo["parents"]
    norms = R.norm(dim=1).float().cpu()
    fine = torch.tensor([par[o] >= 0 for o in obs_of])
    q = torch.tensor([0.05, 0.5, 0.95])
    info = {
        "cond": cond, "context": context, "B": len(order), "parents": par, "ranks": rinfo["ranks"],
        "F": round(float(sum(gains)), 4),
        "resid_norm_fine": ([round(float(v), 4) for v in torch.quantile(norms[fine], q)]
                            if bool(fine.any()) else None),
        "resid_norm_root": round(float(norms[~fine].mean()), 4) if bool((~fine).any()) else None,
        "gain_first": round(gains[0], 4) if gains else None,
        "gain_last": round(gains[-1], 4) if gains else None,
        "per_crop": [int(keep[offs[o]:offs[o + 1]].sum()) for o in range(len(counts))],
        "order": order,
    }
    if magnifications is not None and len(magnifications) == len(counts):
        mags = torch.tensor([float(magnifications[o]) for o in obs_of], device=keep.device)
        info["kept_by_mag"] = {int(m): int((keep & (mags == m)).sum()) for m in sorted(set(magnifications))}
    return keep, info
