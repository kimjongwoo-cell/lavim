"""Question-Conditioned Provenance Residual Information Selection (Q-PRIS) -- C1 candidate.

Notion: C1 / C2 Method Evolution -> "Q-PRIS -- Question-Conditioned Provenance Residual
Information Selection", sections 2-5 (and the section-9 Korean draft), implemented as written:

  h_hat_i  = h_i / ||h_i||            (C1-boundary visual state, LLM-input space)
  r_i^+    = (I - P_pi(o)) h_hat_i    true-parent residual          [section 2]
  nu_i     = sqrt( mean_{p in P_o^-} ||(I - P_p) h_hat_i||^2 + eps )  within-case null
  r~_i     = r_i^+ / nu_i             provenance-calibrated residual (root: r~_i = h_hat_i)
  U_Q      = orthonormal basis of the question-token hidden states H_Q  [section 3]
  q_i      = ||U_Q^T r~_i||^2         question-subspace relevance
  F_Q(S)   = log det(I + sum_{i in S} q_i r~_i r~_i^T),  |S| <= B      [section 4]
  greedy     j* = argmax marginal gain; one global budget, no scale quota, no q2v salience.

q_i r~ r~^T = (sqrt(q_i) r~)(sqrt(q_i) r~)^T, so the objective is the plain residual log-det
of the weighted rows w_i = sqrt(q_i) r~_i and reuses memory.pcris_select.logdet_greedy.

Not fixed by the spec (implementation choices, reported in info):
  * P_o^- = the other observations that act as a coarse parent somewhere in this case
    (distinct entries of the parent map), minus pi(o). Empty -> nu_i = sqrt(1 + eps), i.e.
    no calibration; info records nullcal_obs.
  * the null always projects on a SINGLE alternative parent's span (the spec writes
    (I - P_p)), even when context="ancestors" is used for the true-parent residual.
  * numerical rank / basis: memory.pcris_select._basis (SVD at float32 tolerance).
  * ||h_hat_i|| = 1, so ||(I - P_p) h_hat_i||^2 = 1 - ||P_p h_hat_i||^2 (used for speed).
  * greedy ties -> lowest token index (inherited from logdet_greedy).

Arms (section 7): q="true" (question subspace), q="none" (q_i = 1 -> PCRIS v2, no question
term), and a shuffled-Q arm driven by the caller handing in another case's H_Q. Provenance
controls cond="true|none|wrong" are inherited from PCRIS.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from memory.pcris_select import _ancestors, _basis, logdet_greedy, logdet_value  # noqa: F401
from memory.pcsi_select import _sanitize, wrong_parents


def _offsets(token_counts) -> list[int]:
    offs = [0]
    for c in token_counts:
        offs.append(offs[-1] + int(c))
    return offs


def null_calibrated_residuals(z, token_counts, parents, cond: str = "true",
                              context: str = "parent", nullcal: bool = True,
                              tol: float | None = None, eps: float = 1e-12):
    """r~_i for every token (float64 [N, d]) + info.

    Returns (R_tilde, info) with info = {parents, ranks, nu (per token), nullcal_obs}.
    """
    counts = [int(c) for c in token_counts]
    n_obs, n_tok = len(counts), sum(counts)
    if int(z.shape[0]) != n_tok:
        raise ValueError(f"z has {int(z.shape[0])} rows but token_counts sum to {n_tok}")
    if context not in ("parent", "ancestors"):
        raise ValueError(f"context must be parent|ancestors, got {context!r}")
    base = _sanitize(parents, n_obs)
    if cond == "true":
        par = base
    elif cond == "none":
        par = (-1,) * n_obs
    elif cond == "wrong":
        par = wrong_parents(base)
    else:
        raise ValueError(f"cond must be true|none|wrong, got {cond!r}")

    offs = _offsets(counts)
    H = F.normalize(z.double(), dim=-1)
    R = H.clone()
    nu = torch.ones(n_tok, dtype=torch.float64, device=H.device)
    ranks = [0] * n_obs

    cand = sorted({int(p) for p in par if p >= 0})
    bases: dict[int, torch.Tensor] = {}

    def span(o: int) -> torch.Tensor:
        if o not in bases:
            bases[o] = _basis(H[offs[o]:offs[o + 1]], tol)
        return bases[o]

    nullcal_obs = 0
    for o in range(n_obs):
        p = par[o]
        if p < 0:
            continue                                     # root: r~ = h_hat
        ctx = [p] if context == "parent" else _ancestors(par, o)
        U = (span(p) if context == "parent"
             else _basis(torch.cat([H[offs[c]:offs[c + 1]] for c in ctx]), tol))
        ranks[o] = int(U.shape[1])
        Ho = H[offs[o]:offs[o + 1]]
        R[offs[o]:offs[o + 1]] = Ho - (Ho @ U) @ U.T
        alts = [a for a in cand if a != p]
        if nullcal and alts:
            # ||(I - P_a) h_hat||^2 = 1 - ||U_a^T h_hat||^2   (h_hat is unit norm)
            acc = torch.zeros(Ho.shape[0], dtype=torch.float64, device=H.device)
            for a in alts:
                Ua = span(a)
                acc = acc + (1.0 - ((Ho @ Ua) ** 2).sum(dim=1)).clamp(min=0.0)
            nu[offs[o]:offs[o + 1]] = torch.sqrt(acc / len(alts) + eps)
            nullcal_obs += 1
    R = R / nu[:, None]
    return R, {"parents": par, "ranks": ranks, "nu": nu, "nullcal_obs": nullcal_obs}


def question_basis(q_states, tol: float | None = None):
    """U_Q [d, m]: orthonormal basis of the question-token hidden states (SVD)."""
    if q_states is None:
        return None
    Q = q_states.double()
    if Q.ndim != 2 or Q.shape[0] == 0:
        return None
    return _basis(Q, tol)


def question_relevance(R, U_Q):
    """q_i = ||U_Q^T r~_i||^2 (float64 [N]); U_Q None or rank 0 -> all ones."""
    if U_Q is None or int(U_Q.shape[1]) == 0:
        return torch.ones(int(R.shape[0]), dtype=torch.float64, device=R.device)
    return ((R @ U_Q) ** 2).sum(dim=1)


def qpris_keep_mask(z, q_states, token_counts, parents, budget, magnifications=None,
                    cond: str = "true", context: str = "parent", q_mode: str = "true",
                    nullcal: bool = True, return_info: bool = False,
                    tol: float | None = None, eps: float = 1e-12):
    """Section 4: argmax_{|S| <= B} log det(I + sum_{i in S} q_i r~_i r~_i^T), greedy."""
    if q_mode not in ("true", "none"):
        raise ValueError(f"q_mode must be true|none, got {q_mode!r}")
    R, rinfo = null_calibrated_residuals(z, token_counts, parents, cond=cond,
                                         context=context, nullcal=nullcal, tol=tol, eps=eps)
    U_Q = question_basis(q_states, tol) if q_mode == "true" else None
    qi = question_relevance(R, U_Q)
    W = torch.sqrt(qi.clamp(min=0.0))[:, None] * R
    keep, order, gains = logdet_greedy(W, budget)
    keep = keep.to(z.device)
    if not return_info:
        return keep

    counts = [int(c) for c in token_counts]
    offs = _offsets(counts)
    obs_of = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).tolist()
    par = rinfo["parents"]
    fine = torch.tensor([par[o] >= 0 for o in obs_of])
    nu, qv = rinfo["nu"].cpu(), qi.cpu()
    rn = R.norm(dim=1).cpu()
    quant = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)

    def q3(x):
        return [round(float(v), 4) for v in torch.quantile(x.double(), quant)] if x.numel() else None

    info = {
        "cond": cond, "context": context, "q": (q_mode if U_Q is not None else "none"),
        "q_rank": (int(U_Q.shape[1]) if U_Q is not None else 0),
        "nullcal": bool(nullcal), "nullcal_obs": rinfo["nullcal_obs"],
        "B": len(order), "ranks": rinfo["ranks"], "parents": par,
        "F": round(float(sum(gains)), 4),
        "gain_first": round(gains[0], 4) if gains else None,
        "gain_last": round(gains[-1], 4) if gains else None,
        "zero_gain": int(sum(1 for g in gains if g <= 0.0)),
        "nu_fine": q3(nu[fine]) if bool(fine.any()) else None,
        "rtilde_fine": q3(rn[fine]) if bool(fine.any()) else None,
        "rtilde_root": round(float(rn[~fine].mean()), 4) if bool((~fine).any()) else None,
        "q_fine": q3(qv[fine]) if bool(fine.any()) else None,
        "q_root": q3(qv[~fine]) if bool((~fine).any()) else None,
        "per_crop": [int(keep[offs[o]:offs[o + 1]].sum()) for o in range(len(counts))],
        "order": order,
    }
    if magnifications is not None and len(magnifications) == len(counts):
        mags = torch.tensor([float(magnifications[o]) for o in obs_of], device=keep.device)
        info["kept_by_mag"] = {int(m): int((keep & (mags == m)).sum())
                               for m in sorted(set(magnifications))}
    return keep, info


__all__ = ["null_calibrated_residuals", "question_basis", "question_relevance",
           "qpris_keep_mask", "logdet_value"]
