"""Provenance-Conditioned Submodular Information Selection (PCSI) -- C1 candidate.

Notion: C1 / C2 Method Evolution -> "Provenance-Conditioned Submodular Information
Selection (PCSI)", section 10 (Korean paper-facing draft), implemented as written:

  observations o (crops) with visual states V_o; coarse parent pi(o) from the actual
  acquisition path (Navigator source_x5_anchor -> _prune_image_parent_indices);
  P_o = V_pi(o), or empty when o has no parent;
  F_o(A_o) = I_f(A_o; Q | P_o)
           = sum_{i in V_o} [ min( max_{j in A_o} s_ij , max_{q in Q} s_iq ) - max_{p in P_o} s_ip ]_+
    (facility-location conditional mutual information, eta = nu = 1; the parent
     term is 0 when P_o is empty);
  s_ij = [cos(h_i, h_j)]_+ in the C1-boundary shared LLM-input space;
  F(S) = sum_o F_o(S & V_o),  |S| <= B  (c_i = 1 -> cardinality constraint);
  greedy  j* = argmax_{j not in S} F(S + j) - F(S), one global budget, no quotas.

Not fixed by the spec (implementation choices, reported in info):
  * Q missing -> R_Q = 1 (no relevance ceiling), info q="none".
  * When every marginal gain is 0 before |S| = B (F saturated), the remaining slots are
    filled lexicographically so the KV budget stays matched: first the same objective
    without the parent term (generic SMI, section-8 arm A), then without the relevance
    ceiling (within-observation facility-location coverage), then index order.
    fill="none" stops at saturation instead (literal |S| <= B, fewer than B kept).
  * ties -> lowest token index.
Controls (section 8): cond="none" -> P_o empty for every o (generic SMI, arm A);
cond="wrong" -> each fine observation conditioned on a different parent of the same
hierarchy level (arm C).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-9


def cos_plus(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """s_ab = [cos(a, b)]_+  -> [len(a), len(b)]."""
    return (F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T).clamp(min=0.0)


def _depths(parents: tuple[int, ...]) -> list[int]:
    depth = [-1] * len(parents)

    def walk(o: int, seen: tuple = ()) -> int:
        if depth[o] >= 0:
            return depth[o]
        p = parents[o]
        if p < 0 or p in seen:
            depth[o] = 0
        else:
            depth[o] = walk(p, seen + (o,)) + 1
        return depth[o]

    for o in range(len(parents)):
        walk(o)
    return depth


def _sanitize(parents, n: int) -> tuple[int, ...]:
    if parents is None or len(parents) != n:
        return (-1,) * n
    out = []
    for o, p in enumerate(parents):
        p = int(p)
        out.append(p if 0 <= p < n and p != o else -1)
    return tuple(out)


def wrong_parents(parents, magnifications=None) -> tuple[int, ...]:
    """Arm C: rotate every fine observation's parent to the next observation of the same
    hierarchy level as its true parent (never the true one). Single candidate -> -1."""
    par = _sanitize(parents, len(parents))
    depth = _depths(par)
    out = list(par)
    for o, p in enumerate(par):
        if p < 0:
            continue
        cands = [k for k in range(len(par)) if depth[k] == depth[p] and k != o]
        if len(cands) < 2:
            out[o] = -1
            continue
        out[o] = cands[(cands.index(p) + 1) % len(cands)]
    return tuple(out)


def locate_question_rows(pieces, marker: str = "Question stem:") -> list[int]:
    """Rows (indices into pieces) of the question text that follows `marker` up to the end
    of that line; the marker itself and whitespace-only pieces are excluded."""
    text = "".join(pieces)
    k = text.find(marker)
    if k < 0:
        return []
    start = k + len(marker)
    end = text.find("\n", start)
    end = len(text) if end < 0 else end
    rows, pos = [], 0
    for idx, piece in enumerate(pieces):
        a, b = pos, pos + len(piece)
        pos = b
        if b <= start or a >= end or not piece.strip():
            continue
        rows.append(idx)
    return rows


def _effective_parents(parents, n: int, cond: str) -> tuple[int, ...]:
    base = _sanitize(parents, n)
    if cond == "none":
        return (-1,) * n
    if cond == "wrong":
        return wrong_parents(base)
    if cond != "true":
        raise ValueError(f"VLMAS_PCSI_COND must be true|none|wrong, got {cond!r}")
    return base


def _setup(z, q, token_counts, parents, cond):
    counts = [int(c) for c in token_counts]
    n_tok = sum(counts)
    if int(z.shape[0]) != n_tok:
        raise ValueError(f"z has {int(z.shape[0])} rows but token_counts sum to {n_tok}")
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    par = _effective_parents(parents, len(counts), cond)
    zn = F.normalize(z.float(), dim=-1)
    if q is not None and int(q.numel()) > 0:
        qn = F.normalize(q.float().reshape(-1, z.shape[-1]).to(zn.device), dim=-1)
        rq = (zn @ qn.T).clamp(min=0.0).max(dim=1).values
    else:
        rq = torch.ones(n_tok, device=zn.device)
    rp = torch.zeros(n_tok, device=zn.device)
    blocks = []
    for o in range(len(counts)):
        a, b = offs[o], offs[o + 1]
        blocks.append((zn[a:b] @ zn[a:b].T).clamp(min=0.0))
        if par[o] >= 0:
            pa, pb = offs[par[o]], offs[par[o] + 1]
            rp[a:b] = (zn[a:b] @ zn[pa:pb].T).clamp(min=0.0).max(dim=1).values
    return counts, offs, par, rq, rp, blocks


def _coverage(block: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    if bool(sel.any()):
        return block[:, sel].max(dim=1).values
    return torch.zeros(block.shape[0], device=block.device)


def _block_gain(block, cov, r, p) -> torch.Tensor:
    """F_o(A + j) - F_o(A) for every column j of the observation (each term >= 0)."""
    new = torch.minimum(torch.maximum(cov[:, None], block), r[:, None])
    base = (torch.minimum(cov, r) - p).clamp(min=0.0)
    return ((new - p[:, None]).clamp(min=0.0) - base[:, None]).sum(dim=0)


def flcmi_value(z, q, token_counts, parents, keep, cond: str = "true") -> float:
    """F(S) = sum_o I_f(S & V_o; Q | P_o) for the boolean selection `keep`."""
    counts, offs, _par, rq, rp, blocks = _setup(z, q, token_counts, parents, cond)
    keep = keep.to(rq.device).bool()
    total = 0.0
    for o in range(len(counts)):
        a, b = offs[o], offs[o + 1]
        cov = _coverage(blocks[o], keep[a:b])
        total += float((torch.minimum(cov, rq[a:b]) - rp[a:b]).clamp(min=0.0).double().sum())
    return total


def pcsi_marginal_gains(z, q, token_counts, parents, keep, cond: str = "true") -> torch.Tensor:
    """F(S + j) - F(S) for every token j (selected tokens -> -inf)."""
    counts, offs, _par, rq, rp, blocks = _setup(z, q, token_counts, parents, cond)
    keep = keep.to(rq.device).bool()
    gains = torch.empty(sum(counts), device=rq.device)
    for o in range(len(counts)):
        a, b = offs[o], offs[o + 1]
        gains[a:b] = _block_gain(blocks[o], _coverage(blocks[o], keep[a:b]), rq[a:b], rp[a:b])
    return gains.masked_fill(keep, float("-inf"))


def pcsi_keep_mask(z, q, token_counts, parents, budget, magnifications=None,
                   cond: str = "true", fill: str = "lex", return_info: bool = False):
    """Greedy argmax-marginal-gain selection of |S| = B states (see module docstring)."""
    if fill not in ("lex", "none"):
        raise ValueError(f"VLMAS_PCSI_FILL must be lex|none, got {fill!r}")
    counts, offs, par, rq, rp, blocks = _setup(z, q, token_counts, parents, cond)
    dev = rq.device
    n_tok = sum(counts)
    n_obs = len(counts)
    budget = max(0, min(int(budget), n_tok))
    keep = torch.zeros(n_tok, dtype=torch.bool, device=dev)
    obs_of = torch.repeat_interleave(torch.arange(n_obs), torch.tensor(counts)).tolist()
    cov = [torch.zeros(c, device=dev) for c in counts]
    r_cur, p_cur = rq, rp
    stage, saturated_at = "pcsi", None
    fill_counts = {"smi": 0, "cov": 0, "index": 0}
    order: list[int] = []

    def all_gains():
        return torch.cat([
            _block_gain(blocks[o], cov[o], r_cur[offs[o]:offs[o + 1]], p_cur[offs[o]:offs[o + 1]])
            for o in range(n_obs)])

    if budget >= n_tok:
        keep[:] = True
        order = list(range(n_tok))
    else:
        gains = all_gains()
        while len(order) < budget:
            g = gains.masked_fill(keep, float("-inf"))
            m = float(g.max())
            if m <= _EPS:
                if stage == "pcsi":
                    saturated_at = len(order)
                    if fill == "none":
                        break
                    stage, p_cur = "smi", torch.zeros_like(rp)
                elif stage == "smi":
                    stage, r_cur = "cov", torch.ones_like(rq)
                else:
                    rest = (~keep).nonzero().flatten()[: budget - len(order)]
                    keep[rest] = True
                    order += rest.tolist()
                    fill_counts["index"] += int(rest.numel())
                    break
                gains = all_gains()
                continue
            j = int((g >= m).nonzero()[0])
            keep[j] = True
            order.append(j)
            if stage != "pcsi":
                fill_counts[stage] += 1
            o = obs_of[j]
            a, b = offs[o], offs[o + 1]
            cov[o] = torch.maximum(cov[o], blocks[o][:, j - a])
            gains[a:b] = _block_gain(blocks[o], cov[o], r_cur[a:b], p_cur[a:b])
    if not return_info:
        return keep
    f_val = 0.0
    for o in range(n_obs):
        a, b = offs[o], offs[o + 1]
        c = _coverage(blocks[o], keep[a:b])
        f_val += float((torch.minimum(c, rq[a:b]) - rp[a:b]).clamp(min=0.0).double().sum())
    f_max = float((rq - rp).clamp(min=0.0).double().sum())
    obs_t = torch.tensor(obs_of, device=dev)
    parented = torch.tensor([par[o] >= 0 for o in obs_of], device=dev)
    info = {
        "q": "question" if q is not None and int(q.numel()) > 0 else "none",
        "nq": int(q.reshape(-1, z.shape[-1]).shape[0]) if q is not None and int(q.numel()) > 0 else 0,
        "cond": cond, "fill_mode": fill, "B": budget, "parents": par,
        "saturated_at": saturated_at, "fill": dict(fill_counts),
        "F": round(f_val, 4), "F_max": round(f_max, 4),
        "F_frac": round(f_val / f_max, 4) if f_max > 0 else None,
        "rq_mean": round(float(rq.mean()), 4),
        "rp_mean_parented": round(float(rp[parented].mean()), 4) if bool(parented.any()) else None,
        "open_frac_parented": (round(float((rq[parented] > rp[parented]).float().mean()), 4)
                               if bool(parented.any()) else None),
        "per_crop": [int(keep[offs[o]:offs[o + 1]].sum()) for o in range(n_obs)],
        "order": order,
    }
    if magnifications is not None and len(magnifications) == n_obs:
        mags = torch.tensor([float(magnifications[o]) for o in obs_of], device=dev)
        info["kept_by_mag"] = {int(m): int((keep & (mags == m)).sum()) for m in sorted(set(magnifications))}
        info["rq_mean_by_mag"] = {int(m): round(float(rq[mags == m].mean()), 4)
                                  for m in sorted(set(magnifications))}
    del obs_t
    return keep, info
