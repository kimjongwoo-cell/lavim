"""Directed Cross-Scale Submodular Visual KV Selection (C1 candidate, Notion 0912).

Env-gated at the call site (VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=dcs,
Reasoner consolidation boundary, once); this module is pure functions.

Objective (one global case-level budget, uniform KV cost):

    F(S) = sum_{i in V} s_i * max_{j in S} a_ij ,   |S| <= B

Demand s (VLMAS_DCS_SAL):
    debiased (default): cross-observation debiased salience
        b_r     = median_p u_{p,r}                      (same relative grid position r)
        s_{p,r} = [ log(u_{p,r} + eps) - log(b_r + eps) ]_+
      u = question->vision attention captured at the Reasoner prefill
      (VLMAS_WSI_CONSOL_SC=1). eps = 1e-3 * mean(u) (scale-free; the spec
      leaves eps open). Crops whose grid shape is unique in the case fall back
      to b = median of that crop's own u (counted in info["b_fallback_crops"]).
    raw: s = u.        uniform: s = 1.

Directed WSI evidence substitutability a_ij (demand i <- representative j):
    k_sem(i,j)      = max(0, cos(z_i, z_j)),   z = frozen merger output at the visual positions
    k_phys(i <- j)  = |Om_i ∩ Om_j| / |Om_i|   (level-0 token footprints; a_ij != a_ji)
    a_ij = k_sem(i,j)                if m_i == m_j
         = k_sem(i,j) * k_phys(i<-j) if m_i != m_j          (paper-facing draft)
  controls:
    VLMAS_DCS_AFF=symmetric : cross-scale k_phys -> IoU(Om_i, Om_j) (generic symmetric)
    VLMAS_DCS_Z=none        : spatial-only, a_ij = k_phys(i<-j) for ALL pairs
    wrong-parent            : VLMAS_WSI_CONSOL_CONTROL=shuffled_parent remaps the
                              fine boxes upstream (memory.wsi_memory.apply_parent_shuffle)
  a_ii = 1. Tokens of a box-less crop join no cross-crop physical relation.

Selection: plain greedy on the marginal gain
    Delta(v | S) = sum_i s_i * max(0, a_iv - C_i(S)),   C_i(S) = max_{j in S} a_ij
(uniform cost => ratio rule = gain rule). No hard 5x->20x precedence.
If every remaining gain is 0 before B (all positive demand already covered),
the rest is selected with uniform demand (logged as saturated_at).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# demand
# ---------------------------------------------------------------------------
def debiased_salience(
    u: torch.Tensor,
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    eps: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """s_{p,r} = [log(u_{p,r}+eps) - log(b_r+eps)]_+ ,  b_r = median_p u_{p,r}."""
    u = u.float().clamp(min=0.0)
    if eps is None:
        eps = max(1e-3 * float(u.mean()), 1e-12)
    starts, off = [], 0
    for c in token_counts:
        starts.append(off)
        off += c
    b = torch.empty_like(u)
    groups: dict[tuple[int, int], list[int]] = {}
    for p, g in enumerate(grid_hws):
        groups.setdefault((int(g[0]), int(g[1])), []).append(p)
    fallback = 0
    for shape, crops in groups.items():
        n_tok = shape[0] * shape[1]
        if len(crops) >= 2:
            stack = torch.stack([u[starts[p]:starts[p] + n_tok] for p in crops])
            b_r = torch.quantile(stack, 0.5, dim=0)
            for p in crops:
                b[starts[p]:starts[p] + n_tok] = b_r
        else:
            p = crops[0]
            fallback += 1
            b[starts[p]:starts[p] + n_tok] = torch.quantile(
                u[starts[p]:starts[p] + n_tok], 0.5)
    s = (torch.log(u + eps) - torch.log(b + eps)).clamp(min=0.0)
    return s, {"eps": eps, "b_groups": len(groups), "b_fallback_crops": fallback}


# ---------------------------------------------------------------------------
# affinity
# ---------------------------------------------------------------------------
def physical_overlap(rects: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """rects [N,4] (x0,y0,x1,y1), NaN for box-less crops -> (inter [N,N], area [N])."""
    valid = ~torch.isnan(rects).any(dim=1)
    r = torch.nan_to_num(rects, nan=0.0)
    ix = (torch.minimum(r[:, None, 2], r[None, :, 2])
          - torch.maximum(r[:, None, 0], r[None, :, 0])).clamp(min=0.0)
    iy = (torch.minimum(r[:, None, 3], r[None, :, 3])
          - torch.maximum(r[:, None, 1], r[None, :, 1])).clamp(min=0.0)
    inter = ix * iy
    inter = inter * (valid[:, None] & valid[None, :]).to(inter.dtype)
    area = ((r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])).clamp(min=0.0)
    return inter, area


def physical_kernel(rects: torch.Tensor, aff: str = "directed") -> torch.Tensor:
    """k_phys[i, j]: directed |Om_i∩Om_j|/|Om_i|, or symmetric IoU."""
    inter, area = physical_overlap(rects)
    if aff == "symmetric":
        union = area[:, None] + area[None, :] - inter
        k = torch.where(union > 0, inter / union.clamp(min=1e-12),
                        torch.zeros_like(inter))
    elif aff == "directed":
        k = torch.where(area[:, None] > 0, inter / area[:, None].clamp(min=1e-12),
                        torch.zeros_like(inter))
    else:
        raise ValueError(f"VLMAS_DCS_AFF must be directed|symmetric, got {aff!r}")
    return k.clamp(0.0, 1.0)


def dcs_affinity(
    z: torch.Tensor | None,
    magnifications: torch.Tensor,
    rects: torch.Tensor,
    aff: str = "directed",
    z_mode: str = "merger",
) -> torch.Tensor:
    """A[i, j] = a_ij (demand i represented by j)."""
    k_phys = physical_kernel(rects, aff).to(magnifications.device)
    n = k_phys.shape[0]
    if z_mode == "none":
        a = k_phys
    elif z_mode == "merger":
        if z is None:
            raise ValueError("z_mode=merger needs token representations z")
        zn = F.normalize(z.float(), dim=-1)
        k_sem = (zn @ zn.T).clamp(min=0.0)
        same = magnifications[:, None] == magnifications[None, :]
        a = torch.where(same, k_sem, k_sem * k_phys)
    else:
        raise ValueError(f"VLMAS_DCS_Z must be merger|none, got {z_mode!r}")
    a = a.clone()
    a.fill_diagonal_(1.0)
    return a[:n, :n]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def facility_value(a: torch.Tensor, s: torch.Tensor, keep: torch.Tensor) -> float:
    """Explicit F(S) = sum_i s_i max_{j in S} a_ij (test oracle)."""
    if int(keep.sum()) == 0:
        return 0.0
    return float((s * a[:, keep].max(dim=1).values).sum())


def dcs_greedy(
    a: torch.Tensor,
    s: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, dict]:
    n = a.shape[0]
    info = {"saturated_at": None}
    if budget >= n:
        return torch.ones(n, dtype=torch.bool, device=a.device), info
    s = s.to(a.device, torch.float32)
    demand = s.clone()
    keep = torch.zeros(n, dtype=torch.bool, device=a.device)
    cur = torch.zeros(n, dtype=torch.float32, device=a.device)
    for _ in range(int(budget)):
        gain = (demand[:, None] * (a - cur[:, None]).clamp(min=0.0)).sum(dim=0)
        gain[keep] = -1.0
        v = int(gain.argmax())
        if float(gain[v]) <= 1e-12:
            if info["saturated_at"] is None:
                info["saturated_at"] = int(keep.sum())
                demand = torch.ones_like(demand)     # uniform demand for the rest
                gain = (demand[:, None] * (a - cur[:, None]).clamp(min=0.0)).sum(dim=0)
                gain[keep] = -1.0
                v = int(gain.argmax())
            if float(gain[v]) <= 1e-12:              # everything fully covered
                rest = (~keep).nonzero(as_tuple=True)[0]
                need = int(budget) - int(keep.sum())
                keep[rest[:need]] = True
                break
        keep[v] = True
        cur = torch.maximum(cur, a[:, v])
    return keep, info


def dcs_keep_mask(
    z: torch.Tensor | None,
    relevance: torch.Tensor | None,
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    magnifications: tuple[int, ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    budget: int,
    aff: str = "directed",
    z_mode: str = "merger",
    sal: str = "debiased",
    return_info: bool = False,
):
    from memory.wsi_memory import token_footprints

    n = sum(token_counts)
    device = z.device if isinstance(z, torch.Tensor) else torch.device("cpu")
    info: dict = {"aff": aff, "z": z_mode, "sal": sal, "B": int(budget)}
    # ---- demand
    if sal == "uniform" or relevance is None or int(relevance.numel()) != n:
        if sal != "uniform":
            info["sal"] = f"uniform(no_q2v; requested {sal})"
        s = torch.ones(n, device=device)
    elif sal == "raw":
        s = relevance.float().clamp(min=0.0).to(device)
    elif sal == "debiased":
        s, sinfo = debiased_salience(relevance.to(device), token_counts, grid_hws)
        info.update({k: (round(v, 8) if isinstance(v, float) else v)
                     for k, v in sinfo.items()})
    else:
        raise ValueError(f"VLMAS_DCS_SAL must be debiased|raw|uniform, got {sal!r}")
    info["s_pos"] = int((s > 0).sum())
    # ---- affinity
    rects, _centers = token_footprints(boxes, grid_hws)
    mags = torch.repeat_interleave(
        torch.tensor([float(m) for m in magnifications]),
        torch.tensor(token_counts)).to(device)
    a = dcs_affinity(z, mags, rects.to(device), aff=aff, z_mode=z_mode)
    cross = mags[:, None] != mags[None, :]
    info["cross_pairs"] = int(((a > 0) & cross).sum())
    if z_mode == "merger":
        zn = F.normalize(z.float(), dim=-1)
        off = ~torch.eye(n, dtype=torch.bool, device=device)
        info["ksem_mean"] = round(float((zn @ zn.T)[off].clamp(min=0).mean()), 4)
    # ---- selection
    keep, ginfo = dcs_greedy(a, s, budget)
    info["saturated_at"] = ginfo["saturated_at"]
    f_val = facility_value(a, s, keep)
    f_max = float(s.sum())
    info["F"] = round(f_val, 4)
    info["F_frac"] = round(f_val / f_max, 4) if f_max > 0 else None
    starts, off_ = [], 0
    for c in token_counts:
        starts.append(off_)
        off_ += c
    info["per_crop"] = [int(keep[st:st + c].sum()) for st, c in zip(starts, token_counts)]
    return (keep, info) if return_info else keep
