"""Cross-Scale Innovation-Support Selection (CSIS) -- C1 candidate.

Notion: C1 / C2 Method Evolution -> "Cross-Scale Innovation-Support Selection"
(Development order 13), implemented as written:

  h_hat_i = h_i / ||h_i||                                        [section 1]
  r_i     = (I - U_pi(o) U_pi(o)^T) h_hat_i                      [section 2]
  nu_i    = sqrt( mean_{p in P_o^-} ||(I - P_p) h_hat_i||^2 + eps )   (PCRIS v2 null-cal)
  r~_i    = r_i / nu_i                       (root: r~_i = h_hat_i)
  G^res   = r~ r~^T                                              [section 3]
  G^spa   = kappa_x(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 sigma_x^2))   (slide coords)
  L       = G^res (.) G^spa                  Hadamard; PSD by Schur product theorem
  F(S)    = log det(I + L_S),  |S| <= B                          [section 4]
  greedy    j* = argmax marginal gain

The point of the Hadamard mask: the diagonal is untouched (kappa(x,x)=1), so each token
keeps its PCRIS residual evidence strength, and only the redundancy between tokens becomes
spatially localized. Similar morphology repeated far apart stays selectable (extent /
distribution evidence); similar morphology repeated nearby is discounted.

Limits (unit-tested):
  sigma_x -> inf  =>  G^spa = 1  =>  L = G^res           => identical to PCRIS v2
  sigma_x -> 0    =>  G^spa = I  =>  L = diag(||r~||^2)  => top-B by residual norm

Not fixed by the spec (implementation choices, reported in info):
  * sigma_x. The spec gives the RBF form but no value. Default = the fine-observation
    footprint (the smallest crop side in the case, i.e. one x20 field), so "local" means
    "inside roughly one high-magnification field". Override: VLMAS_CSIS_SIGMA (slide px),
    or VLMAS_CSIS_SIGMA_SCALE to multiply the default.
  * token coordinate x_i = the centre of the token's cell in its crop's grid, mapped to
    slide coordinates via the crop box. Tokens of an x5 crop are therefore 256 px apart
    and tokens of an x20 crop 64 px apart on a 16x16 grid.
  * P_o^- and the null reference set are inherited verbatim from memory.qpris_select.
  * greedy ties -> lowest token index (same rule as memory.pcris_select.logdet_greedy).

support="branch" -- the current canonical formulation, Notion "Question-Agnostic
Cross-Scale Innovation-Support Selection" (the RBF form above is kept there as the archived
draft and stays the default here, so existing csis arms are unchanged):

  b(o)    = top ancestor of observation o in the acquisition tree (the 5x root branch)  [0]
  a_i     = ||r~_i||^2,   u_i = r~_i / (||r~_i|| + eps)                              [1]
  B_ij    = 1[b(i) = b(j)]                  branch indicator, PSD                     [2]
  C_ij    = u_i^T u_j
  L       = D_a^{1/2} (C (.) B) D_a^{1/2}   PSD by Schur; L_ii = a_i                  [3]
  F(S)    = log det(I + L_S),  |S| <= B, greedy (exact Schur-complement gain)          [4]
No question, q2v attention, Q-V similarity or answer candidate enters anywhere [5], and there
is no sigma. Implementation choices for this form:
  * b(o) is taken from the TRUE parents after the self-parent rule, also under
    cond=none|wrong: those arms perturb the innovation reference, not the physical support.
  * a Nav2 x20 whose parent x5 was not acquired (self-parent -> root) has no 5x ancestor and
    forms its own singleton branch.
  * L is block-diagonal by branch, so F decomposes into a sum over branches (unit-tested).
"""
from __future__ import annotations

import torch

from memory.qpris_select import null_calibrated_residuals


def drop_self_parents(parents):
    """par[o] == o -> -1.

    Nav2 assigns source_x5_anchor = the crop's own patch id when the chosen x20's parent
    x5 was not itself selected (onepass_navigation_nav2.py: anchor_ids.get(parent,
    patch_id)). That lands in _prune_image_parent_indices as "my parent is me", and then
    r = (I - P_self) h_hat collapses to ~0, which silently makes the token unselectable.
    A self-parent is not a provenance relation, so treat it as a root instead: the token
    keeps r~ = h_hat and simply carries no cross-scale conditioning. Measured on the
    existing Nav2 runs: 330 / 1234 fine patches (27%), touching 75 / 156 cases.
    Returns (parents, n_dropped).
    """
    par = [int(p) for p in parents]
    dropped = sum(1 for o, p in enumerate(par) if p == o)
    return tuple(-1 if p == o else p for o, p in enumerate(par)), dropped


def token_coordinates(token_counts, grid_hws, boxes) -> torch.Tensor:
    """Slide-space centre of every visual token, float64 [N, 2].

    Token j of crop o sits at grid cell (row, col) of that crop's grid_hw, and the crop
    covers slide box (x, y, w, h). The centre of the cell is therefore
    x + (col + 0.5) * w / cols,  y + (row + 0.5) * h / rows.
    """
    counts = [int(c) for c in token_counts]
    if len(counts) != len(grid_hws) or len(counts) != len(boxes):
        raise ValueError(
            f"token_counts/grid_hws/boxes disagree: {len(counts)}/{len(grid_hws)}/{len(boxes)}")
    out = []
    for count, hw, box in zip(counts, grid_hws, boxes):
        rows, cols = int(hw[0]), int(hw[1])
        if rows * cols != count:
            raise ValueError(f"grid {rows}x{cols} does not match {count} tokens")
        bx, by, bw, bh = (float(v) for v in box)
        idx = torch.arange(count, dtype=torch.float64)
        r, c = torch.div(idx, cols, rounding_mode="floor"), idx % cols
        out.append(torch.stack([bx + (c + 0.5) * bw / cols,
                                by + (r + 0.5) * bh / rows], dim=1))
    return torch.cat(out, dim=0)


def spatial_kernel(coords: torch.Tensor, sigma: float) -> torch.Tensor:
    """kappa_x(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 sigma^2)), PSD, unit diagonal."""
    if not sigma > 0.0:
        return torch.eye(coords.shape[0], dtype=coords.dtype, device=coords.device)
    d2 = torch.cdist(coords, coords) ** 2
    return torch.exp(-d2 / (2.0 * sigma * sigma))


def default_sigma(boxes) -> float:
    """Fine-observation footprint: the smallest crop side present in this case."""
    sides = [min(float(b[2]), float(b[3])) for b in boxes if float(b[2]) > 0 and float(b[3]) > 0]
    return min(sides) if sides else 1.0


def support_branches(parents) -> tuple:
    """b(o): the top ancestor of every observation (itself for a root).

    Walks the parent chain; a parent outside [0, n), a self-parent, or a cycle ends the walk
    at the current observation, so the result is always defined.
    """
    par = [int(p) for p in parents]
    n = len(par)
    out = []
    for o in range(n):
        cur, seen = o, {o}
        while 0 <= par[cur] < n and par[cur] not in seen:
            cur = par[cur]
            seen.add(cur)
        out.append(cur)
    return tuple(out)


def support_kernel(token_counts, branches, device=None) -> torch.Tensor:
    """B_ij = 1[b(i) = b(j)] over tokens, float64 [N, N]."""
    counts = torch.tensor([int(c) for c in token_counts])
    b_tok = torch.repeat_interleave(torch.tensor([int(b) for b in branches]), counts)
    if device is not None:
        b_tok = b_tok.to(device)
    return (b_tok[:, None] == b_tok[None, :]).double()


def innovation_support_kernel(R: torch.Tensor, B: torch.Tensor, eps: float = 1e-12):
    """L = D_a^{1/2} (C (.) B) D_a^{1/2} with a = ||r~||^2, C = Gram of r~ directions.

    Returns (L, a, C).
    """
    R = R.double()
    norm = R.norm(dim=1)
    a = norm * norm
    U = R / (norm + eps)[:, None]
    C = U @ U.T
    s = torch.sqrt(a)
    L = s[:, None] * (C * B.to(C.device, C.dtype)) * s[None, :]
    return L, a, C


def logdet_greedy_kernel(L: torch.Tensor, budget: int):
    """Greedy argmax of F(S) = log det(I + L_S); returns (keep bool [N], order, gains).

    Same incremental Cholesky as memory.pcris_select.logdet_greedy, but driven by an
    explicit PSD kernel instead of a low-rank row factor, because L is a Hadamard product
    and has no cheap factor. Row j of (I + L) is L[:, j] with 1 added at position j.
    """
    L = L.double()
    n_tok = int(L.shape[0])
    budget = max(0, min(int(budget), n_tok))
    keep = torch.zeros(n_tok, dtype=torch.bool, device=L.device)
    order: list[int] = []
    gains: list[float] = []
    if budget == 0:
        return keep, order, gains
    d2 = 1.0 + torch.diagonal(L).clone()
    C = L.new_zeros(budget, n_tok)
    for t in range(budget):
        g = torch.log(d2.clamp(min=1.0)).masked_fill(keep, float("-inf"))
        m = float(g.max())
        j = int((g >= m).nonzero()[0])
        keep[j] = True
        order.append(j)
        gains.append(m)
        if t == budget - 1:
            break
        e = L[:, j].clone()
        e[j] += 1.0
        e = (e - C[:t].T @ C[:t, j]) / torch.sqrt(d2[j])
        C[t] = e
        d2 = d2 - e * e
    return keep, order, gains


def csis_keep_mask(z, token_counts, parents, budget, *, grid_hws, boxes,
                   magnifications=None, sigma: float | None = None,
                   cond: str = "true", context: str = "parent", nullcal: bool = True,
                   self_parent: str = "root", support: str = "rbf", return_info: bool = False,
                   tol: float | None = None, eps: float = 1e-12):
    """Section 4: argmax_{|S| <= B} log det(I + L_S), greedy.

    support="rbf":    L = (r~ r~^T) (.) kappa_x          (archived RBF-distance draft)
    support="branch": L = D_a^{1/2} (C (.) B) D_a^{1/2}  (canonical question-agnostic form)
    """
    if self_parent not in ("root", "keep"):
        raise ValueError(f"self_parent must be root|keep, got {self_parent!r}")
    if support not in ("rbf", "branch"):
        raise ValueError(f"support must be rbf|branch, got {support!r}")
    n_self = 0
    if self_parent == "root":
        parents, n_self = drop_self_parents(parents)
    R, rinfo = null_calibrated_residuals(z, token_counts, parents, cond=cond,
                                         context=context, nullcal=nullcal, tol=tol, eps=eps)
    coords = token_coordinates(token_counts, grid_hws, boxes).to(R.device)
    if support == "branch":
        branches = support_branches(parents)
        G_spa = support_kernel(token_counts, branches, device=R.device)
        L, a_tok, C_dir = innovation_support_kernel(R, G_spa, eps=eps)
        G_res = None
        sigma_x = None
    else:
        sigma_x = float(default_sigma(boxes) if sigma is None else sigma)
        G_res = R @ R.T
        G_spa = spatial_kernel(coords, sigma_x)
        L = G_res * G_spa
    keep, order, gains = logdet_greedy_kernel(L, budget)
    keep = keep.to(z.device)
    if not return_info:
        return keep

    counts = [int(c) for c in token_counts]
    obs_of = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    mags = (torch.tensor([int(m) for m in magnifications])[obs_of]
            if magnifications is not None else None)
    kept = keep.cpu()
    off = ~torch.eye(L.shape[0], dtype=torch.bool, device=L.device)
    quant = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)

    def q3(x):
        if not x.numel():
            return None
        x = x.reshape(-1)
        if x.numel() > 16_000_000:          # torch.quantile input limit: even stride subsample
            x = x[:: -(-x.numel() // 16_000_000)]
        return [round(float(v), 4) for v in torch.quantile(x.double(), quant)]

    info = {
        "mode": "csis", "support": support, "cond": cond, "context": context,
        "self_parent": self_parent, "self_parent_dropped": n_self,
        "sigma_x": None if sigma_x is None else round(sigma_x, 2), "B": len(order),
        "ranks": rinfo["ranks"], "parents": rinfo["parents"],
        "nullcal": bool(nullcal), "nullcal_obs": rinfo["nullcal_obs"],
        "F": round(float(sum(gains)), 4),
        "gain_first": round(gains[0], 4) if gains else None,
        "gain_last": round(gains[-1], 4) if gains else None,
    }
    if support == "rbf":
        # how much the spatial mask actually damped redundancy, off-diagonal only
        info["gspa_offdiag"] = q3(G_spa[off].cpu())
        info["gres_offdiag_abs"] = q3(G_res[off].abs().cpu())
    info["l_offdiag_abs"] = q3(L[off].abs().cpu())
    info["rtilde_norm"] = q3(R.norm(dim=1).cpu())
    info["per_crop"] = [int(kept[obs_of == o].sum()) for o in range(len(counts))]
    if support == "branch":
        n_obs = len(counts)
        info["branches"] = list(branches)
        roots = sorted(set(branches))
        tok_per_branch = {b: sum(counts[o] for o in range(n_obs) if branches[o] == b) for b in roots}
        info["n_branches"] = len(roots)
        info["singleton_branches"] = sum(1 for b in roots if sum(1 for x in branches if x == b) == 1)
        info["same_branch_pair_frac"] = round(float(G_spa[off].mean()), 4) if off.any() else None
        # innovation-direction similarity inside vs across branches (what B keeps / removes)
        same, cross = off & (G_spa > 0.5), off & (G_spa < 0.5)
        info["c_abs_same_branch"] = q3(C_dir[same].abs().cpu())
        info["c_abs_cross_branch"] = q3(C_dir[cross].abs().cpu())
        info["a"] = q3(a_tok.cpu())
        info["kept_by_branch"] = {int(b): [int(kept[torch.isin(obs_of, torch.tensor(
            [o for o in range(n_obs) if branches[o] == b]))].sum()), int(tok_per_branch[b])]
            for b in roots}
    if mags is not None:
        info["kept_by_mag"] = {int(m): int(kept[(mags == m)].sum())
                               for m in sorted({int(v) for v in mags.tolist()})}
        # spatial spread of what we kept, per magnification (extent-evidence proxy)
        sel = coords.cpu()[kept]
        info["kept_spread_px"] = (round(float(torch.cdist(sel, sel).max()), 1)
                                  if sel.shape[0] > 1 else 0.0)
    return keep, info
