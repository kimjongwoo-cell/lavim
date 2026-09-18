"""C1 = Observation-Stratified Visual Compression (method draft 2026-09-07).

No importance estimation of any kind (saliency/EMA/reconstruction/medoid all
measured dead). Each Navigator-selected crop keeps its 2-D spatial support and
is DOWNSAMPLED (spatial thinning), never deduplicated. One fixed global budget
B shared across crops, b_p ∝ |V_p| (>=1 each). Deterministic.

Also provides the Stage-1 falsification controls:
  random_budget  — uniform random keep of B tokens (seeded, matched budget)
  flat           — every-k-th token over the concatenated sequence, ignoring
                   crop boundaries
"""
from __future__ import annotations

import torch


def allocate_budget(counts: list[int], budget: int) -> list[int]:
    """b_p ∝ |V_p| with integer rounding, each >=1, sum == budget."""
    total = sum(counts)
    budget = max(budget, len(counts))
    raw = [budget * c / total for c in counts]
    alloc = [max(1, int(r)) for r in raw]
    # distribute the remainder by largest fractional part; trim from largest
    # allocations if over (never below 1).
    frac = sorted(range(len(counts)), key=lambda i: raw[i] - int(raw[i]),
                  reverse=True)
    i = 0
    while sum(alloc) < budget:
        alloc[frac[i % len(frac)]] += 1
        i += 1
    order = sorted(range(len(counts)), key=lambda i: alloc[i], reverse=True)
    i = 0
    while sum(alloc) > budget:
        j = order[i % len(order)]
        if alloc[j] > 1:
            alloc[j] -= 1
        i += 1
    return alloc


def equal_area_grid_indices(grid_h: int, grid_w: int, keep: int) -> list[int]:
    """Deterministic equal-area sampling of `keep` positions on an H×W grid.

    Chooses a (r, c) cell resolution near `keep` with the grid's aspect ratio,
    takes the token nearest each cell center, then pads (largest spatial gap
    via evenly spaced leftovers) or trims (row-major tail) to exactly `keep`."""
    total = grid_h * grid_w
    if keep >= total:
        return list(range(total))
    import math
    r = max(1, min(grid_h, round(math.sqrt(keep * grid_h / max(grid_w, 1)))))
    c = max(1, min(grid_w, math.ceil(keep / r)))
    picked: list[int] = []
    seen = set()
    for i in range(r):
        for j in range(c):
            y = min(grid_h - 1, int((i + 0.5) * grid_h / r))
            x = min(grid_w - 1, int((j + 0.5) * grid_w / c))
            idx = y * grid_w + x
            if idx not in seen:
                seen.add(idx)
                picked.append(idx)
    if len(picked) > keep:
        picked = picked[:keep]
    elif len(picked) < keep:
        rest = [i for i in range(total) if i not in seen]
        step = max(1, len(rest) // (keep - len(picked)))
        for idx in rest[::step]:
            picked.append(idx)
            if len(picked) == keep:
                break
    return sorted(picked)


def stratified_keep_mask(
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    budget: int,
) -> torch.Tensor:
    """Observation-stratified keep mask over the concatenated token axis."""
    n = sum(token_counts)
    if budget >= n:
        return torch.ones(n, dtype=torch.bool)
    alloc = allocate_budget(list(token_counts), budget)
    keep = torch.zeros(n, dtype=torch.bool)
    offset = 0
    for count, (gh, gw), b in zip(token_counts, grid_hws, alloc):
        for idx in equal_area_grid_indices(gh, gw, b):
            keep[offset + idx] = True
        offset += count
    return keep


def flat_keep_mask(n: int, budget: int) -> torch.Tensor:
    """Control C: every-k-th token over the flat sequence (no crop awareness)."""
    keep = torch.zeros(n, dtype=torch.bool)
    if budget >= n:
        return torch.ones(n, dtype=torch.bool)
    idx = torch.linspace(0, n - 1, budget).round().long()
    keep[idx] = True
    while int(keep.sum()) < budget:  # rounding collisions
        holes = (~keep).nonzero(as_tuple=True)[0]
        keep[holes[0]] = True
    return keep


def random_keep_mask(n: int, budget: int, seed: int = 0) -> torch.Tensor:
    """Control B: uniform random keep at matched budget (seeded)."""
    keep = torch.zeros(n, dtype=torch.bool)
    if budget >= n:
        return torch.ones(n, dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)
    keep[torch.randperm(n, generator=g)[:budget]] = True
    return keep


def topology_keep_mask(
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    parents: tuple[int, ...],
    budget: int,
    shuffle_edges: bool = False,
    seed: int = 42,
) -> torch.Tensor:
    """C1 upgrade (0908 spec): Navigation-Topology Preservation.

    Mandatory set R0 = R_obs (one spatial representative per crop: the token
    nearest the crop's grid center) ∪ R_edge (for every REAL Navigator zoom
    edge p→c, the parent token whose level-0 footprint overlaps the child's
    footprint Ω_c the most — pure geometry, no scores). The remaining budget
    is filled by GLOBAL max-gap (farthest-point) sampling on [0,1]²-normalized
    within-crop coordinates — no per-crop quota.

    shuffle_edges=True: falsification control — parent ids deranged, so the
    edge constraint anchors geometrically meaningless parent tokens.
    """
    n = sum(token_counts)
    if budget >= n:
        return torch.ones(n, dtype=torch.bool)
    n_img = len(token_counts)
    starts, off = [], 0
    for c in token_counts:
        starts.append(off)
        off += c
    par = list(parents) if len(parents) == n_img else [-1] * n_img
    if shuffle_edges:
        import random as _random
        child_ids = [i for i in range(n_img) if par[i] >= 0]
        wrong = child_ids[:]
        _random.Random(seed).shuffle(wrong)
        # reassign each child's parent to ANOTHER child's true parent (shifted)
        par = list(par)
        for a, b in zip(child_ids, wrong[1:] + wrong[:1]):
            par[a] = parents[b]

    # normalized within-crop token coordinates [G, 2]
    coords = torch.zeros(n, 2)
    for img, (gh, gw) in enumerate(grid_hws):
        ys = (torch.arange(gh).repeat_interleave(gw).float() + 0.5) / gh
        xs = (torch.arange(gw).repeat(gh).float() + 0.5) / gw
        coords[starts[img]:starts[img] + gh * gw, 0] = xs
        coords[starts[img]:starts[img] + gh * gw, 1] = ys
    img_of = torch.repeat_interleave(
        torch.arange(n_img), torch.tensor(token_counts))

    keep = torch.zeros(n, dtype=torch.bool)
    # R_obs: token nearest each crop's center
    for img, (gh, gw) in enumerate(grid_hws):
        c = coords[starts[img]:starts[img] + gh * gw]
        keep[starts[img] + int(((c - 0.5) ** 2).sum(1).argmin())] = True
    # R_edge: max-overlap parent token per real zoom edge
    for child in range(n_img):
        p = par[child]
        if p < 0 or p >= n_img or boxes[child] is None or boxes[p] is None:
            continue
        gh, gw = grid_hws[p]
        px, py, pw, ph = (float(v) for v in boxes[p])
        cx, cy, cw, ch = (float(v) for v in boxes[child])
        cell_w, cell_h = pw / gw, ph / gh
        best, best_ov = 0, -1.0
        for r in range(gh):
            for col in range(gw):
                x0, y0 = px + col * cell_w, py + r * cell_h
                ov = (max(0.0, min(x0 + cell_w, cx + cw) - max(x0, cx))
                      * max(0.0, min(y0 + cell_h, cy + ch) - max(y0, cy)))
                if ov > best_ov:
                    best_ov, best = ov, r * gw + col
        keep[starts[p] + best] = True
    # global max-gap filling (within-crop nearest retained distance)
    dmin = torch.full((n,), float("inf"))
    for j in keep.nonzero(as_tuple=True)[0].tolist():
        same = img_of == img_of[j]
        d = (coords - coords[j]).pow(2).sum(1).sqrt()
        dmin = torch.where(same, torch.minimum(dmin, d), dmin)
    kept_per_img = torch.zeros(n_img)
    for j in keep.nonzero(as_tuple=True)[0].tolist():
        kept_per_img[img_of[j]] += 1
    while int(keep.sum()) < budget:
        dsel = dmin.clone()
        dsel[keep] = -1.0
        # fair tie-break: identical grids give identical gap profiles, so
        # without this the argmax fills crops in index order; prefer the crop
        # with the fewest retained tokens at equal gap.
        dsel = dsel - kept_per_img[img_of] * 1e-6
        j = int(dsel.argmax())
        kept_per_img[img_of[j]] += 1
        keep[j] = True
        same = img_of == img_of[j]
        d = (coords - coords[j]).pow(2).sum(1).sqrt()
        dmin = torch.where(same, torch.minimum(dmin, d), dmin)
    return keep


def reassembly_order(
    kept_counts: list[int],
    parents: list[int],
    magnifications: list[int],
    mode: str = "hierarchy",
    seed: int = 42,
) -> list[int]:
    """C2: permutation over the kept token axis arranging crop blocks.

    mode: original | hierarchy (= Coarse-to-Fine Reassembly: root followed by
    its children, acquisition order) | projective (Projective Tree
    Reassembly: subtrees contiguous, parent placed mid-children —
    minimizes sum |sigma(p)-sigma(c)| without imposing coarse-first
    direction) | shuffled_hier (children reattached to a derangement of
    parents) | random (crop blocks shuffled = Shuffled Reassembly)
    | magnification (coarse first)."""
    n_img = len(kept_counts)
    starts, off = [], 0
    for c in kept_counts:
        starts.append(off)
        off += c

    def block(i):
        return list(range(starts[i], starts[i] + kept_counts[i]))

    if mode == "original":
        image_order = list(range(n_img))
    elif mode == "random":
        import random as _random
        image_order = list(range(n_img))
        _random.Random(seed).shuffle(image_order)
    elif mode == "magnification":
        image_order = sorted(range(n_img), key=lambda i: (magnifications[i], i))
    elif mode == "projective":
        # subtree-contiguous, parent mid-children (⌊k/2⌋ before, rest after):
        # exact minimizer of Σ|σ(p)−σ(c)| at block level for equal block
        # sizes, and never interleaves different subtrees.
        image_order = []
        for root in range(n_img):
            if parents[root] >= 0:
                continue
            kids = [i for i in range(n_img) if parents[i] == root]
            half = len(kids) // 2
            image_order.extend(kids[:half] + [root] + kids[half:])
        for i in range(n_img):
            if i not in image_order:
                image_order.append(i)
    else:  # hierarchy / shuffled_hier
        par = list(parents)
        if mode == "shuffled_hier":
            import random as _random
            roots = sorted({p for p in par if p >= 0})
            if len(roots) >= 2:
                shuffled = roots[:]
                _random.Random(seed).shuffle(shuffled)
                remap = {shuffled[i]: shuffled[(i + 1) % len(shuffled)]
                         for i in range(len(shuffled))}
                par = [remap.get(p, p) if p >= 0 else p for p in par]
        image_order = []
        for root in range(n_img):
            if par[root] >= 0:
                continue
            image_order.append(root)
            image_order.extend(i for i in range(n_img) if par[i] == root)
        # orphans whose parent index is out of range / not a root
        for i in range(n_img):
            if i not in image_order:
                image_order.append(i)
    out: list[int] = []
    for i in image_order:
        out.extend(block(i))
    return out


# ---------------------------------------------------------------------------
# Hierarchy-Constrained Spatial Pruning (0910 spec; env VLMAS_WSI_CONSOL_MODE=hier)
# ---------------------------------------------------------------------------

def _derange_parents(parents: list[int], seed: int) -> list[int]:
    """Shuffled-hierarchy control: every child keeps a parent, but the parent
    is a DIFFERENT anchor (cyclic derangement over the distinct real parents).
    Crops/tokens untouched — only the parent-child relation is wrong."""
    import random as _random
    roots = sorted({p for p in parents if p >= 0})
    if len(roots) < 2:
        return list(parents)
    shuffled = roots[:]
    _random.Random(seed).shuffle(shuffled)
    remap = {shuffled[i]: shuffled[(i + 1) % len(shuffled)]
             for i in range(len(shuffled))}
    return [remap.get(p, p) if p >= 0 else p for p in parents]


def _max_overlap_parent_token(parent_box, child_box, grid_hw) -> int:
    """pi(c): index (row-major, on the parent's merged grid) of the parent
    token whose level-0 footprint overlaps the child's box the most.
    Pure geometry — no scores."""
    gh, gw = grid_hw
    px, py, pw, ph = (float(v) for v in parent_box)
    cx, cy, cw, ch = (float(v) for v in child_box)
    cell_w, cell_h = pw / gw, ph / gh
    best, best_ov = 0, -1.0
    for r in range(gh):
        for col in range(gw):
            x0, y0 = px + col * cell_w, py + r * cell_h
            ov = (max(0.0, min(x0 + cell_w, cx + cw) - max(x0, cx))
                  * max(0.0, min(y0 + cell_h, cy + ch) - max(y0, cy)))
            if ov > best_ov:
                best_ov, best = ov, r * gw + col
    return best


def hierarchy_keep_mask(
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    parents: tuple[int, ...],
    budget: int,
    min_per_crop: int = 1,
    shuffle_parents: bool = False,
    seed: int = 42,
    return_info: bool = False,
):
    """Hierarchy-Constrained Spatial Pruning (0910 spec).

    Selection principle = hierarchy preservation + spatial coverage, nothing
    else (no attention / cosine / morphology scores).

    1. Mandatory set S_hier:
         * every crop keeps >= `min_per_crop` tokens (equal-area centers), and
         * for every 20x child c with a real parent p (both boxed), the parent
           token pi(c) covering the child's footprint is kept, so the
           5x-context <-> 20x-detail link survives pruning.
    2. Remaining budget B - |S_hier| is spent by the EXISTING strat rule:
       per-crop quota b_p ∝ |V_p| (allocate_budget) filled with equal-area
       grid thinning (equal_area_grid_indices). Mandatory tokens count toward
       their crop's quota; a crop whose mandatory set exceeds its quota keeps
       the mandatory set and the excess is taken from crops with slack.

    shuffle_parents=True: Shuffled-Hierarchy control — same crops, same
    tokens, same budget, but each child is attached to a different anchor.
    """
    n = sum(token_counts)
    n_img = len(token_counts)
    if budget >= n:
        keep = torch.ones(n, dtype=torch.bool)
        return (keep, {"mandatory": n, "edges": 0, "per_crop": list(token_counts)}) if return_info else keep
    starts, off = [], 0
    for c in token_counts:
        starts.append(off)
        off += c
    par_true = list(parents) if len(parents) == n_img else [-1] * n_img
    par = _derange_parents(par_true, seed) if shuffle_parents else par_true

    def _child_box_in(child, p):
        """Child footprint expressed in parent p's frame. For the real parent
        this is the level-0 box itself; for a shuffled parent the child's
        RELATIVE position inside its true parent is transplanted into the
        wrong parent's frame (same geometry, wrong provenance) so the control
        anchors exactly one geometrically meaningless token per edge."""
        cb = boxes[child]
        p0 = par_true[child]
        if p == p0 or p0 < 0 or p0 >= n_img or boxes[p0] is None:
            return cb
        px0, py0, pw0, ph0 = (float(v) for v in boxes[p0])
        px1, py1, pw1, ph1 = (float(v) for v in boxes[p])
        cx, cy, cw, ch = (float(v) for v in cb)
        return (px1 + (cx - px0) / max(pw0, 1e-6) * pw1,
                py1 + (cy - py0) / max(ph0, 1e-6) * ph1,
                cw / max(pw0, 1e-6) * pw1, ch / max(ph0, 1e-6) * ph1)

    # --- 1. mandatory set -------------------------------------------------
    mand: list[set[int]] = [set() for _ in range(n_img)]
    for img, (gh, gw) in enumerate(grid_hws):
        for idx in equal_area_grid_indices(gh, gw, max(1, min_per_crop)):
            mand[img].add(idx)
    edges = 0
    for child in range(n_img):
        p = par[child]
        if p < 0 or p >= n_img or p == child:
            continue
        if boxes[child] is None or boxes[p] is None:
            continue
        mand[p].add(_max_overlap_parent_token(
            boxes[p], _child_box_in(child, p), grid_hws[p]))
        edges += 1
    n_mand = sum(len(s) for s in mand)

    # --- 2. quota: b_p ∝ |V_p| (existing strat rule), mandatory-aware -----
    target = allocate_budget(list(token_counts), budget)
    target = [max(t, len(m)) for t, m in zip(target, mand)]
    # trim crops with slack (largest slack first) until the budget is met
    while sum(target) > budget:
        slack = [t - len(m) for t, m in zip(target, mand)]
        j = max(range(n_img), key=lambda i: slack[i])
        if slack[j] <= 0:
            break  # mandatory set alone exceeds the budget; keep it whole
        target[j] -= 1
    # (allocate_budget already guarantees sum == budget; the >= step above can
    # only have raised it, so no top-up is needed here.)

    # --- 3. per-crop equal-area fill around the mandatory tokens -----------
    keep = torch.zeros(n, dtype=torch.bool)
    per_crop = []
    for img, ((gh, gw), t) in enumerate(zip(grid_hws, target)):
        total = gh * gw
        chosen = set(mand[img])
        k = t
        picks: list[int] = []
        while len(chosen | set(picks)) < t and k <= total:
            picks = equal_area_grid_indices(gh, gw, k)
            k += 1
        extra = [i for i in picks if i not in chosen]
        need = t - len(chosen)
        if need > 0 and len(extra) > need:
            # drop the extras that are spatially closest to a mandatory token
            # (least coverage gain); keep the rest, row-major.
            def _dist(i):
                y, x = divmod(i, gw)
                return min((y - my) ** 2 + (x - mx) ** 2
                           for my, mx in (divmod(m, gw) for m in chosen))
            extra = sorted(sorted(extra, key=_dist, reverse=True)[:need])
        elif need <= 0:
            extra = []
        for i in list(chosen) + extra:
            keep[starts[img] + i] = True
        per_crop.append(int(keep[starts[img]:starts[img] + total].sum()))
    info = {"mandatory": n_mand, "edges": edges, "per_crop": per_crop,
            "parents": par}
    return (keep, info) if return_info else keep


# ---------------------------------------------------------------------------
# WSI-Aware Visual KV Compression (0910 spec v2; env VLMAS_WSI_CONSOL_MODE=wsi)
#   seeds = center token per observation ∪ bridge token per child (parent
#   token nearest the child's footprint CENTER); fill = greedy covering-radius
#   (observation with the largest R_p gets its farthest token) until budget.
#   Requires B >= B0; below B0 the seed set is kept whole (no branch dropped).
#   wsi_flat = same fill, no bridge constraint; wsi_shuf = parents deranged
#   with the child's relative position transplanted into the wrong parent.
# ---------------------------------------------------------------------------

def _bridge_token(parent_box, child_center_xy, grid_hw) -> int:
    """Parent token whose slide-coordinate center is nearest the child's
    footprint center (spec: argmin_j ||Phi_a(x_aj) - ctr(Omega_c)||)."""
    gh, gw = grid_hw
    px, py, pw, ph = (float(v) for v in parent_box)
    ccx, ccy = child_center_xy
    # normalized position inside the parent, clipped to the grid
    u = min(max((ccx - px) / max(pw, 1e-6), 0.0), 1.0 - 1e-9)
    v = min(max((ccy - py) / max(ph, 1e-6), 0.0), 1.0 - 1e-9)
    col = min(gw - 1, int(u * gw))
    row = min(gh - 1, int(v * gh))
    return row * gw + col


def wsi_aware_keep_mask(
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    parents: tuple[int, ...],
    budget: int,
    constraint: str = "hier",   # hier | flat | shuffled
    seed: int = 42,
    return_info: bool = False,
    normalize_diam: bool = True,   # False = spec "scale_v2": raw ||u_j - u_k||, no diam scaling
):
    n = sum(token_counts)
    n_img = len(token_counts)
    info = {"B0": 0, "edges": 0, "infeasible": False, "per_crop": [],
            "parents": list(parents), "constraint": constraint, "normalize_diam": normalize_diam}
    if budget >= n:
        keep = torch.ones(n, dtype=torch.bool)
        info["B0"], info["per_crop"] = n, list(token_counts)
        return (keep, info) if return_info else keep
    starts, off = [], 0
    for c in token_counts:
        starts.append(off)
        off += c
    par_true = list(parents) if len(parents) == n_img else [-1] * n_img
    par = _derange_parents(par_true, seed) if constraint == "shuffled" else par_true
    if constraint == "flat":
        par = [-1] * n_img

    # normalized token-center coordinates, per observation
    coords = torch.zeros(n, 2)
    for img, (gh, gw) in enumerate(grid_hws):
        ys = (torch.arange(gh).repeat_interleave(gw).float() + 0.5) / gh
        xs = (torch.arange(gw).repeat(gh).float() + 0.5) / gw
        coords[starts[img]:starts[img] + gh * gw, 0] = xs
        coords[starts[img]:starts[img] + gh * gw, 1] = ys
    img_of = torch.repeat_interleave(torch.arange(n_img), torch.tensor(token_counts))
    diam = torch.ones(n_img)
    for img, (gh, gw) in enumerate(grid_hws):
        c = coords[starts[img]:starts[img] + gh * gw]
        diam[img] = max(float((c[0] - c[-1]).norm()), 1e-6) if normalize_diam else 1.0

    keep = torch.zeros(n, dtype=torch.bool)
    # seeds (1): center representative per observation
    for img, (gh, gw) in enumerate(grid_hws):
        c = coords[starts[img]:starts[img] + gh * gw]
        keep[starts[img] + int(((c - 0.5) ** 2).sum(1).argmin())] = True
    # seeds (2): bridge token per real (or transplanted) zoom edge
    edges = 0
    for child in range(n_img):
        p = par[child]
        p0 = par_true[child]
        if p < 0 or p >= n_img or p == child or boxes[child] is None or boxes[p] is None:
            continue
        cx, cy, cw, ch = (float(v) for v in boxes[child])
        ccx, ccy = cx + cw / 2.0, cy + ch / 2.0
        if p != p0 and 0 <= p0 < n_img and boxes[p0] is not None:
            # shuffled control: transplant the child's RELATIVE center from its
            # true parent into the wrong parent's frame
            px0, py0, pw0, ph0 = (float(v) for v in boxes[p0])
            px1, py1, pw1, ph1 = (float(v) for v in boxes[p])
            ccx = px1 + (ccx - px0) / max(pw0, 1e-6) * pw1
            ccy = py1 + (ccy - py0) / max(ph0, 1e-6) * ph1
        keep[starts[p] + _bridge_token(boxes[p], (ccx, ccy), grid_hws[p])] = True
        edges += 1
    B0 = int(keep.sum())
    info["B0"], info["edges"] = B0, edges
    if budget < B0:
        info["infeasible"] = True  # seed set kept whole; nothing dropped

    # greedy covering-radius fill
    dmin = torch.full((n,), float("inf"))
    for j in keep.nonzero(as_tuple=True)[0].tolist():
        same = img_of == img_of[j]
        d = (coords - coords[j]).norm(dim=1) / diam[img_of[j]]
        dmin = torch.where(same, torch.minimum(dmin, d), dmin)
    kept_per_img = torch.zeros(n_img)
    for j in keep.nonzero(as_tuple=True)[0].tolist():
        kept_per_img[img_of[j]] += 1
    while int(keep.sum()) < budget:
        dsel = dmin.clone()
        dsel[keep] = -1.0
        # R_p = max gap per observation; p* = argmax R_p (ties: fewest kept,
        # then lowest index); i* = farthest token in p* (ties: row-major)
        R = torch.full((n_img,), -1.0)
        R.scatter_reduce_(0, img_of, dsel, reduce="amax", include_self=True)
        R = R - kept_per_img * 1e-6
        p_star = int(R.argmax())
        cand = dsel.clone()
        cand[img_of != p_star] = -2.0
        i_star = int(cand.argmax())
        keep[i_star] = True
        kept_per_img[p_star] += 1
        same = img_of == p_star
        d = (coords - coords[i_star]).norm(dim=1) / diam[p_star]
        dmin = torch.where(same, torch.minimum(dmin, d), dmin)
    info["per_crop"] = [int(keep[s:s + c].sum()) for s, c in zip(starts, token_counts)]
    info["parents"] = par
    return (keep, info) if return_info else keep


# ---------------------------------------------------------------------------
# Scale-Structured Visual KV Compression (0910 spec v3; MODE=scale)
#   B_C = floor(alpha*B) coarse / B_F = B - B_C fine.
#   coarse (anchors): mandatory = bridge token per child (parent token nearest
#     the child's footprint center, WSI coords) ∪ one-center of childless
#     anchors; then GLOBAL farthest-point coverage in the common WSI frame.
#   fine (children): b_c >= 1, ∝ N_c; inside each child one-center then FPS in
#     parent-relative normalized coordinates.
#   scale_shuf control: parents deranged, child's relative center transplanted.
# ---------------------------------------------------------------------------

def _token_wsi_coords(box, grid_hw) -> torch.Tensor:
    gh, gw = grid_hw
    x, y, w, h = (float(v) for v in box)
    ys = y + (torch.arange(gh).repeat_interleave(gw).float() + 0.5) / gh * h
    xs = x + (torch.arange(gw).repeat(gh).float() + 0.5) / gw * w
    return torch.stack([xs, ys], dim=1)


def _grid_norm_coords(grid_hw) -> torch.Tensor:
    gh, gw = grid_hw
    ys = (torch.arange(gh).repeat_interleave(gw).float() + 0.5) / gh
    xs = (torch.arange(gw).repeat(gh).float() + 0.5) / gw
    return torch.stack([xs, ys], dim=1)


def _one_center(coords: torch.Tensor) -> int:
    """Discrete 1-center: token minimizing the max distance to all others."""
    d = torch.cdist(coords, coords)
    return int(d.max(dim=1).values.argmin())


def _fps_fill(coords: torch.Tensor, keep: torch.Tensor, target: int) -> torch.Tensor:
    """Greedy farthest-point fill over `coords` (local index space) until
    keep.sum()==target. Ties -> lowest index. keep is modified and returned."""
    n = coords.shape[0]
    if int(keep.sum()) == 0 and target > 0:
        keep[_one_center(coords)] = True
    dmin = torch.full((n,), float("inf"))
    for j in keep.nonzero(as_tuple=True)[0].tolist():
        dmin = torch.minimum(dmin, (coords - coords[j]).norm(dim=1))
    while int(keep.sum()) < min(target, n):
        dsel = dmin.clone()
        dsel[keep] = -1.0
        j = int(dsel.argmax())
        keep[j] = True
        dmin = torch.minimum(dmin, (coords - coords[j]).norm(dim=1))
    return keep


def scale_structured_keep_mask(
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    parents: tuple[int, ...],
    magnifications: tuple[int, ...],
    budget: int,
    alpha: float = 0.5,
    shuffle_parents: bool = False,
    seed: int = 42,
    return_info: bool = False,
):
    n = sum(token_counts)
    n_img = len(token_counts)
    info = {"B": budget, "B_C": 0, "B_F": 0, "coarse": [], "fine": [],
            "bridges": 0, "mandatory_C": 0, "infeasible": False,
            "wsi_coords": True, "per_crop": [], "parents": list(parents)}
    if budget >= n:
        keep = torch.ones(n, dtype=torch.bool)
        info["per_crop"] = list(token_counts)
        return (keep, info) if return_info else keep
    starts, off = [], 0
    for c in token_counts:
        starts.append(off)
        off += c
    par_true = list(parents) if len(parents) == n_img else [-1] * n_img
    par = _derange_parents(par_true, seed) if shuffle_parents else par_true
    mags = list(magnifications) if len(magnifications) == n_img else [0] * n_img
    # scale split: fine = magnification > 5 (fallback: has a parent)
    if any(m > 0 for m in mags):
        fine = [i for i in range(n_img) if mags[i] > 5]
    else:
        fine = [i for i in range(n_img) if par_true[i] >= 0]
    coarse = [i for i in range(n_img) if i not in fine]
    info["coarse"], info["fine"], info["parents"] = coarse, fine, par
    keep = torch.zeros(n, dtype=torch.bool)

    B_C = int(alpha * budget)
    B_F = budget - B_C
    if not fine:            # nothing to refine: all capacity is coarse
        B_C, B_F = budget, 0
    if not coarse:          # no anchors: all capacity is fine
        B_C, B_F = 0, budget
    info["B_C"], info["B_F"] = B_C, B_F

    # ---------------- coarse ----------------
    if coarse:
        have_boxes = all(boxes[i] is not None for i in coarse)
        info["wsi_coords"] = have_boxes
        if have_boxes:
            xs = [float(boxes[i][0]) for i in coarse]
            ys = [float(boxes[i][1]) for i in coarse]
            x1 = [float(boxes[i][0] + boxes[i][2]) for i in coarse]
            y1 = [float(boxes[i][1] + boxes[i][3]) for i in coarse]
            ox, oy = min(xs), min(ys)
            scale = max(max(x1) - ox, max(y1) - oy, 1e-6)
            cparts = [(_token_wsi_coords(boxes[i], grid_hws[i]) - torch.tensor([ox, oy])) / scale
                      for i in coarse]
        else:               # fallback: per-crop normalized grids, offset by index
            cparts = [_grid_norm_coords(grid_hws[i]) + torch.tensor([2.0 * k, 0.0])
                      for k, i in enumerate(coarse)]
        ccoords = torch.cat(cparts, dim=0)
        cstarts, o = {}, 0
        for i, part in zip(coarse, cparts):
            cstarts[i] = o
            o += part.shape[0]
        ckeep = torch.zeros(ccoords.shape[0], dtype=torch.bool)
        # bridges
        has_child = set()
        for c in fine:
            p = par[c]
            p0 = par_true[c]
            if p < 0 or p >= n_img or p not in cstarts or boxes[c] is None or boxes[p] is None:
                continue
            cx, cy, cw, ch = (float(v) for v in boxes[c])
            ccx, ccy = cx + cw / 2.0, cy + ch / 2.0
            if p != p0 and 0 <= p0 < n_img and boxes[p0] is not None:
                px0, py0, pw0, ph0 = (float(v) for v in boxes[p0])
                px1, py1, pw1, ph1 = (float(v) for v in boxes[p])
                ccx = px1 + (ccx - px0) / max(pw0, 1e-6) * pw1
                ccy = py1 + (ccy - py0) / max(ph0, 1e-6) * ph1
            ckeep[cstarts[p] + _bridge_token(boxes[p], (ccx, ccy), grid_hws[p])] = True
            has_child.add(p)
            info["bridges"] += 1
        # childless anchors: discrete one-center
        for i in coarse:
            if i not in has_child:
                gc = _grid_norm_coords(grid_hws[i])
                ckeep[cstarts[i] + _one_center(gc)] = True
        info["mandatory_C"] = int(ckeep.sum())
        if B_C < int(ckeep.sum()):
            info["infeasible"] = True
        ckeep = _fps_fill(ccoords, ckeep, B_C)
        for i in coarse:
            cnt = token_counts[i]
            keep[starts[i]:starts[i] + cnt] = ckeep[cstarts[i]:cstarts[i] + cnt]

    # ---------------- fine ----------------
    if fine and B_F > 0:
        alloc = allocate_budget([token_counts[i] for i in fine], B_F)
        for i, b in zip(fine, alloc):
            gc = _grid_norm_coords(grid_hws[i])
            fkeep = torch.zeros(token_counts[i], dtype=torch.bool)
            fkeep = _fps_fill(gc, fkeep, b)
            keep[starts[i]:starts[i] + token_counts[i]] = fkeep
    elif fine and B_F == 0:
        info["infeasible"] = True   # every child must keep >= 1
        for i in fine:
            keep[starts[i] + _one_center(_grid_norm_coords(grid_hws[i]))] = True
    info["per_crop"] = [int(keep[s:s + c].sum()) for s, c in zip(starts, token_counts)]
    return (keep, info) if return_info else keep


# ---------------------------------------------------------------------------
# Scale-Structured Visual KV Compression, v3 (0911 spec; MODE=scale_v3)
#   Precedence-constrained context coverage.
#   horizontal: A_uv = 1 iff u == v or the token footprints are directly
#     adjacent (edge-sharing, adjacency=4; adjacency=8 adds corner contact)
#     at the same magnification -- i.e. inside one observation grid.
#   utility:    F(S) = sum_p (1/|V_p|) sum_{u in V_p} max_{v in S ∩ V_p} A_uv
#     (monotone submodular coverage, per-observation normalised)
#   vertical:   for each 20x child c, p_c = 5x token containing c's acquisition
#     center (bridge token); S ∩ V_c != ∅  =>  p_c ∈ S  (precedence constraint)
#   greedy:     v* = argmax_v [F(S ∪ cl(v|S)) - F(S)] / |cl(v|S) \ S|,
#     cl(v|S) = {v} ∪ {p_c} if v is the first token kept in child c and p_c ∉ S.
#   When every remaining candidate has zero marginal coverage (F saturated
#   before B is reached) the spec leaves the choice open; we then fill by
#   farthest-point in grid coordinates (largest gap first) and record
#   info["saturated_at"].  constraint: hier | flat (no precedence) | shuffled
#   (deranged parents, child center transplanted) as in the other specs.
# ---------------------------------------------------------------------------
def _coverage_gain(uncovered: torch.Tensor, adjacency: int) -> torch.Tensor:
    """uncovered: [gh, gw] bool -> number of uncovered cells in N(v) ∪ {v}."""
    u = uncovered.float()
    p = torch.nn.functional.pad(u, (1, 1, 1, 1))
    g = u + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    if adjacency == 8:
        g = g + p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    return g


def _cover(covered: torch.Tensor, r: int, c: int, adjacency: int) -> None:
    gh, gw = covered.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if adjacency == 4 and dr != 0 and dc != 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < gh and 0 <= cc < gw:
                covered[rr, cc] = True


def precedence_coverage_keep_mask(
    token_counts: tuple[int, ...],
    grid_hws: tuple[tuple[int, int], ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    parents: tuple[int, ...],
    budget: int,
    constraint: str = "hier",   # hier | flat | shuffled
    adjacency: int = 4,
    seed: int = 42,
    return_info: bool = False,
):
    n = sum(token_counts)
    n_img = len(token_counts)
    info = {"B": int(budget), "edges": 0, "closure_adds": 0, "saturated_at": None,
            "per_crop": [], "parents": list(parents), "constraint": constraint,
            "adjacency": int(adjacency), "F": 0.0}
    if budget >= n:
        keep = torch.ones(n, dtype=torch.bool)
        info["per_crop"] = list(token_counts); info["F"] = float(n_img)
        return (keep, info) if return_info else keep
    starts, off = [], 0
    for c in token_counts:
        starts.append(off); off += c
    par_true = list(parents) if len(parents) == n_img else [-1] * n_img
    par = _derange_parents(par_true, seed) if constraint == "shuffled" else par_true
    if constraint == "flat":
        par = [-1] * n_img
    # p_c: bridge token (absolute index) per child with a valid parent
    bridge = [-1] * n_img
    for child in range(n_img):
        p, p0 = par[child], par_true[child]
        if p < 0 or p >= n_img or p == child or boxes[child] is None or boxes[p] is None:
            continue
        cx, cy, cw, ch = (float(v) for v in boxes[child])
        ccx, ccy = cx + cw / 2.0, cy + ch / 2.0
        if p != p0 and 0 <= p0 < n_img and boxes[p0] is not None:
            px0, py0, pw0, ph0 = (float(v) for v in boxes[p0])
            px1, py1, pw1, ph1 = (float(v) for v in boxes[p])
            ccx = px1 + (ccx - px0) / max(pw0, 1e-6) * pw1
            ccy = py1 + (ccy - py0) / max(ph0, 1e-6) * ph1
        bridge[child] = starts[p] + _bridge_token(boxes[p], (ccx, ccy), grid_hws[p])
        info["edges"] += 1

    keep = torch.zeros(n, dtype=torch.bool)
    covered = [torch.zeros(gh, gw, dtype=torch.bool) for (gh, gw) in grid_hws]
    kept_in = [0] * n_img
    img_of = torch.repeat_interleave(torch.arange(n_img), torch.tensor(token_counts))

    def add(j: int) -> None:
        img = int(img_of[j]); gh, gw = grid_hws[img]
        loc = j - starts[img]
        keep[j] = True; kept_in[img] += 1
        _cover(covered[img], loc // gw, loc % gw, adjacency)

    def needs_parent(img: int) -> bool:
        return bridge[img] >= 0 and kept_in[img] == 0 and not bool(keep[bridge[img]])

    while int(keep.sum()) < budget:
        room = budget - int(keep.sum())
        gain = torch.zeros(n); cost = torch.ones(n)
        for img, (gh, gw) in enumerate(grid_hws):
            g = _coverage_gain(~covered[img], adjacency).reshape(-1) / float(gh * gw)
            gain[starts[img]:starts[img] + gh * gw] = g
        # closure: first token of a child whose bridge is not kept also brings
        # the bridge token (its own coverage gain in the parent grid, cost +1)
        for img in range(n_img):
            if needs_parent(img):
                b = bridge[img]
                gain[starts[img]:starts[img] + token_counts[img]] += float(gain[b])
                cost[starts[img]:starts[img] + token_counts[img]] = 2.0
        ratio = gain / cost
        ratio[keep] = -1.0
        ratio[cost > room] = -1.0
        j = int(ratio.argmax())
        if float(ratio[j]) < 0:
            break                      # nothing structurally valid fits
        if float(ratio[j]) == 0.0:
            # saturated: farthest-point fill (largest grid gap), precedence kept
            if info["saturated_at"] is None:
                info["saturated_at"] = int(keep.sum())
            best, bestd = -1, -1.0
            for img, (gh, gw) in enumerate(grid_hws):
                c_ = 2 if needs_parent(img) else 1
                if c_ > room:
                    continue
                ks = keep[starts[img]:starts[img] + gh * gw]
                rows = torch.arange(gh).repeat_interleave(gw).float()
                cols = torch.arange(gw).repeat(gh).float()
                if int(ks.sum()) == 0:
                    d = torch.full((gh * gw,), float(gh + gw))
                else:
                    kr, kc = rows[ks], cols[ks]
                    d = ((rows[:, None] - kr[None]) ** 2 + (cols[:, None] - kc[None]) ** 2).sqrt().min(1).values
                d[ks] = -1.0
                i = int(d.argmax())
                if float(d[i]) > bestd:
                    bestd, best = float(d[i]), starts[img] + i
            if best < 0:
                break
            j = best
        img = int(img_of[j])
        if needs_parent(img):
            add(bridge[img]); info["closure_adds"] += 1
        add(j)
    info["per_crop"] = [int(keep[s:s + c].sum()) for s, c in zip(starts, token_counts)]
    info["parents"] = par
    info["F"] = float(sum(float(cv.float().mean()) for cv in covered))
    return (keep, info) if return_info else keep


def coverage_utility(keep: torch.Tensor, token_counts, grid_hws, adjacency: int = 4) -> float:
    """Explicit F(S) via the adjacency definition (test oracle)."""
    F, off = 0.0, 0
    for (gh, gw), cnt in zip(grid_hws, token_counts):
        ks = keep[off:off + cnt].reshape(gh, gw)
        cov = torch.zeros(gh, gw, dtype=torch.bool)
        for r in range(gh):
            for c in range(gw):
                if ks[r, c]:
                    _cover(cov, r, c, adjacency)
        F += float(cov.float().mean()); off += cnt
    return F
