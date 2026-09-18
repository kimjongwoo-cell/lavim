"""WSI Anchor-Residual Visual Compression (C1, user spec 2026-09-07).

Per-family (physical coarse ROI + its fine children) compression:
  anchor  = medoid of the coarse tokens (a REAL token, no averaging)
  residual= remaining budget filled greedily by
            g~_i * min_{s in R ∩ family(i)} D^res(i, s)
  D^res   = 1 - cos on per-family standardized-centered features (Cen-Prune:
            raw Qwen geometry is concentrated; per-dim standardization fixes it)
  g_i     = token distinctiveness  -mean_{j != i in family} cos(x_i, x_j)
            (NOT semantic importance), min-max shifted to nonnegative.

The WSI hierarchy defines the selection DOMAIN (families), not a score. The
only knob is the total budget B. Pure functions; env-gating at the call site.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_families(
    token_counts: tuple[int, ...],
    magnifications: tuple[int, ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
) -> tuple[list[list[int]], list[int]]:
    """Group image indices into physical ROI families.

    A coarse image (finer images exist whose box centers fall inside it) roots
    a family containing its children. Any image without such a relation
    (thumbnail, boxless crop, orphan) is its own single-image family.
    Returns (families as lists of image indices, coarse root per family).
    """
    n = len(token_counts)

    def center(idx):
        x, y, w, h = boxes[idx]
        return x + w / 2.0, y + h / 2.0

    parent = [-1] * n
    for i in range(n):
        if boxes[i] is None:
            continue
        cx, cy = center(i)
        best = -1
        for j in range(n):
            if j == i or boxes[j] is None:
                continue
            if magnifications[j] >= magnifications[i] or magnifications[i] == 0:
                continue
            jx, jy, jw, jh = boxes[j]
            if jx <= cx < jx + jw and jy <= cy < jy + jh:
                if best < 0 or magnifications[j] > magnifications[best]:
                    best = j  # nearest coarser scale containing i
        parent[i] = best
    families: dict[int, list[int]] = {}
    for i in range(n):
        root = parent[i] if parent[i] >= 0 else i
        families.setdefault(root, [])
    for i in range(n):
        root = parent[i] if parent[i] >= 0 else i
        if i != root:
            families[root].append(i)
    fam_lists, roots = [], []
    for root, children in families.items():
        fam_lists.append([root] + children)
        roots.append(root)
    return fam_lists, roots


def _token_slices(token_counts: tuple[int, ...]) -> list[tuple[int, int]]:
    out, off = [], 0
    for c in token_counts:
        out.append((off, off + c))
        off += c
    return out


def anchor_residual_keep_mask(
    features: torch.Tensor,               # [N, D] frozen visual states
    token_counts: tuple[int, ...],
    magnifications: tuple[int, ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
    budget: int,
) -> torch.Tensor:
    """Boolean keep mask [N]: per-family medoid anchors first, then a single
    global greedy over residual novelty x distinctiveness. Only knob: budget."""
    device = features.device
    X = features.float()
    N = X.shape[0]
    if budget >= N:
        return torch.ones(N, dtype=torch.bool, device=device)
    fams, roots = build_families(token_counts, magnifications, boxes)
    slices = _token_slices(token_counts)

    fam_of = torch.full((N,), -1, dtype=torch.long, device=device)
    for f, imgs in enumerate(fams):
        for img in imgs:
            lo, hi = slices[img]
            fam_of[lo:hi] = f

    Xn = F.normalize(X, dim=1)
    keep = torch.zeros(N, dtype=torch.bool, device=device)

    # per-family standardized-centered residual directions + distinctiveness
    Ybar = torch.zeros_like(X)
    g = torch.zeros(N, device=device)
    anchors: list[int] = []
    for f, imgs in enumerate(fams):
        idx = (fam_of == torch.tensor(f, device=device)).nonzero(as_tuple=True)[0]
        Xf = X[idx]
        mu = Xf.mean(0, keepdim=True)
        sd = Xf.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)
        Ybar[idx] = F.normalize((Xf - mu) / sd, dim=1)
        if idx.numel() > 1:
            cosf = Xn[idx] @ Xn[idx].T
            g[idx] = -(cosf.sum(1) - 1.0) / (idx.numel() - 1)
        # anchor = medoid of the COARSE image's tokens (raw distance)
        lo, hi = slices[roots[f]]
        cidx = torch.arange(lo, hi, device=device)
        cn = Xn[cidx]
        med = int(torch.argmax((cn @ cn.T).sum(1)))  # max total cos = min dist
        anchors.append(int(cidx[med]))
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)  # nonneg, cross-family comparable

    for a in anchors[: max(1, budget)]:
        keep[a] = True

    # global greedy on residual novelty within each candidate's own family
    dmin = torch.full((N,), float("inf"), device=device)
    for a in anchors[: max(1, budget)]:
        same = fam_of == fam_of[a]
        d = 1.0 - (Ybar @ Ybar[a])
        dmin = torch.where(same, torch.minimum(dmin, d), dmin)
    dmin = torch.where(torch.isinf(dmin), torch.ones_like(dmin), dmin)
    while int(keep.sum()) < budget:
        gain = g * dmin
        gain[keep] = -1.0
        pick = int(torch.argmax(gain))
        if gain[pick] < 0:
            break
        keep[pick] = True
        same = fam_of == fam_of[pick]
        d = 1.0 - (Ybar @ Ybar[pick])
        dmin = torch.where(same, torch.minimum(dmin, d), dmin)
    return keep


@torch.no_grad()
def geometry_audit(
    features: torch.Tensor,
    token_counts: tuple[int, ...],
    magnifications: tuple[int, ...],
    boxes: tuple[tuple[int, int, int, int] | None, ...],
) -> dict[str, float]:
    """Cen-Prune-style assumption check (method validation, not tuning):
    raw/centered mean cosine, top-1 variance fraction, effective rank
    (participation ratio) raw vs standardized-centered — per scale group and
    averaged over ROI families."""
    X = features.float()

    def stats(idx: torch.Tensor) -> dict[str, float]:
        Xf = X[idx]
        n = Xf.shape[0]
        if n < 3:
            return {}
        Xn = F.normalize(Xf, dim=1)
        raw_cos = float((Xn @ Xn.T).sum() - n) / (n * (n - 1))
        C = Xf - Xf.mean(0, keepdim=True)
        Cn = F.normalize(C, dim=1)
        cen_cos = float((Cn @ Cn.T).sum() - n) / (n * (n - 1))
        ev = torch.linalg.eigvalsh(C.T @ C / max(n - 1, 1)).clamp(min=0)
        top1 = float(ev.max() / ev.sum().clamp(min=1e-12))
        er_raw = float(ev.sum() ** 2 / (ev * ev).sum().clamp(min=1e-12))
        S = C / Xf.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)
        ev2 = torch.linalg.eigvalsh(S.T @ S / max(n - 1, 1)).clamp(min=0)
        er_std = float(ev2.sum() ** 2 / (ev2 * ev2).sum().clamp(min=1e-12))
        return {"raw_cos": raw_cos, "cen_cos": cen_cos, "top1_var": top1,
                "eff_rank_raw": er_raw, "eff_rank_std": er_std}

    slices = _token_slices(token_counts)
    out: dict[str, float] = {}
    for label, mags in (("x5", (5,)), ("x20", (20, 40))):
        idx = torch.cat([
            torch.arange(lo, hi) for (lo, hi), m
            in zip(slices, magnifications) if m in mags
        ]) if any(m in mags for m in magnifications) else torch.empty(0, dtype=torch.long)
        for k, v in stats(idx).items():
            out[f"{label}.{k}"] = v
    fams, _roots = build_families(token_counts, magnifications, boxes)
    fam_stats: list[dict[str, float]] = []
    for imgs in fams:
        if len(imgs) < 2:
            continue
        idx = torch.cat([torch.arange(*slices[i]) for i in imgs])
        s = stats(idx)
        if s:
            fam_stats.append(s)
    if fam_stats:
        for k in fam_stats[0]:
            out[f"family.{k}"] = sum(s[k] for s in fam_stats) / len(fam_stats)
    return out
