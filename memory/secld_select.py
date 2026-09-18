"""C1 #19 Spatial Evidence-Coverage Log-Det Selection (SEC-LD). Notion 3ddd771b-c9e2-8157.

Opt-in; off = no call, byte-identical. Sits at the Pruning-B early-observation site
(backbone/qwen3vl.py::_hierarchy_pruned_vision_features, same place as EOVC) and replaces the
keep-mask rule only: the same `observe_blocks` vision blocks, the same budget, no quota.

Page §1-§4 (cell = one merged visual token = merge x merge patches = 32 x 32 px):

  r_i = mean_{x in cell i} 1[gray(x) > tau_p]          bright-pixel ratio, from pixel_values itself
  m_i = 1[r_i >= tau_c]                                 binary bright-cell map
  b_g = G_sigma * Pad_rep(m_g)                          Gaussian occupancy on the crop's 2-D token grid
  e_i = 1 - b_i                                         spatial evidence demand, NOT support-normalised
  z_i = h_i / ||h_i||                                   morphology direction at the pruning boundary
  q_i = sqrt(e_i) z_i
  F(S) = sum_g log det(I + sum_{j in S cap V_g} q_j q_j^T),   |S| <= B, no per-crop quota

  greedy  Delta(j|S) = log(1 + e_j z_j^T M_g(S)^{-1} z_j)   exact, incremental Cholesky on the dual
          kernel I + Q Q^T restricted to same-support pairs (block-diagonal => sum of per-support
          log-dets), so one global argmax per pick decides both the crop and the token.

Values the page leaves open, fixed here: h_i = mean of the merge-group patch states after the
observation blocks (Pruning-B's `pooled`); gray = ITU-R 601 luminance of the un-normalised pixels
(PIL convert("L") integer formula, bit-identical to the 50-case audit glass_fat_5ds.py); Gaussian taps = 2*round(4 sigma)+1 (cv2's rule
for float images, so sigma=2 -> 17 taps); budget = round(keep_ratio * N).

Env
  VLMAS_SECLD=1            enable
  VLMAS_SECLD_TAU_P=215    bright-pixel gray threshold
  VLMAS_SECLD_TAU_C=0.9    bright-cell ratio threshold
  VLMAS_SECLD_SIGMA=2.0    Gaussian sigma in grid cells
  VLMAS_SECLD_KEEP=        keep ratio override (unset = the variant's prune keep_ratio, 0.25 for pruning_b)
  VLMAS_SECLD_MIN=0        per-support minimum (page: 0; >0 only as a runtime-safety fallback)
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

def enabled() -> bool:
    return os.environ.get("VLMAS_SECLD", "").strip() == "1"


def config() -> dict:
    def f(name, default):
        v = os.environ.get(name, "").strip()
        return float(v) if v else default
    keep = os.environ.get("VLMAS_SECLD_KEEP", "").strip()
    return {
        "tau_p": f("VLMAS_SECLD_TAU_P", 215.0),
        "tau_c": f("VLMAS_SECLD_TAU_C", 0.9),
        "sigma": f("VLMAS_SECLD_SIGMA", 2.0),
        "keep": float(keep) if keep else None,
        "min_per_support": int(os.environ.get("VLMAS_SECLD_MIN", "0") or 0),
    }


# --------------------------------------------------------------------------- pure


def cell_bright_ratio(pixel_values: torch.Tensor, grid_thw, *, patch: int = 16, temporal: int = 2,
                      merge: int = 2, mean: float = 0.5, std: float = 0.5,
                      tau_p: float = 215.0) -> torch.Tensor:
    """pixel_values [Np, C*T*P*P] (HF Qwen2-VL layout: per patch [C, T, P, P]; consecutive
    merge*merge patches form one merged token, merged grid in raster order) -> r_i [N] in [0, 1].
    Uses temporal frame 0 (frames are copies for still images)."""
    unit = merge * merge
    pv = pixel_values.detach().to(torch.float32)
    n_patch = int(pv.shape[0])
    if pv.shape[1] != 3 * temporal * patch * patch:
        raise ValueError(f"unexpected patch vector length {int(pv.shape[1])}")
    expected = sum(int(t * h * w) for t, h, w in torch.as_tensor(grid_thw).tolist())
    if n_patch != expected or n_patch % unit:
        raise ValueError(f"{n_patch} patches vs grid {expected} (unit {unit})")
    x = pv.reshape(n_patch, 3, temporal, patch, patch)[:, :, 0]         # [Np, 3, P, P]
    rgb = torch.round((x * std + mean) * 255.0).clamp_(0.0, 255.0)      # exact uint8 values back
    # PIL Image.convert("L") integer luminance (ITU-R 601-2): (R*19595 + G*38470 + B*7471 + 2^15) >> 16.
    # All intermediates are < 2^24 so float32 is exact; this is the audit's gray, bit for bit.
    gray = torch.floor((rgb[:, 0] * 19595.0 + rgb[:, 1] * 38470.0 + rgb[:, 2] * 7471.0 + 32768.0) / 65536.0)
    bright = (gray > tau_p).to(torch.float32).mean(dim=(1, 2))           # per patch
    return bright.reshape(n_patch // unit, unit).mean(dim=1)


def gaussian_kernel(sigma: float, radius: int | None = None) -> torch.Tensor:
    if radius is None:
        radius = int(round(4.0 * sigma))
    radius = max(1, radius)
    ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    return (g1[:, None] * g1[None, :]).to(torch.float32)


def occupancy(m: torch.Tensor, grids, sigma: float, radius: int | None = None) -> torch.Tensor:
    """m [N] binary/float cell map over all crops (concatenated raster grids) -> b [N] in [0, 1],
    Gaussian-smoothed with replicate padding inside each crop."""
    k = gaussian_kernel(sigma, radius).to(m.device)
    r = (int(k.shape[0]) - 1) // 2
    out = torch.empty_like(m, dtype=torch.float32)
    off = 0
    for gh, gw in grids:
        n = int(gh * gw)
        img = m[off:off + n].to(torch.float32).reshape(1, 1, int(gh), int(gw))
        img = F.pad(img, (r, r, r, r), mode="replicate")
        out[off:off + n] = F.conv2d(img, k[None, None])[0, 0].reshape(-1)
        off += n
    if off != int(m.numel()):
        raise ValueError("grids do not cover the cell map")
    return out.clamp_(0.0, 1.0)


def evidence(pixel_values: torch.Tensor, grid_thw, cfg: dict | None = None, *, merge: int = 2,
             patch: int = 16, temporal: int = 2) -> dict:
    """Page §1: r, m, b, e for every merged token (all crops concatenated)."""
    cfg = config() if cfg is None else cfg
    thw = [tuple(int(v) for v in row) for row in torch.as_tensor(grid_thw).tolist()]
    grids = []
    for t, h, w in thw:
        grids.extend([(h // merge, w // merge)] * t)
    r = cell_bright_ratio(pixel_values, grid_thw, patch=patch, temporal=temporal, merge=merge,
                          tau_p=float(cfg["tau_p"]))
    m = (r >= float(cfg["tau_c"])).to(torch.float32)
    b = occupancy(m, grids, float(cfg["sigma"]))
    return {"r": r, "m": m, "b": b, "e": (1.0 - b).clamp_(0.0, 1.0), "grids": grids}


@torch.no_grad()
def support_logdet_greedy(q: torch.Tensor, support: torch.Tensor, budget: int,
                          first_per_support: int = 0):
    """Greedy max of sum_g log det(I + Q_{S_g} Q_{S_g}^T) over rows q [N, d] with support ids [N].

    Incremental Cholesky of the dual kernel K = I + Q Q^T masked to same-support pairs (block
    diagonal, so the log-det of K equals the page's sum over supports). Gain of j given S is
    log(d_j^2), d_j^2 = 1 + |q_j|^2 - |c_j|^2 = 1 + q_j^T (I + Q_{S_g}^T Q_{S_g})^{-1} q_j.
    Ties -> lowest index. Returns (order LongTensor, gains list, d2_at_pick list).
    """
    q = q.double() if q.device.type == "cpu" else q.float()
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
        per = torch.zeros(n_sup, dtype=torch.long, device=q.device)
        for _ in range(first_per_support):
            for g in range(n_sup):
                forced.append(g)
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
        kj = (q @ q[j]) * (support == support[j]).to(q.dtype)
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


@torch.no_grad()
def select(pixel_values: torch.Tensor, grid_thw, states: torch.Tensor, token_counts, budget: int,
           cfg: dict | None = None, *, merge: int = 2, patch: int = 16, temporal: int = 2) -> dict:
    """Page §3-§4 on one Reasoner prefill: states [N, d] = merge-group states at the pruning
    boundary (one row per merged token, same order as pixel_values groups), token_counts per crop."""
    t0 = time.time()
    cfg = config() if cfg is None else cfg
    counts = [int(c) for c in token_counts]
    N = int(states.shape[0])
    if sum(counts) != N:
        raise ValueError(f"token_counts {sum(counts)} != states {N}")
    ev = evidence(pixel_values, grid_thw, cfg, merge=merge, patch=patch, temporal=temporal)
    e = ev["e"].to(states.device)
    if int(e.numel()) != N:
        raise ValueError(f"evidence cells {int(e.numel())} != states {N}")
    z = F.normalize(states.to(torch.float32), dim=-1)
    q = e.sqrt().unsqueeze(1) * z
    support = torch.repeat_interleave(torch.arange(len(counts), device=states.device),
                                      torch.tensor(counts, dtype=torch.long, device=states.device))
    B = max(0, min(int(budget), N))
    order, gains, d2s = support_logdet_greedy(q, support, B,
                                              first_per_support=int(cfg.get("min_per_support", 0)))
    keep = torch.zeros(N, dtype=torch.bool, device=states.device)
    keep[order.to(states.device)] = True
    per = [int(keep[support == g].sum()) for g in range(len(counts))]
    e_cpu, b_cpu, m_cpu = e.detach().cpu(), ev["b"].detach().cpu(), ev["m"].detach().cpu()
    sel = order.cpu()
    e_sel = e_cpu[sel]
    # §11 diagnostics: does the greedy collapse to the plain e-ranking?
    top_e = torch.topk(e_cpu, B).indices if B > 0 else torch.empty(0, dtype=torch.long)
    erank_overlap = float(keep.cpu()[top_e].float().mean()) if B > 0 else float("nan")
    nov = [((d - 1.0) / float(x)) if float(x) > 1e-12 else float("nan") for d, x in zip(d2s, e_sel.tolist())]
    nov_t = torch.tensor([v for v in nov if v == v], dtype=torch.float64)
    k10 = max(1, B // 10)
    return {
        "keep": keep, "order": order, "gains": gains, "per_support": per, "budget": B, "n": N,
        "e": e_cpu, "b": b_cpu, "m": m_cpu,
        "empty_crops": sum(1 for v in per if v == 0),
        "e_all_mean": float(e_cpu.mean()), "e_all_lt01": float((e_cpu < 0.1).float().mean()),
        "m_all_frac": float(m_cpu.mean()), "m_sel_frac": float(m_cpu[sel].mean()) if B else float("nan"),
        "e_sel_mean": float(e_sel.mean()) if B else float("nan"),
        "e_sel_min": float(e_sel.min()) if B else float("nan"),
        "b_sel_mean": float(b_cpu[sel].mean()) if B else float("nan"),
        "gain_first": float(sum(gains[:k10]) / k10) if B else float("nan"),
        "gain_last": float(sum(gains[-k10:]) / k10) if B else float("nan"),
        "nov_first": float(nov_t[:k10].mean()) if nov_t.numel() else float("nan"),
        "nov_last": float(nov_t[-k10:].mean()) if nov_t.numel() else float("nan"),
        "nov_min": float(nov_t.min()) if nov_t.numel() else float("nan"),
        "erank_overlap": erank_overlap,
        "sec": time.time() - t0,
    }


def summary(info: dict) -> str:
    per = info["per_support"]
    sd = torch.tensor(per, dtype=torch.float32).std(unbiased=False) if len(per) > 1 else torch.tensor(0.0)
    return (f"kept {info['budget']}/{info['n']} per_support={per} sd={float(sd):.1f} "
            f"empty_crops={info['empty_crops']} m_all={info['m_all_frac']:.3f} m_sel={info['m_sel_frac']:.3f} "
            f"e_all={info['e_all_mean']:.3f} e_all_lt01={info['e_all_lt01']:.3f} "
            f"e_sel={info['e_sel_mean']:.3f} e_sel_min={info['e_sel_min']:.3f} b_sel={info['b_sel_mean']:.3f} "
            f"gain_first={info['gain_first']:.4f} gain_last={info['gain_last']:.4f} "
            f"nov_first={info['nov_first']:.3f} nov_last={info['nov_last']:.3f} nov_min={info['nov_min']:.3f} "
            f"erank_overlap={info['erank_overlap']:.3f} sec={info['sec']:.2f}")


__all__ = ["enabled", "config", "cell_bright_ratio", "gaussian_kernel", "occupancy", "evidence",
           "support_logdet_greedy", "select", "summary"]
