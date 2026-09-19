"""C1 #14.8 Spectral Support-Demand Allocation + Residual-Salience Representative Selection (SSDA).

Notion C1/C2 Method Evolution 14.8. Two separate decisions under one global visual budget B:

  how many  k_g   support spectral demand: residual-salience-weighted, centred covariance of the unit merger
                  features of support g,
                      p_{g,i} = softmax_g(x_{g,i} - gamma b_i),  mu_g = sum_i p z,  C_g = sum_i p (z - mu)(z - mu)^T,
                  eigenvalues lambda_{g,1} >= ...; discrete water-filling: the B largest eigenvalues over all
                  supports, counted per support (k_g <= N_g), i.e. max sum_g sum_{r<=k_g} lambda_{g,r} s.t. sum k_g = B.
  which    S_g    weighted facility location inside the support with exactly k_g picks,
                      max sum_i p_{g,i} max_{j in S_g} (1 + cos(z_i, z_j)) / 2.

Residual salience = #14.6 null-template residual: a = vision block-23 received attention (uniform = 1 per crop),
x = centred log(a + eps), b = mean centred log map of 5 content-free inputs (offline, memory/ssda_null_block23.npz,
relative-coordinate resampled for other grids), gamma = max(0, sum_g x_g.b / (G b.b)) per slide.

Runtime: VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ssda (dispatched from memory/qasc_select.select_vision_features, same
Reasoner-prefill site and same survivor bookkeeping). VLMAS_QASC_KEEP (default 0.25) sets B. Log line [SSDA].

Also C1 #14.6 Null-Template Residual Salience + Physical-Support Constrained Morphology Coverage (NTRS):
VLMAS_C1_SELECT=ntrs — the same residual weights p and similarity, but ONE global greedy over all supports on
    F(S) = sum_g sum_{i in V_g} p_{g,i} max_{j in S cap V_g} (1 + cos(z_i, z_j)) / 2,   |S| = B,
so the per-support counts are an outcome of the greedy (no k_g stage). Log line [NTRS].
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch

EPS = 1e-3
NULL_FILE = Path(__file__).with_name("ssda_null_block23.npz")


# --------------------------------------------------------------------------- pure (numpy)


def centred_log(maps, eps: float = EPS) -> np.ndarray:
    lg = np.log(np.asarray(maps, dtype=np.float64) + eps)
    return lg - lg.mean(axis=-1, keepdims=True)


def null_template(null_maps_uniform) -> np.ndarray:
    """[n_null, gh*gw] content-free maps (uniform = 1) -> b (mean centred log map)."""
    return centred_log(null_maps_uniform).mean(axis=0)


def resample_template(b: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    """Bilinear resample of a raster template on relative cell-centre coordinates, re-centred."""
    sh, sw = src_hw
    dh, dw = dst_hw
    if (sh, sw) == (dh, dw):
        return np.asarray(b, dtype=np.float64)
    g = np.asarray(b, dtype=np.float64).reshape(sh, sw)
    ys = (np.arange(dh) + 0.5) / dh * sh - 0.5
    xs = (np.arange(dw) + 0.5) / dw * sw - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, sh - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, sw - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    x1 = np.clip(x0 + 1, 0, sw - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    out = (g[np.ix_(y0, x0)] * (1 - wy) * (1 - wx) + g[np.ix_(y1, x0)] * wy * (1 - wx)
           + g[np.ix_(y0, x1)] * (1 - wy) * wx + g[np.ix_(y1, x1)] * wy * wx).ravel()
    return out - out.mean()


def residual_weights(a_uniform_crops, b_crops):
    """Per-support softmax of (centred log salience - gamma * template). Returns (p list, gamma)."""
    xs = [centred_log(a) for a in a_uniform_crops]
    num = sum(float(x @ b) for x, b in zip(xs, b_crops))
    den = sum(float(b @ b) for b in b_crops)
    gamma = max(0.0, num / den) if den > 0 else 0.0
    ps = []
    for x, b in zip(xs, b_crops):
        r = x - gamma * b
        e = np.exp(r - r.max())
        ps.append(e / e.sum())
    return ps, gamma


def weighted_spectrum(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of C = sum_i p_i (z_i - mu)(z_i - mu)^T, mu = sum_i p_i z_i."""
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mu = p @ z
    y = np.sqrt(p)[:, None] * (z - mu)
    # nonzero eigenvalues of Y^T Y (d x d) equal those of the N x N Gram Y Y^T; eigvalsh on the Gram avoids an
    # N x d SVD, which took ~10 s per crop inside the GPU pipeline process (BLAS thread contention)
    vals = np.linalg.eigvalsh(y @ y.T)[::-1][: min(y.shape)]
    return np.clip(vals, 0.0, None)


def water_fill(spectra, budget: int, caps):
    """B largest eigenvalues over all supports (per-support cap). Returns (k list, water level tau)."""
    vals = np.concatenate([np.asarray(s, dtype=np.float64) for s in spectra])
    owner = np.concatenate([np.full(len(s), g) for g, s in enumerate(spectra)])
    k = [0] * len(spectra)
    tau = float("nan")
    taken = 0
    budget = int(min(budget, sum(int(min(c, len(s))) for c, s in zip(caps, spectra))))
    for idx in np.argsort(-vals, kind="mergesort"):
        g = int(owner[idx])
        if k[g] >= caps[g]:
            continue
        k[g] += 1
        taken += 1
        tau = float(vals[idx])
        if taken == budget:
            break
    return k, tau


def facility_location(z: np.ndarray, p: np.ndarray, k: int) -> list[int]:
    """Greedy max sum_i p_i max_{j in S} (1 + cos(z_i, z_j)) / 2 with |S| = k (z unit rows). Pick order."""
    k = int(min(k, len(z)))
    if k <= 0:
        return []
    s = (1.0 + np.asarray(z, dtype=np.float64) @ np.asarray(z, dtype=np.float64).T) / 2.0
    p = np.asarray(p, dtype=np.float64)
    cov = np.zeros(len(z))
    chosen = np.zeros(len(z), dtype=bool)
    order = []
    for _ in range(k):
        gain = (p[:, None] * np.clip(s - cov[:, None], 0.0, None)).sum(axis=0)
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        order.append(j)
        chosen[j] = True
        cov = np.maximum(cov, s[:, j])
    return order


def global_facility_location(z_all: np.ndarray, ps, counts, budget: int) -> list[int]:
    """#14.6: one global greedy over supports; demand only inside the candidate's own support. Pick order."""
    offs = np.cumsum([0] + [int(c) for c in counts])
    sims, cov, gains = [], [], []
    for g in range(len(counts)):
        zg = np.asarray(z_all[offs[g]:offs[g + 1]], dtype=np.float64)
        s = (1.0 + zg @ zg.T) / 2.0
        sims.append(s)
        cov.append(np.zeros(len(zg)))
        gains.append((np.asarray(ps[g], dtype=np.float64)[:, None] * s).sum(axis=0))
    N = int(offs[-1])
    chosen = np.zeros(N, dtype=bool)
    order = []
    for _ in range(int(min(budget, N))):
        flat = np.concatenate(gains)
        flat[chosen] = -np.inf
        j = int(np.argmax(flat))
        g = int(np.searchsorted(offs, j, side="right") - 1)
        order.append(j)
        chosen[j] = True
        cov[g] = np.maximum(cov[g], sims[g][:, j - offs[g]])
        pg = np.asarray(ps[g], dtype=np.float64)
        gains[g] = (pg[:, None] * np.clip(sims[g] - cov[g][:, None], 0.0, None)).sum(axis=0)
    return order


def select_ntrs(z_all: np.ndarray, a_uniform_all: np.ndarray, counts, grids, null_b16: np.ndarray, budget: int):
    """Full #14.6 selection. Returns (keep indices sorted, info dict with per-support counts k)."""
    offs = np.cumsum([0] + [int(c) for c in counts])
    b_crops = [resample_template(null_b16, (16, 16), g) for g in grids]
    ps, gamma = residual_weights([a_uniform_all[offs[g]:offs[g + 1]] for g in range(len(counts))], b_crops)
    order = global_facility_location(z_all, ps, counts, budget)
    keep = np.array(sorted(order), dtype=np.int64)
    k = [int(((keep >= offs[g]) & (keep < offs[g + 1])).sum()) for g in range(len(counts))]
    return keep, {"k": k, "gamma": gamma, "p": ps, "tau": float("nan"), "energy_kept": [float("nan")] * len(counts)}


def select(z_all: np.ndarray, a_uniform_all: np.ndarray, counts, grids, null_b16: np.ndarray, budget: int):
    """Full #14.8 selection. z_all unit rows [N, d]; a_uniform_all [N] (uniform = 1 inside each crop).

    Returns (keep indices sorted, info dict).
    """
    offs = np.cumsum([0] + [int(c) for c in counts])
    b_crops = [resample_template(null_b16, (16, 16), g) for g in grids]
    a_crops = [a_uniform_all[offs[g]:offs[g + 1]] for g in range(len(counts))]
    ps, gamma = residual_weights(a_crops, b_crops)
    spectra = [weighted_spectrum(z_all[offs[g]:offs[g + 1]], ps[g]) for g in range(len(counts))]
    k, tau = water_fill(spectra, budget, [int(c) for c in counts])
    keep = []
    for g in range(len(counts)):
        local = facility_location(z_all[offs[g]:offs[g + 1]], ps[g], k[g])
        keep.extend(int(offs[g]) + j for j in local)
    info = {"k": k, "tau": tau, "gamma": gamma, "trace": [float(s.sum()) for s in spectra],
            "energy_kept": [float(s[:kk].sum() / s.sum()) if s.sum() > 0 else 0.0 for s, kk in zip(spectra, k)],
            "p": ps}
    return np.array(sorted(keep), dtype=np.int64), info


def load_null_b16(path: Path | str = NULL_FILE) -> np.ndarray:
    return null_template(np.load(path)["maps_block23"])


# --------------------------------------------------------------------------- runtime


@torch.no_grad()
def select_vision_features(bb, pixel_values, grid_thw):
    """Vision tower + #14.8 selection. Returns (retained merged features [K, d], keep mask [N])."""
    from memory.qasc_select import merged_mean, received_attention
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    t0 = time.time()
    visual = bb.visual
    layer = int(os.environ.get("VLMAS_QASC_VLAYER", "-1")) % len(visual.blocks)
    grabbed = {}

    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cu = kwargs.get("cu_seqlens")
        pos = kwargs.get("position_embeddings")
        if hidden is None or cu is None or pos is None:
            return None
        seq = hidden.shape[0]
        qkv = module.qkv(hidden).reshape(seq, 3, module.num_heads, -1).permute(1, 0, 2, 3)
        q, k, _ = qkv.unbind(0)
        cos, sin = pos
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        out = torch.zeros(seq, dtype=torch.float32, device=hidden.device)
        cuts = [int(v) for v in cu]
        for a, e in zip(cuts[:-1], cuts[1:]):
            if e > a:
                out[a:e] = received_attention(q[:, a:e], k[:, a:e], module.scaling)
        grabbed["a"] = out
        return None

    handle = visual.blocks[layer].attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        feats = bb.vlm.get_image_features(pixel_values, grid_thw, return_dict=True).pooler_output
    finally:
        handle.remove()
    feats = torch.cat(list(feats), dim=0) if isinstance(feats, (list, tuple)) else feats
    if "a" not in grabbed:
        raise RuntimeError("SSDA: vision attention hook did not fire")
    unit = int(visual.spatial_merge_unit)
    merge = int(round(math.sqrt(unit)))
    a = merged_mean(grabbed["a"], unit).double().cpu().numpy()
    N = int(feats.shape[0])
    if a.size != N:
        raise RuntimeError(f"SSDA: salience {a.size} != features {N}")
    thw = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
    counts = [t * h * w // unit for t, h, w in thw]
    grids = [(h // merge, w // merge) for _, h, w in thw]
    offs = np.cumsum([0] + counts)
    a_u = np.concatenate([a[offs[g]:offs[g + 1]] / max(a[offs[g]:offs[g + 1]].sum(), 1e-30) * counts[g]
                          for g in range(len(counts))])
    z = torch.nn.functional.normalize(feats.float(), dim=-1).double().cpu().numpy()
    keep_ratio = float(os.environ.get("VLMAS_QASC_KEEP", "0.25"))
    budget = max(1, math.ceil(N * keep_ratio))
    mode = os.environ.get("VLMAS_C1_SELECT", "ssda").strip() or "ssda"
    fn = select_ntrs if mode == "ntrs" else select
    keep_idx, info = fn(z, a_u, counts, grids, load_null_b16(os.environ.get("VLMAS_SSDA_NULL", str(NULL_FILE))), budget)
    keep = torch.zeros(N, dtype=torch.bool, device=feats.device)
    keep[torch.as_tensor(keep_idx, device=feats.device)] = True
    kept_cpu = keep.detach().cpu()
    image_of = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long))
    weight = torch.as_tensor(np.concatenate(info["p"]), dtype=torch.float32)
    bb._prefill_prune_survivor_scores = weight[kept_cpu]
    bb._prefill_prune_survivor_group_ids = torch.nonzero(kept_cpu, as_tuple=True)[0]
    bb._prefill_prune_survivor_image_ids = image_of[kept_cpu]
    mags = list(getattr(bb, "_prune_image_magnifications", ()) or ())
    tag = "NTRS" if mode == "ntrs" else "SSDA"
    print(f"[{tag}] kept {int(keep.sum())}/{N} layer={layer} k={info['k']} mags={mags} gamma={info['gamma']:.3f} "
          f"tau={info['tau']:.3e} empty_crops={sum(1 for v in info['k'] if v == 0)} "
          f"energy_kept_min={min(info['energy_kept']):.3f} sec={time.time() - t0:.2f}", flush=True)
    return feats[keep], keep


__all__ = ["centred_log", "null_template", "resample_template", "residual_weights", "weighted_spectrum",
           "water_fill", "facility_location", "global_facility_location", "select", "select_ntrs", "load_null_b16",
           "select_vision_features"]
