"""C1 #21 NOVA v2 — threshold-free spatial null. Notion 3ddd771b-c9e2-81e1 (09-17 15:13 revision).

v1 (memory/nova_select.py + memory/secld_select.py) is NOT modified. v2 changes only the §2 estimator of
the non-null mass e_i; §1 (full vision first), §3-§5 (q_i = sqrt(e_i) z_i, support-wise log-det, exact
greedy, no quota, no support normalisation) are v1's code, reused as is.

Opt-in: VLMAS_NOVA_V2=1 on top of the v1 switches (VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova). The import
hook tmp_hj/nova_v2_hooks/sitecustomize.py then calls `patch(memory.secld_select)`, which swaps that
module's `evidence` for `evidence_v2` in this process only. Flag off = nothing is imported or patched.

Page §2.1  o(x) = -log((I(x) + eps) / I_0)  per RGB channel,  u_i = mean_{x in cell i} o(x)
           p(u) = pi_B N(u; 0, Sigma_B) + (1 - pi_B) N(u; mu_T, Sigma_T)   fitted per case by EM,
           background mean pinned at 0 (zero absorption),   w_i = P(B | u_i)
Page §2.2  d_grid = median nearest-neighbour spacing of the token centres = 1 cell on the regular lattice,
           K_ij = exp(-|p_i - p_j|^2 / (2 d_grid^2)) / sum_{k in V_g(i)} exp(...)   (normalised INSIDE the
           crop, so no padding value is invented at the border),  b_i = sum_j K_ij w_j,  e_i = 1 - b_i

Values the page leaves open, fixed here:
  I_0 = 255, eps = 1 (8-bit levels; pure white -> o = 0 exactly, black -> log 256);
  pixels = the same exact uint8 RGB recovery as v1 (frame 0 of the fp32 pixel_values);
  EM: full 3x3 covariances, ridge 1e-6 I, <= 200 iterations, stop at |d loglik| / N < 1e-7;
  EM start (no threshold): T = all cells (mean, covariance), B = second moment about 0 of the half of the
      cells with the smallest |u|_2, pi_B = 0.5. The start only seeds EM; the fit is maximum likelihood.
  Gaussian taps: radius 4 d_grid = 4 cells (the v1 rule 2*round(4 sigma)+1 with sigma = 1).
The v1 binary bright-cell map m (tau_p 215, tau_c 0.9) is still computed, but ONLY as the logged
diagnostic m_all / m_sel so v1 and v2 logs stay comparable; it does not enter e_i.

Log (one line per Reasoner prefill, before v1's `[NOVA] kept ...` line):
  [NOVA2] n= piB= muT=[..] trB= trT= w_mean= w_gt05= b_mean= e_mean= e_lt01= w_vs_m_agree= em_iter= sec=
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

I0 = 255.0
EPS = 1.0
RIDGE = 1e-6
EM_MAX_ITER = 200
EM_TOL = 1e-7


def enabled() -> bool:
    return os.environ.get("VLMAS_NOVA_V2", "").strip() == "1"


def cell_od(pixel_values: torch.Tensor, grid_thw, *, patch: int = 16, temporal: int = 2, merge: int = 2,
            mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """pixel_values [Np, C*T*P*P] (HF Qwen2-VL layout, consecutive merge*merge patches = one merged token)
    -> u [N, 3] mean optical density per merged cell (float64)."""
    unit = merge * merge
    pv = pixel_values.detach().to(torch.float32)
    n_patch = int(pv.shape[0])
    if pv.shape[1] != 3 * temporal * patch * patch:
        raise ValueError(f"unexpected patch vector length {int(pv.shape[1])}")
    expected = sum(int(t * h * w) for t, h, w in torch.as_tensor(grid_thw).tolist())
    if n_patch != expected or n_patch % unit:
        raise ValueError(f"{n_patch} patches vs grid {expected} (unit {unit})")
    x = pv.reshape(n_patch, 3, temporal, patch, patch)[:, :, 0]
    rgb = torch.round((x * std + mean) * 255.0).clamp_(0.0, 255.0).to(torch.float64)
    od = -torch.log((rgb + EPS) / (I0 + EPS))                           # white 255 -> exactly 0
    per_patch = od.mean(dim=(2, 3))                                     # [Np, 3]
    return per_patch.reshape(n_patch // unit, unit, 3).mean(dim=1)


def _log_gauss(u: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    d = u.shape[1]
    L = torch.linalg.cholesky(cov)
    y = torch.linalg.solve_triangular(L, (u - mu).T, upper=False)
    return -0.5 * (y * y).sum(dim=0) - torch.log(torch.diagonal(L)).sum() - 0.5 * d * math.log(2.0 * math.pi)


def fit_null_em(u: torch.Tensor, max_iter: int = EM_MAX_ITER, tol: float = EM_TOL) -> dict:
    """Two-state Gaussian mixture on u [N, 3]; the background mean is pinned at 0. Returns the posterior
    w = P(B | u) [N] and the fitted parameters."""
    u = u.to(torch.float64)
    N, d = int(u.shape[0]), int(u.shape[1])
    eye = torch.eye(d, dtype=torch.float64, device=u.device)
    zero = torch.zeros(d, dtype=torch.float64, device=u.device)
    if N < 2 * (d + 1):
        w = torch.zeros(N, dtype=torch.float64, device=u.device)
        return {"w": w, "pi_b": 0.0, "mu_t": u.mean(dim=0) if N else zero, "cov_b": eye * RIDGE,
                "cov_t": eye * RIDGE, "iters": 0, "loglik": float("nan"), "degenerate": True}
    norm = u.norm(dim=1)
    low = norm <= norm.median()
    cov_b = (u[low].T @ u[low]) / max(1, int(low.sum())) + RIDGE * eye
    mu_t = u.mean(dim=0)
    c = u - mu_t
    cov_t = (c.T @ c) / N + RIDGE * eye
    pi_b = 0.5
    prev = None
    it = 0
    for it in range(1, max_iter + 1):
        lb = math.log(max(pi_b, 1e-300)) + _log_gauss(u, zero, cov_b)
        lt = math.log(max(1.0 - pi_b, 1e-300)) + _log_gauss(u, mu_t, cov_t)
        tot = torch.logaddexp(lb, lt)
        w = torch.exp(lb - tot)
        ll = float(tot.sum())
        if prev is not None and abs(ll - prev) / N < tol:
            break
        prev = ll
        nb, nt = float(w.sum()), float((1.0 - w).sum())
        if nb < 1e-8 or nt < 1e-8:                                      # one state is empty: stop, keep w
            break
        pi_b = nb / N
        cov_b = (u.T * w) @ u / nb + RIDGE * eye
        mu_t = ((1.0 - w)[:, None] * u).sum(dim=0) / nt
        c = u - mu_t
        cov_t = (c.T * (1.0 - w)) @ c / nt + RIDGE * eye
    lb = math.log(max(pi_b, 1e-300)) + _log_gauss(u, zero, cov_b)
    lt = math.log(max(1.0 - pi_b, 1e-300)) + _log_gauss(u, mu_t, cov_t)
    tot = torch.logaddexp(lb, lt)
    return {"w": torch.exp(lb - tot), "pi_b": float(pi_b), "mu_t": mu_t, "cov_b": cov_b, "cov_t": cov_t,
            "iters": it, "loglik": float(tot.sum()), "degenerate": False}


def heat_kernel(d_grid: float = 1.0, radius: int | None = None) -> torch.Tensor:
    if radius is None:
        radius = int(round(4.0 * d_grid))
    radius = max(1, radius)
    ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
    g1 = torch.exp(-(ax ** 2) / (2.0 * d_grid * d_grid))
    return g1[:, None] * g1[None, :]                                    # unnormalised: normalised per cell below


def local_null(w: torch.Tensor, grids, d_grid: float = 1.0) -> torch.Tensor:
    """b_i = sum_{j in V_g(i)} K_ij w_j with K normalised over the cells of the same crop (page §2.2)."""
    k = heat_kernel(d_grid).to(w.device)
    r = (int(k.shape[0]) - 1) // 2
    out = torch.empty(int(w.numel()), dtype=torch.float64, device=w.device)
    off = 0
    for gh, gw in grids:
        n = int(gh * gw)
        img = w[off:off + n].to(torch.float64).reshape(1, 1, int(gh), int(gw))
        num = F.conv2d(F.pad(img, (r, r, r, r)), k[None, None])
        den = F.conv2d(F.pad(torch.ones_like(img), (r, r, r, r)), k[None, None])
        out[off:off + n] = (num / den)[0, 0].reshape(-1)
        off += n
    if off != int(w.numel()):
        raise ValueError("grids do not cover the cell map")
    return out.clamp_(0.0, 1.0)


_V1_EVIDENCE = None


def evidence_v2(pixel_values: torch.Tensor, grid_thw, cfg: dict | None = None, *, merge: int = 2,
                patch: int = 16, temporal: int = 2) -> dict:
    """Drop-in for secld_select.evidence: same keys (r, m, b, e, grids) + w, u, fit."""
    t0 = time.time()
    thw = [tuple(int(v) for v in row) for row in torch.as_tensor(grid_thw).tolist()]
    grids = []
    for t, h, w_ in thw:
        grids.extend([(h // merge, w_ // merge)] * t)
    u = cell_od(pixel_values, grid_thw, patch=patch, temporal=temporal, merge=merge)
    fit = fit_null_em(u)
    w = fit["w"]
    b = local_null(w, grids, 1.0)
    e = (1.0 - b).clamp_(0.0, 1.0)
    # v1 bright-cell map: logged diagnostic only (m_all / m_sel), never used for e.
    from memory import secld_select as _sec
    v1cfg = dict(cfg or {})
    r = _sec.cell_bright_ratio(pixel_values, grid_thw, patch=patch, temporal=temporal, merge=merge,
                               tau_p=float(v1cfg.get("tau_p", 215.0)))
    m = (r >= float(v1cfg.get("tau_c", 0.9))).to(torch.float32)
    agree = float(((w > 0.5).to(torch.float32).cpu() == m.cpu()).float().mean()) if int(m.numel()) else float("nan")
    mu = [round(float(v), 3) for v in fit["mu_t"].tolist()]
    print(f"[NOVA2] n={int(w.numel())} piB={fit['pi_b']:.3f} muT={mu} trB={float(torch.trace(fit['cov_b'])):.5f} "
          f"trT={float(torch.trace(fit['cov_t'])):.5f} w_mean={float(w.mean()):.3f} "
          f"w_gt05={float((w > 0.5).double().mean()):.3f} b_mean={float(b.mean()):.3f} e_mean={float(e.mean()):.3f} "
          f"e_lt01={float((e < 0.1).double().mean()):.3f} w_vs_m_agree={agree:.3f} em_iter={fit['iters']} "
          f"degenerate={fit['degenerate']} sec={time.time() - t0:.2f}", flush=True)
    return {"r": r, "m": m, "b": b.to(torch.float32), "e": e.to(torch.float32), "grids": grids,
            "w": w.to(torch.float32), "u": u, "fit": fit}


def patch(module) -> None:
    """Import-hook entry: swap memory.secld_select.evidence for evidence_v2 (this process only)."""
    global _V1_EVIDENCE
    if not enabled() or getattr(module, "_nova_v2_patched", False):
        return
    _V1_EVIDENCE = module.evidence
    module.evidence = evidence_v2
    module._nova_v2_patched = True
    print("[NOVA2] patched memory.secld_select.evidence -> memory.nova_v2_select.evidence_v2 "
          "(OD two-state EM posterior + in-crop heat kernel, d_grid=1)", flush=True)


__all__ = ["enabled", "cell_od", "fit_null_em", "heat_kernel", "local_null", "evidence_v2", "patch"]
