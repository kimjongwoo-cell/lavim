"""C1 #21.3 NOVA v3 — Optional Transparent-State Null Allocation. Notion 3ded771b-c9e2-81ba (09-17 17:30).

v1 (memory/nova_select.py, memory/secld_select.py) and v2 (memory/nova_v2_select.py) are NOT modified. v3 changes only the
cell background evidence w_i; the heat kernel (v2 local_null, d_grid = 1 cell), e_i = 1 - b_i and the support-wise log-det
greedy are reused as is. Opt-in: VLMAS_NOVA_V3=1 on top of VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova; the import hook
tmp_hj/nova_v3_hooks/sitecustomize.py swaps memory.secld_select.evidence for evidence_v3.

Page §2-§4 per physical support g (one crop), cells u_i = mean optical density (RGB, I0 = 255, eps = 1):
  H0: u ~ p_T (Gaussian mixture, K = 1..KMAX)          H1: u ~ pi_B p_B + (1 - pi_B) p_T, background = lightest state
  gamma_g = P(H1 | U_g) = sigmoid(log E1 - log E0),  r_i = P(c_i = B | u_i, H1),  w_i = gamma_g r_i
Evidence (Laplace / BIC form, filled in here because the page leaves the inference family open):
  log E(H, K) = L_K - (6K + K - 1)/2 log N + sum_k [ log prior_H(mu_k) + 3/2 log 2pi + 1/2 log det(Sigma_k / n_k) ]
  H0: every state mean ~ Uniform on the physical OD box [0, log 256]^3
  H1: the lightest state mean ~ half-normal N+(0, tau^2 I) (glass absorbs ~nothing), the rest Uniform; K >= 2
  log E_H = logsumexp over K (uniform prior on K).
Why the prior (user decision 09-17): without a prior that separates p_B from p_T, H1(B + K tissue states) is the same model
as H0(K + 1 tissue states) and gamma = 0.5 on 61/72 real crops (offline, scratchpad nova_v3/v3_offline.py).
tau: VLMAS_NOVA3_TAU (default 0.07). Offline on the 6 dumped GPU cases (72 crops): tau 0.05, 0.07, 0.10 give the same
glass / no-glass split; 0.15 flips pale TCGA crops to background, 0.25 flips most tissue crops.

Mixture fit: full-covariance EM in float64, ridge 1e-4, k-means++-style deterministic init (farthest-point from the
lowest-OD cell), 2 restarts (second init from the darkest cell), <= 100 iterations. Only cells, no pixels beyond u_i.
Log: `[NOVA3] n= gamma=[..] glass_supports= w_mean= w_gt05= e_mean= tau= sec=` per Reasoner prefill.
"""
from __future__ import annotations

import math
import os
import time

import torch

from memory import nova_v2_select as _v2

KMAX = 4
RIDGE = 1e-4
LOGBOX = 3.0 * math.log(math.log(256.0))


def enabled() -> bool:
    return os.environ.get("VLMAS_NOVA_V3", "").strip() == "1"


def tau() -> float:
    v = os.environ.get("VLMAS_NOVA3_TAU", "").strip()
    return float(v) if v else 0.07


def _log_gauss(x, mu, cov):
    d = x.shape[1]
    L = torch.linalg.cholesky(cov)
    y = torch.linalg.solve_triangular(L, (x - mu).T, upper=False)
    return -0.5 * (y * y).sum(0) - torch.log(torch.diagonal(L)).sum() - 0.5 * d * math.log(2 * math.pi)


def _init_means(x, K, start):
    idx = [start]
    dist = ((x - x[start]) ** 2).sum(1)
    for _ in range(1, K):
        j = int(torch.argmax(dist))
        idx.append(j)
        dist = torch.minimum(dist, ((x - x[j]) ** 2).sum(1))
    return x[idx].clone()


def fit_gmm(x: torch.Tensor, K: int, iters: int = 100, tol: float = 1e-6):
    """Full-covariance EM. Returns dict(loglik, weights, means, covs, resp)."""
    x = x.to(torch.float64)
    N, d = x.shape
    I = torch.eye(d, dtype=torch.float64)
    norms = x.norm(dim=1)
    best = None
    for start in (int(torch.argmin(norms)), int(torch.argmax(norms))):
        mu = _init_means(x, K, start)
        lab = torch.cdist(x, mu).argmin(1)
        w = torch.stack([(lab == k).double().mean() for k in range(K)]).clamp_min(1e-3)
        w = w / w.sum()
        cov = torch.stack([(torch.cov(x[lab == k].T) if int((lab == k).sum()) > d else torch.cov(x.T)) + RIDGE * I
                           for k in range(K)])
        prev = None
        for _ in range(iters):
            lp = torch.stack([torch.log(w[k]) + _log_gauss(x, mu[k], cov[k]) for k in range(K)])
            tot = torch.logsumexp(lp, 0)
            ll = float(tot.sum())
            R = torch.exp(lp - tot)
            if prev is not None and abs(ll - prev) < tol * N:
                break
            prev = ll
            nk = R.sum(1).clamp_min(1e-8)
            w = nk / N
            mu = (R @ x) / nk[:, None]
            cov = torch.stack([((x - mu[k]).T * R[k]) @ (x - mu[k]) / nk[k] + RIDGE * I for k in range(K)])
        lp = torch.stack([torch.log(w[k]) + _log_gauss(x, mu[k], cov[k]) for k in range(K)])
        tot = torch.logsumexp(lp, 0)
        ll = float(tot.sum())
        if best is None or ll > best["loglik"]:
            best = {"loglik": ll, "weights": w, "means": mu, "covs": cov, "resp": torch.exp(lp - tot)}
    return best


def support_gamma(u: torch.Tensor, tau_: float, kmax: int = KMAX):
    """u [N, 3] mean OD of one support -> (gamma, r [N], info)."""
    u = u.to(torch.float64)
    N = int(u.shape[0])
    e0, e1, best1 = [], [], None
    for K in range(1, kmax + 1):
        if N < 10 * K:
            break
        g = fit_gmm(u, K)
        nk = (g["weights"] * N).clamp_min(1.0)
        pen = -(K * 6 + (K - 1)) / 2.0 * math.log(N)
        occ = sum(1.5 * math.log(2 * math.pi) + 0.5 * float(torch.linalg.slogdet(g["covs"][k] / nk[k])[1]) for k in range(K))
        base = g["loglik"] + pen + occ
        e0.append(base - K * LOGBOX)
        if K >= 2:
            b = int(torch.argmin(g["means"].norm(dim=1)))
            mu = g["means"][b].clamp_min(0.0)
            lp_b = 3 * math.log(2.0) + float((-0.5 * (mu / tau_) ** 2 - math.log(tau_ * math.sqrt(2 * math.pi))).sum())
            v = base - (K - 1) * LOGBOX + lp_b
            e1.append(v)
            if best1 is None or v > best1[0]:
                best1 = (v, g, b, K)
    if not e1:
        return 0.0, torch.zeros(N, dtype=torch.float64), {"K1": 0, "muB": float("nan")}
    t0 = torch.tensor(e0, dtype=torch.float64)
    t1 = torch.tensor(e1, dtype=torch.float64)
    dlog = float(torch.logsumexp(t1, 0) - torch.logsumexp(t0, 0))
    gamma = 1.0 / (1.0 + math.exp(-max(min(dlog, 50.0), -50.0)))
    _, g, b, K = best1
    return gamma, g["resp"][b], {"K1": K, "muB": float(g["means"][b].norm()), "dlog": dlog}


def evidence_v3(pixel_values, grid_thw, cfg=None, *, merge: int = 2, patch: int = 16, temporal: int = 2) -> dict:
    """Drop-in for secld_select.evidence (keys r, m, b, e, grids) + w, gamma."""
    t0 = time.time()
    tau_ = tau()
    thw = [tuple(int(v) for v in row) for row in torch.as_tensor(grid_thw).tolist()]
    grids = []
    for t, h, w_ in thw:
        grids.extend([(h // merge, w_ // merge)] * t)
    u = _v2.cell_od(pixel_values, grid_thw, patch=patch, temporal=temporal, merge=merge).detach().cpu()  # EM on CPU float64
    ws, gammas, off = [], [], 0
    for gh, gw in grids:
        n = gh * gw
        gam, r, _ = support_gamma(u[off:off + n], tau_)
        gammas.append(gam)
        ws.append(gam * r)
        off += n
    w = torch.cat(ws)
    b = _v2.local_null(w, grids, 1.0)
    e = (1.0 - b).clamp_(0.0, 1.0)
    from memory import secld_select as _sec
    v1cfg = dict(cfg or {})
    r1 = _sec.cell_bright_ratio(pixel_values, grid_thw, patch=patch, temporal=temporal, merge=merge,
                                tau_p=float(v1cfg.get("tau_p", 215.0)))
    m = (r1 >= float(v1cfg.get("tau_c", 0.9))).to(torch.float32)
    print(f"[NOVA3] n={int(w.numel())} gamma={[round(g, 2) for g in gammas]} glass_supports={sum(g > 0.5 for g in gammas)} "
          f"w_mean={float(w.mean()):.3f} w_gt05={float((w > 0.5).double().mean()):.3f} m_all={float(m.mean()):.3f} "
          f"e_mean={float(e.mean()):.3f} tau={tau_} sec={time.time() - t0:.2f}", flush=True)
    return {"r": r1, "m": m, "b": b.to(torch.float32), "e": e.to(torch.float32), "grids": grids,
            "w": w.to(torch.float32), "gamma": gammas}


def patch(module) -> None:
    if not enabled() or getattr(module, "_nova_v3_patched", False):
        return
    module.evidence = evidence_v3
    module._nova_v3_patched = True
    print(f"[NOVA3] patched memory.secld_select.evidence -> memory.nova_v3_select.evidence_v3 (tau={tau()})", flush=True)


__all__ = ["enabled", "tau", "fit_gmm", "support_gamma", "evidence_v3", "patch"]
