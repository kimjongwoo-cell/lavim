"""NOVA rho + representation neighbourhood coherence (09-18, offline check sender_relay_exp/runs/nova_coh_offline).
Existing files are not modified. Background evidence b_i = NOVA rho (memory/nova_rho_select.py: continuous white fraction,
in-crop normalised Gaussian sigma). New: coherence of the merged vision features around each token,
  coh_i = || sum_j K_ij z_j ||,  z = L2-normalised merged feature, K = same in-crop normalised Gaussian (sigma cells),
glass tokens carry content-free, mutually inconsistent embeddings (small coh); fat / tissue tokens share content (large coh).
c_i = rank of coh_i within the prefill in [0, 1];  e_i = 1 - b_i (1 - c_i)  -> a white region is kept when its
neighbourhood embeddings agree. Selection = the unchanged support-wise log-det greedy (secld_select.select).
Opt-in VLMAS_NOVA_COH=1 (+ VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova). Log `[NOVACOH] n= b_mean= c_white= e_mean= ...`."""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from memory import nova_rho_select as _rho

_CUR: dict = {}


def enabled() -> bool:
    return os.environ.get("VLMAS_NOVA_COH", "").strip() == "1"


def coherence(z: torch.Tensor, grids, sigma: float) -> torch.Tensor:
    rad = max(1, int(round(4.0 * sigma)))
    ax = torch.arange(-rad, rad + 1, dtype=torch.float32, device=z.device)
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    k = (g1[:, None] * g1[None, :])[None, None]
    out, off = [], 0
    for gh, gw in grids:
        n = gh * gw
        zz = z[off:off + n].T.reshape(-1, 1, gh, gw)
        num = F.conv2d(zz, k, padding=rad)[:, 0]
        den = F.conv2d(torch.ones(1, 1, gh, gw, device=z.device), k, padding=rad)[0, 0]
        out.append((num / den).norm(dim=0).reshape(-1))
        off += n
    return torch.cat(out)


def evidence_coh(pixel_values, grid_thw, cfg=None, *, merge: int = 2, patch: int = 16, temporal: int = 2) -> dict:
    t0 = time.time()
    ev = _rho.evidence_rho(pixel_values, grid_thw, cfg, merge=merge, patch=patch, temporal=temporal)
    states = _CUR.get("states")
    if states is None:
        raise RuntimeError("NOVACOH: evidence called outside secld_select.select")
    z = F.normalize(states.detach().to(torch.float32), dim=-1)
    sigma = float((cfg or {}).get("sigma", 2.0))
    coh = coherence(z, ev["grids"], sigma).cpu()
    if int(coh.numel()) != int(ev["b"].numel()):
        raise RuntimeError(f"NOVACOH: coherence {coh.numel()} != cells {ev['b'].numel()}")
    c = coh.argsort().argsort().to(torch.float32) / max(1, coh.numel() - 1)
    b = ev["b"].detach().cpu().to(torch.float32)
    e = (1.0 - b * (1.0 - c)).clamp_(0.0, 1.0)
    w = b > 0.5
    print(f"[NOVACOH] n={int(e.numel())} b_mean={float(b.mean()):.3f} white_frac={float(w.float().mean()):.3f} "
          f"c_white={float(c[w].mean()) if bool(w.any()) else float('nan'):.3f} e_rho_mean={float((1 - b).mean()):.3f} "
          f"e_mean={float(e.mean()):.3f} sigma={sigma} sec={time.time() - t0:.2f}", flush=True)
    ev["e"] = e.to(ev["e"].device)
    ev["c"] = c
    return ev


def patch(module) -> None:
    if not enabled() or getattr(module, "_nova_coh_patched", False):
        return
    orig_select = module.select

    def select(pixel_values, grid_thw, states, token_counts, budget, cfg=None, **kw):
        _CUR["states"] = states
        try:
            return orig_select(pixel_values, grid_thw, states, token_counts, budget, cfg, **kw)
        finally:
            _CUR.clear()

    module.select = select
    module.evidence = evidence_coh
    module._nova_coh_patched = True
    print("[NOVACOH] patched memory.secld_select.select/evidence -> nova_rho b x (1 - neighbourhood coherence rank)", flush=True)


__all__ = ["enabled", "coherence", "evidence_coh", "patch"]
