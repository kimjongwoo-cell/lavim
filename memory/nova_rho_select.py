"""C1 canonical draft — Local-Structure-Aware Visual Memory Selection (노션 "0917 3. Method" §2-§5).

v1 (memory/nova_select.py + secld_select.py), v2, v3 파일은 무수정. 이 모듈은 §2의 local background-evidence field만
바꾼다: v1은 cell white fraction rho_i를 tau_c로 한 번 더 이진화한 m_i를 Gaussian에 넣지만, draft는 rho_i를 그대로 넣는다.
커널도 draft 식 그대로 같은 observation(crop) 안에서만 정규화한다(패딩 없음).

  rho_i = mean_x 1[gray(x) > tau_p]                      (v1과 같은 gray·tau_p)
  K_ij^(o) = exp(-|p_i-p_j|^2 / 2 sigma^2) / sum_{k in V_o} exp(...)   (crop 내부 정규화)
  b_i = sum_j K_ij^(o) rho_j,   e_i = 1 - b_i
§3-§5(q_i = sqrt(e_i) z_i, observation별 log-det, greedy, 예산 B)는 v1 코드를 그대로 쓴다.

Opt-in: VLMAS_NOVA_RHO=1 (+ VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova). sigma는 VLMAS_NOVA_SIGMA(기본 2.0)를 따른다.
Log: `[NOVARHO] n= rho_mean= b_mean= e_mean= e_lt01= sigma= vs_v1_e_corr= sec=`
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F


def enabled() -> bool:
    return os.environ.get("VLMAS_NOVA_RHO", "").strip() == "1"


def local_occupancy(rho: torch.Tensor, grids, sigma: float) -> torch.Tensor:
    """b_i = sum_j K_ij^(o) rho_j with K normalised inside each crop (draft §2)."""
    radius = max(1, int(round(4.0 * sigma)))
    ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    k = (g1[:, None] * g1[None, :]).to(device=rho.device, dtype=rho.dtype)
    out, off = torch.empty_like(rho), 0
    for gh, gw in grids:
        n = int(gh * gw)
        img = rho[off:off + n].reshape(1, 1, int(gh), int(gw))
        pad = (radius,) * 4
        num = F.conv2d(F.pad(img, pad), k[None, None])
        den = F.conv2d(F.pad(torch.ones_like(img), pad), k[None, None])
        out[off:off + n] = (num / den)[0, 0].reshape(-1)
        off += n
    if off != int(rho.numel()):
        raise ValueError("grids do not cover the cells")
    return out.clamp_(0.0, 1.0)


def evidence_rho(pixel_values, grid_thw, cfg=None, *, merge: int = 2, patch: int = 16, temporal: int = 2) -> dict:
    """Drop-in for secld_select.evidence: same keys (r, m, b, e, grids); b uses rho, not the binary m."""
    t0 = time.time()
    from memory import secld_select as _sec
    cfg = _sec.config() if cfg is None else cfg
    thw = [tuple(int(v) for v in row) for row in torch.as_tensor(grid_thw).tolist()]
    grids = []
    for t, h, w in thw:
        grids.extend([(h // merge, w // merge)] * t)
    rho = _sec.cell_bright_ratio(pixel_values, grid_thw, patch=patch, temporal=temporal, merge=merge,
                                 tau_p=float(cfg["tau_p"]))
    sigma = float(cfg["sigma"])
    b = local_occupancy(rho.to(torch.float32), grids, sigma)
    e = (1.0 - b).clamp_(0.0, 1.0)
    m = (rho >= float(cfg["tau_c"])).to(torch.float32)          # v1 mask: logged for comparison only
    b_v1 = _sec.occupancy(m, grids, sigma)
    e_v1 = (1.0 - b_v1).clamp_(0.0, 1.0)
    corr = float(torch.corrcoef(torch.stack([e.double(), e_v1.double()]))[0, 1]) if e.numel() > 1 else float("nan")
    print(f"[NOVARHO] n={int(rho.numel())} rho_mean={float(rho.mean()):.3f} b_mean={float(b.mean()):.3f} "
          f"e_mean={float(e.mean()):.3f} e_lt01={float((e < 0.1).float().mean()):.3f} sigma={sigma} "
          f"v1_e_mean={float(e_v1.mean()):.3f} vs_v1_e_corr={corr:.3f} sec={time.time() - t0:.2f}", flush=True)
    return {"r": rho, "m": m, "b": b, "e": e, "grids": grids, "e_v1": e_v1}


def patch(module) -> None:
    if not enabled() or getattr(module, "_nova_rho_patched", False):
        return
    module.evidence = evidence_rho
    module._nova_rho_patched = True
    print("[NOVARHO] patched memory.secld_select.evidence -> memory.nova_rho_select.evidence_rho "
          "(continuous rho field, in-crop normalised kernel)", flush=True)


__all__ = ["enabled", "local_occupancy", "evidence_rho", "patch"]
