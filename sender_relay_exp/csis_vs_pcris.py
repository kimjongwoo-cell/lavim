#!/usr/bin/env python3
"""CSIS vs PCRIS v2 — 실제 덤프에서 무엇이 달라지는가."""
import sys, glob, torch
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.csis_select import csis_keep_mask, default_sigma, token_coordinates, spatial_kernel
from memory.qpris_select import qpris_keep_mask, null_calibrated_residuals

DUMP = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris/smoke_dump"
B = 512


def ov(a, b):
    return (a & b).sum().item() / max(a.sum().item(), 1)


def spread(coords, keep):
    sel = coords[keep]
    return float(torch.cdist(sel, sel).max()) if sel.shape[0] > 1 else 0.0


def nn_median(coords, keep):
    """선택된 토큰끼리의 최근접 거리 중앙값 — 국소 중복이 줄면 커진다."""
    sel = coords[keep]
    d = torch.cdist(sel, sel)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.median())


for f in sorted(glob.glob(f"{DUMP}/qpris_*.pt")):
    D = torch.load(f, map_location="cpu", weights_only=False)
    z, counts, parents = D["z"], D["token_counts"], D["parents"]
    grid, boxes, mags = D["grid_hws"], D["boxes"], D["magnifications"]
    coords = token_coordinates(counts, grid, boxes)
    print(f"=== {f.split('/')[-1]}  기본 sigma={default_sigma(boxes):.0f}px")

    k_pc = qpris_keep_mask(z.double(), None, counts, parents, B, cond="true",
                           context="parent", q_mode="none", nullcal=True).bool()
    print(f"    {'arm':22s} {'겹침 vs PCRIS':>13s} {'x5/x20':>10s} "
          f"{'최대 퍼짐(px)':>13s} {'최근접 중앙(px)':>15s}")
    obs = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor([int(c) for c in counts]))
    mg = torch.tensor([int(m) for m in mags])[obs]

    def row(tag, k):
        n5, n20 = int(k[mg == 5].sum()), int(k[mg == 20].sum())
        print(f"    {tag:22s} {ov(k_pc, k):13.3f} {f'{n5}/{n20}':>10s} "
              f"{spread(coords, k):13.0f} {nn_median(coords, k):15.0f}")

    row("PCRIS v2 (기준)", k_pc)
    for s in (256.0, 1024.0, 4096.0, 16384.0):
        k = csis_keep_mask(z.double(), counts, parents, B, grid_hws=grid, boxes=boxes,
                           magnifications=mags, sigma=s).bool()
        row(f"CSIS sigma={s:.0f}", k)
    # 공간 커널이 실제 좌표에서 어떤 범위인가
    Kx = spatial_kernel(coords, 1024.0)
    off = ~torch.eye(Kx.shape[0], dtype=torch.bool)
    q = torch.quantile(Kx[off].double(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    print(f"    kappa_x off-diag (sigma=1024): p5={q[0]:.4f} p50={q[1]:.4f} p95={q[2]:.4f}")
    print()
