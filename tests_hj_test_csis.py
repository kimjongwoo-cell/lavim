#!/usr/bin/env python3
"""CSIS 유닛 — 스펙 식과 두 극한이 실제로 성립하는지.

usage: python3 tests_hj_test_csis.py
"""
import sys
import torch

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory.csis_select import (csis_keep_mask, default_sigma, logdet_greedy_kernel,
                                spatial_kernel, token_coordinates)
from memory.pcris_select import logdet_greedy
from memory.qpris_select import null_calibrated_residuals

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def toy(seed=0, n_obs=8, tok=16):
    """x5 3장 + x20 5장, 실제 트리와 같은 parents/boxes 모양."""
    torch.manual_seed(seed)
    counts = (tok,) * n_obs
    grid = ((4, 4),) * n_obs
    boxes = ((7303, 1108, 4096, 4096), (7303, 5204, 4096, 4096), (21911, 15614, 4096, 4096),
             (9351, 2132, 1024, 1024), (10375, 6228, 1024, 1024), (9351, 4180, 1024, 1024),
             (10375, 7252, 1024, 1024), (10375, 3156, 1024, 1024))
    parents = (-1, -1, -1, 0, 1, 0, 1, 0)
    mags = (5, 5, 5, 20, 20, 20, 20, 20)
    z = torch.randn(n_obs * tok, 32, dtype=torch.float64)
    return z, counts, parents, grid, boxes, mags


print("== token_coordinates")
z, counts, parents, grid, boxes, mags = toy()
xy = token_coordinates(counts, grid, boxes)
check("shape [N,2]", tuple(xy.shape) == (128, 2), tuple(xy.shape))
# crop 0: box (7303,1108,4096,4096), 4x4 grid -> 셀 폭 1024, 첫 셀 중심 +512
check("첫 토큰 중심", torch.allclose(xy[0], torch.tensor([7303 + 512.0, 1108 + 512.0], dtype=torch.float64)), xy[0])
check("같은 행 두 번째 칸은 +1024", abs(float(xy[1][0] - xy[0][0]) - 1024.0) < 1e-9)
check("다음 행은 y +1024", abs(float(xy[4][1] - xy[0][1]) - 1024.0) < 1e-9)
# crop 3 은 x20 1024px -> 셀 256
check("x20 셀 간격 256", abs(float(xy[49][0] - xy[48][0]) - 256.0) < 1e-9)
try:
    token_coordinates((16,), ((3, 3),), ((0, 0, 10, 10),))
    check("grid 불일치 ValueError", False)
except ValueError:
    check("grid 불일치 ValueError", True)

print("== spatial_kernel")
K = spatial_kernel(xy, 1024.0)
check("대각 1", torch.allclose(torch.diagonal(K), torch.ones(128, dtype=torch.float64)))
check("대칭", torch.allclose(K, K.T))
check("PSD", float(torch.linalg.eigvalsh(K).min()) > -1e-8, float(torch.linalg.eigvalsh(K).min()))
check("가까울수록 큼", float(K[0, 1]) > float(K[0, 100]))
check("sigma=0 이면 항등", torch.allclose(spatial_kernel(xy, 0.0), torch.eye(128, dtype=torch.float64)))

print("== Schur product: L = G_res (.) G_spa 는 PSD")
R, _ = null_calibrated_residuals(z, counts, parents)
L = (R @ R.T) * K
check("L PSD", float(torch.linalg.eigvalsh(L).min()) > -1e-8, float(torch.linalg.eigvalsh(L).min()))
check("L 대각 = ||r~||^2 (증거 강도 보존)",
      torch.allclose(torch.diagonal(L), (R * R).sum(dim=1)))

print("== logdet_greedy_kernel 이 기존 logdet_greedy 와 일치 (L = R R^T 일 때)")
k1, o1, g1 = logdet_greedy(R, 20)
k2, o2, g2 = logdet_greedy_kernel(R @ R.T, 20)
check("같은 keep", bool((k1 == k2).all()))
check("같은 순서", o1 == o2, f"{o1[:5]} vs {o2[:5]}")
check("같은 gain", max(abs(a - b) for a, b in zip(g1, g2)) < 1e-9)

print("== 극한 1: sigma -> inf 이면 PCRIS v2 와 동일")
keep_inf = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes, sigma=1e12)
check("PCRIS v2 와 같은 선택", bool((keep_inf == k1).all()))

print("== 극한 2: sigma -> 0 이면 잔차 norm top-B")
keep_zero = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes, sigma=0.0)
topb = torch.zeros(128, dtype=torch.bool)
topb[R.norm(dim=1).argsort(descending=True)[:20]] = True
check("norm top-B 와 같은 선택", bool((keep_zero == topb).all()))

print("== default_sigma = 가장 작은 crop 변 (fine observation footprint)")
check("1024", abs(default_sigma(boxes) - 1024.0) < 1e-9, default_sigma(boxes))

print("== csis_keep_mask 기본 동작")
keep, info = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes,
                            magnifications=mags, return_info=True)
check("예산만큼 선택", int(keep.sum()) == 20, int(keep.sum()))
check("info mode", info["mode"] == "csis")
check("sigma 기록", info["sigma_x"] == 1024.0, info["sigma_x"])
check("ranks 기록 (root 0, child >0)",
      info["ranks"][:3] == [0, 0, 0] and all(r > 0 for r in info["ranks"][3:]), info["ranks"])
check("per_crop 합 = B", sum(info["per_crop"]) == 20, info["per_crop"])
check("kept_by_mag 합 = B", sum(info["kept_by_mag"].values()) == 20, info["kept_by_mag"])
check("공간 마스크가 off-diagonal 을 실제로 줄임",
      info["l_offdiag_abs"][1] < info["gres_offdiag_abs"][1],
      f"{info['l_offdiag_abs']} vs {info['gres_offdiag_abs']}")

print("== 경계")
check("budget 0", int(csis_keep_mask(z, counts, parents, 0, grid_hws=grid, boxes=boxes).sum()) == 0)
check("budget > N 이면 전부", int(csis_keep_mask(z, counts, parents, 999,
                                              grid_hws=grid, boxes=boxes).sum()) == 128)
k_a = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes)
k_b = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes)
check("결정적", bool((k_a == k_b).all()))

print("== cond 통제군이 실제로 다른 선택을 냄")
k_true = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes, cond="true")
k_wrong = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes, cond="wrong")
k_none = csis_keep_mask(z, counts, parents, 20, grid_hws=grid, boxes=boxes, cond="none")
check("true != wrong", not bool((k_true == k_wrong).all()))
check("true != none", not bool((k_true == k_none).all()))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
