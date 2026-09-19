#!/usr/bin/env python3
"""CSIS support=branch 유닛 — 노션 "Question-Agnostic Cross-Scale Innovation-Support Selection"
canonical 식(L = D_a^{1/2}(C (.) B)D_a^{1/2}, F = logdet(I + L_S))이 코드와 맞는지.

usage: python3 tests_hj_test_csis_branch.py
"""
import importlib.util
import inspect
import itertools
import os
import sys

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, TREE)
from memory.csis_select import (csis_keep_mask, innovation_support_kernel, logdet_greedy_kernel,
                                spatial_kernel, support_branches, support_kernel)
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
    """x5 3장 + x20 5장 (tests_hj_test_csis.py 와 같은 모양)."""
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


def logdet_F(L, idx):
    if not idx:
        return 0.0
    sub = L[idx][:, idx]
    return float(torch.logdet(torch.eye(len(idx), dtype=L.dtype) + sub))


print("[support_branches]")
check("toy roots/children", support_branches((-1, -1, -1, 0, 1, 0, 1, 0)) == (0, 1, 2, 0, 1, 0, 1, 0))
check("chain goes to the top", support_branches((-1, 0, 1, 2)) == (0, 0, 0, 0))
check("self-parent is its own branch", support_branches((-1, 1, 0)) == (0, 1, 0))
check("out-of-range parent = root", support_branches((-1, 9, -5)) == (0, 1, 2))
check("cycle terminates", len(support_branches((1, 0, 0))) == 3)

print("[kernel]")
z, counts, parents, grid, boxes, mags = toy()
R, _ = null_calibrated_residuals(z, counts, parents)
br = support_branches(parents)
B = support_kernel(counts, br)
L, a, C = innovation_support_kernel(R, B)
check("B symmetric 0/1", torch.equal(B, B.T) and set(B.unique().tolist()) <= {0.0, 1.0})
check("B PSD", float(torch.linalg.eigvalsh(B).min()) > -1e-9)
check("L PSD", float(torch.linalg.eigvalsh(L).min()) > -1e-9, float(torch.linalg.eigvalsh(L).min()))
check("diag L = a = ||r~||^2", torch.allclose(torch.diagonal(L), (R * R).sum(1), atol=1e-9))
check("L == (r~ r~^T) (.) B", torch.allclose(L, (R @ R.T) * B, atol=1e-8),
      float((L - (R @ R.T) * B).abs().max()))
obs_tok = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
b_tok = torch.tensor(br)[obs_tok]
cross = b_tok[:, None] != b_tok[None, :]
check("cross-branch entries are exactly 0", float(L[cross].abs().max()) == 0.0)
check("C unit diagonal", torch.allclose(torch.diagonal(C), torch.ones(C.shape[0], dtype=C.dtype), atol=1e-9))

print("[greedy]")
budget = 24
keep, order, gains = logdet_greedy_kernel(L, budget)
check("budget respected", int(keep.sum()) == budget and len(order) == budget)
check("sum of gains = F(S)", abs(sum(gains) - logdet_F(L, order)) < 1e-8, (sum(gains), logdet_F(L, order)))
# exact Schur-complement marginal gain at every step
ok_schur, ok_argmax = True, True
for t in range(budget):
    S = order[:t]
    j = order[t]
    if S:
        LS = L[S][:, S]
        inv = torch.linalg.inv(torch.eye(len(S), dtype=L.dtype) + LS)
        schur = float(torch.log(1 + L[j, j] - L[j, S] @ inv @ L[S, j]))
    else:
        schur = float(torch.log(1 + L[j, j]))
    ok_schur &= abs(schur - gains[t]) < 1e-8
    base = logdet_F(L, S)
    best = max(logdet_F(L, S + [i]) - base for i in range(L.shape[0]) if i not in S)
    ok_argmax &= abs(best - gains[t]) < 1e-8
check("gain = log(1 + L_ii - L_iS (I+L_S)^-1 L_Si)", ok_schur)
check("greedy picks the brute-force argmax each step", ok_argmax)
# block-diagonal: F decomposes over branches
per_branch = sum(logdet_F(L, [i for i in order if int(b_tok[i]) == b]) for b in set(br))
check("F(S) = sum over branches", abs(per_branch - logdet_F(L, order)) < 1e-8)

print("[limits / arms]")
one_branch = (-1, 0, 0, 0, 0, 0, 0, 0)          # every observation under root 0
R1, _ = null_calibrated_residuals(z, counts, one_branch)
kb = csis_keep_mask(z, counts, one_branch, budget, grid_hws=grid, boxes=boxes, support="branch")
ki = csis_keep_mask(z, counts, one_branch, budget, grid_hws=grid, boxes=boxes, support="rbf", sigma=1e12)
check("single branch == rbf sigma->inf (PCRIS v2 global log-det)", torch.equal(kb, ki))
keepA, infoA = csis_keep_mask(z, counts, parents, budget, grid_hws=grid, boxes=boxes,
                              magnifications=mags, support="branch", return_info=True)
check("info carries support/branches", infoA["support"] == "branch" and infoA["branches"] == list(br))
check("info n_branches", infoA["n_branches"] == 3 and infoA["sigma_x"] is None)
check("kept_by_branch sums to B", sum(v[0] for v in infoA["kept_by_branch"].values()) == budget)
_, infoW = csis_keep_mask(z, counts, parents, budget, grid_hws=grid, boxes=boxes,
                          support="branch", cond="wrong", return_info=True)
check("cond=wrong keeps TRUE branches", infoW["branches"] == list(br) and infoW["parents"] != parents)
_, infoS = csis_keep_mask(z, counts, (-1, -1, -1, 3, 1, 0, 1, 0), budget, grid_hws=grid, boxes=boxes,
                          support="branch", return_info=True)
check("self-parent x20 -> singleton branch", infoS["self_parent_dropped"] == 1
      and infoS["branches"][3] == 3 and infoS["singleton_branches"] == 2)  # obs 2 (lone x5) and obs 3
try:
    csis_keep_mask(z, counts, parents, budget, grid_hws=grid, boxes=boxes, support="gauss")
    check("bad support rejected", False)
except ValueError:
    check("bad support rejected", True)
sig = inspect.signature(csis_keep_mask).parameters
check("no question / candidate argument in the selector",
      not any(k in sig for k in ("question", "q", "q_ids", "candidates", "q2v", "answer")))

print("[rbf default unchanged]")
bak = os.path.join(TREE, "memory", "csis_select.py.bak_0914_branch")
if os.path.exists(bak):
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("csis_old", bak)
    spec = importlib.util.spec_from_loader("csis_old", loader)
    old = importlib.util.module_from_spec(spec)
    loader.exec_module(old)
    same_keep, same_info = True, True
    for seed, (cond, ctx, nc) in itertools.product(range(3), [("true", "parent", True), ("wrong", "parent", True),
                                                            ("none", "parent", False), ("true", "ancestors", True)]):
        zz, cc, pp, gg, bb, mm = toy(seed)
        for sigma in (None, 512.0, 1e9):
            k_new, i_new = csis_keep_mask(zz, cc, pp, 20, grid_hws=gg, boxes=bb, magnifications=mm, sigma=sigma,
                                          cond=cond, context=ctx, nullcal=nc, return_info=True)
            k_old, i_old = old.csis_keep_mask(zz, cc, pp, 20, grid_hws=gg, boxes=bb, magnifications=mm, sigma=sigma,
                                              cond=cond, context=ctx, nullcal=nc, return_info=True)
            same_keep &= torch.equal(k_new, k_old)
            i_new = dict(i_new)
            check_support = i_new.pop("support") == "rbf"
            same_info &= check_support and i_new == i_old
    check("rbf keep masks identical to pre-change module (36 configs)", same_keep)
    check("rbf info identical apart from the added support key", same_info)
else:
    check("pre-change backup present for the regression check", False, bak)

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
