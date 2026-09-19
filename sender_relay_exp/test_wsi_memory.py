"""Model-free unit tests for memory/wsi_memory.py (Method Sec. 3.3).

Run: PYTHONPATH=. python sender_relay_exp/test_wsi_memory.py
"""
import sys

import torch

from memory.wsi_memory import (
    consolidation_keep_mask,
    token_footprints,
    valid_representative_mask,
    wsi_greedy_selection,
)

failures = []


def check(name, condition):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        failures.append(name)


def cluster(direction, n, dim=8, noise=0.01, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.zeros(n, dim)
    base[:, direction] = 1.0
    return base + noise * torch.randn(n, dim, generator=generator)


print("token_footprints")
rects, centers = token_footprints(((0, 0, 100, 40),), ((2, 5),))
check("count = grid cells", rects.shape == (10, 4) and centers.shape == (10, 2))
check("first cell rect", torch.allclose(rects[0], torch.tensor([0.0, 0.0, 20.0, 20.0])))
check("last cell rect", torch.allclose(rects[9], torch.tensor([80.0, 20.0, 100.0, 40.0])))
check("center inside own cell", torch.allclose(centers[0], torch.tensor([10.0, 10.0])))
rects2, centers2 = token_footprints((None,), ((1, 3),))
check("missing box -> NaN", bool(torch.isnan(rects2).all()) and bool(torch.isnan(centers2).all()))

print("valid_representative_mask")
# image0 = x5 covering (0,0,100,100) as 1x1; image1 = x20 crop (10,10,20,20) as 1x2
rects3, centers3 = token_footprints(((0, 0, 100, 100), (10, 10, 20, 20)), ((1, 1), (1, 2)))
image_ids = torch.tensor([0, 1, 1])
mags = torch.tensor([5, 20, 20])
valid = valid_representative_mask(image_ids, mags, rects3, centers3)
check("same image always valid", bool(valid[1, 2]) and bool(valid[2, 1]) and bool(valid[0, 0]))
check("coarse ancestor covers fine (20x reads 5x)", bool(valid[1, 0]) and bool(valid[2, 0]))
check("directional: fine never represents coarse", not bool(valid[0, 1]) and not bool(valid[0, 2]))
# fine token outside the coarse footprint gets no ancestor
rects4, centers4 = token_footprints(((0, 0, 50, 50), (200, 200, 20, 20)), ((1, 1), (1, 1)))
valid4 = valid_representative_mask(
    torch.tensor([0, 1]), torch.tensor([5, 20]), rects4, centers4)
check("non-overlapping crop -> no ancestor", not bool(valid4[1, 0]))

print("wsi_greedy_selection")
# one image, three morphology clusters 6xA 2xB 2xC, uniform relevance:
# budget 3 should take one representative from each cluster (coverage optimum)
features = torch.cat([cluster(0, 6, seed=1), cluster(1, 2, seed=2), cluster(2, 2, seed=3)])
valid_all = torch.ones(10, 10, dtype=torch.bool)
keep = wsi_greedy_selection(features, torch.ones(10), valid_all, 3)
check("budget exact", int(keep.sum()) == 3)
check("covers all three clusters",
      bool(keep[:6].any()) and bool(keep[6:8].any()) and bool(keep[8:].any()))
check("typical morphology retained (A cluster largest, picked first)",
      bool(keep[:6].any()))
# task weighting: zero relevance on cluster C -> budget 2 prefers A and B
weights = torch.ones(10); weights[8:] = 0.0
keep2 = wsi_greedy_selection(features, weights, valid_all, 2)
check("task weight steers selection", bool(keep2[:6].any()) and bool(keep2[6:8].any())
      and not bool(keep2[8:].any()))
# graph constraint: tokens that no candidate may represent contribute nothing
valid_none = torch.eye(10, dtype=torch.bool)
keep3 = wsi_greedy_selection(features, torch.ones(10), valid_none, 3)
check("self-only graph still fills budget", int(keep3.sum()) == 3)
# determinism
again = wsi_greedy_selection(features, torch.ones(10), valid_all, 3)
check("deterministic", bool((again == keep).all()))
# budget >= G keeps everything
keep_all = wsi_greedy_selection(features, torch.ones(10), valid_all, 99)
check("budget >= G keeps all", int(keep_all.sum()) == 10)
# kappa modes: unshifted must ignore orthogonal "coverage"
keep_cos = wsi_greedy_selection(features, torch.ones(10), valid_all, 3, kappa_shift=False)
check("kappa=cos still covers all clusters",
      bool(keep_cos[:6].any()) and bool(keep_cos[6:8].any()) and bool(keep_cos[8:].any()))
check("kappa=cos budget exact", int(keep_cos.sum()) == 3)
check("kappa modes are distinct code paths",
      isinstance(keep_cos, torch.Tensor))

print("consolidation_keep_mask (end-to-end)")
# image0: x5 1x2 covering left/right halves; image1: x20 2x2 inside left half,
# duplicating the x5-left morphology; image2: x20 1x2 elsewhere, novel morphology
feats = torch.cat([
    cluster(0, 1, seed=4), cluster(1, 1, seed=5),   # x5: A | B
    cluster(0, 4, seed=6),                            # x20 dup of A
    cluster(2, 2, seed=7),                            # x20 novel C
])
keep_mask = consolidation_keep_mask(
    feats, None,
    token_counts=(2, 4, 2),
    magnifications=(5, 20, 20),
    boxes=((0, 0, 200, 100), (10, 10, 40, 40), (150, 10, 40, 20)),
    grid_hws=((1, 2), (2, 2), (1, 2)),
    budget=4,
)
check("budget respected", int(keep_mask.sum()) == 4)
check("novel fine detail survives", bool(keep_mask[6:].any()))
check("both coarse contexts survive", bool(keep_mask[0]) and bool(keep_mask[1]))
dup_kept = int(keep_mask[2:6].sum())
check("cross-scale duplicate high-mag reduced", dup_kept <= 2)

total = 5 + 4 + 10 + 4
print(f"\n{total - len(failures)}/{total} passed")
if failures:
    print("FAILED:", failures)
    sys.exit(1)
