"""Model-free unit tests for memory/consolidate.py (Components 1+2).

Run: PYTHONPATH=. python sender_relay_exp/test_consolidate.py
"""
import sys

import torch

from memory.consolidate import (
    cross_scale_consolidate,
    morphology_coverage_rescue,
)

failures = []


def check(name, condition):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        failures.append(name)


def cluster(direction, n, noise=0.01, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.zeros(n, 8)
    base[:, direction] = 1.0
    return base + noise * torch.randn(n, 8, generator=generator)


def coverage(features, keep):
    z = torch.nn.functional.normalize(features.float(), dim=-1)
    kept = torch.nonzero(keep, as_tuple=True)[0]
    return float((z @ z[kept].T).max(dim=-1).values.sum())


print("Component 1 — morphology_coverage_rescue")
# one image: 6xA, 2xB, 2xC; pruning kept 4 (3 A + 1 B) — redundant A's should
# be traded for the uncovered C cluster while budget stays 4.
features = torch.cat([cluster(0, 6, seed=1), cluster(1, 2, seed=2), cluster(2, 2, seed=3)])
keep = torch.zeros(10, dtype=torch.bool)
keep[[0, 1, 2, 6]] = True
new_keep, swaps, gain = morphology_coverage_rescue(features, keep, (10,))
check("budget preserved", int(new_keep.sum()) == 4)
check("swaps happened", swaps >= 1)
check("coverage increased", coverage(features, new_keep) > coverage(features, keep) - 1e-6)
check("gain positive", gain > 0)
check("C cluster now covered", bool(new_keep[8] or new_keep[9]))
check("an A representative survives", bool(new_keep[:6].any()))
check("original mask untouched", int(keep.sum()) == 4 and bool(keep[6]))

# already-perfect coverage -> no swaps, identical mask
features2 = torch.cat([cluster(0, 3, seed=4), cluster(1, 3, seed=5)])
keep2 = torch.zeros(6, dtype=torch.bool)
keep2[[0, 3]] = True
new_keep2, swaps2, _ = morphology_coverage_rescue(features2, keep2, (6,))
check("no-gain -> no swaps", swaps2 == 0)
check("no-gain -> mask identical", bool((new_keep2 == keep2).all()))

# per-image isolation: two images, second has nothing to fix
features3 = torch.cat([features, features2])
keep3 = torch.cat([keep, keep2])
new_keep3, _, _ = morphology_coverage_rescue(features3, keep3, (10, 6))
check("image budgets preserved independently",
      int(new_keep3[:10].sum()) == 4 and int(new_keep3[10:].sum()) == 2)
check("second image untouched", bool((new_keep3[10:] == keep2).all()))

# determinism
again, again_swaps, _ = morphology_coverage_rescue(features, keep, (10,))
check("deterministic", bool((again == new_keep).all()) and again_swaps == swaps)

# max_swaps cap
capped, capped_swaps, _ = morphology_coverage_rescue(
    features, keep, (10,), max_swaps_per_image=1)
check("swap cap respected", capped_swaps <= 1)

print("Component 2 — cross_scale_consolidate")
# image 0 = x5 parent (clusters A,B); image 1 = x20 child with parent 0:
# kept child groups duplicate parent content (A,B), dropped ones are novel (C,D).
parent_features = torch.cat([cluster(0, 2, seed=6), cluster(1, 2, seed=7)])
child_features = torch.cat([
    cluster(0, 1, seed=8), cluster(1, 1, seed=9),   # redundant with parent
    cluster(2, 1, seed=10), cluster(3, 1, seed=11),  # novel detail
])
features4 = torch.cat([parent_features, child_features])
keep4 = torch.zeros(8, dtype=torch.bool)
keep4[[0, 2]] = True      # parent keeps
keep4[[4, 5]] = True      # child keeps = the redundant pair
new_keep4, cross_swaps = cross_scale_consolidate(
    features4, keep4, (4, 4), (0, 0))
check("child budget preserved", int(new_keep4[4:].sum()) == 2)
check("parent untouched", bool((new_keep4[:4] == keep4[:4]).all()))
check("swaps happened", cross_swaps >= 1)
check("novel detail revived", bool(new_keep4[6] or new_keep4[7]))

# margin blocks swaps when redundancy differences are tiny
same = torch.cat([cluster(0, 4, seed=12), cluster(0, 4, seed=13)])
keep5 = torch.zeros(8, dtype=torch.bool)
keep5[[0, 1, 4, 5]] = True
new_keep5, swaps5 = cross_scale_consolidate(same, keep5, (4, 4), (0, 0), margin=0.05)
check("margin blocks near-tie swaps", swaps5 == 0 and bool((new_keep5 == keep5).all()))

# root/self-parent images are never touched
new_keep6, swaps6 = cross_scale_consolidate(features4, keep4, (4, 4), (0, 1))
check("self-parent child skipped", swaps6 == 0 and bool((new_keep6 == keep4).all()))
new_keep7, swaps7 = cross_scale_consolidate(features4, keep4, (4, 4), (-1, -1))
check("rootless images skipped", swaps7 == 0 and bool((new_keep7 == keep4).all()))

# determinism
again4, again_cross = cross_scale_consolidate(features4, keep4, (4, 4), (0, 0))
check("deterministic", bool((again4 == new_keep4).all()) and again_cross == cross_swaps)

print("Pipeline order — C1 then C2 keeps budget")
piped, _, _ = morphology_coverage_rescue(features4, keep4, (4, 4))
piped, _ = cross_scale_consolidate(piped_features := features4, piped, (4, 4), (0, 0))
check("E-arm budget preserved", int(piped.sum()) == int(keep4.sum()))

total = 12 + 8 + 1
print(f"\n{total - len(failures)}/{total} passed")
if failures:
    print("FAILED:", failures)
    sys.exit(1)
