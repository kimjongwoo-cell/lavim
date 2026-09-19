"""#14.6 clamped-coverage variant unit tests. python3 tests_hj_test_146clamp.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_146clamp_gtex import credit, first_pick_share, per_crop_counts, weighted_greedy  # noqa

PASS = FAIL = 0
rng = np.random.default_rng(0)


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


def near(a, b, tol=1e-9):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) < tol


# ---------------------------------------------------------------- credit()
cos = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0])
check("tau=2 is exactly max(cos,0)", near(credit(cos, 2.0), np.clip(cos, 0, None)))
check("tau=1 pays nothing below cos 0.5",
      near(credit(cos, 1.0), np.clip(2.0 * cos - 1.0, 0, None)), credit(cos, 1.0))
check("tau=0.5 cutoff at cos 0.75",
      near(credit(cos, 0.5), np.clip(4.0 * cos - 3.0, 0, None)))
check("self-similarity is 1 for every tau",
      all(near(credit(np.array([1.0]), t), 1.0) for t in (2.0, 1.0, 0.5, 0.25)))
check("cutoff formula 1 - tau/2",
      all(near(credit(np.array([1 - t / 2]), t), 0.0) for t in (2.0, 1.0, 0.5)))
check("never negative", bool((credit(cos, 1.0) >= 0).all()))
check("monotone in cos", bool((np.diff(credit(cos, 1.0)) >= -1e-12).all()))
check("smaller tau is stricter", bool((credit(cos, 0.5) <= credit(cos, 1.0) + 1e-12).all()))
# the clamp is the distortion form written as a similarity: 1 - min(1, d/tau), d = 2(1-cos)
d = 2.0 * (1.0 - cos)
check("clamp == 1 - min(1, d/tau)", near(credit(cos, 1.0), 1.0 - np.minimum(1.0, d / 1.0)))

# ---------------------------------------------------------------- greedy
counts = [6, 5, 4]
z = rng.normal(size=(15, 5))
z = z / np.linalg.norm(z, axis=1, keepdims=True)
w = np.concatenate([np.full(c, 1.0 / c) for c in counts])       # support-local demand sums to 1
k = 7
order, gains, F, Fmax = weighted_greedy(z, w, counts, k, 1.0)
check("budget respected", len(order) == k and len(set(order.tolist())) == k)
check("Fmax == total demand", near(Fmax, w.sum(), 1e-12))
check("coverage <= Fmax", F <= Fmax + 1e-12)
check("gains non-increasing (single greedy, no feasibility phase)",
      all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1)), gains)
check("gains sum to the coverage", near(sum(gains), F, 1e-9), (sum(gains), F))
check("counts sum to k", sum(per_crop_counts(order, counts)) == k)

# tau=2 must reproduce the audit's V1 selection exactly on the same input
o2, _, _, _ = weighted_greedy(z, w, counts, k, 2.0)
sims = []
offs = np.cumsum([0] + counts)
for g in range(3):
    zg = z[offs[g]:offs[g + 1]]
    s = np.clip(zg @ zg.T, 0.0, None)
    np.fill_diagonal(s, 1.0)
    sims.append(s)
cov = [np.zeros(c) for c in counts]
chosen = np.zeros(15, dtype=bool)
ref = []
for _ in range(k):
    gains_r = [(w[offs[g]:offs[g + 1]][:, None]
                * np.clip(sims[g] - cov[g][:, None], 0, None)).sum(axis=0) for g in range(3)]
    flat = np.concatenate(gains_r)
    flat[chosen] = -np.inf
    j = int(np.argmax(flat))
    g = int(np.searchsorted(offs, j, side="right") - 1)
    ref.append(j)
    chosen[j] = True
    cov[g] = np.maximum(cov[g], sims[g][:, j - offs[g]])
check("tau=2 == independent max(cos,0) reference", o2.tolist() == ref, (o2.tolist(), ref))

# a stricter tau must not cover more
_, _, F_strict, _ = weighted_greedy(z, w, counts, k, 0.5)
check("stricter tau covers no more", F_strict <= F + 1e-12, (F_strict, F))

# ---------------------------------------------------------------- first_pick_share
fs2 = first_pick_share(z, w, counts, k, 2.0)
fs1 = first_pick_share(z, w, counts, k, 1.0)
fs_h = first_pick_share(z, w, counts, k, 0.25)
check("first share in [0,1]", all(0.0 <= x <= 1.0 + 1e-12 for x in (fs2, fs1, fs_h)),
      (fs2, fs1, fs_h))
check("stricter tau -> smaller first-pick share", fs_h <= fs1 <= fs2 + 1e-12, (fs_h, fs1, fs2))
# with a huge tau every representative covers everything -> first pick takes all of it
fs_big = first_pick_share(z, w, counts, k, 1e6)
check("tau -> inf gives first share 1", near(fs_big, 1.0, 1e-4), fs_big)
# hand case: one support of two identical tokens, first pick must cover the whole demand
zz = np.array([[1.0, 0.0], [1.0, 0.0]])
check("identical tokens: first pick covers all",
      near(first_pick_share(zz, np.array([0.5, 0.5]), [2], 1, 1.0), 1.0, 1e-12))
# orthogonal tokens under tau=1 (cutoff cos 0.5): a representative covers only itself
zo = np.eye(2)
check("orthogonal tokens under tau=1: first pick covers only itself",
      near(first_pick_share(zo, np.array([0.5, 0.5]), [2], 1, 1.0), 0.5, 1e-12))

print(f"#14.6-clamp tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
