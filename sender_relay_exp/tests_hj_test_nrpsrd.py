"""NR-PSRD (C1 #16) unit tests for the pure functions. python3 tests_hj_test_nrpsrd.py

The projection, the support distortion D_g, the greedy and the AUC statistic are checked against
hand written references on small synthetic inputs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_nrpsrd_gtex import (  # noqa: E402
    auc, distortion_of, null_basis, per_crop_counts, rd_greedy, residualize, support_distortion)

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


# ------------------------------------------------------------------ null basis / projection
d = 8
nu = rng.normal(size=(3, d))
U, sv = null_basis(nu)
check("rank of 3 generic vectors", U.shape == (d, 3), U.shape)
check("orthonormal", near(U.T @ U, np.eye(3), 1e-10))
nu_dep = np.vstack([nu, nu[0] + nu[1], 2.0 * nu[2]])
Ud, _ = null_basis(nu_dep)
check("dependent rows do not raise the rank", Ud.shape[1] == 3, Ud.shape)
check("same span", near(Ud @ Ud.T, U @ U.T, 1e-9))

z = rng.normal(size=(5, d))
z = z / np.linalg.norm(z, axis=1, keepdims=True)
phi = residualize(z, U)
check("phi orthogonal to U", near(phi @ U, 0.0, 1e-10))
check("idempotent", near(residualize(phi, U), phi, 1e-12))
check("r=0 control is the identity", near(residualize(z, np.zeros((d, 0))), z))
check("null-spanned token has ~zero residual",
      near(residualize((nu[0] / np.linalg.norm(nu[0]))[None, :], U), 0.0, 1e-10))
check("residual norm <= 1 for unit z", bool(((phi ** 2).sum(1) <= 1 + 1e-12).all()))
r2 = null_basis(nu, rank=2)[0]
check("rank truncation", r2.shape == (d, 2) and near(r2.T @ r2, np.eye(2), 1e-10))
check("smaller subspace keeps more residual",
      bool(((residualize(z, r2) ** 2).sum(1) >= (phi ** 2).sum(1) - 1e-12).all()))

# ------------------------------------------------------------------ D_g (page §4)
P = rng.normal(size=(6, 4)) * 0.3
null = (P ** 2).sum(1)
check("empty set = null cost", near(support_distortion(P, []), null.sum(), 1e-12))
S = [1, 4]
ref = 0.0
for i in range(6):
    best = min(float(((P[i] - P[j]) ** 2).sum()) for j in S)
    ref += min(float(null[i]), best)
check("D_g == definition", near(support_distortion(P, S), ref, 1e-12), (support_distortion(P, S), ref))
check("selected token costs 0 (it explains itself)",
      near(support_distortion(P[[1]], [0]), 0.0, 1e-12))
check("monotone non-increasing in S",
      support_distortion(P, [1, 4]) <= support_distortion(P, [1]) + 1e-12
      and support_distortion(P, [1]) <= support_distortion(P, []) + 1e-12)
# a background-like support (phi ~ 0) costs ~0 with no selection at all
Pz = np.zeros((5, 4))
check("zero residual support needs no token", near(support_distortion(Pz, []), 0.0, 1e-12))

# ------------------------------------------------------------------ greedy (page §5-§6)
counts = [6, 5, 4]
phi_all = np.vstack([rng.normal(size=(6, 4)) * 0.6,
                     rng.normal(size=(5, 4)) * 0.05,     # near-null support
                     rng.normal(size=(4, 4)) * 0.6])
k = 7
order, gains, D_after, D0 = rd_greedy(phi_all, counts, k)
check("budget respected", len(order) == k, len(order))
check("no duplicates", len(set(order.tolist())) == k)
check("min 1 per support", all(c >= 1 for c in per_crop_counts(order, counts)),
      per_crop_counts(order, counts))
check("D0 == total null cost", near(D0, (phi_all ** 2).sum(), 1e-12))
check("D_after == D of the chosen set", near(D_after, distortion_of(phi_all, counts, order), 1e-9),
      (D_after, distortion_of(phi_all, counts, order)))
check("distortion decreases", D_after < D0)
check("gains sum to the reduction", near(sum(gains), D0 - D_after, 1e-9), (sum(gains), D0 - D_after))
check("gains non-negative", all(g >= -1e-12 for g in gains))
# the first pick of a support is the argmax of its own reduction (§6)
g0 = phi_all[:6]
best0 = int(np.argmax([support_distortion(g0, []) - support_distortion(g0, [j]) for j in range(6)]))
check("first pick of support 0 == argmax single-token reduction", int(order[0]) == best0,
      (int(order[0]), best0))
# after the feasibility picks the greedy is global: gains must be non-increasing from there
tail = gains[len(counts):]
check("global greedy gains non-increasing", all(tail[i] >= tail[i + 1] - 1e-9 for i in range(len(tail) - 1)),
      tail)
# the near-null support should get only its one mandatory token when the budget is tight
check("near-null support gets the minimum", per_crop_counts(order, counts)[1] == 1,
      per_crop_counts(order, counts))
# budget larger than N
o2, _, _, _ = rd_greedy(phi_all, counts, 999)
check("k > N clamps", len(o2) == 15, len(o2))
# k smaller than the number of supports still returns k picks
o3, _, _, _ = rd_greedy(phi_all, counts, 2)
check("k < G", len(o3) == 2 and len(set(o3.tolist())) == 2)

# a greedy step can never beat the exhaustive best first pick
flat_best = max((support_distortion(phi_all[:6], []) - support_distortion(phi_all[:6], [j]))
                for j in range(6))
check("first gain == best single-token gain in that support", near(gains[0], flat_best, 1e-9),
      (gains[0], flat_best))

# ------------------------------------------------------------------ distortion_of / counts
check("distortion_of empty == D0", near(distortion_of(phi_all, counts, []), D0, 1e-12))
check("per_crop_counts sums", sum(per_crop_counts(order, counts)) == k)
check("per_crop_counts places tokens", per_crop_counts([0, 6, 11], counts) == [1, 1, 1])

# ------------------------------------------------------------------ AUC
check("auc perfect", near(auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])), 1.0, 1e-12))
check("auc reversed", near(auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0, 1e-12))
check("auc all ties", near(auc(np.ones(3), np.ones(4)), 0.5, 1e-12))
check("auc one tie", near(auc(np.array([2.0, 0.0]), np.array([2.0, 1.0])), 0.375, 1e-12),
      auc(np.array([2.0, 0.0]), np.array([2.0, 1.0])))
check("auc empty -> nan", np.isnan(auc(np.array([]), np.array([1.0]))))
# brute force check against the definition on random data
a, bq = rng.normal(size=40), rng.normal(size=30)
brute = np.mean([[1.0 if x > y else (0.5 if x == y else 0.0) for y in bq] for x in a])
check("auc == brute force", near(auc(a, bq), brute, 1e-12), (auc(a, bq), brute))

print(f"NR-PSRD tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
