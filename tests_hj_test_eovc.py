"""EOVC (C2 #14) unit tests. python3 tests_hj_test_eovc.py

The representation, the exact variance decomposition, the gain formula and the greedy are checked
against hand written references on small synthetic supports.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import eovc  # noqa: E402

PASS = FAIL = 0
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


def near(a, b, tol=1e-9):
    return float(torch.as_tensor(a - b).abs().max()) < tol


# ------------------------------------------------------------------ config
os.environ.pop("VLMAS_EOVC", None)
check("off", not eovc.enabled())
os.environ["VLMAS_EOVC"] = "1"
check("on", eovc.enabled())
check("default tol", eovc.tol() == 1e-6)

# ------------------------------------------------------------------ representation (§1)
d, n = 7, 5
gp = torch.randn(n, 4, d) * 2.0 + 0.5
Y = eovc.support_matrices(gp)
check("shape [n, d, 4]", tuple(Y.shape) == (n, d, 4), Y.shape)
u = gp / gp.norm(dim=-1, keepdim=True)
mu = u.reshape(-1, d).mean(0)
ref = (u - mu).transpose(1, 2)
check("Y == normalise then centre on the support mean", near(Y, ref, 1e-12))
check("patch states are unit norm before centring",
      near((Y.transpose(1, 2) + mu).norm(dim=-1), torch.ones(n, 4), 1e-12))
check("support mean is removed: sum over all 4n columns is 0",
      near(Y.sum(dim=(0, 2)), torch.zeros(d), 1e-11))
# scale invariance of the raw states (direction only)
check("raw scale does not change Y", near(eovc.support_matrices(gp * 3.7), Y, 1e-11))

# ------------------------------------------------------------------ exact decomposition (§2)
bet, wit = eovc.decompose(Y)
check("||Y||_F^2 == between + within (no coefficient)",
      near(eovc.variation(Y), bet + wit, 1e-11), (eovc.variation(Y) - bet - wit).abs().max())
ubar = Y.mean(dim=-1)
check("between == 4||ubar - mu||^2", near(bet, 4.0 * ubar.pow(2).sum(-1), 1e-11))
check("within == sum_r ||u - ubar||^2",
      near(wit, (Y - ubar.unsqueeze(-1)).pow(2).sum(dim=(-2, -1)), 1e-11))
# a group whose 4 patches are identical has zero within-variation
same = torch.randn(1, 1, d).repeat(1, 4, 1)
gp2 = torch.cat([same, torch.randn(3, 4, d)], dim=0)
b2, w2 = eovc.decompose(eovc.support_matrices(gp2))
check("identical patches -> within 0", near(w2[0], torch.zeros(()), 1e-12), float(w2[0]))

# ------------------------------------------------------------------ unexplained / orth
U0 = Y.new_zeros(d, 0)
check("empty basis -> D_g = total variation",
      near(eovc.unexplained(Y, U0), float(eovc.variation(Y).sum()), 1e-11))
Q = eovc.orth(Y[0])
check("orth is orthonormal", near(Q.T @ Q, torch.eye(Q.shape[1]), 1e-10))
check("rank of one group <= 4", Q.shape[1] <= 4, Q.shape)
check("selected group is fully explained by its own span",
      near(eovc.unexplained(Y[0:1], Q), torch.zeros(()), 1e-10), eovc.unexplained(Y[0:1], Q))
full = eovc.orth(Y.transpose(0, 1).reshape(d, -1))
check("full span explains everything", eovc.unexplained(Y, full) < 1e-9,
      eovc.unexplained(Y, full))
check("adding a basis never raises D_g",
      eovc.unexplained(Y, Q) <= eovc.unexplained(Y, U0) + 1e-12)

# ------------------------------------------------------------------ gain formula (§5)
A = Y.transpose(0, 1).reshape(d, -1)
g_impl = eovc._gains_res(Y, 1e-9)
# reference: D_g(empty) - D_g({j}) with an explicit orth basis per candidate
g_ref = torch.tensor([eovc.unexplained(Y, U0) - eovc.unexplained(Y, eovc.orth(Y[j], 1e-9))
                      for j in range(n)])
check("gain == D_g(empty) - D_g({j})", near(g_impl, g_ref, 1e-8),
      (g_impl - g_ref).abs().max())
# with a non-empty basis
U1 = eovc.orth(Y[2], 1e-9)
Yr1 = Y - torch.einsum('dk,nke->nde', U1, torch.einsum('dk,nde->nke', U1, Y))
g_impl2 = eovc._gains_res(Yr1, 1e-9)
g_ref2 = torch.tensor([
    eovc.unexplained(Y, U1) - eovc.unexplained(
        Y, eovc.orth(torch.cat([U1, Y[j] - U1 @ (U1.T @ Y[j])], dim=1), 1e-9))
    for j in range(n)])
check("gain with a basis == reference", near(g_impl2, g_ref2, 1e-8),
      (g_impl2 - g_ref2).abs().max())
check("already-selected group has ~zero gain", float(g_impl2[2]) < 1e-9, float(g_impl2[2]))
check("gains non-negative", bool((g_impl2 >= -1e-12).all()))

# ------------------------------------------------------------------ greedy (§4-§5)
counts = [5, 4, 3]
# d must exceed the 4*n columns of a support, otherwise the span reaches full rank after a
# couple of picks and the remaining budget is filled by the saturation path (real d is 1024).
d = 64
gp3 = torch.cat([torch.randn(5, 4, d) * 1.5,
                 torch.randn(1, 1, d).repeat(4, 4, 1) + 1e-6 * torch.randn(4, 4, d),  # near-1D
                 torch.randn(3, 4, d) * 1.5])
info = eovc.select(gp3, counts, budget=7, rtol=1e-9)
keep = info["keep"]
check("budget respected", int(keep.sum()) == 7, int(keep.sum()))
check("one per support guaranteed", all(c >= 1 for c in info["per_support"]),
      info["per_support"])
check("per_support sums to budget", sum(info["per_support"]) == 7)
check("D decreases", info["D_end"] < info["D0"])
check("explained in [0, 1]", 0.0 <= info["explained_frac"] <= 1.0 + 1e-9,
      info["explained_frac"])
# the first pick of support 0 must be the argmax of its own single-group reduction (§5)
Y0 = eovc.support_matrices(gp3[:5])
A0 = Y0.transpose(0, 1).reshape(d, -1)
best0 = int(torch.argmax(eovc._gains_res(Y0, 1e-9)))
check("first pick of support 0 == argmax single-group gain",
      int(keep[:5].nonzero()[0]) == best0 or info["picks"][0][1] == best0,
      (info["picks"][0], best0))
# the near-degenerate support should get only its mandatory token at a tight budget
check("near-1D support gets the minimum while real gains remain",
      info["per_support"][1] == 1, (info["per_support"], info["saturated_at"]))
# global greedy gains are non-increasing after the feasibility phase
tail = [p[2] for p in info["picks"][len(counts):]]
check("global greedy gains non-increasing",
      all(tail[i] >= tail[i + 1] - 1e-8 for i in range(len(tail) - 1)), tail)

# budget = number of supports -> exactly the feasibility picks
info_min = eovc.select(gp3, counts, budget=3, rtol=1e-9)
check("budget == G gives one per support", info_min["per_support"] == [1, 1, 1],
      info_min["per_support"])
# full budget explains (almost) everything
info_full = eovc.select(gp3, counts, budget=12, rtol=1e-9)
check("full budget keeps all", int(info_full["keep"].sum()) == 12)
check("full budget explains ~all", info_full["D_end"] < 1e-6 * max(info_full["D0"], 1.0),
      (info_full["D_end"], info_full["D0"]))
# budget over N clamps
info_over = eovc.select(gp3, counts, budget=99, rtol=1e-9)
check("budget > N clamps", int(info_over["keep"].sum()) == 12)
# a support of identical groups saturates after one pick -> the tie path fires
gp4 = torch.randn(1, 4, d).repeat(6, 1, 1)
info_sat = eovc.select(gp4, [6], budget=4, rtol=1e-9)
check("identical groups: saturation recorded", info_sat["saturated_at"] is not None,
      info_sat["saturated_at"])
check("identical groups: budget still filled", int(info_sat["keep"].sum()) == 4)
check("summary text", "explained=" in eovc.summary(info) and "sd=" in eovc.summary(info))

print(f"EOVC tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
