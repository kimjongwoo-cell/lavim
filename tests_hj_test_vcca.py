"""Unit tests for memory/vcca_diag.py (Visual-to-Candidate Contrast Alignment).

Pure-function level only: trie construction, contrast subspace, and the three
ratios. Nothing here needs a model or a GPU.

run: python3 tests_hj_test_vcca.py
"""
import sys

import torch

from memory.vcca_diag import (
    accessible_fraction,
    branching_nodes,
    contrast_basis,
    disc_ratio,
    seq_ratio,
)

FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAIL.append(name)


print("branching_nodes")

nodes = branching_nodes([[1, 2, 3], [1, 2, 4], [1, 5]])
check("finds exactly the two branching prefixes", [n["prefix"] for n in nodes] == [(1,), (1, 2)],
      f"got {[n['prefix'] for n in nodes]}")
check("root is not a node when all candidates share the first token",
      () not in [n["prefix"] for n in nodes])
n0 = nodes[0]
check("node (1,) sees all three candidates", n0["members"] == [0, 1, 2], f"got {n0['members']}")
check("node (1,) distinct next tokens are {2,5}", n0["distinct"] == [2, 5], f"got {n0['distinct']}")
check("node (1,) next map is per candidate", n0["next"] == {0: 2, 1: 2, 2: 5}, f"got {n0['next']}")
n1 = nodes[1]
check("node (1,2) sees only the two candidates through it", n1["members"] == [0, 1])
check("node (1,2) distinct next tokens are {3,4}", n1["distinct"] == [3, 4])

check("identical candidates yield no node", branching_nodes([[7, 8], [7, 8]]) == [])
check("single candidate yields no node", branching_nodes([[7, 8]]) == [])

pref = branching_nodes([[1, 2], [1, 2, 3]])
check("a candidate that merely ends early is not a branch (k=1)", pref == [], f"got {pref}")

ended = branching_nodes([[1, 2], [1, 3], [1]])
check("ended candidate is recorded, not counted as a next token",
      len(ended) == 1 and ended[0]["ended"] == [2] and ended[0]["distinct"] == [2, 3],
      f"got {ended}")

deep = branching_nodes([[1, 1, 1], [1, 1, 2], [1, 3], [4]])
check("nodes are ordered by increasing prefix length",
      [len(n["prefix"]) for n in deep] == sorted(len(n["prefix"]) for n in deep),
      f"got {[n['prefix'] for n in deep]}")
check("root branches when candidates differ at token 0",
      deep[0]["prefix"] == () and deep[0]["distinct"] == [1, 4])

print("contrast_basis")

D = 6
rows2 = torch.zeros(2, D, dtype=torch.float64)
rows2[0, 0] = 1.0
rows2[1, 1] = 1.0
U2 = contrast_basis(rows2)
check("two candidates give a rank-1 contrast subspace", U2.shape == (D, 1), f"got {U2.shape}")
diff = (rows2[0] - rows2[1]).to(torch.float64)
cos = torch.abs(U2[:, 0] @ diff / diff.norm())
check("the basis vector spans w1-w2", bool(torch.isclose(cos, torch.tensor(1.0, dtype=torch.float64))),
      f"cos={float(cos)}")

rows3 = torch.eye(3, D, dtype=torch.float64)
U3 = contrast_basis(rows3)
check("k candidates give rank k-1 after centering", U3.shape == (D, 2), f"got {U3.shape}")
check("basis is orthonormal",
      bool(torch.allclose(U3.T @ U3, torch.eye(2, dtype=torch.float64), atol=1e-10)))

dup = torch.zeros(3, D, dtype=torch.float64)
dup[0, 0] = 1.0
dup[1, 0] = 1.0          # duplicate row -> rank drops
dup[2, 1] = 1.0
check("duplicate unembedding rows drop the rank", contrast_basis(dup).shape == (D, 1),
      f"got {contrast_basis(dup).shape}")

print("accessible_fraction")

U = torch.zeros(D, 2, dtype=torch.float64)
U[0, 0] = 1.0
U[1, 1] = 1.0
inside = torch.zeros(D, dtype=torch.float64)
inside[0] = 3.0
check("a delta inside the subspace has rho = 1",
      abs(accessible_fraction(U, inside) - 1.0) < 1e-12, f"got {accessible_fraction(U, inside)}")
outside = torch.zeros(D, dtype=torch.float64)
outside[4] = 5.0
check("a delta orthogonal to the subspace has rho = 0",
      abs(accessible_fraction(U, outside)) < 1e-12, f"got {accessible_fraction(U, outside)}")
mixed = torch.zeros(D, dtype=torch.float64)
mixed[0] = 1.0
mixed[4] = 1.0
check("a half-in delta has rho = 0.5", abs(accessible_fraction(U, mixed) - 0.5) < 1e-12,
      f"got {accessible_fraction(U, mixed)}")
check("a zero delta gives 0, not nan",
      accessible_fraction(U, torch.zeros(D, dtype=torch.float64)) == 0.0)

print("disc_ratio / seq_ratio")

check("a pure common shift is fully non-discriminative",
      abs(disc_ratio([2.0, 2.0, 2.0])) < 1e-12, f"got {disc_ratio([2.0, 2.0, 2.0])}")
check("a zero-mean shift is fully discriminative",
      abs(disc_ratio([1.0, -1.0]) - 1.0) < 1e-12, f"got {disc_ratio([1.0, -1.0])}")
check("a half-and-half shift lands between 0 and 1",
      0.0 < disc_ratio([2.0, 0.0]) < 1.0, f"got {disc_ratio([2.0, 0.0])}")
check("disc_ratio of all zeros is 0, not nan", disc_ratio([0.0, 0.0]) == 0.0)
check("seq_ratio uses the same centering rule",
      abs(seq_ratio([1.0, -1.0]) - 1.0) < 1e-12 and abs(seq_ratio([3.0, 3.0])) < 1e-12)

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1)
print("all tests passed")
