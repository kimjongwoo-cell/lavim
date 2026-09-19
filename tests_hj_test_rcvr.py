"""Unit tests for memory/rcvr_diag.py (Receiver-Conditioned Visual KV Re-Read probe).

Pure-function level only: the attention split, the non-visual contribution, the
reallocation ceiling and the query gap. No model, no GPU.

run: python3 tests_hj_test_rcvr.py
"""
import sys

import torch

from memory.rcvr_diag import (
    context_contribution,
    cos_gap,
    realloc_ceiling,
    visual_read,
    visual_split,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"  ok   {name}" if cond else f"  FAIL {name} {detail}")
    if not cond:
        FAIL.append(name)


f64 = torch.float64
alpha = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=f64)
mask = torch.tensor([True, False, True, False])

print("visual_split")
m_v, m_c = visual_split(alpha, mask)
check("visual mass sums the masked columns", abs(m_v - 0.4) < 1e-12, f"got {m_v}")
check("context mass sums the rest", abs(m_c - 0.6) < 1e-12, f"got {m_c}")
check("the two masses recover the total", abs((m_v + m_c) - 1.0) < 1e-12)
mv0, mc0 = visual_split(alpha, torch.zeros(4, dtype=torch.bool))
check("no visual column gives zero visual mass", abs(mv0) < 1e-12 and abs(mc0 - 1.0) < 1e-12)

print("context_contribution")
values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=f64)
rc = context_contribution(alpha, values, mask)
# non-visual rows are index 1 (0.2 * [0,1]) and 3 (0.4 * [0,2]) -> [0, 1.0]
check("it sums only the non-visual rows, alpha-weighted",
      torch.allclose(rc, torch.tensor([0.0, 1.0], dtype=f64)), f"got {rc.tolist()}")
check("masking everything as visual gives a zero contribution",
      torch.allclose(context_contribution(alpha, values, torch.ones(4, dtype=torch.bool)),
                     torch.zeros(2, dtype=f64)))
check("this is m_C * c, i.e. NOT renormalised",
      abs(float(rc.norm()) - 1.0) < 1e-12, "should be 0.2*1 + 0.4*2 = 1.0 on axis 1")

print("visual_read")
beta, v, m_v2 = visual_read(alpha, values, mask)
check("beta renormalises within the visual columns only", abs(float(beta.sum()) - 1.0) < 1e-12)
check("beta is proportional to the visual alphas",
      torch.allclose(beta, torch.tensor([0.25, 0.75], dtype=f64)), f"got {beta.tolist()}")
check("v is the beta-weighted mean of the visual values",
      torch.allclose(v, torch.tensor([0.25 * 1.0 + 0.75 * 2.0, 0.0], dtype=f64)), f"got {v.tolist()}")
check("visual_read reports the same mass as visual_split", abs(m_v2 - m_v) < 1e-12)

print("realloc_ceiling")
same = torch.tensor([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], dtype=f64)
b3 = torch.tensor([0.2, 0.3, 0.5], dtype=f64)
shift, vn = realloc_ceiling(b3, same)
check("identical visual values leave NO room for any re-weighting", abs(shift) < 1e-12,
      f"got {shift}")
check("the reported |v| is the weighted mean's norm", abs(vn - 5.0) < 1e-12, f"got {vn}")

ortho = torch.eye(2, dtype=f64)
shift2, vn2 = realloc_ceiling(torch.tensor([0.5, 0.5], dtype=f64), ortho)
check("two orthonormal rows at equal weight give a 0.7071 ceiling",
      abs(shift2 - (0.5 ** 0.5)) < 1e-12, f"got {shift2}")

shift3, _ = realloc_ceiling(torch.tensor([1.0, 0.0], dtype=f64), ortho)
check("all mass on one row -> ceiling is the distance to the farthest row",
      abs(shift3 - (2 ** 0.5)) < 1e-12, f"got {shift3}")
check("the ceiling is an upper bound on an actual re-weighting",
      shift2 + 1e-12 >= float((torch.tensor([0.9, 0.1], dtype=f64) @ ortho
                               - torch.tensor([0.5, 0.5], dtype=f64) @ ortho).norm()))

print("cos_gap")
q = torch.tensor([1.0, 2.0, 3.0], dtype=f64)
check("an unchanged query has zero gap", abs(cos_gap(q, q.clone())) < 1e-12, f"got {cos_gap(q, q)}")
check("rescaling alone is not a direction change",
      abs(cos_gap(q, 7.5 * q)) < 1e-12, f"got {cos_gap(q, 7.5 * q)}")
check("an orthogonal query has gap 1",
      abs(cos_gap(torch.tensor([1.0, 0.0], dtype=f64), torch.tensor([0.0, 1.0], dtype=f64)) - 1.0) < 1e-12)
check("an opposed query has gap 2",
      abs(cos_gap(q, -q) - 2.0) < 1e-12, f"got {cos_gap(q, -q)}")
check("a zero query gives 0, not nan", cos_gap(q, torch.zeros(3, dtype=f64)) == 0.0)

print("RoPE invariance (why this probe needs no rotary embedding)")
theta = 0.7
rot = torch.tensor([[torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
                    [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))]], dtype=f64)
a = torch.tensor([0.3, -1.2], dtype=f64)
b = torch.tensor([1.1, 0.4], dtype=f64)
check("an orthogonal rotation applied to BOTH queries preserves the gap",
      abs(cos_gap(a, b) - cos_gap(rot @ a, rot @ b)) < 1e-12,
      f"{cos_gap(a, b)} vs {cos_gap(rot @ a, rot @ b)}")

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1)
print("all tests passed")
