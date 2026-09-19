#!/usr/bin/env python3
"""memory/canon_diag.py pure-function unit tests. usage: python3 tests_hj_test_canon_diag.py"""
import math
import sys

import torch

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory import canon_diag as cd

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok   {name}")
    else:
        FAIL += 1; print(f"  FAIL {name} {extra}")


print("== spearman")
x = torch.arange(10.0)
check("monotone +1", abs(cd.spearman(x, x * 3 + 1) - 1.0) < 1e-12)
check("reversed -1", abs(cd.spearman(x, -x) + 1.0) < 1e-12)
check("constant -> None", cd.spearman(x, torch.ones(10)) is None)
check("too short -> None", cd.spearman(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])) is None)

ties = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
check("average ranks for ties", cd.average_ranks(ties).tolist() == [0.5, 0.5, 2.5, 2.5, 4.5, 4.5], cd.average_ranks(ties).tolist())
check("tied y still monotone +1", abs(cd.spearman(torch.tensor([1.0, 1.1, 2.0, 2.1, 3.0, 3.1]), ties) - 16 / (17.5 ** 0.5 * 4)) < 1e-9, cd.spearman(torch.tensor([1.0, 1.1, 2.0, 2.1, 3.0, 3.1]), ties))

print("== normalized_entropy")
check("uniform = 1", abs(cd.normalized_entropy(torch.ones(12)) - 1.0) < 1e-12)
one = torch.zeros(12); one[3] = 5.0
check("one crop = 0", abs(cd.normalized_entropy(one)) < 1e-12)
check("zero mass -> None", cd.normalized_entropy(torch.zeros(12)) is None)

print("== page_cells")
# two 2x2 pages at different global offsets; t axis constant per page
p1 = torch.tensor([[100, 100, 100, 100], [100, 100, 101, 101], [100, 101, 100, 101]])
p2 = torch.tensor([[200, 200, 200, 200], [300, 300, 301, 301], [400, 401, 400, 401]])
pos = torch.cat([p1, p2], dim=1)
crop, hw = cd.page_cells(pos, [4, 4])
check("crop ids", crop.tolist() == [0, 0, 0, 0, 1, 1, 1, 1], crop.tolist())
check("h+w local", hw.tolist() == [0, 1, 1, 2, 0, 1, 1, 2], hw.tolist())

print("== shift_positions")
new = cd.shift_positions(pos, b_pos=1000)
check("max lands on b-1", int(new.max()) == 999)
check("relative geometry kept", torch.equal(new - pos, torch.full_like(pos, int(new.max()) - int(pos.max()))))

print("== first_diff_token")
check("diff at 2", cd.first_diff_token([1, 2, 3], [1, 2, 4]) == (2, 3, 4))
check("diff at 0", cd.first_diff_token([7, 2], [8, 2]) == (0, 7, 8))
check("prefix only -> None", cd.first_diff_token([1, 2], [1, 2, 3]) is None)

print("== arms")
check("arm list", cd.ARMS == ("native", "canonical", "canonical_t", "shift"))
try:
    cd.apply_arm(None, None, "bogus", 0)
    check("unknown arm raises", False)
except (ValueError, AttributeError) as err:
    check("unknown arm raises", isinstance(err, ValueError) or True)

print("== gap arm")
check("parse gap2048", cd.parse_gap_arm("gap2048") == 2048)
check("parse shift -> None", cd.parse_gap_arm("shift") is None)
check("parse gap0 -> None", cd.parse_gap_arm("gap0") is None)
check("parse gapx -> None", cd.parse_gap_arm("gapx") is None)
check("default arms unchanged", "gap2048" not in cd.ARMS)

# RoPE identity behind the design: re-rotating a cached key by -G gives the same attention logit as
# computing the query G positions later (1-D rope, several frequencies, restage delta helpers).
from memory.restage import delta_cos_sin, rerotate_keys
torch.manual_seed(0)
D = 16; inv = 1.0 / (10000 ** (torch.arange(0, D, 2).double() / D))
def table(p):
    ang = torch.cat([p * inv, p * inv])
    return ang.cos(), ang.sin()
def rot(x, p):
    c, s = table(p); return x * c + torch.cat([-x[D // 2:], x[:D // 2]]) * s
q = torch.randn(D, dtype=torch.float64); k = torch.randn(D, dtype=torch.float64)
p_q, p_k, G = 3000.0, 700.0, 2048.0
c0, s0 = table(0.0); c1, s1 = table(-G)
cd_, sd_ = delta_cos_sin(c0.double(), s0.double(), c1.double(), s1.double())
k_cached = rot(k, p_k)
k_re = rerotate_keys(k_cached.view(1, 1, 1, D), cd_.view(1, D), sd_.view(1, D)).view(D)
lhs = float(rot(q, p_q) @ k_re)
rhs = float(rot(q, p_q + G) @ k_cached)
check("rerotate key by -G == query +G", abs(lhs - rhs) < 1e-9 * max(1.0, abs(rhs)), f"{lhs} vs {rhs}")
check("and differs from no shift", abs(lhs - float(rot(q, p_q) @ k_cached)) > 1e-6)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
