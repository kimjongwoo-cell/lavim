"""Unit tests for memory/qasc_select.py (C1 #14). Run: python tests_hj_test_qasc.py"""
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory import qasc_select as Q  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


torch.manual_seed(0)


def brute_greedy(b, budget):
    N, d = b.shape
    S = []
    for _ in range(budget):
        best, bi = -1e18, None
        for i in range(N):
            if i in S:
                continue
            M = torch.eye(d, dtype=torch.float64) + b[S + [i]].T @ b[S + [i]]
            v = float(torch.logdet(M))
            if v > best + 1e-12:
                best, bi = v, i
        S.append(bi)
    return S


for trial in range(4):
    N, d, B = 14, 5, 6
    z = F.normalize(torch.randn(N, d, dtype=torch.float64), dim=-1)
    a = torch.rand(N, dtype=torch.float64) * (3.0 if trial % 2 else 0.05)
    b = a.sqrt().unsqueeze(1) * z
    order, gains = Q.logdet_greedy(b, B)
    ref = brute_greedy(b, B)
    check(f"greedy == brute force (trial {trial})", order.tolist() == ref, f"{order.tolist()} vs {ref}")
    total = float(torch.logdet(torch.eye(d, dtype=torch.float64) + b[order].T @ b[order]))
    check(f"sum of gains == logdet (trial {trial})", abs(sum(gains) - total) < 1e-8, f"{sum(gains)} vs {total}")
    S = order.tolist()
    M = torch.eye(d, dtype=torch.float64) + b[S[:3]].T @ b[S[:3]]
    j = S[3]
    formula = math.log(1 + float(a[j]) * float(z[j] @ torch.linalg.solve(M, z[j])))
    check(f"gain formula log(1 + a z^T M^-1 z) (trial {trial})", abs(gains[3] - formula) < 1e-9, f"{gains[3]} vs {formula}")

# budget edge cases
o, g = Q.logdet_greedy(torch.zeros(5, 3, dtype=torch.float64), 3)
check("zero salience still fills budget", len(o) == 3 and all(abs(x) < 1e-12 for x in g))
o, g = Q.logdet_greedy(torch.randn(4, 3, dtype=torch.float64), 10)
check("budget capped at N", len(o) == 4 and len(set(o.tolist())) == 4)

# duplicates: a repeated direction loses to a new direction of equal salience
z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
o, _ = Q.logdet_greedy(z, 2)
check("coverage prefers new direction", sorted(o.tolist()) in ([0, 2], [1, 2]), str(o.tolist()))

# salience / merge
q = torch.randn(2, 8, 4)
k = torch.randn(2, 8, 4)
ra = Q.received_attention(q, k, 0.5)
check("received attention sums to 1", abs(float(ra.sum()) - 1.0) < 1e-5)
check("merged mean", Q.merged_mean(torch.arange(8.0), 4).tolist() == [1.5, 5.5])

# token centres
c = Q.token_centers((100, 200, 40, 20), 2, 4)
check("token centres raster", c[0].tolist() == [105.0, 205.0] and c[4].tolist() == [105.0, 215.0] and c[7].tolist() == [135.0, 215.0])

# cross-scale factor: x20 crop (1) inside x5 crop (0) left half; x20 crop (2) outside
z5 = F.normalize(torch.randn(16, 6), dim=-1)          # x5 grid 4x4 over box (0,0,400,400)
z20a = z5[[0, 1, 4, 5]].clone()                         # x20 2x2 over box (0,0,200,200): centres fall in x5 cells 0,1,4,5
z20b = F.normalize(torch.randn(4, 6), dim=-1)           # x20 2x2 outside
zall = torch.cat([z5, z20a, z20b])
n, related = Q.cross_scale_factor(zall, [16, 4, 4], [(4, 4), (2, 2), (2, 2)], (5, 20, 20),
                                  ((0, 0, 400, 400), (0, 0, 200, 200), (1000, 1000, 200, 200)))
check("cross related count", related == 4, str(related))
check("identical feature -> n = 0", torch.allclose(n[16:20], torch.zeros(4), atol=1e-6), str(n[16:20]))
check("no relation -> n = 1", torch.allclose(n[20:24], torch.ones(4)) and torch.allclose(n[:16], torch.ones(16)))
zall2 = torch.cat([z5, -z20a, z20b])
n2, _ = Q.cross_scale_factor(zall2, [16, 4, 4], [(4, 4), (2, 2), (2, 2)], (5, 20, 20),
                             ((0, 0, 400, 400), (0, 0, 200, 200), (1000, 1000, 200, 200)))
check("opposite feature -> n = 1", torch.allclose(n2[16:20], torch.ones(4), atol=1e-6), str(n2[16:20]))
n3, r3 = Q.cross_scale_factor(zall, [16, 4, 4], [(4, 4), (2, 2), (2, 2)], (5, 20, 20), (None, None, None))
check("no boxes -> all ones", r3 == 0 and torch.allclose(n3, torch.ones(24)))

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
