"""Receiver-Conditioned Visual Paging unit tests (model-free math invariants)."""
import math
import os
import sys

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")

import torch

from memory.paging import demand_token_permutation, page_anchor_positions
from memory.restage import rerotate_keys

torch.manual_seed(0)
ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL {name}"
    ok += 1


# 1. ascending demand -> token permutation, page blocks contiguous
demands = [0.5, -1.0, 2.0]
counts = [2, 3, 1]
perm, new_counts = demand_token_permutation(demands, counts)
check("page order ascending", new_counts == [3, 2, 1])
check("token perm", perm == [2, 3, 4, 0, 1, 5])
check("perm is permutation", sorted(perm) == list(range(6)))

# 2. tie/degenerate: single page returns identity-ish handling upstream
perm1, counts1 = demand_token_permutation([1.0], [4])
check("single page identity", perm1 == [0, 1, 2, 3] and counts1 == [4])

# 3. per-page rigid anchors (MROPE=1): geometry preserved, pages strictly later
os.environ["VLMAS_KV_RESTAGE_MROPE"] = "1"
old = torch.tensor([
    [5, 5, 5, 9, 9],     # t
    [2, 3, 4, 0, 1],     # h
    [7, 7, 7, 3, 4],     # w
], dtype=torch.long)
pages = [3, 2]
new, cursor = page_anchor_positions(old, pages, 100)
check("within-page geometry", torch.equal(
    new[:, 1] - new[:, 0], old[:, 1] - old[:, 0]))
check("page1 min at cursor", int(new[:, :3].min()) == 100)
p1_max = int(new[:, :3].max())
check("page2 after page1", int(new[:, 3:].min()) == p1_max + 1)
check("cursor = global max + 1", cursor == int(new.max()) + 1)
check("span compressed", cursor < 100 + int(old.max()) + 1)

# 4. MROPE off: sequential 1-D over permuted token order
os.environ["VLMAS_KV_RESTAGE_MROPE"] = ""
new0, cursor0 = page_anchor_positions(old, pages, 100)
check("1-D sequential", torch.equal(
    new0[0], torch.arange(100, 105)) and cursor0 == 105)
check("1-D three axes equal", torch.equal(new0[0], new0[1]))

# 5. inverse rotation identity: un-rotate(rotate(k)) == k
H, n, D = 2, 6, 16
k = torch.randn(1, H, n, D)
theta_half = torch.rand(n, D // 2) * 3.0
theta = torch.cat([theta_half, theta_half], dim=-1)  # RoPE half-duplicated table
cos, sin = torch.cos(theta), torch.sin(theta)
rotated = rerotate_keys(k, cos, sin)
restored = rerotate_keys(rotated, cos, -sin)
check("inverse rotation", torch.allclose(restored, k, atol=1e-5))

# 6. log-mean-exp property: one strong key beats a larger uniform page
q = torch.zeros(D)
q[0] = 1.0
strong = torch.zeros(2, D)
strong[0, 0] = 10.0
uniform = torch.zeros(50, D)
def lme(page):
    logits = page @ q / math.sqrt(D)
    return float(torch.logsumexp(logits, 0) - math.log(page.shape[0]))
check("strong small page wins", lme(strong) > lme(uniform))
check("count alone does not win", abs(lme(uniform) - lme(uniform[:5])) < 1e-6)

print(f"PAGING UNIT PASS: {ok}/{ok} invariants")
