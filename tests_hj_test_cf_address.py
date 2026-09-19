"""Counterfactual Dual-Address unit tests (model-free math invariants)."""
import sys

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")

import torch

from memory.cf_address import _js_divergence, _page_spans, _receiver_coords
from memory.restage import delta_cos_sin, rerotate_keys

torch.manual_seed(0)
ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL {name}"
    ok += 1


# 1. page spans contiguous and cover the token axis
check("spans", _page_spans([3, 2, 1]) == [(0, 3), (3, 5), (5, 6)])

# 2. JS divergence: identical dists -> 0, disjoint -> ln2
p = torch.tensor([0.25, 0.25, 0.25, 0.25])
check("js self zero", abs(_js_divergence(p, p)) < 1e-9)
a = torch.tensor([1.0, 0.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0, 0.0])
import math
check("js disjoint = ln2", abs(_js_divergence(a, b) - math.log(2)) < 1e-6)
check("js symmetric", abs(_js_divergence(a, b) - _js_divergence(b, a)) < 1e-9)

# 3. receiver coords: rigid translation, geometry preserved, min at anchor
pos = torch.tensor([[5, 5, 5], [2, 3, 4], [7, 8, 9]], dtype=torch.long).T  # [3,3]
recv = _receiver_coords(pos, 100)
check("min at anchor", int(recv.min()) == 100)
check("geometry preserved", torch.equal(recv[:, 1] - recv[:, 0],
                                        pos[:, 1] - pos[:, 0]))
check("all axes shifted equally within token", torch.equal(
    recv - recv.min(dim=1, keepdim=True).values,
    pos - pos.min(dim=1, keepdim=True).values))

# 4. native address is identity: delta(pos->pos) leaves keys unchanged
H, n, D = 2, 4, 16
k = torch.randn(1, H, n, D)
theta_half = torch.rand(n, D // 2) * 3.0
theta = torch.cat([theta_half, theta_half], dim=-1)
cos, sin = torch.cos(theta), torch.sin(theta)
cos_d, sin_d = delta_cos_sin(cos, sin, cos, sin)  # native->native
k_native = rerotate_keys(k, cos_d, sin_d)
check("native identity", torch.allclose(k_native, k, atol=1e-5))

# 5. routing rule is a pure comparison (receiver wins iff U_R > U_N)
def route(u_N, u_R):
    return "R" if u_R > u_N else "N"
check("route to receiver", route(0.1, 0.3) == "R")
check("route stays native", route(0.3, 0.1) == "N")
check("tie stays native", route(0.2, 0.2) == "N")

print(f"CF-ADDRESS UNIT PASS: {ok}/{ok} invariants")
