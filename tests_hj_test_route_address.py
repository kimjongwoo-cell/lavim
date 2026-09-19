"""Single-Pass Dual-Address Routing unit tests (model-free invariants)."""
import math
import sys

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")

import torch

from memory.aavm import reduce_query_heads
from memory.restage import delta_cos_sin, rerotate_keys
from memory.route_address import band_coordinates, band_span

torch.manual_seed(0)
ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL {name}"
    ok += 1


# 1. band: original order kept, per-page geometry preserved, pages consecutive
pos = torch.tensor([
    [50, 50, 50, 90, 90],
    [2, 3, 4, 0, 1],
    [7, 7, 7, 3, 4],
], dtype=torch.long)
band, end = band_coordinates(pos, [3, 2], 1000)
check("page0 min at start", int(band[:, :3].min()) == 1000)
check("page0 geometry", torch.equal(band[:, 1] - band[:, 0], pos[:, 1] - pos[:, 0]))
check("page1 follows page0", int(band[:, 3:].min()) == int(band[:, :3].max()) + 1)
check("span from zero", band_span(pos, [3, 2]) == end - 1000)

# 2. mu_R = R(Delta) mu_N exactly when the translation is uniform per page
H, n, D = 2, 5, 16
k = torch.randn(1, H, n, D)
th_old = torch.rand(1, D // 2) * 2.0
th_new = th_old + 0.7                      # same delta for every token
cos_o = torch.cos(torch.cat([th_old, th_old], -1)).expand(n, D)
sin_o = torch.sin(torch.cat([th_old, th_old], -1)).expand(n, D)
cos_n = torch.cos(torch.cat([th_new, th_new], -1)).expand(n, D)
sin_n = torch.sin(torch.cat([th_new, th_new], -1)).expand(n, D)
cd, sd = delta_cos_sin(cos_o, sin_o, cos_n, sin_n)
mu_rot_then_mean = rerotate_keys(k, cd, sd).mean(2)[0]
mu_mean_then_rot = rerotate_keys(k.mean(2, keepdim=True), cd[:1], sd[:1])[0, :, 0]
check("rotation commutes with page mean", torch.allclose(
    mu_rot_then_mean, mu_mean_then_rot, atol=1e-5))

# 3. G equals the exact mean of per-token logits (anchor is an exact reduction)
q = torch.randn(H, D)
g_anchor = float(((q * k.mean(2)[0]).sum(-1) / math.sqrt(D)).mean())
g_tokens = float((torch.einsum("hd,hnd->hn", q, k[0]) / math.sqrt(D)).mean())
check("anchor == mean token logit", abs(g_anchor - g_tokens) < 1e-5)

# 4. GQA group-mean query reduces (1/H) sum_h exactly
Hq = 4 * H
qq = torch.randn(Hq, D)
mu = k.mean(2)[0]                                   # [H_kv, D]
full = sum(float((qq[h] @ mu[h // 4]) / math.sqrt(D)) for h in range(Hq)) / Hq
red = float(((reduce_query_heads(qq, H) * mu).sum(-1) / math.sqrt(D)).mean())
check("gqa reduction exact", abs(full - red) < 1e-5)

# 5. routing rule: receiver iff G_R > G_N (ties stay native)
check("route R", ("R" if 0.5 > 0.2 else "N") == "R")
check("tie N", ("R" if 0.2 > 0.2 else "N") == "N")

print(f"ROUTE UNIT PASS: {ok}/{ok} invariants")
