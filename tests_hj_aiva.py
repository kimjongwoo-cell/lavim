"""AIVA: the bias must reproduce pi_g * beta_{j|g} exactly."""
import sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
import torch
from memory.aiva import group_logit_bias

torch.manual_seed(0)
ok = 0
def check(name, cond):
    global ok
    assert cond, f"FAIL {name}"
    ok += 1

kv = 12
vis = torch.tensor([2, 3, 4, 5, 6, 7, 8])      # |V| = 7
past_len, q = kv - 1, 1                         # one query at the tail
scores = torch.randn(1, 1, q, kv)

bias = group_logit_bias(vis, past_len=past_len, query_len=q, kv_len=kv,
                        device=scores.device, dtype=torch.float32)
alpha = torch.softmax(scores + bias, dim=-1)[0, 0, 0]

# reference: explicit pi_g * beta_{j|g}
s = scores[0, 0, 0]
vmask = torch.zeros(kv, dtype=torch.bool); vmask[vis] = True
t = past_len
visible = torch.arange(kv) <= t
V = vmask & visible
C = (~vmask) & visible
zbar_v = torch.exp(s[V]).mean()
zbar_c = torch.exp(s[C]).mean()
pi_v = zbar_v / (zbar_v + zbar_c)
pi_c = zbar_c / (zbar_v + zbar_c)
ref = torch.zeros(kv)
ref[V] = pi_v * torch.softmax(s[V], dim=-1)
ref[C] = pi_c * torch.softmax(s[C], dim=-1)

check("matches pi_g * beta", torch.allclose(alpha, ref, atol=1e-6))
check("sums to 1", abs(float(alpha.sum()) - 1.0) < 1e-6)
check("group mass = pi_V", abs(float(alpha[V].sum()) - float(pi_v)) < 1e-6)

# within-group ordering untouched
order_before = torch.argsort(s[V], descending=True)
order_after = torch.argsort(alpha[V], descending=True)
check("within-group order preserved", torch.equal(order_before, order_after))

# append-invariance: duplicating the visual block must not change pi_V
kv2 = kv + 7
vis2 = torch.cat([vis, vis + 7])
s2 = torch.cat([s[:9], s[2:9], s[9:]])          # same scores, doubled visual
bias2 = group_logit_bias(vis2, past_len=kv2 - 1, query_len=1, kv_len=kv2,
                         device=s.device, dtype=torch.float32)
alpha2 = torch.softmax(s2.view(1, 1, 1, kv2) + bias2, dim=-1)[0, 0, 0]
vmask2 = torch.zeros(kv2, dtype=torch.bool); vmask2[vis2] = True
check("append-invariant group mass",
      abs(float(alpha2[vmask2].sum()) - float(alpha[V].sum())) < 1e-5)

# vanilla WOULD have grown: same doubling with no bias
plain = torch.softmax(s2, dim=-1)
plain_one = torch.softmax(s, dim=-1)
check("vanilla is not append-invariant",
      float(plain[vmask2].sum()) - float(plain_one[V].sum()) > 0.05)

# inverse control flips the sign
import os
os.environ["VLMAS_AIVA_CONTROL"] = "inverse"
bias_inv = group_logit_bias(vis, past_len=past_len, query_len=q, kv_len=kv,
                            device=s.device, dtype=torch.float32)
check("inverse control flips", torch.allclose(bias_inv, -bias, atol=1e-6))
del os.environ["VLMAS_AIVA_CONTROL"]

# multi-row prefill: each row uses its own causally visible counts
bias_rows = group_logit_bias(vis, past_len=0, query_len=kv, kv_len=kv,
                             device=s.device, dtype=torch.float32)[0, 0]
row_vals = [float(bias_rows[r, vis[0]]) for r in range(kv)]
check("row-dependent bias", len(set(round(v, 6) for v in row_vals)) > 1)
check("non-visual columns unbiased", float(bias_rows[:, 0].abs().max()) == 0.0)

print(f"AIVA UNIT PASS: {ok}/{ok} — bias == pi_g*beta, append-invariant, "
      f"within-group order intact")
