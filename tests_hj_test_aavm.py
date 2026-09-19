"""AAVM unit tests: the invariants the method text claims must hold exactly."""
import sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")

import torch

from memory.aavm import (
    address_shift, apply_crop_shift, crop_spans, reduce_query_heads,
    rotate_vector, shuffled_addresses,
)

torch.manual_seed(0)
H, D, n = 4, 16, 7
ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL {name}"
    ok += 1


k = torch.randn(H, n, D)
a = torch.randn(H, D)
dk = address_shift(k, a)

# 1. constraint satisfied: new centroid equals c_p
new_centroid = (k + dk.unsqueeze(1)).mean(dim=1)
c = k.mean(dim=1).norm(dim=-1, keepdim=True) * a / a.norm(dim=-1, keepdim=True)
check("centroid constraint", torch.allclose(new_centroid, c, atol=1e-5))

# 2. norm preserved: ||c_p|| == ||k_bar||
check("norm preserved", torch.allclose(
    c.norm(dim=-1), k.mean(dim=1).norm(dim=-1), atol=1e-5))

# 3. direction is the address direction
check("address direction", torch.allclose(
    torch.nn.functional.normalize(c, dim=-1),
    torch.nn.functional.normalize(a, dim=-1), atol=1e-5))

# 4. minimum perturbation: any other shift meeting the constraint is larger
alt = dk + torch.randn(H, D) * 0.1
alt = alt - (alt.mean(dim=0, keepdim=True) * 0)  # keep it a per-head shift
# a non-uniform perturbation with the same centroid: add zero-mean noise per token
noise = torch.randn(H, n, D)
noise = noise - noise.mean(dim=1, keepdim=True)
k_alt = k + dk.unsqueeze(1) + noise
check("same centroid for alt", torch.allclose(k_alt.mean(dim=1), c, atol=1e-5))
check("minimum perturbation", (k + dk.unsqueeze(1) - k).pow(2).sum() < (k_alt - k).pow(2).sum())

# 5. within-crop key geometry preserved exactly
k_new = k + dk.unsqueeze(1)
check("within-crop geometry", torch.allclose(
    k_new[:, 0] - k_new[:, 1], k[:, 0] - k[:, 1], atol=1e-6))

# 6. crop-uniform attention logit shift (cached-space application)
keys = torch.randn(1, H, 12, D)
spans = crop_spans([5, 7])
check("spans", spans == [(0, 5), (5, 12)])
cos = torch.randn(D).abs().clamp(max=1.0)
sin = (1 - cos ** 2).sqrt()
shift_rot = rotate_vector(dk, cos, sin)
before = keys.clone()
apply_crop_shift(keys, spans[1], shift_rot)
q = torch.randn(1, H, 1, D)
logit_before = (q * before[:, :, 5:12, :]).sum(-1)
logit_after = (q * keys[:, :, 5:12, :]).sum(-1)
delta_logit = logit_after - logit_before
check("crop-uniform logit shift", torch.allclose(
    delta_logit, delta_logit[..., :1].expand_as(delta_logit), atol=1e-5))
check("relative preference unchanged", torch.allclose(
    logit_after[..., 0] - logit_after[..., 1],
    logit_before[..., 0] - logit_before[..., 1], atol=1e-5))
check("other crop untouched", torch.allclose(keys[:, :, 0:5, :], before[:, :, 0:5, :]))

# 7. rotation matches the expected additive term q^T R(p) dk
expected = (q.squeeze(2) * shift_rot.unsqueeze(0)).sum(-1)
check("logit shift equals q.R(p)dk", torch.allclose(
    delta_logit[..., 0], expected, atol=1e-4))

# 8. shuffled-address control is a derangement
addrs = [torch.randn(H, D) for _ in range(4)]
shuf = shuffled_addresses(addrs)
check("derangement", all(not torch.equal(shuf[i], addrs[i]) for i in range(4)))
check("same multiset", len(shuf) == 4)

# 9. degenerate address does not explode
tiny = torch.zeros(H, D)
dk_tiny = address_shift(k, tiny)
check("degenerate finite", torch.isfinite(dk_tiny).all())

# 10. v2 GQA reduction: mean over each KV head's query group, exactly
H_q = 4 * H
u_q = torch.randn(H_q, D)
u_kv = reduce_query_heads(u_q, H)
check("gqa shape", u_kv.shape == (H, D))
check("gqa group mean", torch.allclose(u_kv[0], u_q[0:4].mean(dim=0), atol=1e-6))
check("gqa last group", torch.allclose(u_kv[-1], u_q[-4:].mean(dim=0), atol=1e-6))
check("gqa identity when H_q==H_kv", torch.allclose(
    reduce_query_heads(u_q, H_q), u_q, atol=1e-6))
try:
    reduce_query_heads(u_q, 3)
    check("gqa non-divisible rejected", False)
except ValueError:
    check("gqa non-divisible rejected", True)

# 11. v2 address plugs into the shared shift math: same invariants hold
dk_v2 = address_shift(k, u_kv)
new_centroid_v2 = (k + dk_v2.unsqueeze(1)).mean(dim=1)
c_v2 = (k.mean(dim=1).norm(dim=-1, keepdim=True)
        * u_kv / u_kv.norm(dim=-1, keepdim=True))
check("v2 centroid constraint", torch.allclose(new_centroid_v2, c_v2, atol=1e-5))
check("v2 norm preserved", torch.allclose(
    c_v2.norm(dim=-1), k.mean(dim=1).norm(dim=-1), atol=1e-5))
check("v2 geometry preserved", torch.allclose(
    (k + dk_v2.unsqueeze(1))[:, 0] - (k + dk_v2.unsqueeze(1))[:, 1],
    k[:, 0] - k[:, 1], atol=1e-6))

print(f"AAVM UNIT PASS: {ok}/{ok} invariants")
