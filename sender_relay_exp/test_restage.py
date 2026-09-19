"""Model-free unit tests for memory/restage.py (delta-RoPE re-rotation).

Run: PYTHONPATH=. python sender_relay_exp/test_restage.py
"""
import sys

import torch

from memory.restage import delta_cos_sin, rerotate_keys, rotate_half

failures = []


def check(name, condition):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        failures.append(name)


def rope_tables(positions, dim=32, theta=10000.0):
    """Reference GPTNeoX-style rope: cos/sin [n, dim] (freqs duplicated)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq  # [n, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(keys, cos, sin):
    return keys * cos.unsqueeze(0).unsqueeze(0) + rotate_half(keys) * sin.unsqueeze(0).unsqueeze(0)


torch.manual_seed(0)
n, dim = 6, 32
raw = torch.randn(1, 4, n, dim)
pos_a = torch.tensor([3, 10, 47, 100, 512, 4096])
pos_b = torch.tensor([6000, 6001, 6002, 6003, 6004, 6005])
cos_a, sin_a = rope_tables(pos_a, dim)
cos_b, sin_b = rope_tables(pos_b, dim)
cached = apply_rope(raw, cos_a, sin_a)          # what the KV cache holds

print("delta rotation")
cos_d, sin_d = delta_cos_sin(cos_a, sin_a, cos_b, sin_b)
restaged = rerotate_keys(cached, cos_d, sin_d)
direct = apply_rope(raw, cos_b, sin_b)
check("R(b-a)·R(a)k == R(b)k", torch.allclose(restaged, direct, atol=1e-5))

cos_i, sin_i = delta_cos_sin(cos_a, sin_a, cos_a, sin_a)
identity = rerotate_keys(cached, cos_i, sin_i)
check("identity delta is a no-op", torch.allclose(identity, cached, atol=1e-6))
check("identity tables are cos=1,sin=0",
      torch.allclose(cos_i, torch.ones_like(cos_i), atol=1e-6)
      and torch.allclose(sin_i, torch.zeros_like(sin_i), atol=1e-6))

back = rerotate_keys(restaged, *delta_cos_sin(cos_b, sin_b, cos_a, sin_a))
check("round trip returns original", torch.allclose(back, cached, atol=1e-5))

print("shapes/broadcast")
batched = rerotate_keys(cached, cos_d.unsqueeze(0), sin_d.unsqueeze(0))
check("3-D cos/sin accepted", torch.allclose(batched, restaged, atol=1e-6))
check("dtype preserved (bf16)",
      rerotate_keys(cached.to(torch.bfloat16), cos_d, sin_d).dtype == torch.bfloat16)

print("relative-phase invariance (the property the attention depends on)")
# scores between a moved key and a fixed query must match a genuinely
# re-positioned key: q at position p attends to k via phase (p - pos_k).
q_pos = torch.tensor([7000] * n)
cos_q, sin_q = rope_tables(q_pos, dim)
query = apply_rope(torch.randn(1, 4, n, dim), cos_q, sin_q)
score_restaged = (query * restaged).sum(-1)
score_direct = (query * direct).sum(-1)
check("attention scores identical", torch.allclose(score_restaged, score_direct, atol=1e-4))

total = 4 + 2 + 1
print(f"\n{total - len(failures)}/{total} passed")
if failures:
    print("FAILED:", failures)
    sys.exit(1)
