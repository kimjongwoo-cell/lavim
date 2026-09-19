"""Model-free unit tests for the Visual Binding Step helpers.

Run: PYTHONPATH=. python sender_relay_exp/test_visual_bind.py
"""
import sys

import torch

from vision_text_mas.visual_bind import (
    build_bind_mask,
    matched_noise_like,
    parse_bind_layers,
    shuffle_permutation,
)

failures = []


def check(name, condition):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        failures.append(name)


print("parse_bind_layers")
check("single int", parse_bind_layers("18", 36) == [18])
check("comma list sorted+dedup", parse_bind_layers("22,14,18,18", 36) == [14, 18, 22])
check("fraction f0.5 of 36", parse_bind_layers("f0.5", 36) == [round(0.5 * 35)])
check("fraction f0.8 of 36", parse_bind_layers("f0.8", 36) == [28])
check("mixed int+fraction", parse_bind_layers("8,f1.0", 36) == [8, 35])
check("empty -> middle", parse_bind_layers("", 36) == [18])
check("garbage -> middle", parse_bind_layers("abc,,", 36) == [18])
check("out of range dropped", parse_bind_layers("40,-1,10", 36) == [10])
check("all invalid -> middle", parse_bind_layers("99", 36) == [18])
check("'all' -> every layer", parse_bind_layers("all", 36) == list(range(36)))
check("'ALL' case-insensitive", parse_bind_layers(" ALL ", 4) == [0, 1, 2, 3])

print("build_bind_mask")
vis = torch.tensor([3, 5, 6], dtype=torch.long)
mask = build_bind_mask(10, vis, 9, torch.float32, "cpu")
check("shape [1,1,1,kv]", tuple(mask.shape) == (1, 1, 1, 10))
allowed = (mask[0, 0, 0] == 0).nonzero().flatten().tolist()
check("zeros exactly at vis+self", allowed == [3, 5, 6, 9])
blocked = mask[0, 0, 0, [0, 1, 2, 4, 7, 8]]
check("blocked at dtype min", bool((blocked == torch.finfo(torch.float32).min).all()))
mask16 = build_bind_mask(4, torch.tensor([1]), 3, torch.bfloat16, "cpu")
check("bf16 dtype respected", mask16.dtype == torch.bfloat16)
mask_ns = build_bind_mask(10, vis, None, torch.float32, "cpu")
ns_allowed = (mask_ns[0, 0, 0] == 0).nonzero().flatten().tolist()
check("self_col=None -> R*-only read", ns_allowed == [3, 5, 6])

print("shuffle_permutation")
p1 = shuffle_permutation(8, 42)
p2 = shuffle_permutation(8, 42)
check("deterministic", bool((p1 == p2).all()))
check("is permutation", sorted(p1.tolist()) == list(range(8)))
check("not identity (n=8, seed=42)", p1.tolist() != list(range(8)))

print("matched_noise_like")
values = torch.randn(1, 4, 16, 32) * 3 + 1.5
noise1 = matched_noise_like(values, 7)
noise2 = matched_noise_like(values, 7)
check("deterministic", bool((noise1 == noise2).all()))
check("shape preserved", noise1.shape == values.shape)
check("mean matched (±0.2)", abs(noise1.float().mean() - values.float().mean()) < 0.2)
check("std matched (±0.2)", abs(noise1.float().std() - values.float().std()) < 0.2)
check("content destroyed", not torch.allclose(noise1, values))
bf16 = values.to(torch.bfloat16)
check("dtype follows input", matched_noise_like(bf16, 1).dtype == torch.bfloat16)

total = 11 + 5 + 3 + 6
print(f"\n{total - len(failures)}/{total} passed")
if failures:
    print("FAILED:", failures)
    sys.exit(1)
