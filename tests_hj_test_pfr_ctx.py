#!/usr/bin/env python3
"""memory/pfr_attention.py — matched-context specificity 확장(arm A record / D shuffled) 유닛.

usage: python3 tests_hj_test_pfr_ctx.py
"""
import json
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from memory import pfr_attention as pfr

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


print("== parse_layers")
check("unset -> all", pfr.parse_layers("", 36) == tuple(range(36)))
check("single 27", pfr.parse_layers("27", 36) == (27,))
check("list+range", pfr.parse_layers("30,24-26", 36) == (24, 25, 26, 30))
for bad in ("36", "-1", "a"):
    try:
        pfr.parse_layers(bad, 36)
        check(f"bad {bad!r} raises", False)
    except ValueError:
        check(f"bad {bad!r} raises", True)

print("== modes")
for m in ("1", "identity", "record", "shuffled"):
    os.environ["VLMAS_ANSWERER_PFR"] = m
    check(f"enabled {m}", pfr.enabled())
os.environ["VLMAS_ANSWERER_PFR"] = ""
check("unset disabled", not pfr.enabled())

print("== js_divergence")
torch.manual_seed(0)
p = torch.softmax(torch.randn(4, 50), -1)
q = torch.softmax(torch.randn(4, 50), -1)
check("JS(p,p)=0", float(pfr.js_divergence(p, p).abs().max()) < 1e-12)
check("symmetric", torch.allclose(pfr.js_divergence(p, q), pfr.js_divergence(q, p)))
check("bounded by ln2", float(pfr.js_divergence(p, q).max()) <= math.log(2) + 1e-9)
onehot_a = torch.zeros(1, 3); onehot_a[0, 0] = 1
onehot_b = torch.zeros(1, 3); onehot_b[0, 1] = 1
check("disjoint -> ln2", abs(float(pfr.js_divergence(onehot_a, onehot_b)) - math.log(2)) < 1e-9)

print("== fixed_derangement")
d = pfr.fixed_derangement(range(20), seed=0)
check("bijection", sorted(d.keys()) == list(range(20)) and sorted(d.values()) == list(range(20)))
check("no fixed point", all(k != v for k, v in d.items()))
check("deterministic", d == pfr.fixed_derangement(range(20), seed=0))
check("two cases swap", pfr.fixed_derangement([0, 1]) == {0: 1, 1: 0})
try:
    pfr.fixed_derangement([3])
    check("one case raises", False)
except ValueError:
    check("one case raises", True)

print("== donor_call_index (relative progress, no tail repeat)")
check("prefill k -> k", pfr.donor_call_index(0, True, 50, 1, 80) == 0)
check("prefill clamped", pfr.donor_call_index(2, True, 50, 1, 80) == 0)
check("gen start -> 0", pfr.donor_call_index(0, False, 50, 1, 80) == 0)
check("gen end -> donor end", pfr.donor_call_index(49, False, 50, 1, 80) == 79)
check("gen middle -> middle", pfr.donor_call_index(25, False, 51, 1, 101) == 50)
check("beyond target native length clamps to donor end", pfr.donor_call_index(70, False, 50, 1, 80) == 79)
idx = [pfr.donor_call_index(g, False, 10, 1, 100) for g in range(10)]
check("monotone spread over whole donor", idx[0] == 0 and idx[-1] == 99 and idx == sorted(idx), idx)
check("short donor", pfr.donor_call_index(5, False, 10, 1, 1) == 0)
check("no donor gen -> None", pfr.donor_call_index(3, False, 10, 1, 0) is None)

print("== re-read identities (synthetic attention row)")
torch.manual_seed(1)
H, L, D = 4, 40, 8
q = torch.randn(1, H, 1, D); K = torch.randn(1, H, L, D); V = torch.randn(1, H, L, D)
vmask = torch.zeros(L, dtype=torch.bool); vmask[5:25] = True
scale = 1 / math.sqrt(D)
alpha = torch.softmax(q @ K.transpose(-2, -1) * scale, -1)
rho = alpha[..., vmask].sum(-1, keepdim=True)
b_nat = alpha[..., vmask] / rho
b_id = torch.softmax(q @ K[..., vmask, :].transpose(-2, -1) * scale, -1)
check("same-query re-read == native visual conditional", float(pfr.js_divergence(b_nat, b_id).max()) < 1e-10)
m_C = (alpha * (~vmask).float()) @ V
o_nat = alpha @ V
o_id = m_C + rho * (b_id @ V[..., vmask, :])
check("identity output == native output", torch.allclose(o_nat, o_id, atol=1e-6))

print("== donor loading")
with tempfile.TemporaryDirectory() as tmp:
    for case, n_gen in ((0, 5), (1, 9)):
        calls = [{"T": 300, "kind": "prefill", "r_C": torch.full((6,), float(case))}]
        calls += [{"T": 1, "kind": "gen", "r_C": torch.full((6,), float(case) + g / 100)} for g in range(n_gen)]
        torch.save({"case": case, "layer": 27, "mode": "record", "calls": calls, "stats": []},
                   os.path.join(tmp, f"case_{case:04d}.pt"))
    perm = os.path.join(tmp, "perm.json")
    json.dump({"0": 1, "1": 0}, open(perm, "w"))
    os.environ["VLMAS_ANSWERER_PFR_DONOR_DIR"] = tmp
    os.environ["VLMAS_ANSWERER_PFR_PERM"] = perm
    dn = pfr._load_donor(0)
    check("donor case", dn["case"] == 1)
    check("donor banks", len(dn["pre"]) == 1 and len(dn["gen"]) == 9)
    check("target native gen count from own dump", dn["n_target_gen"] == 5)
    json.dump({"0": 0, "1": 1}, open(perm, "w"))
    try:
        pfr._load_donor(0)
        check("self-donor rejected", False)
    except RuntimeError:
        check("self-donor rejected", True)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
