"""Model-free unit tests for the sender-priority relay experiment (§9).

Run with the project venv from the tree root:
    PYTHONPATH=. <python> sender_relay_exp/test_sender_relay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.reallocate import reallocate  # noqa: E402
from vision_text_mas.relay_destination import (  # noqa: E402
    build_destination_weights,
    deterministic_shuffle_seed,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def synthetic_attention(batch=1, heads=2, queries=3, keys=12, seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, heads, queries, keys, generator=generator)
    return torch.softmax(logits, dim=-1)


def test_1_mapping() -> None:
    print("Test 1 — sender_priority mapping (score[keep] alignment)")
    scores = torch.tensor([0.1, 0.9, 0.4, 0.8])
    keep = torch.tensor([False, True, False, True])
    sender_priority = scores[keep]
    check("survivor scores", torch.equal(sender_priority, torch.tensor([0.9, 0.8])))
    # survivor order must follow original token order, exactly like
    # qwen3vl._hierarchy_pruned_vision_features's scores[keep_groups] stash
    group_ids = torch.nonzero(keep, as_tuple=True)[0]
    check("survivor original ids", group_ids.tolist() == [1, 3])


def test_2_normalization() -> None:
    print("Test 2 — normalization")
    columns = torch.tensor([5, 9, 11])
    weights, audit = build_destination_weights(
        relay_source="sender_priority",
        columns=columns,
        sender_priority=torch.tensor([2.0, 1.0, 1.0]),
        base_seed=42,
    )
    check("no fallback", not audit["fallback_used"])
    check(
        "weights [0.5, 0.25, 0.25]",
        weights is not None
        and torch.allclose(weights, torch.tensor([0.5, 0.25, 0.25]), atol=1e-6),
        detail=str(weights),
    )
    check("weights sum to 1", weights is not None and abs(float(weights.sum()) - 1.0) < 1e-6)
    # negative scores clamp to zero before normalizing
    weights2, _ = build_destination_weights(
        relay_source="sender_priority",
        columns=columns,
        sender_priority=torch.tensor([2.0, -5.0, 2.0]),
        base_seed=42,
    )
    check(
        "negatives clamped",
        weights2 is not None
        and torch.allclose(weights2, torch.tensor([0.5, 0.0, 0.5]), atol=1e-6),
    )
    # degenerate → explicit fallback, no weights
    weights3, audit3 = build_destination_weights(
        relay_source="sender_priority",
        columns=columns,
        sender_priority=torch.zeros(3),
        base_seed=42,
    )
    check("degenerate fallback", weights3 is None and audit3["fallback_used"])
    check("fallback reason recorded", audit3["fallback_reason"] == "degenerate_priority")
    # length mismatch → explicit fallback
    weights4, audit4 = build_destination_weights(
        relay_source="sender_priority",
        columns=columns,
        sender_priority=torch.tensor([1.0, 2.0]),
        base_seed=42,
    )
    check(
        "length-mismatch fallback",
        weights4 is None and "priority_length_mismatch" in str(audit4["fallback_reason"]),
    )


def test_3_shuffled_control() -> None:
    print("Test 3 — shuffled control determinism")
    columns = torch.arange(24)
    priority = torch.linspace(0.0, 1.0, 24)
    w_a, audit_a = build_destination_weights(
        relay_source="shuffled_sender_priority",
        columns=columns, sender_priority=priority, base_seed=42,
    )
    w_b, audit_b = build_destination_weights(
        relay_source="shuffled_sender_priority",
        columns=columns, sender_priority=priority, base_seed=42,
    )
    w_c, audit_c = build_destination_weights(
        relay_source="shuffled_sender_priority",
        columns=columns, sender_priority=priority, base_seed=43,
    )
    check("same seed → same permutation", w_a is not None and torch.equal(w_a, w_b))
    check("seed logged", audit_a["shuffle_seed"] == audit_b["shuffle_seed"] is not None)
    check(
        "different seed → different permutation (or logged distinct seed)",
        audit_c["shuffle_seed"] != audit_a["shuffle_seed"],
    )
    check(
        "shuffle preserves the weight multiset",
        w_a is not None and torch.allclose(w_a.sort().values, w_b.sort().values),
    )
    check(
        "case-deterministic seed depends on columns",
        deterministic_shuffle_seed(42, torch.arange(24))
        != deterministic_shuffle_seed(42, torch.arange(1, 25)),
    )


def test_4_mass_conservation() -> None:
    print("Test 4 — attention mass conservation for every relay source")
    aw = synthetic_attention()
    vis = torch.tensor([2, 5, 7, 9])
    alpha = 0.05
    rows_before = aw.float().sum(-1)
    n = vis.numel()
    variants = {
        "canonical(existing-attention)": dict(vis_w=None, static_destination=False),
        "uniform": dict(vis_w=torch.full((n,), 1.0 / n), static_destination=True),
        "sender_priority": dict(
            vis_w=torch.tensor([0.4, 0.3, 0.2, 0.1]), static_destination=True
        ),
        "shuffled_sender_priority": dict(
            vis_w=torch.tensor([0.1, 0.4, 0.2, 0.3]), static_destination=True
        ),
    }
    for name, kwargs in variants.items():
        out = reallocate(
            aw.clone(), vis, alpha,
            kwargs["vis_w"], 1.0, None, "proportional",
            static_destination=kwargs["static_destination"],
        )
        rows_after = out.float().sum(-1)
        drift = float((rows_after - rows_before).abs().max())
        check(f"row-sum conserved [{name}]", drift < 1e-5, f"drift={drift:.2e}")
        vis_mass_before = float(aw.float()[..., vis].sum(-1).mean())
        vis_mass_after = float(out.float()[..., vis].sum(-1).mean())
        check(
            f"visual mass increased [{name}]",
            vis_mass_after > vis_mass_before,
        )


def test_4b_same_total_gain() -> None:
    print("Test 4b — total visual gain G is destination-invariant")
    aw = synthetic_attention(seed=7)
    vis = torch.tensor([1, 4, 8, 10])
    alpha = 0.05
    n = vis.numel()
    gains = []
    for vis_w, static in (
        (None, False),
        (torch.full((n,), 1.0 / n), True),
        (torch.tensor([0.7, 0.1, 0.1, 0.1]), True),
    ):
        out = reallocate(aw.clone(), vis, alpha, vis_w, 1.0, None, "proportional",
                         static_destination=static)
        gains.append(
            float((out.float()[..., vis].sum(-1) - aw.float()[..., vis].sum(-1)).mean())
        )
    spread = max(gains) - min(gains)
    check("G identical across destinations", spread < 1e-6, f"gains={gains}")


def test_5_key_norm_backward_compat() -> None:
    print("Test 5 — canonical path is numerically unchanged")
    aw = synthetic_attention(seed=3)
    vis = torch.tensor([0, 3, 6])
    alpha = 0.05
    # Expected canonical formula (existing-attention destination, ratio 1.0):
    k = aw.shape[-1]
    rec = torch.zeros(k, dtype=torch.bool)
    rec[vis] = True
    donor = ~rec
    af = aw.float()
    moved = alpha * (af * donor).sum(-1, keepdim=True)
    af2 = af * torch.where(donor, 1.0 - alpha, torch.ones(())).float()
    rec_w = af2 * rec
    expected = af2 + moved * (rec_w / rec_w.sum(-1, keepdim=True).clamp_min(1e-9))
    # (a) default args — the exact call canonical key_norm makes at ratio 1.0
    key_norm_scores = torch.rand(1, k)
    out_canonical = reallocate(aw.clone(), vis, alpha, None, 1.0, key_norm_scores,
                               "proportional")
    check(
        "canonical == expected formula",
        torch.allclose(out_canonical.float(), expected, atol=1e-6),
    )
    # (b) new relay source WITHOUT a weight vector falls through to canonical
    out_fallthrough = reallocate(aw.clone(), vis, alpha, None, 1.0, None,
                                 "proportional", static_destination=True)
    check(
        "static source without weights == canonical",
        torch.allclose(out_fallthrough.float(), out_canonical.float(), atol=1e-6),
    )
    # (c) degenerate static weights fall through to canonical
    out_degenerate = reallocate(aw.clone(), vis, alpha, torch.zeros(3), 1.0, None,
                                "proportional", static_destination=True)
    check(
        "degenerate static weights == canonical",
        torch.allclose(out_degenerate.float(), out_canonical.float(), atol=1e-6),
    )
    # (d) static destination actually differs from canonical when weights differ
    out_static = reallocate(aw.clone(), vis, alpha,
                            torch.tensor([0.9, 0.05, 0.05]), 1.0, None,
                            "proportional", static_destination=True)
    check(
        "sender destination differs from canonical",
        not torch.allclose(out_static.float(), out_canonical.float(), atol=1e-6),
    )


def test_6_uniform_semantics() -> None:
    print("Test 6 — uniform destination puts equal gain on every visual column")
    aw = synthetic_attention(seed=11)
    vis = torch.tensor([2, 6, 9])
    alpha = 0.05
    n = vis.numel()
    out = reallocate(aw.clone(), vis, alpha, torch.full((n,), 1.0 / n), 1.0, None,
                     "proportional", static_destination=True)
    gain = out.float()[..., vis] - (1.0) * aw.float()[..., vis]
    per_col = gain.reshape(-1, n)
    spread = float((per_col - per_col.mean(-1, keepdim=True)).abs().max())
    check("equal per-column gain", spread < 1e-6, f"spread={spread:.2e}")


if __name__ == "__main__":
    test_1_mapping()
    test_2_normalization()
    test_3_shuffled_control()
    test_4_mass_conservation()
    test_4b_same_total_gain()
    test_5_key_norm_backward_compat()
    test_6_uniform_semantics()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
