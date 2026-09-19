"""Behavioral contracts for cross-scale consistency-gated latent evidence."""

import torch
from spatial_latent.relation_consistency_attention import (
    ConsistencyStep,
    consistency_gate_sdpa,
)
from test_spatial_latent import TinyAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def test_consistent_parent_enables_child_evidence() -> None:
    # Given one agreeing parent, one child, and a contradictory shuffled parent.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 3, 2)
    value = torch.tensor([[[[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]]])
    real = ConsistencyStep(torch.tensor([0]), torch.tensor([1]), strength=0.1)
    shuffled = ConsistencyStep(torch.tensor([2]), torch.tensor([1]), strength=0.1)
    # When the relation latent reads an agreeing or contradictory parent.
    with consistency_gate_sdpa(real) as real_probe:
        real_output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, None
        )
    with consistency_gate_sdpa(shuffled) as shuffled_probe:
        shuffled_output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, None
        )
    # Then agreement gates in child evidence only for the real parent-child pair.
    assert torch.all(real_output > shuffled_output)
    assert real_probe.mean_agreement > shuffled_probe.mean_agreement


def test_contradictory_pair_preserves_native_attention() -> None:
    # Given a parent whose evidence opposes its child.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 2, 2)
    value = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]])
    step = ConsistencyStep(torch.tensor([0]), torch.tensor([1]), strength=0.1)
    native, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # When the consistency gate sees a contradictory pair.
    with consistency_gate_sdpa(step) as probe:
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then it adds no high-power evidence beyond the native attention output.
    assert torch.equal(output, native)
    assert probe.mean_agreement == 0.0
