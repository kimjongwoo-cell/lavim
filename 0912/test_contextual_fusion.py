"""Behavioral contract for Reasoner-side cross-scale latent fusion."""

import torch
from spatial_latent.contextual_fusion_attention import (
    CrossScaleFusionStep,
    cross_scale_fusion_sdpa,
)
from test_spatial_latent import TinyAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def test_aligned_child_is_fused_with_parent_context() -> None:
    # Given an aligned parent-child pair and a contradictory parent control.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 3, 2)
    value = torch.tensor([[[[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]]])
    real = CrossScaleFusionStep(torch.tensor([0]), torch.tensor([1]), strength=0.4)
    shuffled = CrossScaleFusionStep(torch.tensor([2]), torch.tensor([1]), strength=0.4)
    # When a Reasoner latent step reads the paired tissue scales.
    with cross_scale_fusion_sdpa(real) as real_probe:
        real_output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, None
        )
    with cross_scale_fusion_sdpa(shuffled) as shuffled_probe:
        shuffled_output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, None
        )
    # Then only the aligned pair receives joint parent-child fusion.
    assert torch.all(real_output > shuffled_output)
    assert real_probe.mean_agreement > shuffled_probe.mean_agreement
