"""Behavioral contracts for explicit cross-scale contrast injection."""

import torch
from spatial_latent.relation_contrast_attention import (
    ContrastStep,
    cross_scale_contrast_sdpa,
)
from test_spatial_latent import TinyAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def test_contrast_output_changes_when_parent_pairing_changes() -> None:
    # Given one low-power parent, two high-power children, and one distractor.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 4)
    key = torch.zeros(1, 1, 4, 4)
    value = torch.tensor(
        [[[[0.0] * 4, [1.0] * 4, [1.0] * 4, [1.0] * 4]]]
    )
    real = ContrastStep(
        parent_columns=torch.tensor([0]),
        child_columns=torch.tensor([1, 2]),
        strength=1.0,
    )
    shuffled = ContrastStep(
        parent_columns=torch.tensor([3]),
        child_columns=torch.tensor([1, 2]),
        strength=1.0,
    )
    # When the same latent query uses real and shuffled parent-child pairings.
    with cross_scale_contrast_sdpa(real):
        real_output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, None
        )
    with cross_scale_contrast_sdpa(shuffled):
        shuffled_output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, None
        )
    # Then only the pairing with a parent-child contrast changes the hidden update.
    assert torch.all(real_output > shuffled_output)


def test_zero_contrast_preserves_native_output() -> None:
    # Given parent and children with identical attended values.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 3, 2)
    value = torch.ones_like(key)
    step = ContrastStep(
        parent_columns=torch.tensor([0]),
        child_columns=torch.tensor([1, 2]),
        strength=0.2,
    )
    native, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # When explicit contrast injection sees no cross-scale difference.
    with cross_scale_contrast_sdpa(step) as probe:
        contrasted, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, None
        )
    # Then the original SDPA result is preserved exactly.
    assert torch.equal(contrasted, native)
    assert probe.applied_calls == 1
