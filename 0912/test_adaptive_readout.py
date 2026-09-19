"""Behavioral contracts for agreement-weighted Answerer readout."""

import torch
from spatial_latent.adaptive_readout_attention import adaptive_pair_sdpa
from test_spatial_latent import TinyAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def test_high_agreement_pair_biases_single_pass_readout() -> None:
    # Given two latent pairs with distinct values and a dominant first weight.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 5, 2)
    value = torch.tensor(
        [[[[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]]
    )
    pairs = torch.tensor([[1, 2], [3, 4]])
    weights = torch.tensor([0.9, 0.1])
    # When the agreement-weighted pair bias is applied.
    with adaptive_pair_sdpa(pairs, weights, alpha=0.1) as probe:
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then the first pair raises its share above unweighted single-pass 0.4.
    assert output.mean().item() > 0.4
    assert probe.applied_calls == 1


def test_pair_bias_preserves_bfloat16_attention_dtype() -> None:
    # Given Qwen-style bfloat16 states and float32 agreement weights.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16)
    key = torch.zeros(1, 1, 5, 2, dtype=torch.bfloat16)
    value = torch.zeros(1, 1, 5, 2, dtype=torch.bfloat16)
    pairs = torch.tensor([[1, 2], [3, 4]])
    weights = torch.tensor([0.9, 0.1])
    # When the adaptive readout constructs its pair bias.
    with adaptive_pair_sdpa(pairs, weights, alpha=0.1):
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then the single-pass attention retains the model dtype.
    assert output.dtype == torch.bfloat16
