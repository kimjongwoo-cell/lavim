"""Behavioral contracts for cross-scale relation latents."""

import math

import torch
from spatial_latent.relation_attention import (
    RelationStep,
    relation_step_sdpa,
    reserved_relation_sdpa,
)
from test_spatial_latent import TinyAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def test_context_step_reads_parent_without_previous_latent() -> None:
    # Given equal visual keys and a distinct parent value.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 4)
    key = torch.zeros(1, 1, 5, 4)
    value = torch.zeros_like(key)
    value[:, :, 1] = 1
    # When the context step boosts only the parent column.
    step = RelationStep(torch.tensor([1]), include_previous=False, log_gain=math.log(4))
    with relation_step_sdpa(step):
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then the parent contributes 4 / (4 + 4 other keys).
    assert torch.allclose(output, torch.full_like(output, 0.5))


def test_relation_step_reads_children_and_previous_latent() -> None:
    # Given two child keys and the previous latent immediately before this query.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 4)
    key = torch.zeros(1, 1, 6, 4)
    value = torch.zeros_like(key)
    value[:, :, 1] = 1
    value[:, :, 2] = 1
    value[:, :, 4] = 1
    # When the relation step boosts both children and the preceding latent key.
    step = RelationStep(
        torch.tensor([1, 2]), include_previous=True, log_gain=math.log(4)
    )
    with relation_step_sdpa(step) as probe:
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then those three sources contribute 12 / (12 + 3 other keys).
    assert torch.allclose(output, torch.full_like(output, 0.8))
    assert probe.previous_columns == (4,)


def test_synthesis_step_reads_prior_relation_rows() -> None:
    # Given four relation rows at alternating positions in the latent tail.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 10, 2)
    value = torch.zeros_like(key)
    value[:, :, [2, 4, 6, 8]] = 1
    # When synthesis targets the four relation offsets from the current key.
    step = RelationStep(
        torch.empty(0, dtype=torch.long),
        include_previous=False,
        log_gain=math.log(4),
        tail_offsets=(2, 4, 6, 8),
    )
    with relation_step_sdpa(step) as probe:
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then all four dynamically addressed rows contribute to synthesis.
    assert torch.allclose(output, torch.full_like(output, 16 / 22))
    assert probe.previous_columns == (8, 6, 4, 2)


def test_answerer_reserves_relation_mass() -> None:
    # Given one ordinary source and one relation source with distinct values.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 2, 2)
    value = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    # When 20 percent of output attention is reserved for the relation source.
    with reserved_relation_sdpa(torch.tensor([1]), alpha=0.2) as probe:
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then the answerer receives exactly the relation value's reserved contribution.
    assert torch.allclose(output, torch.full_like(output, 0.2))
    assert probe.applied_calls == 1


def test_answerer_relation_budget_preserves_forbidden_mask() -> None:
    # Given a relation source forbidden by the original mask.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 2, 2)
    value = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    mask = torch.tensor([[[[True, False]]]])
    # When the reserved relation adapter runs.
    with reserved_relation_sdpa(torch.tensor([1]), alpha=0.2):
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, mask)
    # Then it cannot revive a key that was inaccessible to the query.
    assert torch.equal(output, torch.zeros_like(output))
