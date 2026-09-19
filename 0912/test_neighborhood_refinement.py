"""Behavioral contract for reading an already-created regional latent."""

import math

import torch
from spatial_latent.neighborhood_refinement_runtime import refinement_definition
from spatial_latent.relation_attention import RelationStep, relation_step_sdpa
from test_spatial_latent import TinyAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def test_refinement_reads_same_neighborhood_and_previous_latent() -> None:
    # Given two local visual rows and the preceding latent summary row.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 5, 2)
    value = torch.zeros_like(key)
    value[:, :, [0, 1, 3]] = 1
    step = refinement_definition(torch.tensor([0, 1]), step_index=1, gain=4)
    # When the second regional step refines its first latent summary.
    with relation_step_sdpa(step) as probe:
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then both visual neighborhood rows and the prior latent are emphasized.
    assert torch.allclose(output, torch.full_like(output, 12 / 14))
    assert probe.previous_columns == (3,)


def test_first_regional_step_has_no_prior_latent_to_read() -> None:
    # Given the first regional latent's visual neighborhood.
    columns = torch.tensor([0, 1])
    # When the first step's read definition is made.
    step = refinement_definition(columns, step_index=0, gain=4)
    # Then it only reads the visual neighborhood, not a nonexistent latent.
    assert isinstance(step, RelationStep)
    assert step.include_previous is False
    assert step.log_gain == math.log(4)
