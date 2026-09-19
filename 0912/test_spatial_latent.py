"""Behavioral contracts for the isolated spatial latent experiment."""

import math

import pytest
import torch
from spatial_latent import attention, grouping
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def test_real_groups_follow_parent_relationships() -> None:
    # Given four parents and interleaved children, as in the live pipeline.
    parents = (-1, -1, -1, -1, 0, 1, 2, 3, 0, 1, 2, 3)
    # When neighborhoods are constructed from parent identity.
    groups = grouping.make_groups(parents, grouping.GroupMode.SPATIAL)
    # Then each low-power context travels with its actual high-power children.
    assert groups == ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11))


def test_shuffled_groups_preserve_coverage_without_real_parents() -> None:
    # Given the same physical patch set and a fixed control seed.
    parents = (-1, -1, -1, -1, 0, 1, 2, 3, 0, 1, 2, 3)
    # When spatial relationships are shuffled.
    groups = grouping.make_groups(parents, grouping.GroupMode.SHUFFLED)
    # Then sizes/scales are matched, every patch occurs once, and parents differ.
    assert sorted(patch for group in groups for patch in group) == list(range(12))
    assert all(len(group) == 3 and group[0] < 4 for group in groups)
    assert all(parents[child] != group[0] for group in groups for child in group[1:])
    assert groups == grouping.make_groups(parents, grouping.GroupMode.SHUFFLED)


def test_column_runs_retain_absolute_cache_offsets() -> None:
    # Given visual runs separated by prompt tokens and a carried-cache prefix.
    columns = torch.tensor([100, 101, 108, 109, 110, 200])
    # When patch token ranges are resolved.
    runs = grouping.patch_columns(columns, expected_patches=3)
    # Then the mapping preserves the actual cache addresses.
    assert [part.tolist() for part in runs] == [[100, 101], [108, 109, 110], [200]]


def test_invalid_patch_layout_fails_instead_of_guessing() -> None:
    # Given an incomplete patch acquisition.
    parents = (-1, -1, 0)
    # When constructing the fixed 12-patch intervention.
    with pytest.raises(grouping.LayoutError):
        # Then the run fails explicitly rather than silently using another method.
        grouping.make_groups(parents, grouping.GroupMode.SPATIAL)


class TinyAttention(torch.nn.Module):
    """Real SDPA-compatible attention metadata for a controlled tensor example."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = 1
        self.is_causal = True


def test_bias_changes_only_selected_latent_attention() -> None:
    # Given equal key scores, with one context value carrying nonzero evidence.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 4)
    key = torch.zeros(1, 1, 4, 4)
    value = torch.zeros_like(key)
    value[:, :, 1] = 1
    settings = attention.BiasSettings(log_gain=math.log(2))
    # When the latent query is guided toward that context key.
    with attention.guided_sdpa(torch.tensor([1]), settings) as probe:
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then its contribution is 2/(1+2+1+1), without removing other keys.
    assert torch.allclose(output, torch.full_like(output, 0.4))
    assert probe.applied_calls == 1


@pytest.mark.parametrize("layer,queries", [(7, 1), (21, 1), (8, 2)])
def test_other_layers_and_prefill_are_unchanged(layer: int, queries: int) -> None:
    # Given an out-of-scope layer or a multi-token prefill.
    module = TinyAttention(layer)
    query = torch.zeros(1, 1, queries, 4)
    key = torch.zeros(1, 1, 4, 4)
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    expected, _ = sdpa_attention_forward(module, query, key, value, None)
    # When the experiment's SDPA adapter is installed.
    with attention.guided_sdpa(torch.tensor([1]), attention.BiasSettings()) as probe:
        actual, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, None)
    # Then ordinary model computation is unchanged.
    assert torch.equal(actual, expected)
    assert probe.applied_calls == 0


def test_bias_preserves_existing_attention_mask() -> None:
    # Given a selected key that the original attention mask forbids.
    module = TinyAttention(8)
    query = torch.zeros(1, 1, 1, 4)
    key = torch.zeros(1, 1, 4, 4)
    value = torch.zeros_like(key)
    value[:, :, 1] = 1
    mask = torch.tensor([[[[True, False, True, True]]]])
    # When a positive spatial bias targets that key.
    with attention.guided_sdpa(torch.tensor([1]), attention.BiasSettings()):
        result, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, mask)
    # Then the originally forbidden key stays inaccessible.
    assert torch.equal(result, torch.zeros_like(result))


def test_adapter_restores_sdpa_after_exit() -> None:
    # Given the registered SDPA implementation.
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    # When a scoped bias operation completes.
    with attention.guided_sdpa(torch.tensor([1]), attention.BiasSettings()):
        pass
    # Then no intervention leaks into Answerer or another role.
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
