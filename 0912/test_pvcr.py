"""Contracts for visual contribution extraction and receiver cache substitution."""

import pytest
import torch
from pvcr.core import contribution, full_attention
from pvcr.layout import ImageMetadata, Neighborhoods, build_neighborhoods
from pvcr.receiver import RelayBank, layer_values, receiver_mask, substitute_values
from transformers.cache_utils import DynamicCache, DynamicLayer


def test_full_denominator_retains_text_competition() -> None:
    query = torch.ones(1, 4, 1, 2)
    keys = torch.zeros(1, 2, 3, 2)
    keys[:, :, 2] = 10
    attention = full_attention(query, keys, None, scale=1.0)
    assert attention.shape == (1, 2, 3)
    assert attention[..., :2].sum().item() < 1e-6
    assert torch.allclose(attention.sum(-1), torch.ones(1, 2))


def test_grouped_contribution_excludes_direct_text_values() -> None:
    attention = torch.full((1, 2, 3), 1 / 3)
    values = torch.tensor(
        [[[[2.0, 0.0], [0.0, 4.0], [1e4, 1e4]], [[6.0, 0.0], [0.0, 8.0], [1e4, 1e4]]]]
    )
    layout = Neighborhoods(
        torch.tensor([0, 1]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        "test",
    )
    result = contribution(attention, values, layout)
    assert torch.allclose(result.value, torch.tensor([[[2 / 3, 4 / 3], [2.0, 8 / 3]]]))
    assert torch.allclose(result.visual_mass, torch.full((1, 2), 2 / 3))


def test_both_mask_formats_preserve_forbidden_columns() -> None:
    query = torch.zeros(1, 4, 1, 2, dtype=torch.bfloat16)
    keys = torch.zeros(1, 2, 3, 2, dtype=torch.bfloat16)
    for mask in (
        torch.tensor([[[[True, False, True]]]]),
        torch.tensor([[[[0.0, float("-inf"), 0.0]]]]),
    ):
        attention = full_attention(query, keys, mask, scale=1.0)
        assert torch.equal(attention[..., 1], torch.zeros(1, 2))


def test_value_substitution_keeps_keys_length_and_restores_on_error() -> None:
    cache = DynamicCache()
    keys = torch.arange(12.0).reshape(1, 2, 3, 2)
    values = torch.ones_like(keys)
    cache.update(keys.clone(), values.clone(), 0)
    layer = cache.layers[0]
    assert isinstance(layer, DynamicLayer)
    assert layer.keys is not None and layer.values is not None
    bank = RelayBank(torch.tensor([2]), {0: torch.full((1, 2, 1, 2), 7.0)})
    with (
        pytest.raises(RuntimeError, match="receiver failed"),
        substitute_values(cache, bank),
    ):
        assert torch.equal(layer.keys, keys)
        assert cache.get_seq_length() == 3
        assert torch.equal(layer.values[:, :, 2], torch.full((1, 2, 2), 7.0))
        raise RuntimeError("receiver failed")
    assert torch.equal(layer.values, values)


def test_gqa_mapping_matches_repeated_keys_reference() -> None:
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(2, 8, 1, 4, generator=generator)
    keys = torch.randn(2, 2, 7, 4, generator=generator)
    reference = query @ keys.repeat_interleave(4, dim=1).transpose(-2, -1) * 0.5
    reference = reference.softmax(-1).reshape(2, 2, 4, 7).mean(2)
    assert torch.allclose(full_attention(query, keys, None, scale=0.5), reference)


def test_receiver_prefill_remains_causal_with_past_cache() -> None:
    query = torch.zeros(1, 4, 3, 2)
    keys = torch.zeros(1, 2, 5, 2)
    mask = receiver_mask(
        query, keys, None, columns=torch.tensor([1]), log_gain=0.7, causal=True
    )
    assert torch.equal(
        mask[0, 0].isneginf(),
        torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, False, False, True],
                [False, False, False, False, False],
            ]
        ),
    )
    assert torch.allclose(mask[..., 1], torch.full((1, 1, 3), 0.7))


def test_restoration_uses_grown_cache_storage() -> None:
    cache = DynamicCache()
    initial = torch.ones(1, 2, 3, 2)
    cache.update(initial.clone(), initial.clone(), 0)
    bank = RelayBank(torch.tensor([2]), {0: torch.full((1, 2, 1, 2), 7.0)})
    with substitute_values(cache, bank) as audit:
        extra = torch.full((1, 2, 1, 2), 3.0)
        cache.update(extra, extra, 0)
        assert layer_values(cache, 0)[0, 0, 2, 0].item() == 7.0
    assert cache.get_seq_length() == 4
    assert (
        audit.before_substitution,
        audit.after_substitution,
        audit.before_restoration,
        audit.after_restoration,
    ) == (3, 3, 4, 4)
    assert torch.equal(layer_values(cache, 0)[:, :, :3], initial)
    assert layer_values(cache, 0)[0, 0, 3, 0].item() == 3.0


def test_no_joint_spatial_support_returns_zero() -> None:
    layout = Neighborhoods(
        torch.tensor([0, 1]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        "test",
    )
    result = contribution(torch.tensor([[[1.0, 0.0]]]), torch.ones(1, 1, 2, 3), layout)
    assert torch.equal(result.value, torch.zeros(1, 1, 3))
    assert result.support.item() == 0.0


def test_shuffled_thumbnail_preserves_neighborhood_sizes() -> None:
    columns = torch.cat((torch.arange(16), torch.arange(20, 36)))
    grids = torch.tensor([[1, 8, 8], [1, 8, 8]])
    metadata = (ImageMetadata(), ImageMetadata())
    real = build_neighborhoods(columns, grids, metadata, variant="spatial")
    shuffled = build_neighborhoods(columns, grids, metadata, variant="shuffled")
    assert torch.equal(real.context.sum(-1), shuffled.context.sum(-1))
    assert not torch.equal(real.context, shuffled.context)
    assert torch.equal(real.columns, torch.arange(16))
