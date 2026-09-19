"""Behavioral contracts: latent provenance, actual insertion, selective reads."""

import pytest
import torch
from latent_visual_relay import cache as relay_cache
from latent_visual_relay import selection
from latent_visual_relay.capture import select_prompt_query_rows
from latent_visual_relay.attention import memory_bias, relation_family_read_bias
from pvcr.errors import LayoutError
from pvcr.layout import ImageMetadata, Neighborhoods, build_neighborhoods
from pvcr.receiver import layer_values
from transformers.cache_utils import DynamicCache
from latent_visual_relay.cache import InsertedBlock, InsertionAudit


def test_adaptive_gate_selects_spikes_without_forcing_top_k() -> None:
    scores = torch.tensor([[[1.0, 1.0, 8.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    chosen = selection.select_heads(scores)
    assert torch.equal(
        chosen,
        torch.tensor([[[False, False, True, False], [False, False, False, False]]]),
    )


def test_flat_scores_do_not_force_a_latent_row() -> None:
    chosen = selection.select_heads(torch.ones(1, 2, 10))
    assert not chosen.any()


def test_contrastive_query_relay_keeps_only_latent_kv_closer_to_visual_than_text() -> (
    None
):
    # Given actual visual/text query vectors and two latent K rows.
    visual_queries = torch.tensor([[[[1.0, 0.0]]]])
    text_queries = torch.tensor([[[[0.0, 1.0]]]])
    latent_keys = torch.tensor([[[[0.8, 0.2], [0.2, 0.8]]]])

    # When each latent KV is scored against actual visual and text queries.
    margin = selection.contrastive_query_margin(
        visual_queries, text_queries, latent_keys
    )
    chosen = selection.select_visual_closer_pairs(margin)

    # Then only the visual-closer K/V pair passes to the next visual cache.
    assert margin[0, 0, 0] > 0
    assert margin[0, 0, 1] < 0
    assert torch.equal(chosen, torch.tensor([[[True, False]]]))


def test_contrastive_key_relay_keeps_every_visual_closer_pair_per_step() -> None:
    # Given multiple qualifying K/V pairs across two latent steps.
    margins = torch.tensor([[[0.8, -0.1], [0.3, 0.2]]])

    # When each head-step is gated independently without a global top-percent cap.
    chosen = selection.select_visual_closer_pairs(margins)

    # Then all and only visual-closer K/V pairs pass to the next visual cache.
    assert torch.equal(chosen, torch.tensor([[[True, False], [True, True]]]))


def test_contextual_query_margin_requires_local_and_neighbor_visual_affinity() -> None:
    # Given local tissue queries, surrounding-context queries, text queries, and latent K.
    local_queries = torch.tensor([[[[1.0, 0.0]]]])
    context_queries = torch.tensor([[[[0.8, 0.2]]]])
    text_queries = torch.tensor([[[[0.0, 1.0]]]])
    latent_keys = torch.tensor([[[[0.8, 0.2], [0.2, 0.8]]]])

    # When both pathology scales contribute to the contrastive query score.
    margins = selection.contrastive_context_margin(
        local_queries, context_queries, text_queries, latent_keys
    )

    # Then context-consistent latent K/V passes, while text-aligned K/V is rejected.
    assert margins[0, 0, 0, 0] > 0
    assert margins[0, 0, 0, 1] < 0


def test_contextual_query_rejects_latent_kv_missing_neighbor_support() -> None:
    # Given a latent key closer to local visual Q than text Q but far from context Q.
    local_queries = torch.tensor([[[[1.0, 0.0]]]])
    context_queries = torch.tensor([[[[0.5, 3**0.5 / 2]]]])
    text_queries = torch.tensor([[[[0.7, 0.51**0.5]]]])
    latent_keys = torch.tensor([[[[1.0, 0.0]]]])

    # When local, context, and text affinities are compared for this latent row.
    margins = selection.contrastive_context_margin(
        local_queries, context_queries, text_queries, latent_keys
    )

    # Then local-only support is rejected because context affinity is below text.
    assert margins[0, 0, 0, 0] < 0


def test_contextual_adaptive_gate_keeps_every_stepwise_head_outlier() -> None:
    # Given per-head local+context-vs-text margins for two latent steps.
    margins = torch.tensor(
        [[[0.01, 0.10], [0.02, 0.12], [0.03, -0.10], [0.50, 0.11]]]
    )

    # When each step uses its head-wise median-plus-MAD threshold.
    selected = selection.select_contextual_pairs(margins)

    # Then every positive outlier passes without a fixed top-k count.
    assert torch.equal(
        selected,
        torch.tensor([[[False, False], [False, True], [False, False], [True, False]]]),
    )


def test_actual_attention_scores_gate_family_specific_kv_pairs() -> None:
    # Given GQA-reduced latent visual attention for two KV heads and two families.
    scores = torch.tensor(
        [[[[0.0, 0.8], [0.2, 0.0]], [[0.7, 0.0], [0.0, 0.3]]]],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # When each latent step is gated across KV heads and assigned its best family.
    selected = selection.select_attention_context_pairs(scores)

    # Then each KV head relays its strongest child-parent-sibling family only.
    assert torch.equal(
        selected,
        torch.tensor([[[[False, True], [False, False]], [[True, False], [False, False]]]]),
    )


def test_pathology_context_family_keeps_child_parent_and_paired_child() -> None:
    # Given four x5 parents and two spatially separate x20 children per parent.
    metadata = tuple(
        ImageMetadata(
            patch_id=f"p{parent}",
            magnification=5,
            box=(parent * 200, 0, 100, 100),
        )
        for parent in range(4)
    ) + tuple(
        ImageMetadata(
            patch_id=f"c{parent}-{child}",
            parent_id=f"p{parent}",
            magnification=20,
            box=(parent * 200 + child * 50, 0, 40, 40),
        )
        for parent in range(4)
        for child in range(2)
    )
    columns = torch.arange(0, 24, 2)
    grids = torch.tensor([[1, 2, 2]] * len(metadata))

    # When pathology neighborhoods are built at individual child resolution.
    layout = build_neighborhoods(columns, grids, metadata, variant="paired_context")

    # Then each family is one child, its paired sibling, and their shared parent.
    assert layout.core.shape == (8, 12)
    for family in range(8):
        child_column = family + 4
        sibling_column = 4 + (family ^ 1)
        parent_column = family // 2
        assert layout.core[family, child_column] == 1
        assert layout.context[family, sibling_column] == 1
        assert layout.context[family, parent_column] == 1
        assert layout.core[family].sum() == 1
        assert layout.context[family].sum() == 2
    assert build_neighborhoods(columns, grids, metadata, variant="visual").core.shape == (
        4,
        12,
    )

def test_prompt_query_selection_accepts_cuda_layout_columns() -> None:
    # Given cached prefill Q on CPU and visual-layout addresses on CUDA.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to reproduce mixed-device layout addresses")
    prompt_queries = torch.arange(10.0).reshape(1, 1, 5, 2)
    visual_columns = torch.tensor([3, 5], device="cuda")

    # When selecting absolute visual query positions from the CPU cache.
    selected = select_prompt_query_rows(prompt_queries, visual_columns, start=2)

    # Then indices are normalized and the requested Q rows are returned.
    assert torch.equal(selected, torch.tensor([[[[2.0, 3.0], [6.0, 7.0]]]]))


def test_inserted_values_come_from_latent_not_visual_cache() -> None:
    cache = DynamicCache()
    keys = torch.arange(30.0).reshape(1, 2, 5, 3)
    values = keys + 100
    cache.update(keys.clone(), values.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([3, 4]), torch.tensor([1, 2]))
    chosen = {0: torch.tensor([[[True, False], [False, True]]])}
    result = relay_cache.insert_selected(cache, source, chosen)
    actual = layer_values(cache, 0)
    assert result.columns.tolist() == [3, 4]
    assert cache.get_seq_length() == 7
    assert torch.equal(actual[:, :, :3], values[:, :, :3])
    assert torch.equal(actual[:, :, 5:], values[:, :, 3:])
    assert torch.equal(actual[0, 0, 3], values[0, 0, 3])
    assert torch.equal(actual[0, 1, 4], values[0, 1, 4])
    assert torch.equal(actual[0, 0, 4], torch.zeros(3))
    assert torch.equal(actual[0, 1, 3], torch.zeros(3))
    assert result.audit.source_equal and result.audit.original_cache_equal


def test_family_aware_insertion_preserves_the_latent_rows_tissue_addresses() -> None:
    # Given family-specific sender selection over two heads and two latent steps.
    cache = DynamicCache()
    values = torch.arange(30.0).reshape(1, 2, 5, 3)
    cache.update(values.clone(), values.clone(), 0)
    layout = Neighborhoods(
        columns=torch.tensor([1, 2]),
        core=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        context=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        kind="test",
    )
    source = relay_cache.SourceBlock(
        torch.tensor([3, 4]), torch.tensor([1, 2]), layout
    )
    selected = {
        0: torch.tensor(
            [[[[True, False], [False, True]], [[False, True], [True, False]]]]
        )
    }

    # When selected rows are inserted into the receiver cache.
    inserted = relay_cache.insert_selected(cache, source, selected)

    # Then each inserted cache column retains its family and source layout.
    assert inserted.family_ids is not None
    assert inserted.family_ids.tolist() == [0, 0, 1, 1]
    assert inserted.layout is layout


def test_empty_selection_leaves_native_cache_unchanged() -> None:
    cache = DynamicCache()
    values = torch.arange(30.0).reshape(1, 2, 5, 3)
    cache.update(values.clone(), values.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([3, 4]), torch.tensor([1, 2]))
    result = relay_cache.insert_selected(
        cache, source, {0: torch.zeros(1, 2, 2, dtype=torch.bool)}
    )
    assert result.columns.numel() == 0
    assert torch.equal(layer_values(cache, 0), values)
    assert cache.get_seq_length() == 5


def test_selected_keys_preserve_source_rotary_values_exactly() -> None:
    cache = DynamicCache()
    keys = torch.arange(30.0).reshape(1, 2, 5, 3)
    cache.update(keys.clone(), keys.clone() + 100, 0)
    source = relay_cache.SourceBlock(torch.tensor([3, 4]), torch.tensor([1, 2]))
    relay_cache.insert_selected(
        cache, source, {0: torch.tensor([[[True, False], [False, True]]])}
    )
    actual = relay_cache.layer_view(cache, 0).keys
    assert torch.equal(actual[0, 0, 3], keys[0, 0, 3])
    assert torch.equal(actual[0, 1, 4], keys[0, 1, 4])


def test_appended_mask_excludes_unselected_gqa_heads() -> None:
    cache = DynamicCache()
    keys = torch.zeros(1, 2, 5, 3)
    cache.update(keys.clone(), keys.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([3, 4]), torch.tensor([1, 2]))
    block = relay_cache.insert_selected(
        cache, source, {0: torch.tensor([[[True, False], [False, True]]])}
    )
    bias = memory_bias([block], torch.zeros(1, 4, 1, 3), 0, 7)
    assert torch.equal(
        bias[0, :, 0, 3:5].isneginf(),
        torch.tensor([[False, True], [False, True], [True, False], [True, False]]),
    )
    assert torch.equal(bias[..., :3], torch.zeros(1, 4, 1, 3))


def test_relay_attention_gain_applies_only_to_selected_kv_heads() -> None:
    # Given two selected K/V pairs and a positive retrieval gain.
    cache = DynamicCache()
    keys = torch.zeros(1, 2, 5, 3)
    cache.update(keys.clone(), keys.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([3, 4]), torch.tensor([1, 2]))
    block = relay_cache.insert_selected(
        cache, source, {0: torch.tensor([[[True, False], [False, True]]])}
    )

    # When the next agent attends through the relay memory bias.
    query = torch.zeros(1, 4, 1, 3)
    bias = memory_bias([block], query, 0, 7, log_gain=2.0)

    # Then selected GQA heads receive the configured gain, while others are blocked.
    assert torch.equal(bias[0, :, 0, 3:5], torch.tensor([[2.0, -float("inf")], [2.0, -float("inf")], [-float("inf"), 2.0], [-float("inf"), 2.0]]))


def test_receiver_bias_follows_query_supported_child_context_family() -> None:
    # Given two child/context families and two KV heads attending opposite families.
    layout = Neighborhoods(
        columns=torch.tensor([0, 1, 2, 3]),
        core=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        context=torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        kind="test_child_context",
    )
    block = InsertedBlock(
        columns=torch.tensor([6, 7]),
        allowed={0: torch.ones(1, 2, 2, dtype=torch.bool)},
        audit=InsertionAudit(6, 8, 4, [5, 5], 4, True, True),
        family_ids=torch.tensor([0, 1]),
        layout=layout,
    )
    visual_attention = torch.tensor(
        [[[0.45, 0.45, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
          [0.01, 0.01, 0.45, 0.45, 0.02, 0.02, 0.02, 0.02]]]
    )
    query = torch.zeros(1, 4, 1, 2, dtype=torch.bfloat16)

    # When receiver family compatibility is converted to a bias over inserted rows.
    bias = relation_family_read_bias(
        [block], visual_attention, query, layer=0, length=8, log_gain=2.0
    )

    # Then each GQA group prefers only the latent row paired with its attended family.
    assert bias[0, 0, 0, 6] > bias[0, 0, 0, 7]
    assert bias[0, 1, 0, 6] > bias[0, 1, 0, 7]
    assert bias[0, 2, 0, 7] > bias[0, 2, 0, 6]
    assert bias[0, 3, 0, 7] > bias[0, 3, 0, 6]
    assert torch.equal(bias[..., :6], torch.zeros(1, 4, 1, 6, dtype=torch.bfloat16))


def test_real_sdpa_reads_copied_latent_values_without_replacing_visual_values() -> None:
    cache = DynamicCache()
    keys = torch.zeros(1, 2, 5, 3)
    values = torch.ones_like(keys)
    values[:, :, 3] = 10
    values[:, :, 4] = 20
    cache.update(keys.clone(), values.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([3, 4]), torch.tensor([1, 2]))
    block = relay_cache.insert_selected(
        cache, source, {0: torch.tensor([[[True, False], [False, True]]])}
    )
    query = torch.zeros(1, 4, 1, 3)
    view = relay_cache.layer_view(cache, 0)
    result = torch.nn.functional.scaled_dot_product_attention(
        query,
        view.keys,
        view.values,
        attn_mask=memory_bias([block], query, 0, 7),
        enable_gqa=True,
    )
    expected = (
        torch.tensor([43 / 6, 43 / 6, 53 / 6, 53 / 6])
        .reshape(1, 4, 1, 1)
        .expand_as(result)
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_spatial_grounding_requires_core_and_context_above_background() -> None:
    layout = Neighborhoods(
        torch.tensor([0, 1]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        "test",
    )
    attention = torch.tensor([[[0.05, 0.05, 0.90], [0.45, 0.45, 0.10]]])
    scores = selection.grounding_scores(attention, layout, spatial=True)
    assert torch.allclose(scores, torch.tensor([[0.0, 3.15]]), atol=1e-5)


def test_relational_grounding_binds_a_latent_row_to_its_supported_patch_family() -> (
    None
):
    # Given two x5-parent/x20-child families and one latent attention row.
    layout = Neighborhoods(
        torch.arange(6),
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "x5_x20",
    )
    attention = torch.tensor([[[0.25, 0.25, 0.25, 0.01, 0.01, 0.01, 0.22]]])

    # When the row jointly attends one family's fine morphology and low-power context.
    scores = selection.relational_grounding_scores(attention, layout)

    # Then only that true family receives a positive relay score.
    assert scores.shape == (1, 1, 2)
    assert scores[0, 0, 0] > 0
    assert scores[0, 0, 1] == 0


def test_relational_gate_keeps_only_cross_family_outliers() -> None:
    # Given one head with an exceptional parent/child family-step score.
    scores = torch.tensor([[[[1.0, 1.0], [1.0, 9.0]]]])

    # When family and step are gated jointly rather than independently per family.
    chosen = selection.select_relational_heads(scores)

    # Then the ordinary family's local maximum cannot become a relay row.
    assert torch.equal(chosen, torch.tensor([[[[False, False], [False, True]]]]))


def test_relational_capacity_keeps_one_adaptive_family_for_each_supported_step() -> (
    None
):
    scores = {0: torch.tensor([[[[1.0, 5.0], [9.0, 7.0]]]])}
    selected = {0: torch.tensor([[[[False, True], [True, True]]]])}

    kept = selection.cap_relational_rows(scores, selected, capacity=2)

    assert torch.equal(kept[0], torch.tensor([[[[False, False], [True, True]]]]))


def test_family_aware_relay_copies_each_selected_family_step_at_visual_boundary() -> (
    None
):
    # Given two relational families that select different latent steps and heads.
    cache = DynamicCache()
    values = torch.arange(30.0).reshape(1, 2, 5, 3)
    cache.update(values.clone(), values.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([3, 4]), torch.tensor([1, 2]))
    selected = {
        0: torch.tensor(
            [[[[True, False], [False, False]], [[False, False], [False, True]]]]
        )
    }

    # When the receiver receives the relational family-step selections.
    block = relay_cache.insert_selected(cache, source, selected)
    actual = relay_cache.layer_view(cache, 0).values

    # Then each row is copied from its selected latent step, only for its owning head.
    assert block.columns.tolist() == [3, 4]
    assert torch.equal(actual[0, 0, 3], values[0, 0, 3])
    assert torch.equal(actual[0, 1, 4], values[0, 1, 4])
    assert torch.equal(actual[0, 0, 4], torch.zeros(3))
    assert torch.equal(actual[0, 1, 3], torch.zeros(3))


def test_family_aware_relay_copies_rows_without_removing_source_latent_memory() -> None:
    # Given a cache where the source latent rows must remain available after relay.
    cache = DynamicCache()
    values = torch.arange(30.0).reshape(1, 1, 10, 3)
    cache.update(values.clone(), values.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([8, 9]), torch.tensor([1, 2]))
    selected = {0: torch.tensor([[[[True, False], [False, True]]]])}

    # When two family-step entries are relayed.
    block = relay_cache.insert_selected(cache, source, selected)

    # Then they are placed after vision while the original latent rows remain intact.
    assert block.columns.tolist() == [3, 4]
    assert cache.get_seq_length() == 12
    assert torch.equal(
        relay_cache.layer_view(cache, 0).values[:, :, 5:], values[:, :, 3:]
    )


def test_insertion_never_expands_fixed_context_limit() -> None:
    cache = DynamicCache()
    keys = torch.zeros(1, 1, 12288, 2)
    cache.update(keys.clone(), keys.clone(), 0)
    source = relay_cache.SourceBlock(torch.tensor([12287]), torch.tensor([1, 2]))
    with pytest.raises(LayoutError, match="fixed context"):
        relay_cache.insert_selected(
            cache, source, {0: torch.ones(1, 1, 1, dtype=torch.bool)}
        )
    assert cache.get_seq_length() == 12288
