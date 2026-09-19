"""Tests for question-guided, rank-adaptive SparseVLM selection."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.sparsevlm import (
    SparseVLMPathologyLayout,
    sparsevlm_keep_mask,
    sparsevlm_pathology_keep_mask,
)
from memory.atp_sap import atp_sap_keep_mask
from backbone.qwen3vl import Qwen3VLBackbone
from vision_text_mas.direct_hf_ablation_presets import build_ablation_preset


def test_sparsevlm_keeps_question_attended_visual_tokens() -> None:
    """Given question raters, retain the visual columns they rate highest."""
    attention = torch.tensor(
        [[[0.90, 0.05, 0.03, 0.02], [0.45, 0.025, 0.015, 0.01]]]
    )

    keep, info = sparsevlm_keep_mask({12: attention}, lambda_scale=1.0)

    assert info.rater_count == 1
    assert info.drop_count == 3
    assert keep.tolist() == [True, False, False, False]


def test_sparsevlm_preserves_all_tokens_for_full_rank_question_attention() -> None:
    """Given independent question evidence, the rank rule does not prune it."""
    attention = (0.3 * torch.eye(3)).unsqueeze(0)

    keep, info = sparsevlm_keep_mask({12: attention}, lambda_scale=1.0)

    assert info.rank == 3
    assert info.drop_count == 0
    assert keep.all()


def test_sparsevlm_uses_only_question_rows_when_spans_are_available() -> None:
    """Given prompt boilerplate, score visual columns from the question rows only."""
    attention = torch.tensor(
        [[[0.01, 0.99], [0.99, 0.01], [0.99, 0.01]]]
    )

    keep, _ = sparsevlm_keep_mask(
        {12: attention}, question_spans=[(0, 1)], lambda_scale=1.0
    )

    assert keep.tolist() == [False, True]


def test_pathology_context_preserves_neighbor_and_parent_scale_tokens() -> None:
    """Given a salient child token, keep local and matched parent context."""
    base_keep = torch.zeros(128, dtype=torch.bool)
    base_keep[64 + 27] = True
    scores = torch.arange(128, dtype=torch.float32)
    layout = SparseVLMPathologyLayout(
        grid_shapes=((8, 8), (8, 8)),
        boxes=((0, 0, 1024, 1024), (256, 256, 256, 256)),
        magnifications=(5, 20),
        parent_indices=(-1, 0),
    )

    keep, info = sparsevlm_pathology_keep_mask(base_keep, scores, layout)

    assert keep[64 + 27]  # selected salient visual token remains
    assert keep[64 + 26] or keep[64 + 28] or keep[64 + 19] or keep[64 + 35]
    assert keep[36]  # x5 token aligned to the x20 crop center
    assert info.spatial_anchor_count > 0
    assert info.neighbor_count > 0
    assert info.parent_anchor_count == 1
    assert int(keep.sum()) >= int(base_keep.sum())


def test_pathology_context_does_not_invent_parent_when_geometry_is_invalid() -> None:
    """Given a self-parent link, preserve local context but reject scale link."""
    layout = SparseVLMPathologyLayout(
        grid_shapes=((2, 2), (1, 2)),
        boxes=((0, 0, 512, 512), (0, 0, 256, 256)),
        magnifications=(20, 20),
        parent_indices=(0, 0),
    )

    keep, info = sparsevlm_pathology_keep_mask(
        torch.tensor([True, False, False, False, False, False]),
        torch.arange(6, dtype=torch.float32),
        layout,
    )

    assert info.parent_anchor_count == 0
    assert keep[0]
    assert int(keep.sum()) > 1


def test_pathology_context_replaces_low_priority_tokens_without_growing_budget() -> None:
    """Given clustered top scores, trade redundant tokens for spatial context."""
    scores = torch.zeros(256, dtype=torch.float32)
    scores[:64] = 1.0
    base_keep = torch.zeros(256, dtype=torch.bool)
    base_keep[:64] = True
    layout = SparseVLMPathologyLayout(
        grid_shapes=((16, 16),),
        boxes=((0, 0, 2048, 2048),),
        magnifications=(5,),
        parent_indices=(-1,),
    )

    keep, info = sparsevlm_pathology_keep_mask(base_keep, scores, layout)

    assert int(keep.sum()) == int(base_keep.sum())
    assert info.context_added_count > 0
    assert info.base_tokens_replaced_count == info.context_added_count


def test_sparsevlm_preset_uses_question_adaptive_reasoner_pruning() -> None:
    """Given the SparseVLM arm, enable its boundary rule only for Reasoner."""
    preset = build_ablation_preset("pruning_sparsevlm", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.query_adaptive
    assert preset.prune_config.query_adaptive_lambda == 0.5
    assert preset.prune_config.reasoner_only
    assert preset.prune_config.keep_ratio is None


def test_sparsevlm_pathology_preset_adds_context_without_fixed_keep_ratio() -> None:
    """Given the pathology arm, compose WSI context with rank-adaptive pruning."""
    preset = build_ablation_preset("pruning_sparsevlm_pathology", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.query_adaptive
    assert preset.prune_config.query_pathology_context
    assert preset.prune_config.query_adaptive_lambda == 0.75
    assert preset.prune_config.keep_ratio is None
    assert preset.prune_config.reasoner_only


def test_sparsevlm_pathology_dual_arm_composes_both_relay_boundaries() -> None:
    """Given the combined arm, preserve WSI context and both latent relays."""
    preset = build_ablation_preset(
        "pruning_sparsevlm_pathology_latent_kv_relay2_dual", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.query_pathology_context
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )


def test_sparsevlm_dual_both_composes_question_pruning_and_both_relays() -> None:
    """Given QGAP + Dual Both, retain both question pruning and handoffs."""
    preset = build_ablation_preset(
        "pruning_sparsevlm_latent_kv_relay2_dual", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.query_adaptive
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )


def test_sparsevlm_upstream_variant_prunes_and_relays_at_planner_and_navigator() -> None:
    """Given the upstream arm, enable question pruning and relay at every handoff."""
    preset = build_ablation_preset(
        "pruning_sparsevlm_latent_kv_relay2_planner_navigator", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.query_adaptive
    assert not preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "evidence_planner",
        "navigator",
        "reasoner",
    )


def test_atp_sap_variant_uses_adaptive_visual_pruning_and_upstream_relays() -> None:
    """Given the ATP-SAP arm, use adaptive scoring and all three handoffs."""
    preset = build_ablation_preset(
        "pruning_atp_sap_latent_kv_relay2_planner_navigator", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.atp_sap
    assert not preset.prune_config.reasoner_only
    assert preset.prune_config.keep_ratio is None
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "evidence_planner",
        "navigator",
        "reasoner",
    )


def test_atp_sap_keeps_score_salient_tokens_and_spatial_scaffold() -> None:
    """Given a coarse WSI grid, retain relevant tokens plus spatial anchors."""
    visual_self_score = torch.arange(64, dtype=torch.float32)
    cross_attention = torch.zeros((2, 3, 64), dtype=torch.float32)
    cross_attention[:, 0, 0] = 1.0
    cross_attention[:, 1, 63] = 1.0

    keep, info = atp_sap_keep_mask(
        visual_self_score,
        {24: cross_attention},
        question_spans=[(0, 1)],
        grid_shapes=((8, 8),),
    )

    assert keep[0]
    assert keep.sum() >= 16
    assert info.retained_by_spatial_scaffold == 16
    assert info.retained_total == int(keep.sum())


def test_atp_sap_computes_a_distinct_adaptive_threshold_for_each_layer() -> None:
    """Given layers with different score sparsity, retain their separate K values."""
    visual_self_score = torch.zeros(32)
    broad = torch.zeros((1, 1, 32))
    broad[..., :16] = 1.0
    concentrated = torch.zeros((1, 1, 32))
    concentrated[..., 0] = 1.0

    keep, info = atp_sap_keep_mask(
        visual_self_score,
        {5: broad, 24: concentrated},
        question_spans=None,
        grid_shapes=((4, 8),),
    )

    assert info.retained_by_layer == ((5, 16), (24, 1))
    assert info.retained_total == int(keep.sum())
    assert info.retained_total >= info.retained_by_spatial_scaffold


def test_atp_sap_scaffold_uses_surviving_eadaprune_spatial_positions() -> None:
    """Given an irregular E-Ada subset, scaffold occupied cells without grid mismatch."""
    spatial_indices = torch.tensor([0, 4, 16, 20, 32, 36, 48, 52])
    visual_self_score = torch.zeros(8)
    cross_attention = torch.zeros((1, 1, 8))
    cross_attention[..., 0] = 1.0

    keep, info = atp_sap_keep_mask(
        visual_self_score,
        {5: cross_attention},
        question_spans=None,
        grid_shapes=((8, 8),),
        spatial_token_indices=(spatial_indices,),
    )

    assert info.retained_by_redundancy == 1
    assert info.retained_by_spatial_scaffold == 8
    assert info.retained_total == 8
    assert bool(keep[0])


def test_atp_sap_boundary_prunes_irregular_prefill_and_runs_pathology_rescue(
    monkeypatch,
) -> None:
    """Given pre-pruned positions, run ATP, morphology rescue, and cache pruning."""
    indices = torch.tensor(
        [row * 16 + col * 2 for row in range(4) for col in range(4)]
        + [1, 3, 5, 7]
    )
    vis_abs = torch.arange(100, 120)
    cross_attention = torch.zeros((1, 1, 20))
    cross_attention[..., 0] = 1.0
    prune_calls = []

    def fake_apply_kv_prune(past_kv, seq_len, vis_cols, keep, spans, device):
        retained = [column for column, selected in zip(vis_cols, keep) if selected]
        prune_calls.append(tuple(retained))
        return "pruned_cache", seq_len - (len(vis_cols) - len(retained)), spans, retained

    monkeypatch.setattr("memory.prune.apply_kv_prune", fake_apply_kv_prune)
    backbone = SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2)
            )
        ),
        _prune_args=SimpleNamespace(
            prune_query_adaptive=False,
            prune_atp_sap=True,
            prune_morphology_coverage_rescue=True,
            prune_cross_scale_consolidation=True,
            prune_cross_scale_margin=0.05,
        ),
        _prune_morphology_enabled=True,
        _prune_image_parent_indices=(),
    )

    result = Qwen3VLBackbone._wsi_consolidate_boundary(
        backbone,
        "original_cache",
        120,
        vis_abs,
        torch.randn(20, 8),
        {5: cross_attention},
        torch.tensor([[1, 16, 16]]),
        [("P", 1), ("V", 20)],
        visual_self_score=torch.zeros(20),
        spatial_token_indices=(indices,),
    )

    assert result[0] == "pruned_cache"
    assert result[3].sum() == 16
    assert len(prune_calls) == 1
    assert len(prune_calls[0]) == 16
    assert result[4].tolist() == list(prune_calls[0])

