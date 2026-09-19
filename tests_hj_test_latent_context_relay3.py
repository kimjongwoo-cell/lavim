"""Behavior tests for pathology-contextual Reasoner latent KV relay."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vision_text_mas.direct_hf_ablation_presets import build_ablation_preset
from vision_text_mas.latent_kv_relay import (
    RelationalVisualContext,
    select_pathology_context_steps,
)


def test_relay_selects_latent_step_representing_related_patches_not_one_patch() -> None:
    """Given parent/child KV groups, relay the step aligned to both groups."""
    keys = torch.zeros((1, 1, 10, 4), dtype=torch.float32)
    keys[0, 0, 0:2, 0] = 1.0  # parent patch
    keys[0, 0, 2:4, 1] = 1.0  # child patch
    keys[0, 0, 4:6, 3] = 1.0  # unrelated patch
    keys[0, 0, 6:8, 2] = 1.0  # text / prompt context
    keys[0, 0, 8] = torch.tensor([1.0, 1.0, 0.0, 0.0])  # parent + child
    keys[0, 0, 9] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # parent only
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys)])
    context = RelationalVisualContext(
        latent_steps=2,
        vision_columns=torch.arange(6),
        text_start=0,
        patch_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        parent_patch_indices=(-1, 0, -1),
    )

    selected = select_pathology_context_steps(cache, context)

    assert selected.tolist() == [0]


def test_relay_selects_nothing_without_related_visual_patch_pairs() -> None:
    """Given isolated patches, do not relay a step as neighborhood context."""
    keys = torch.eye(5, dtype=torch.float32).view(1, 1, 5, 5)
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys)])
    context = RelationalVisualContext(
        latent_steps=1,
        vision_columns=torch.tensor([0, 1]),
        text_start=2,
        patch_ids=torch.tensor([0, 1]),
        parent_patch_indices=(-1, -1),
    )

    selected = select_pathology_context_steps(cache, context)

    assert selected.numel() == 0


def test_relay3_variant_composes_contextual_relay_with_reasoner_pruning_b() -> None:
    """Given the new Both arm, configure B pruning and context relay at Reasoner."""
    preset = build_ablation_preset(
        "pruning_b_reasoner_context_relay3", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.pathology_context_latent_kv_relay
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "reasoner",
    )


def test_relay3_standalone_keeps_pruning_disabled() -> None:
    """Given standalone Relay3, configure contextual Reasoner relay only."""
    preset = build_ablation_preset("reasoner_context_relay3", latent_steps=10)

    assert preset.prune_config is None
    assert preset.reallocation is not None
    assert preset.reallocation.pathology_context_latent_kv_relay
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "reasoner",
    )
