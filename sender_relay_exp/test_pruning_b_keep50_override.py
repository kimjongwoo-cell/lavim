"""Regression tests for the isolated Pruning-B 50% keep-ratio runner."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sender_relay_exp.pruning_b_keep50_override import pruning_b_keep50_preset


def test_pruning_b_keep50_preset_changes_only_visual_retention_ratio() -> None:
    """Given Pruning-B + dual relay, retain 50% and preserve its other rules."""
    preset = pruning_b_keep50_preset(latent_steps=10)

    assert preset.name == "pruning_b_latent_kv_relay2_dual"
    assert preset.prune_config is not None
    assert preset.prune_config.keep_ratio == 0.50
    assert preset.prune_config.prefill_hierarchy_aware
    assert preset.prune_config.reasoner_only
    assert preset.prune_config.morphology_coverage_rescue
    assert preset.prune_config.cross_scale_consolidation
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )
