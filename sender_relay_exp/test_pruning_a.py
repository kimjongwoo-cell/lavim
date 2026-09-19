"""Contract checks for the Pruning A WSI cross-scale preset."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.prune import PruneController, spectral_energy_keep_ratio
from memory.relation_prune import relation_spatial_context_keep
from vision_text_mas.direct_hf_ablation_presets import build_ablation_preset
from vision_text_mas.latent_kv_relay import select_visual_contrast_steps


def test_hierarchy_pruner_binds_prune_args_before_cross_scale_check() -> None:
    """Regression: Pruning A must not reference an unbound `_pa` local."""
    source_path = Path(__file__).resolve().parents[1] / "backbone" / "qwen3vl.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_hierarchy_pruned_vision_features"
    )
    assigned = any(
        isinstance(node, ast.Name) and node.id == "_pa" and isinstance(node.ctx, ast.Store)
        for node in ast.walk(function)
    )
    reads_prune_args = any(
        isinstance(node, ast.Name) and node.id == "_pa" and isinstance(node.ctx, ast.Load)
        for node in ast.walk(function)
    )
    assert not reads_prune_args or assigned, "Pruning A cross-scale path references unbound local `_pa`"


def test_eadaprune_adaptive_budget_has_math_binding() -> None:
    """Regression: E-AdaPrune must resolve math.ceil for its adaptive budget."""
    from backbone import qwen3vl

    function = qwen3vl.Qwen3VLBackbone._hierarchy_pruned_vision_features.__wrapped__

    assert "math" in function.__globals__, (
        "E-AdaPrune references math.ceil but its module has no math binding"
    )


def test_pruning_b_enables_evidence_coverage_and_cross_scale_rescue() -> None:
    """Given Pruning B, when its preset is built, then both safeguards are active."""
    preset = build_ablation_preset("pruning_b", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.prefill_hierarchy_aware
    assert preset.prune_config.morphology_coverage_rescue
    assert preset.prune_config.cross_scale_consolidation
    assert preset.prune_config.keep_ratio == 0.25
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is None


def test_pruning_b_latent_relay2_composes_pruning_and_visual_grounded_relay() -> None:
    """Given the combined arm, preserve Pruning B and relay2 as separate mechanisms."""
    preset = build_ablation_preset("pruning_b_latent_kv_relay2", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.keep_ratio == 0.25
    assert preset.prune_config.morphology_coverage_rescue
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay
    assert preset.reallocation.collect_visual_grounding


def test_pruning_b_latent_relay2_broadens_adaptive_step_selection() -> None:
    """Given the broad arm, keep Pruning B and lower relay2's robust gate."""
    preset = build_ablation_preset(
        "pruning_b_latent_kv_relay2_broad", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.keep_ratio == 0.25
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay
    assert preset.reallocation.visual_grounding_threshold == 0.5
    assert preset.reallocation.visual_grounding_top_k == 10


def test_pruning_b_relay2_all_agents_composes_pruning_and_three_handoffs() -> None:
    """Given the all-agent arm, enable B's visual selector and relay each handoff."""
    preset = build_ablation_preset(
        "pruning_b_latent_kv_relay2_all_agents", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.keep_ratio == 0.25
    assert not preset.prune_config.reasoner_only
    assert preset.prune_config.prune_hierarchy_all_roles
    assert preset.prune_config.prefill_hierarchy_aware
    assert preset.prune_config.morphology_coverage_rescue
    assert preset.prune_config.cross_scale_consolidation
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "evidence_planner",
        "navigator",
        "reasoner",
    )


def test_relay3_prunes_only_navigator_and_reasoner_and_relays_both() -> None:
    """Given Relay3, when built, then only evidence roles prune and relay."""
    preset = build_ablation_preset(
        "pruning_b_nav_reasoner_relay3_dual", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.roles == ("navigator", "reasoner")
    assert preset.reallocation is not None
    assert preset.reallocation.visual_contrast_latent_kv_relay
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )


def test_reasoner_relay3_does_not_prune_navigator_and_relays_both() -> None:
    """Reasoner-only pruning retains dual-stage visual-context relay."""
    preset = build_ablation_preset(
        "pruning_b_reasoner_relay3_dual", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.roles == ("reasoner",)
    assert preset.reallocation is not None
    assert preset.reallocation.visual_contrast_latent_kv_relay
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )


def test_reasoner_relay2_uses_reasoner_only_pruning_and_dual_handoffs() -> None:
    """Relay2 combination prunes only Reasoner and relays at both handoffs."""
    preset = build_ablation_preset(
        "pruning_b_reasoner_relay2_dual", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )


def test_reasoner_pruning_with_triple_relay_covers_all_handoffs() -> None:
    """Triple Relay2 includes Planner→Navigator while pruning only Reasoner."""
    preset = build_ablation_preset(
        "pruning_b_reasoner_relay2_triple", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "evidence_planner",
        "navigator",
        "reasoner",
    )


def test_paired_repeat_variants_match_original_dual_and_triple_configs() -> None:
    """Repeat run labels isolate output without changing either experiment."""
    for original, paired in (
        ("pruning_b_reasoner_relay2_dual", "pruning_b_reasoner_relay2_dual_paired"),
        ("pruning_b_reasoner_relay2_triple", "pruning_b_reasoner_relay2_triple_paired"),
    ):
        original_preset = build_ablation_preset(original, latent_steps=10)
        paired_preset = build_ablation_preset(paired, latent_steps=10)
        assert paired_preset.prune_config == original_preset.prune_config
        assert paired_preset.reallocation == original_preset.reallocation


def test_relay3_selects_latent_rows_closer_to_visual_than_text_keys() -> None:
    """Given visual/text keys, when contrasted, then only visual rows pass."""
    class Layer:
        def __init__(self, keys: torch.Tensor) -> None:
            self.keys = keys
            self.values = keys.clone()

    class Cache:
        def __init__(self, keys: torch.Tensor) -> None:
            self.layers = (Layer(keys),)

    # [batch=1, heads=1, sequence=(visual,text,latent0,latent1), dim=2]
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]]]])
    selected = select_visual_contrast_steps(
        Cache(keys),
        latent_steps=2,
        vision_columns=torch.tensor([0]),
        text_start=0,
    )

    assert selected.tolist() == [0]


def test_dual_relay_targets_only_pathology_evidence_handoffs() -> None:
    """Given dual relay, skip Planner and relay both evidence-bearing senders."""
    relay = build_ablation_preset("latent_kv_relay2_dual", latent_steps=10)
    both = build_ablation_preset(
        "pruning_b_latent_kv_relay2_dual", latent_steps=10
    )

    assert relay.reallocation is not None
    assert relay.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )
    assert both.prune_config is not None
    assert both.prune_config.reasoner_only
    assert both.reallocation is not None
    assert both.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )


def test_cross_scale_router_combines_pruning_b_with_reasoner_to_answerer_relay() -> None:
    """Given the new method, retain B and relay only visual-grounded Reasoner rows."""
    preset = build_ablation_preset(
        "pruning_b_cross_scale_relay2", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "reasoner",
    )


def test_pruning_a_dual_both_composes_hierarchical_pruning_and_dual_relay() -> None:
    """Given Pruning A plus Dual Both, preserve both independent mechanisms."""
    preset = build_ablation_preset(
        "pruning_a_latent_kv_relay2_dual", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.keep_ratio == 0.50
    assert preset.prune_config.hierarchy_aware
    assert preset.prune_config.prefill_hierarchy_aware
    assert preset.prune_config.reasoner_only
    assert preset.reallocation is not None
    assert preset.reallocation.visual_grounded_latent_kv_relay
    assert preset.reallocation.visual_grounded_latent_kv_relay_stages == (
        "navigator",
        "reasoner",
    )


def test_spectral_energy_budget_keeps_more_information_rich_features() -> None:
    """Given equal token counts, a flatter spectrum receives a larger budget."""
    redundant = torch.diag(torch.tensor([10.0, 0.1, 0.1, 0.1]))
    information_rich = torch.eye(4)

    redundant_ratio = spectral_energy_keep_ratio(
        redundant, energy_ratio=0.90, min_keep_ratio=0.25, max_keep_ratio=0.50
    )
    rich_ratio = spectral_energy_keep_ratio(
        information_rich, energy_ratio=0.90, min_keep_ratio=0.25, max_keep_ratio=0.50
    )

    assert redundant_ratio == 0.25
    assert rich_ratio == 0.50


def test_pruning_adaptive_uses_spectral_budget_with_wsi_safeguards() -> None:
    """Given adaptive pruning, retain E-AdaPrune budgeting and WSI constraints."""
    preset = build_ablation_preset("pruning_adaptive", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.keep_ratio == 1.0
    assert preset.prune_config.spectral_energy_ratio == 0.95
    assert preset.prune_config.spectral_min_keep_ratio == 0.25
    assert preset.prune_config.spectral_max_keep_ratio == 0.50
    assert preset.prune_config.prefill_hierarchy_aware
    assert preset.prune_config.morphology_coverage_rescue
    assert preset.prune_config.cross_scale_consolidation
    assert preset.prune_config.reasoner_only


def test_e_ada_prune_pathology_variant_uses_paper_threshold_and_dual_relay2() -> None:
    """Given E-AdaPrune, retain its pathology safeguards and dual Relay2 config."""
    relay = build_ablation_preset("latent_kv_relay2_dual", latent_steps=10)
    preset = build_ablation_preset(
        "pruning_eadaprune_pathology_latent_kv_relay2_dual", latent_steps=10
    )

    assert preset.prune_config is not None
    assert preset.prune_config.spectral_energy_ratio == 0.99
    assert preset.prune_config.spectral_min_keep_ratio == 0.25
    assert preset.prune_config.spectral_max_keep_ratio == 0.50
    assert preset.prune_config.prefill_hierarchy_aware
    assert preset.prune_config.morphology_coverage_rescue
    assert preset.prune_config.cross_scale_consolidation
    assert preset.prune_config.reasoner_only
    assert preset.reallocation == relay.reallocation


def test_atp_sap_pathology_proxy_uses_spatial_scaffold_and_dual_relay2() -> None:
    """Given frozen Qwen, use per-layer ATP-SAP with pathology and dual Relay2."""
    preset = build_ablation_preset(
        "pruning_atp_sap_pathology_latent_kv_relay2_dual", latent_steps=10
    )
    relay = build_ablation_preset("latent_kv_relay2_dual", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.atp_sap
    assert preset.prune_config.atp_all_layers
    assert preset.prune_config.reasoner_only
    assert preset.prune_config.keep_ratio is None
    assert preset.prune_config.probe_layer is None
    assert preset.prune_config.morphology_coverage_rescue
    assert preset.prune_config.cross_scale_consolidation
    assert preset.reallocation == relay.reallocation


def test_factorial_variants_cover_all_justified_stage_combinations() -> None:
    """Given the sweep, expose every Prune-R/Nav-relay/Answer-relay setting."""
    expected = {
        "base": (False, ()),
        "pruning_b": (True, ()),
        "latent_kv_relay2_navigator": (False, ("navigator",)),
        "latent_kv_relay2": (False, ("reasoner",)),
        "latent_kv_relay2_dual": (False, ("navigator", "reasoner")),
        "pruning_b_latent_kv_relay2_navigator": (True, ("navigator",)),
        "pruning_b_latent_kv_relay2": (True, ("reasoner",)),
        "pruning_b_latent_kv_relay2_dual": (True, ("navigator", "reasoner")),
    }

    for name, (pruned, stages) in expected.items():
        preset = build_ablation_preset(name, latent_steps=10)
        assert (preset.prune_config is not None) is pruned
        actual_stages = (
            preset.reallocation.visual_grounded_latent_kv_relay_stages
            if preset.reallocation is not None
            else ()
        )
        assert actual_stages == stages


def test_pruning_c_preserves_parent_token_at_selected_child_location() -> None:
    """Given a selected child token, its mapped parent-scale context survives."""
    keep = np.zeros(32, dtype=bool)
    keep[16] = True
    score = np.zeros(32, dtype=np.float64)
    score[16] = 1.0

    expanded = relation_spatial_context_keep(
        keep=keep,
        score=score,
        roi_of_token=np.asarray([0] * 16 + [1] * 16, dtype=np.int64),
        parent_roi=np.asarray([-1, 0], dtype=np.int64),
        roi_grids=((4, 4), (4, 4)),
        roi_boxes=((0, 0, 400, 400), (100, 100, 100, 100)),
        radius=0,
        per_parent_cap=1,
    )

    assert expanded[16]
    assert expanded[5], "child top-left projects to parent grid cell (1, 1)"
    assert int(expanded.sum()) == 2


def test_pruning_c_enables_coordinate_linked_parent_context() -> None:
    """Given Pruning C, relation-preserving context is active at B's budget."""
    preset = build_ablation_preset("pruning_c", latent_steps=10)

    assert preset.prune_config is not None
    assert preset.prune_config.parent_spatial_context
    assert preset.prune_config.parent_spatial_radius == 1
    assert preset.prune_config.keep_ratio == 0.25
    assert preset.prune_config.reasoner_only


def test_pruning_c_swaps_parent_context_into_fixed_budget() -> None:
    """Given Pruning C's fixed budget, child context replaces an unrelated token."""
    args = SimpleNamespace(
        prune_every=1,
        prune_warmup=0,
        prune_ema_lambda=0.5,
        kv_prune="topk",
        kv_prune_b_span=1,
        prune_keep_ratio=0.125,
        prune_event_steps=(1,),
        prune_hierarchy_aware=True,
        prune_parent_spatial_context=True,
        prune_parent_spatial_radius=0,
    )
    controller = PruneController(args)
    controller.start(
        grids=torch.tensor([[1, 8, 8], [1, 8, 8]]),
        vis_idx=torch.arange(32),
        sms=2,
        images=[],
        q2v={},
        l_mid_layers=[],
        image_parent_indices=(-1, 0),
        image_boxes=((0, 0, 400, 400), (100, 100, 100, 100)),
    )
    controller.s[16] = 1.0

    keep = controller.maybe_keep_mask(0)

    assert keep is not None
    assert keep[16]
    assert keep[5]
    assert int(keep.sum()) == 4


def main() -> None:
    """Verify that Pruning A enables the intended fixed-budget hierarchy path."""
    preset = build_ablation_preset("pruning_a", latent_steps=10)
    config = preset.prune_config
    assert config is not None
    assert config.prefill_hierarchy_aware
    assert config.hierarchy_aware
    assert config.cross_scale_consolidation
    assert config.keep_ratio == 0.50
    assert config.event_steps == (1,)
    assert config.reasoner_only
    assert config.one_shot
    assert preset.reallocation is None
    print("Pruning A preset contract: PASS")


if __name__ == "__main__":
    main()
