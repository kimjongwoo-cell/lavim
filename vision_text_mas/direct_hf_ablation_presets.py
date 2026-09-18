"""Fixed, comparable pruning and reallocation presets for direct-HF Latent WSI-MAS."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, assert_never

from vision_text_mas.latent_batch_reallocation import TerminalReallocationConfig
from vision_text_mas.latent_hybrid_variants import NativeHybridPruneConfig


AblationName = Literal[
    "base", "pruning", "pruning_v2", "pruning_v3", "pruning_v4", "pruning_v5",
    "pruning_v6", "pruning_a", "pruning_b", "pruning_c", "pruning_adaptive",
    "pruning_sparsevlm",
    "pruning_sparsevlm_pathology",
    "reallocation", "reallocation_v2", "reallocation_a", "pathology_reallocation",
    "pathology_context_reallocation", "latent_kv_relay", "latent_kv_relay2", "both",
    "latent_kv_relay2_navigator", "latent_kv_relay2_dual",
    "pruning_b_latent_kv_relay2_navigator", "pruning_b_latent_kv_relay2_dual",
    "pruning_a_latent_kv_relay2_dual",
    "pruning_eadaprune_pathology_latent_kv_relay2_dual",
    "pruning_atp_sap_pathology_latent_kv_relay2_dual",
    "pruning_sparsevlm_latent_kv_relay2_dual",
    "pruning_sparsevlm_pathology_latent_kv_relay2_dual",
    "pruning_sparsevlm_latent_kv_relay2_planner_navigator",
    "pruning_atp_sap_latent_kv_relay2_planner_navigator",
    "pruning_b_latent_kv_relay2_all_agents",
    "pruning_b_nav_reasoner_relay3_dual",
    "pruning_b_reasoner_relay3_dual",
    "pruning_b_reasoner_relay2_dual",
    "pruning_b_reasoner_relay2_triple",
    "pruning_b_reasoner_relay2_dual_paired",
    "pruning_b_reasoner_relay2_triple_paired",
    "pruning_b_reasoner_context_relay3",
    "reasoner_context_relay3",
    "pruning_v3_reallocation", "pruning_b_reallocation_a",
    "pruning_b_latent_kv_relay2", "pruning_b_latent_kv_relay2_broad",
    "reallocation_v2_answerer_only",
    "pruning_v3_reallocation_answerer_only",
]


@dataclass(frozen=True, slots=True)
class DirectHfAblationPreset:
    """Explicit variant state attached only to the HF latent rollout."""

    name: AblationName
    prune_config: NativeHybridPruneConfig | None
    reallocation: TerminalReallocationConfig | None


def build_ablation_preset(
    name: AblationName, *, latent_steps: int = 5
) -> DirectHfAblationPreset:
    """Build one of the four fixed ablations under a common inference protocol."""
    # The previous safe/gradual policy waited for two latent steps and removed
    # only 64 visual columns per event.  On multi-ROI slides that leaves nearly
    # the full visual KV cache alive when the 2048-token answer decode begins.
    # Use the explicit per-ROI top-k floor at the first latent step instead:
    # this is still a pruning-only ablation, but has a bounded cache before the
    # long terminal decode.
    prune_config = NativeHybridPruneConfig(
        mode="topk",
        tokens_per_image=32,
        keep_ratio=0.10,
        neighborhood_radius=1,
        every_latent_steps=5,
        event_steps=(1, 3),
        dynamic_ratio=True,
        warmup_steps=0,
        stream_rel=False,
        recompute_saliency=True,
        saliency_reduce="max",
        stream_mode="snap",
        drop_per_event=64,
    )
    reallocation = TerminalReallocationConfig(
        alpha=0.15,
        first_layer=8,
        last_layer=20,
        # v9: reallocation lives at the handoff boundary only — the Reasoner
        # latent steps run untouched (v6-v8 showed per-step dosing is either
        # invisible or noise there), and the paper's claim is about the
        # receiving agent attending to surviving visual entries.
        adaptive=False,
        latent_roles=(),
        # v9: handoff-scoped relay at full strength — restores the surviving
        # visual entries' share from ~0.5% to ~20% of attention at the moment
        # the Answerer receives the cache (mechanism verified in logs).
        answerer_alpha=0.05,
        answerer_adaptive=False,
        answerer_prefill_only=True,
        relay_ratio=1.0,
        relay_source="key_norm",
        relay_mode="proportional",
    )
    receiver_reallocation = TerminalReallocationConfig(
        alpha=0.05,
        first_layer=8,
        last_layer=20,
        # The receiving Reasoner re-reads the Navigator's retained visual KV
        # together with its newly appended patch KV.  Planner/Navigator remain
        # control-only; Answerer receives the same visual relay at its handoff.
        latent_roles=("reasoner",),
        answerer_alpha=0.05,
        answerer_prefill_only=True,
        relay_ratio=1.0,
        relay_source="key_norm",
        relay_mode="proportional",
    )
    two_stage_pruning = NativeHybridPruneConfig(
        mode="topk",
        tokens_per_image=32,
        keep_ratio=None,
        neighborhood_radius=1,
        every_latent_steps=5,
        event_steps=None,
        dynamic_ratio=True,
        warmup_steps=0,
        stream_rel=False,
        recompute_saliency=True,
        saliency_reduce="max",
        stream_mode="snap",
        drop_per_event=64,
        reasoner_only=False,
        dynamic_step_events=True,
        event_probes_only=True,
        first_event_keep_ratio=0.50,
        probe_layer=24,
    )
    match name:
        case "base":
            return DirectHfAblationPreset(name, None, None)
        case "pruning":
            return DirectHfAblationPreset(name, prune_config, None)
        case "pruning_v2":
            return DirectHfAblationPreset(name, two_stage_pruning, None)
        case "pruning_v3":
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    keep_ratio=None,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    dynamic_ratio=False,
                    warmup_steps=0,
                    stream_rel=False,
                    recompute_saliency=True,
                    saliency_reduce="max",
                    stream_mode="snap",
                    drop_per_event=64,
                    # Planner/Navigator visual context is required to emit the
                    # one-shot navigation control JSON.  Start physical visual-KV
                    # selection when the Navigator has materialized pathology
                    # patches and the Reasoner begins evidence analysis.
                    reasoner_only=True,
                    probe_layer=24,
                    dynamic_step_events=False,
                    event_probes_only=True,
                    one_shot=True,
                    first_event_keep_ratio=None,
                    stable_streaming=False,
                    stable_dynamic_survivors=False,
                    stable_dynamic_min_keep=0.10,
                    stable_dynamic_max_keep=0.45,
                    morphology_aware=True,
                    morphology_quota=1,
                    hierarchy_aware=True,
                    prefill_hierarchy_aware=True,
                    prefill_significance_sigma=0.50,
                    prefill_observe_blocks=2,
                    relative_saliency_floor=0.50,
                ),
                None,
            )
        case "pruning_v3_reallocation":
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    keep_ratio=None,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    dynamic_ratio=False,
                    warmup_steps=0,
                    stream_rel=False,
                    recompute_saliency=True,
                    saliency_reduce="max",
                    stream_mode="snap",
                    drop_per_event=64,
                    reasoner_only=True,
                    probe_layer=24,
                    dynamic_step_events=False,
                    event_probes_only=True,
                    one_shot=True,
                    first_event_keep_ratio=None,
                    stable_streaming=False,
                    stable_dynamic_survivors=False,
                    stable_dynamic_min_keep=0.10,
                    stable_dynamic_max_keep=0.45,
                    morphology_aware=True,
                    morphology_quota=1,
                    hierarchy_aware=True,
                    prefill_hierarchy_aware=True,
                    prefill_significance_sigma=0.50,
                    prefill_observe_blocks=2,
                    relative_saliency_floor=0.50,
                ),
                receiver_reallocation,
            )
        case "pruning_v4":
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=16,
                    # Inspect evidence twice, but never force a token budget.
                    # Only tokens far below their patch's strongest current
                    # reasoning evidence are removed.
                    keep_ratio=None,
                    neighborhood_radius=0,
                    every_latent_steps=1,
                    event_steps=(2, 4),
                    dynamic_ratio=False,
                    warmup_steps=0,
                    stream_rel=False,
                    recompute_saliency=True,
                    saliency_reduce="max",
                    stream_mode="snap",
                    drop_per_event=64,
                    # The hierarchy exists only after Navigator has produced
                    # coarse/fine pathology patches. Mid/late Reasoner attention
                    # refines the visual memory twice before the final readout.
                    reasoner_only=True,
                    one_shot=False,
                    probe_layer=24,
                    dynamic_step_events=False,
                    event_probes_only=True,
                    stable_streaming=False,
                    stable_dynamic_survivors=False,
                    # Ranking is only among visual evidence. Restrict the
                    # selected-layer probe to visual keys instead of rebuilding
                    # a softmax over the full cumulative text/control cache.
                    stable_visual_only_scores=True,
                    morphology_aware=False,
                    hierarchy_aware=True,
                    prefill_hierarchy_aware=False,
                    relative_saliency_floor=0.50,
                ),
                None,
            )
        case "pruning_v5":
            # Promote the historically strong V2 Two-stage schedule under the
            # current step-5 protocol. Event placement remains relative to the
            # rollout horizon, so five latent steps compact at events 1 and 3.
            return DirectHfAblationPreset(name, two_stage_pruning, None)
        case "pruning_v6":
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=32,
                    keep_ratio=None,
                    neighborhood_radius=0,
                    every_latent_steps=1,
                    event_steps=None,
                    dynamic_ratio=True,
                    warmup_steps=0,
                    stream_rel=False,
                    recompute_saliency=True,
                    saliency_reduce="max",
                    stream_mode="snap",
                    drop_per_event=64,
                    reasoner_only=False,
                    probe_layer=24,
                    dynamic_step_events=True,
                    event_probes_only=False,
                    first_event_keep_ratio=0.50,
                    stable_streaming=True,
                    stable_low_steps=3,
                    stable_observe_every=1,
                    stable_single_event=False,
                    stable_dynamic_survivors=False,
                    morphology_aware=False,
                ),
                None,
            )
        case "pruning_a":
            # Pruning A: preserve the WSI coarse/fine evidence relation. The
            # hierarchy-aware prefill keeps a fixed visual budget and guarantees
            # an evidence floor for every image. Cross-scale consolidation then
            # swaps a retained x20 group that repeats its x5 parent for a dropped
            # x20 group carrying detail absent from that parent.
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    keep_ratio=0.50,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    dynamic_ratio=False,
                    warmup_steps=0,
                    stream_rel=False,
                    recompute_saliency=True,
                    saliency_reduce="max",
                    stream_mode="snap",
                    drop_per_event=64,
                    reasoner_only=True,
                    one_shot=True,
                    probe_layer=24,
                    hierarchy_aware=True,
                    prefill_hierarchy_aware=True,
                    prefill_significance_sigma=None,
                    prefill_observe_blocks=4,
                    cross_scale_consolidation=True,
                    cross_scale_margin=0.05,
                ),
                None,
            )
        case "pruning_adaptive":
            # E-AdaPrune budgeting: spectral rank selects how many tokens are
            # retained per case; the existing WSI selector decides which tokens.
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    # Prefill uses the spectral override below. Keeping 100% at
                    # the latent event prevents a second fixed-budget cut.
                    keep_ratio=1.0,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    recompute_saliency=True,
                    saliency_reduce="max",
                    reasoner_only=True,
                    one_shot=True,
                    probe_layer=24,
                    hierarchy_aware=True,
                    prefill_hierarchy_aware=True,
                    prefill_observe_blocks=4,
                    spectral_energy_ratio=0.95,
                    spectral_min_keep_ratio=0.25,
                    spectral_max_keep_ratio=0.50,
                    morphology_coverage_rescue=True,
                    cross_scale_consolidation=True,
                    cross_scale_margin=0.05,
                ),
                None,
            )
        case "pruning_sparsevlm":
            # SparseVLM at the Reasoner boundary: question-attending text rows
            # rate visual KV, and the attention-matrix rank determines how many
            # low-priority rows to remove.
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    keep_ratio=None,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    reasoner_only=True,
                    one_shot=True,
                    probe_layer=24,
                    query_adaptive=True,
                    query_adaptive_lambda=0.5,
                ),
                None,
            )
        case "pruning_sparsevlm_pathology":
            # SparseVLM's query/rank-adaptive budget plus a WSI local/scale
            # context scaffold; token scores and pruning budget stay training-free.
            base = build_ablation_preset("pruning_sparsevlm", latent_steps=latent_steps)
            assert base.prune_config is not None
            return DirectHfAblationPreset(
                name,
                # Match Pruning-B's 25% visual-KV budget while retaining the
                # rank-adaptive SparseVLM rule (lambda=.75 => ~25% when rank << V).
                replace(
                    base.prune_config,
                    query_adaptive_lambda=0.75,
                    query_pathology_context=True,
                ),
                None,
            )
        case "pruning_b":
            # Pruning B: at a matched, smaller visual-KV budget, retain a
            # representative set of morphology and preserve the parent x5
            # context of retained x20 detail.  The two rescue passes exchange
            # groups only within an image, so they cannot inflate KV length.
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    keep_ratio=0.25,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    dynamic_ratio=False,
                    warmup_steps=0,
                    stream_rel=False,
                    recompute_saliency=True,
                    saliency_reduce="max",
                    stream_mode="snap",
                    drop_per_event=64,
                    reasoner_only=True,
                    one_shot=True,
                    probe_layer=24,
                    hierarchy_aware=True,
                    prefill_hierarchy_aware=True,
                    prefill_significance_sigma=None,
                    prefill_observe_blocks=4,
                    morphology_coverage_rescue=True,
                    cross_scale_consolidation=True,
                    cross_scale_margin=0.05,
                ),
                None,
            )
        case "pruning_c":
            # Pruning C: retain Pruning B's fixed global budget, then swap in
            # spatially aligned parent-scale context for surviving x20 detail.
            # The linkage is derived solely from Navigator's real level-0 boxes.
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    keep_ratio=0.25,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    dynamic_ratio=False,
                    warmup_steps=0,
                    stream_rel=False,
                    recompute_saliency=True,
                    saliency_reduce="max",
                    stream_mode="snap",
                    drop_per_event=64,
                    reasoner_only=True,
                    one_shot=True,
                    probe_layer=24,
                    hierarchy_aware=True,
                    prefill_hierarchy_aware=True,
                    prefill_significance_sigma=None,
                    prefill_observe_blocks=4,
                    morphology_coverage_rescue=True,
                    cross_scale_consolidation=True,
                    cross_scale_margin=0.05,
                    parent_spatial_context=True,
                    parent_spatial_radius=1,
                ),
                None,
            )
        case "reallocation":
            return DirectHfAblationPreset(name, None, reallocation)
        case "reallocation_v2":
            return DirectHfAblationPreset(name, None, receiver_reallocation)
        case "reallocation_a":
            return DirectHfAblationPreset(
                name,
                None,
                TerminalReallocationConfig(
                    alpha=0.05,
                    first_layer=8,
                    last_layer=20,
                    latent_roles=(),
                    # The Reasoner supplies a focus map; only the receiving
                    # Answerer is reallocated, over focus plus local tissue context.
                    answerer_alpha=0.05,
                    answerer_prefill_only=True,
                    relay_ratio=1.0,
                    relay_source="sender_priority",
                    relay_mode="proportional",
                    pathology_aware=True,
                    spatial_context_relay=True,
                    parent_anchor_strength=0.35,
                ),
            )
        case "pruning_b_reallocation_a":
            pruning_b = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            reallocation_a = build_ablation_preset(
                "reallocation_a", latent_steps=latent_steps
            )
            assert pruning_b.prune_config is not None
            assert reallocation_a.reallocation is not None
            return DirectHfAblationPreset(
                name,
                pruning_b.prune_config,
                reallocation_a.reallocation,
            )
        case "reallocation_v2_answerer_only":
            # Sender-relay experiment §5: same receiver config, but no
            # Reasoner-boundary intervention — only the Answerer prefill.
            return DirectHfAblationPreset(
                name, None, replace(receiver_reallocation, latent_roles=())
            )
        case "pruning_v3_reallocation_answerer_only":
            combo = build_ablation_preset(
                "pruning_v3_reallocation", latent_steps=latent_steps
            )
            return DirectHfAblationPreset(
                name,
                combo.prune_config,
                replace(receiver_reallocation, latent_roles=()),
            )
        case "pathology_reallocation":
            return DirectHfAblationPreset(
                name,
                None,
                TerminalReallocationConfig(
                    alpha=0.05,
                    first_layer=8,
                    last_layer=20,
                    latent_roles=("reasoner",),
                    answerer_alpha=0.05,
                    answerer_prefill_only=True,
                    relay_ratio=1.0,
                    relay_source="key_norm",
                    relay_mode="proportional",
                    pathology_aware=True,
                    parent_anchor_strength=0.35,
                    fine_evidence_boost=0.50,
                ),
            )
        case "pathology_context_reallocation":
            return DirectHfAblationPreset(
                name,
                None,
                TerminalReallocationConfig(
                    # Keep the established V2 redistribution in Reasoner and
                    # Answerer; the only addition is a question-conditioned
                    # fine-to-parent context recipient at the Answerer handoff.
                    alpha=0.05,
                    first_layer=8,
                    last_layer=20,
                    latent_roles=("reasoner",),
                    answerer_alpha=0.05,
                    answerer_prefill_only=True,
                    relay_ratio=1.0,
                    relay_source="key_norm",
                    relay_mode="proportional",
                    pathology_aware=True,
                    hierarchy_context_relay=True,
                    parent_anchor_strength=0.35,
                ),
            )
        case "latent_kv_relay":
            return DirectHfAblationPreset(
                name, None, TerminalReallocationConfig(
                    alpha=0.05, first_layer=8, last_layer=20, latent_roles=(),
                    # The Answerer reads the appended rows through ordinary
                    # attention.  Do not force eager terminal reallocation:
                    # that materializes a 12k attention matrix for no relay gain.
                    answerer_alpha=None, answerer_prefill_only=True,
                    relay_ratio=1.0, relay_source="key_norm", relay_mode="proportional",
                    latent_kv_relay=True,
                )
            )
        case "latent_kv_relay2":
            return DirectHfAblationPreset(
                name, None, TerminalReallocationConfig(
                    alpha=0.05, first_layer=8, last_layer=20, latent_roles=(),
                    answerer_alpha=None, answerer_prefill_only=True,
                    relay_ratio=1.0, relay_source="key_norm", relay_mode="proportional",
                    visual_grounded_latent_kv_relay=True,
                    collect_visual_grounding=True,
                    visual_grounding_threshold=1.0,
                    visual_grounding_top_k=10,
                )
            )
        case "latent_kv_relay2_dual":
            relay = build_ablation_preset("latent_kv_relay2", latent_steps=latent_steps)
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                None,
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay_stages=("navigator", "reasoner"),
                ),
            )
        case "latent_kv_relay2_navigator":
            relay = build_ablation_preset("latent_kv_relay2", latent_steps=latent_steps)
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                None,
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay_stages=("navigator",),
                ),
            )
        case "pruning_b_latent_kv_relay2":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset("latent_kv_relay2", latent_steps=latent_steps)
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                relay.reallocation,
            )
        case "pruning_b_reasoner_context_relay3":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset("latent_kv_relay2", latent_steps=latent_steps)
            assert pruning.prune_config is not None
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay=False,
                    visual_contrast_latent_kv_relay=False,
                    pathology_context_latent_kv_relay=True,
                    visual_grounded_latent_kv_relay_stages=("reasoner",),
                ),
            )
        case "reasoner_context_relay3":
            relay = build_ablation_preset("latent_kv_relay2", latent_steps=latent_steps)
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                None,
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay=False,
                    visual_contrast_latent_kv_relay=False,
                    pathology_context_latent_kv_relay=True,
                    visual_grounded_latent_kv_relay_stages=("reasoner",),
                ),
            )
        case "pruning_b_latent_kv_relay2_dual":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset("latent_kv_relay2_dual", latent_steps=latent_steps)
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                relay.reallocation,
            )
        case "pruning_a_latent_kv_relay2_dual":
            pruning = build_ablation_preset("pruning_a", latent_steps=latent_steps)
            relay = build_ablation_preset("latent_kv_relay2_dual", latent_steps=latent_steps)
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                relay.reallocation,
            )
        case "pruning_eadaprune_pathology_latent_kv_relay2_dual":
            pruning = build_ablation_preset(
                "pruning_adaptive", latent_steps=latent_steps
            )
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            return DirectHfAblationPreset(
                name,
                replace(pruning.prune_config, spectral_energy_ratio=0.99),
                relay.reallocation,
            )
        case "pruning_atp_sap_pathology_latent_kv_relay2_dual":
            # ATP's learned thresholds are unavailable here. Use the training-
            # free per-layer SAP proxy, pathology rescues, and both Relay2 paths.
            pruning = build_ablation_preset(
                "pruning_atp_sap_latent_kv_relay2_planner_navigator",
                latent_steps=latent_steps,
            )
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            return DirectHfAblationPreset(
                name,
                replace(
                    pruning.prune_config,
                    reasoner_only=True,
                    probe_layer=None,
                    atp_all_layers=True,
                    morphology_coverage_rescue=True,
                    cross_scale_consolidation=True,
                    cross_scale_margin=0.05,
                ),
                relay.reallocation,
            )
        case "pruning_sparsevlm_latent_kv_relay2_dual":
            pruning = build_ablation_preset(
                "pruning_sparsevlm", latent_steps=latent_steps
            )
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                relay.reallocation,
            )
        case "pruning_sparsevlm_pathology_latent_kv_relay2_dual":
            pruning = build_ablation_preset(
                "pruning_sparsevlm_pathology", latent_steps=latent_steps
            )
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                relay.reallocation,
            )
        case "pruning_sparsevlm_latent_kv_relay2_planner_navigator":
            # Keep QGAP's existing Reasoner pruning and additionally apply the
            # same question-guided KV rule to Planner and Navigator visual inputs.
            # Relay each visual-grounded latent state at Planner→Navigator,
            # Navigator→Reasoner, and Reasoner→Answerer boundaries.
            pruning = build_ablation_preset(
                "pruning_sparsevlm", latent_steps=latent_steps
            )
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                replace(pruning.prune_config, reasoner_only=False),
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay_stages=(
                        "evidence_planner",
                        "navigator",
                        "reasoner",
                    ),
                ),
            )
        case "pruning_b_latent_kv_relay2_all_agents":
            # Apply Pruning B's fixed 25% morphology-preserving visual budget
            # to every agent, and relay grounded latent KV across all handoffs.
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                replace(
                    pruning.prune_config,
                    reasoner_only=False,
                    prune_hierarchy_all_roles=True,
                ),
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay_stages=(
                        "evidence_planner",
                        "navigator",
                        "reasoner",
                    ),
                ),
            )
        case "pruning_b_nav_reasoner_relay3_dual":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                replace(
                    pruning.prune_config,
                    reasoner_only=False,
                    roles=("navigator", "reasoner"),
                    prune_hierarchy_all_roles=True,
                ),
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay=False,
                    collect_visual_grounding=False,
                    visual_contrast_latent_kv_relay=True,
                    visual_grounded_latent_kv_relay_stages=(
                        "navigator",
                        "reasoner",
                    ),
                ),
            )
        case "pruning_b_reasoner_relay3_dual":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                replace(
                    pruning.prune_config,
                    reasoner_only=False,
                    roles=("reasoner",),
                    prune_hierarchy_all_roles=True,
                ),
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay=False,
                    collect_visual_grounding=False,
                    visual_contrast_latent_kv_relay=True,
                    visual_grounded_latent_kv_relay_stages=(
                        "navigator",
                        "reasoner",
                    ),
                ),
            )
        case "pruning_b_reasoner_relay2_dual" | "pruning_b_reasoner_relay2_dual_paired":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                relay.reallocation,
            )
        case "pruning_b_reasoner_relay2_triple" | "pruning_b_reasoner_relay2_triple_paired":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert pruning.prune_config is not None
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay_stages=(
                        "evidence_planner",
                        "navigator",
                        "reasoner",
                    ),
                ),
            )
        case "pruning_atp_sap_latent_kv_relay2_planner_navigator":
            # ATP-LLaVA's learned threshold heads are unavailable for frozen
            # Qwen. Use its self/cross-modal SAP scores with an adaptive
            # score-statistic threshold and spatial scaffold instead.
            relay = build_ablation_preset(
                "latent_kv_relay2_dual", latent_steps=latent_steps
            )
            assert relay.reallocation is not None
            return DirectHfAblationPreset(
                name,
                NativeHybridPruneConfig(
                    mode="topk",
                    tokens_per_image=8,
                    keep_ratio=None,
                    neighborhood_radius=0,
                    every_latent_steps=latent_steps,
                    event_steps=(1,),
                    reasoner_only=False,
                    one_shot=True,
                    probe_layer=24,
                    atp_sap=True,
                ),
                replace(
                    relay.reallocation,
                    visual_grounded_latent_kv_relay_stages=(
                        "evidence_planner",
                        "navigator",
                        "reasoner",
                    ),
                ),
            )
        case "pruning_b_latent_kv_relay2_navigator":
            pruning = build_ablation_preset("pruning_b", latent_steps=latent_steps)
            relay = build_ablation_preset(
                "latent_kv_relay2_navigator", latent_steps=latent_steps
            )
            return DirectHfAblationPreset(
                name,
                pruning.prune_config,
                relay.reallocation,
            )
        case "pruning_b_latent_kv_relay2_broad":
            combined = build_ablation_preset(
                "pruning_b_latent_kv_relay2", latent_steps=latent_steps
            )
            assert combined.prune_config is not None
            assert combined.reallocation is not None
            return DirectHfAblationPreset(
                name,
                combined.prune_config,
                replace(combined.reallocation, visual_grounding_threshold=0.5),
            )
        case "both":
            return DirectHfAblationPreset(name, prune_config, reallocation)
        case unreachable:
            assert_never(unreachable)
