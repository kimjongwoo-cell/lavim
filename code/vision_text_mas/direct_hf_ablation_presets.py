"""Fixed, comparable pruning and reallocation presets for direct-HF Latent WSI-MAS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

from vision_text_mas.latent_batch_reallocation import TerminalReallocationConfig
from vision_text_mas.latent_hybrid_variants import NativeHybridPruneConfig


AblationName = Literal[
    "base", "pruning", "pruning_v2", "pruning_v3", "pruning_v4", "pruning_v5",
    "pruning_v6",
    "reallocation", "reallocation_v2", "pathology_reallocation",
    "pathology_context_reallocation", "latent_kv_relay", "both",
    "pruning_v3_reallocation",
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
        case "reallocation":
            return DirectHfAblationPreset(name, None, reallocation)
        case "reallocation_v2":
            return DirectHfAblationPreset(name, None, receiver_reallocation)
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
                name,
                None,
                TerminalReallocationConfig(
                    alpha=0.05,
                    first_layer=8,
                    last_layer=20,
                    latent_roles=(),
                    answerer_alpha=0.05,
                    answerer_prefill_only=True,
                    relay_ratio=1.0,
                    relay_source="key_norm",
                    relay_mode="proportional",
                    latent_kv_relay=True,
                ),
            )
        case "both":
            return DirectHfAblationPreset(name, prune_config, reallocation)
        case unreachable:
            assert_never(unreachable)
