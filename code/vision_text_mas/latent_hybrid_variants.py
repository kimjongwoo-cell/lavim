"""Native-hybrid latent rollout switches shared by the HF backbone and vLLM readout."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Generator, Protocol

import torch

from vision_text_mas.latent_batch_reallocation import TerminalReallocationConfig


@dataclass(frozen=True, slots=True)
class NativeHybridPruneConfig:
    """One visual-KV top-k event per agent's latent rollout by default."""

    mode: str = "topk"
    tokens_per_image: int = 32
    keep_ratio: float | None = None
    neighborhood_radius: int = 1
    every_latent_steps: int = 5
    event_steps: tuple[int, ...] | None = None
    dynamic_ratio: bool = False
    warmup_steps: int = 0
    stream_rel: bool = False
    recompute_saliency: bool = False
    saliency_reduce: str = "mean"
    stream_mode: str = "snap"
    drop_per_event: int = 64
    morphology_aware: bool = False
    morphology_quota: int = 1
    reasoner_only: bool = False
    relevance_weight: float = 1.0
    focus_weight: float = 0.2
    reduce_head_first: bool = False
    rebuild_eviction_score: bool = False
    relevance_precise: bool = False
    background_correct: bool = False
    one_shot: bool = False
    probe_layer: int | None = None
    dynamic_step_events: bool = False
    event_probes_only: bool = False
    first_event_keep_ratio: float | None = None
    stable_streaming: bool = False
    stable_low_steps: int = 3
    stable_observe_every: int = 1
    stable_single_event: bool = False
    stable_dynamic_survivors: bool = False
    stable_dynamic_min_keep: float = 0.10
    stable_dynamic_max_keep: float = 0.45
    stable_visual_only_scores: bool = False
    stable_event_at_end: bool = False
    hierarchy_aware: bool = False
    prefill_hierarchy_aware: bool = False
    prefill_significance_sigma: float | None = None
    prefill_observe_blocks: int = 4
    step_keep_ratios: tuple[float, ...] | None = None
    relative_saliency_floor: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"safe", "topk"}:
            raise ValueError("prune mode must be safe or topk")
        if self.tokens_per_image < 1:
            raise ValueError("prune tokens per image must be positive")
        if self.keep_ratio is not None and not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("prune keep ratio must be in (0, 1]")
        if self.neighborhood_radius < 0:
            raise ValueError("prune neighborhood radius must be non-negative")
        if self.every_latent_steps < 1:
            raise ValueError("prune interval must be positive")
        if self.event_steps is not None and (
            not self.event_steps
            or any(step < 1 for step in self.event_steps)
            or tuple(sorted(set(self.event_steps))) != self.event_steps
        ):
            raise ValueError("prune event steps must be sorted positive unique integers")
        if self.warmup_steps < 0:
            raise ValueError("prune warmup must be non-negative")
        if self.saliency_reduce not in {"mean", "max", "contrast"}:
            raise ValueError("saliency reduce must be mean, max, or contrast")
        if self.stream_mode not in {"snap", "gradual"}:
            raise ValueError("prune stream mode must be snap or gradual")
        if self.step_keep_ratios is not None and (
            not self.step_keep_ratios
            or any(not 0.0 < ratio <= 1.0 for ratio in self.step_keep_ratios)
            or tuple(sorted(self.step_keep_ratios, reverse=True)) != self.step_keep_ratios
        ):
            raise ValueError("step keep ratios must be non-increasing values in (0, 1]")
        if self.relative_saliency_floor is not None and not 0.0 < self.relative_saliency_floor < 1.0:
            raise ValueError("relative saliency floor must be in (0, 1)")
        if self.prefill_significance_sigma is not None and self.prefill_significance_sigma < 0.0:
            raise ValueError("prefill significance sigma must be non-negative")
        if self.prefill_observe_blocks < 1:
            raise ValueError("prefill observation blocks must be positive")
        if self.drop_per_event < 1:
            raise ValueError("prune drop per event must be positive")
        if self.morphology_quota < 1:
            raise ValueError("morphology quota must be positive")
        if self.relevance_weight < 0.0 or self.focus_weight < 0.0:
            raise ValueError("pruning score weights must be non-negative")
        if self.probe_layer is not None and self.probe_layer < 0:
            raise ValueError("pruning probe layer must be non-negative")
        if self.first_event_keep_ratio is not None and not 0.0 < self.first_event_keep_ratio <= 1.0:
            raise ValueError("first-event keep ratio must be in (0, 1]")
        if self.stable_low_steps < 1:
            raise ValueError("stable low-step threshold must be positive")
        if self.stable_observe_every < 1:
            raise ValueError("stable observation interval must be positive")
        if not 0.0 < self.stable_dynamic_min_keep <= self.stable_dynamic_max_keep <= 1.0:
            raise ValueError("stable dynamic keep bounds must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class _BackbonePruneArgs:
    """Subset of the preserved pruning contract required by ``PruneController``."""

    kv_prune: str
    prune_every: int
    kv_prune_b_span: int
    kv_prune_radius: int
    prune_keep_ratio: float | None = None
    prune_event_steps: tuple[int, ...] | None = None
    prune_dynamic_ratio: bool = False
    prune_warmup: int = 0
    prune_ema_lambda: float = 0.5
    kv_prune_tissue_th: float = 0.1
    kv_prune_sink_frac: float = 0.05
    prune_dump: bool = False
    prune_stream_rel: bool = False
    saliency_lam_r: float = 1.0
    saliency_lam_s: float = 0.2
    prune_stream_mode: str = "snap"
    prune_drop_per_event: int = 64
    stream_saliency_reduce: str = "mean"
    stream_reduce_head_first: bool = False
    stream_rebuild_ec: bool = False
    stream_rel_precise: bool = False
    saliency_bg_correct: bool = False
    prune_prefill: str = "none"
    stream_recompute_saliency: bool = False
    prune_morphology_aware: bool = False
    prune_morphology_quota: int = 1
    prune_reasoner_only: bool = False
    prune_one_shot: bool = False
    prune_probe_layer: int | None = None
    prune_step_adaptive: bool = False
    prune_event_probes_only: bool = False
    prune_first_event_keep_ratio: float | None = None
    prune_stable_streaming: bool = False
    prune_stable_low_steps: int = 3
    prune_stable_observe_every: int = 1
    prune_stable_single_event: bool = False
    prune_stable_dynamic_survivors: bool = False
    prune_stable_dynamic_min_keep: float = 0.10
    prune_stable_dynamic_max_keep: float = 0.45
    prune_stable_visual_only_scores: bool = False
    prune_stable_event_at_end: bool = False
    prune_hierarchy_aware: bool = False
    prune_prefill_hierarchy_aware: bool = False
    prune_prefill_significance_sigma: float | None = None
    prune_prefill_observe_blocks: int = 4
    prune_step_keep_ratios: tuple[float, ...] | None = None
    prune_relative_saliency_floor: float | None = None


class _VariantBackbone(Protocol):
    _prune_args: _BackbonePruneArgs
    _latent_reallocation: TerminalReallocationConfig | None
    device: str


def attach_native_hybrid_variants(
    backbone: _VariantBackbone,
    *,
    prune_config: NativeHybridPruneConfig | None,
    reallocation: TerminalReallocationConfig | None,
) -> None:
    """Attach explicit ablation state to the HF latent rollout only."""
    if prune_config is not None:
        backbone._prune_args = _BackbonePruneArgs(
            kv_prune=prune_config.mode,
            prune_every=prune_config.every_latent_steps,
            kv_prune_b_span=prune_config.tokens_per_image,
            prune_keep_ratio=prune_config.keep_ratio,
            prune_event_steps=prune_config.event_steps,
            prune_dynamic_ratio=prune_config.dynamic_ratio,
            kv_prune_radius=prune_config.neighborhood_radius,
            prune_warmup=prune_config.warmup_steps,
            prune_stream_rel=prune_config.stream_rel,
            stream_recompute_saliency=prune_config.recompute_saliency,
            stream_saliency_reduce=prune_config.saliency_reduce,
            prune_stream_mode=prune_config.stream_mode,
            prune_drop_per_event=prune_config.drop_per_event,
            prune_morphology_aware=prune_config.morphology_aware,
            prune_morphology_quota=prune_config.morphology_quota,
            prune_reasoner_only=prune_config.reasoner_only,
            saliency_lam_r=prune_config.relevance_weight,
            saliency_lam_s=prune_config.focus_weight,
            stream_reduce_head_first=prune_config.reduce_head_first,
            stream_rebuild_ec=prune_config.rebuild_eviction_score,
            stream_rel_precise=prune_config.relevance_precise,
            saliency_bg_correct=prune_config.background_correct,
            prune_one_shot=prune_config.one_shot,
            prune_probe_layer=prune_config.probe_layer,
            prune_step_adaptive=prune_config.dynamic_step_events,
            prune_event_probes_only=prune_config.event_probes_only,
            prune_first_event_keep_ratio=prune_config.first_event_keep_ratio,
            prune_stable_streaming=prune_config.stable_streaming,
            prune_stable_low_steps=prune_config.stable_low_steps,
            prune_stable_observe_every=prune_config.stable_observe_every,
            prune_stable_single_event=prune_config.stable_single_event,
            prune_stable_dynamic_survivors=prune_config.stable_dynamic_survivors,
            prune_stable_dynamic_min_keep=prune_config.stable_dynamic_min_keep,
            prune_stable_dynamic_max_keep=prune_config.stable_dynamic_max_keep,
            prune_stable_visual_only_scores=prune_config.stable_visual_only_scores,
            prune_stable_event_at_end=prune_config.stable_event_at_end,
            prune_hierarchy_aware=prune_config.hierarchy_aware,
            prune_prefill_hierarchy_aware=prune_config.prefill_hierarchy_aware,
            prune_prefill_significance_sigma=prune_config.prefill_significance_sigma,
            prune_prefill_observe_blocks=prune_config.prefill_observe_blocks,
            prune_step_keep_ratios=prune_config.step_keep_ratios,
            prune_relative_saliency_floor=prune_config.relative_saliency_floor,
        )
    backbone._latent_reallocation = reallocation


@contextlib.contextmanager
def latent_reallocation(
    backbone: _VariantBackbone,
    vision_columns: torch.Tensor,
) -> Generator[object | None, None, None]:
    """Reweight only image-conditioned HF latent steps; terminal vLLM stays native."""
    config = getattr(backbone, "_latent_reallocation", None)
    if config is None:
        yield None
        return
    from memory.reallocate import reallocating

    probe_mass = getattr(backbone, "_latent_reallocation_probe", None)
    role = getattr(backbone, "_latent_reallocation_role", "")
    alpha = config.alpha_for_role(role, probe_mass)
    if alpha == 0.0 and not config.adaptive:
        yield None
        return
    with reallocating(
        vision_columns,
        alpha,
        config.layers,
        device=backbone.device,
        vis_weight=getattr(backbone, "_reallocation_vision_weights", None),
        decode_only=False,
        relay_ratio=config.relay_ratio,
        relay_source=config.relay_source,
        relay_mode=config.relay_mode,
    ) as probe:
        yield probe
