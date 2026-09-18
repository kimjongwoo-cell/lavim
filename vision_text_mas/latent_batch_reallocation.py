"""Decode-only eager-attention boundary for the reallocation ablation."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import torch

from backbone.qwen3vl import Qwen3VLBackbone


@dataclass(frozen=True, slots=True)
class TerminalReallocationConfig:
    """Original 0717 decode-only S8 attention-reallocation hyperparameters."""

    alpha: float = 0.15
    first_layer: int = 14
    last_layer: int = 30
    adaptive: bool = False
    adaptive_lambda: float = 1.0
    adaptive_target: float = 0.15
    adaptive_alpha_max: float = 0.3
    latent_roles: tuple[str, ...] = ("evidence_planner", "navigator", "reasoner")
    role_alphas: tuple[tuple[str, float], ...] = ()
    answerer_alpha: float | None = None
    answerer_adaptive: bool = False
    answerer_prefill_only: bool = False
    relay_ratio: float = 1.0
    # "uniform" / "sender_priority" / "shuffled_sender_priority" are the
    # sender-relay experiment's Answerer-handoff destination variants; every
    # preexisting preset keeps "attention"/"key_norm" and is byte-identical.
    relay_source: Literal[
        "attention",
        "key_norm",
        "uniform",
        "sender_priority",
        "shuffled_sender_priority",
    ] = "attention"
    relay_mode: Literal["topk", "proportional"] = "topk"
    adaptive_within_role: bool = False
    pathology_aware: bool = False
    latent_kv_relay: bool = False
    visual_grounded_latent_kv_relay: bool = False
    visual_contrast_latent_kv_relay: bool = False
    pathology_context_latent_kv_relay: bool = False
    visual_grounded_latent_kv_relay_stages: tuple[str, ...] = ("reasoner",)
    collect_visual_grounding: bool = False
    visual_grounding_threshold: float = 0.5
    visual_grounding_top_k: int = 3
    spatial_context_relay: bool = False
    hierarchy_context_reasoner: bool = False
    hierarchy_context_relay: bool = False
    parent_anchor_strength: float = 0.35
    fine_evidence_boost: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("reallocation alpha must be in (0, 1]")
        if self.first_layer < 0 or self.last_layer < self.first_layer:
            raise ValueError("reallocation layer range is invalid")
        if self.adaptive_lambda < 0.0:
            raise ValueError("adaptive reallocation lambda must be non-negative")
        if not 0.0 <= self.adaptive_target <= 1.0:
            raise ValueError("adaptive reallocation target must be in [0, 1]")
        if not 0.0 <= self.adaptive_alpha_max <= 1.0:
            raise ValueError("adaptive reallocation alpha max must be in [0, 1]")
        if any(not 0.0 < alpha <= 1.0 for _, alpha in self.role_alphas):
            raise ValueError("role reallocation alpha must be in (0, 1]")
        if self.answerer_alpha is not None and not 0.0 < self.answerer_alpha <= 1.0:
            raise ValueError("answerer reallocation alpha must be in (0, 1]")
        if not 0.0 < self.relay_ratio <= 1.0:
            raise ValueError("reallocation relay ratio must be in (0, 1]")
        if self.relay_mode not in ("topk", "proportional"):
            raise ValueError("reallocation relay mode must be 'topk' or 'proportional'")
        if not 0.0 <= self.parent_anchor_strength <= 1.0:
            raise ValueError("parent anchor strength must be in [0, 1]")
        if self.fine_evidence_boost < 0.0:
            raise ValueError("fine evidence boost must be non-negative")
        if not 0.0 < self.visual_grounding_threshold <= 1.0:
            raise ValueError("visual grounding threshold must be in (0, 1]")
        if self.visual_grounding_top_k < 1:
            raise ValueError("visual grounding top-k must be positive")

    def initial_probe_for_role(
        self, role: str, previous_probe: float | None
    ) -> float | None:
        """Avoid comparing visual mass across roles with different image tokens."""
        if self.adaptive_within_role and role in self.latent_roles:
            return None
        return previous_probe

    def alpha_for_probe(self, probe_mass: float | None) -> float:
        """Move only enough donor mass to approach, never overshoot, the target."""
        if not self.adaptive:
            return self.alpha
        if probe_mass is None:
            return 0.0
        observed = min(1.0, max(0.0, float(probe_mass)))
        deficit = max(0.0, self.adaptive_target - observed)
        if deficit == 0.0:
            return 0.0
        donor_mass = max(1e-9, 1.0 - observed)
        correction = deficit / donor_mass
        gain = min(1.0, self.adaptive_lambda)
        return min(
            self.adaptive_alpha_max,
            gain * correction,
        )

    def alpha_for_role(self, role: str, probe_mass: float | None) -> float:
        """Return zero outside the configured latent roles, else its own alpha."""
        if role == "answerer":
            if self.answerer_alpha is not None:
                return self.answerer_alpha
            return self.alpha_for_probe(probe_mass) if self.answerer_adaptive else 0.0
        if role not in self.latent_roles:
            return 0.0
        role_alpha = dict(self.role_alphas).get(role)
        return role_alpha if role_alpha is not None else self.alpha_for_probe(probe_mass)

    @property
    def layers(self) -> tuple[int, ...]:
        """Expose the inclusive Qwen layer range used by the original preset."""
        return tuple(range(self.first_layer, self.last_layer + 1))


@contextlib.contextmanager
def terminal_reallocation(
    backbone: Qwen3VLBackbone,
    *,
    vision_mask: torch.Tensor,
    vision_weights: torch.Tensor | None = None,
    visual_patch_ids: torch.Tensor | None = None,
    parent_patch_indices: tuple[int, ...] = (),
    config: TerminalReallocationConfig | None,
) -> Generator[None, None, None]:
    """Apply reallocation at the handoff boundary and restore SDPA afterward.

    With ``config.answerer_prefill_only`` the intervention fires once on the
    receiving agent's prefill over the handed-off cache (the paper's handoff
    semantics); otherwise it rides every decode step (legacy behaviour)."""
    if config is None or not bool(vision_mask.any()):
        yield None
        return
    from memory.reallocate import reallocating

    previous_model = backbone.model.config._attn_implementation
    previous_language = backbone.lm.config._attn_implementation
    backbone.model.config._attn_implementation = "eager"
    backbone.lm.config._attn_implementation = "eager"
    try:
        with reallocating(
            vision_mask,
            config.alpha,
            config.layers,
            device=backbone.device,
            vis_weight=vision_weights,
            decode_only=not config.answerer_prefill_only,
            prefill_only=config.answerer_prefill_only,
            relay_ratio=config.relay_ratio,
            relay_source=config.relay_source,
            relay_mode=config.relay_mode,
            visual_patch_ids=(
                visual_patch_ids if config.hierarchy_context_relay else None
            ),
            parent_patch_indices=(
                parent_patch_indices if config.hierarchy_context_relay else ()
            ),
            parent_anchor_strength=(
                config.parent_anchor_strength if config.hierarchy_context_relay else 0.0
            ),
        ) as probe:
            yield probe
    finally:
        backbone.model.config._attn_implementation = previous_model
        backbone.lm.config._attn_implementation = previous_language
