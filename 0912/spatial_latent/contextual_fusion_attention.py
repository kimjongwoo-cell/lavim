"""Reasoner-side fusion of low-power context and high-power morphology."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Unpack

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from spatial_latent.attention import BiasProbe


@dataclass(frozen=True, slots=True)
class CrossScaleFusionStep:
    """Visual addresses for one parent-context and child-morphology fusion."""

    parent_columns: torch.Tensor
    child_columns: torch.Tensor
    strength: float
    first_layer: int = 8
    last_layer: int = 20


class FusionProbe(BiasProbe):
    """Accumulate parent-child alignment during a fused Reasoner step."""

    def __init__(self) -> None:
        super().__init__()
        self.agreement_sum = 0.0

    @property
    def mean_agreement(self) -> float:
        """Return alignment over all active Reasoner layers."""
        return self.agreement_sum / self.applied_calls if self.applied_calls else 0.0


def _additive_mask(
    mask: torch.Tensor | None,
    query: torch.Tensor,
) -> torch.Tensor | None:
    if mask is None or mask.dtype != torch.bool:
        return mask
    additive = torch.zeros_like(mask, dtype=query.dtype)
    return additive.masked_fill(~mask, float("-inf"))


def _source_mask(
    columns: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    existing: torch.Tensor | None,
) -> torch.Tensor:
    mask = torch.full(
        (1, 1, query.shape[-2], key.shape[-2]),
        float("-inf"),
        dtype=query.dtype,
        device=query.device,
    )
    mask[..., columns.to(query.device)] = 0.0
    return mask if existing is None else mask + existing


@contextmanager
def cross_scale_fusion_sdpa(step: CrossScaleFusionStep) -> Iterator[FusionProbe]:
    """Write a latent row containing aligned architectural and cellular evidence."""
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    probe = FusionProbe()

    def fusion_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, None]:
        native, _ = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
        layer = getattr(module, "layer_idx", -1)
        active = query.shape[-2] == 1 and step.first_layer <= layer <= step.last_layer
        if not active:
            return native, None
        existing = _additive_mask(attention_mask, query)
        parent, _ = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            _source_mask(step.parent_columns, query, key, existing),
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
        child, _ = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            _source_mask(step.child_columns, query, key, existing),
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
        agreement = torch.nn.functional.cosine_similarity(parent, child, dim=-1)
        agreement = agreement.clamp_min(0.0).unsqueeze(-1)
        fused = (parent + child * agreement) / (1.0 + agreement)
        probe.agreement_sum += float(agreement.mean().detach().cpu())
        probe.applied_calls += 1
        probe.key_count = int(key.shape[-2])
        return native.mul(1.0 - step.strength).add_(fused, alpha=step.strength), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = fusion_forward
    try:
        yield probe
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
