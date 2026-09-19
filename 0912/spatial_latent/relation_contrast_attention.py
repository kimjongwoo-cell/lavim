"""Question-conditioned cross-scale contrast injection for latent SDPA."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Unpack

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from spatial_latent.attention import BiasProbe

_RMS_EPSILON: Final = 1e-6


@dataclass(frozen=True, slots=True)
class ContrastStep:
    """Parent and child addresses defining one cross-scale relation."""

    parent_columns: torch.Tensor
    child_columns: torch.Tensor
    strength: float
    first_layer: int = 8
    last_layer: int = 20


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
def cross_scale_contrast_sdpa(step: ContrastStep) -> Iterator[BiasProbe]:
    """Add normalized child-minus-parent evidence to each relation hidden update."""
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    probe = BiasProbe()

    def contrast_forward(
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
        delta = child - parent
        delta_rms = delta.square().mean(dim=-1, keepdim=True).sqrt()
        source_rms = (
            child.square().mean(dim=-1, keepdim=True).sqrt()
            + parent.square().mean(dim=-1, keepdim=True).sqrt()
        ) * 0.5
        relation = delta * source_rms / delta_rms.clamp_min(_RMS_EPSILON)
        probe.applied_calls += 1
        probe.key_count = int(key.shape[-2])
        return native + relation * step.strength, None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = contrast_forward
    try:
        yield probe
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
