"""Scoped, single-query SDPA logit bias for designated Reasoner layers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Unpack

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@dataclass(frozen=True, slots=True)
class BiasSettings:
    """Scientific intervention settings, independent of runtime cache state."""

    log_gain: float = math.log(2)
    first_layer: int = 8
    last_layer: int = 20


@dataclass(slots=True)  # noqa: MUTABLE_OK
class BiasProbe:
    """Mutable counter used to verify the live attention adapter actually fired."""

    applied_calls: int = 0
    key_count: int = 0


@contextmanager
def guided_sdpa(columns: torch.Tensor, settings: BiasSettings) -> Iterator[BiasProbe]:
    """Install an additive spatial prior for this latent forward, then restore SDPA."""
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    probe = BiasProbe()

    def spatial_forward(
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
        """Mirror the Transformers attention interface at its existing boundary."""
        layer = getattr(module, "layer_idx", -1)
        active = (
            query.shape[-2] == 1
            and settings.first_layer <= layer <= settings.last_layer
            and columns.numel() > 0
            and settings.log_gain != 0.0
        )
        mask = attention_mask
        if active:
            bias = query.new_zeros((1, 1, 1, key.shape[-2]))
            bias[..., columns.to(query.device)] = settings.log_gain
            if mask is not None:
                if mask.dtype == torch.bool:
                    allowed = torch.zeros_like(mask, dtype=query.dtype)
                    mask = allowed.masked_fill(~mask, float("-inf"))
                bias = bias + mask
            mask = bias
            probe.applied_calls += 1
            probe.key_count = int(key.shape[-2])
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    ALL_ATTENTION_FUNCTIONS["sdpa"] = spatial_forward
    try:
        yield probe
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
