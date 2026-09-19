"""Agreement-weighted Answerer attention over paired latent evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Unpack

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from spatial_latent.attention import BiasProbe


def _additive_mask(
    mask: torch.Tensor | None,
    query: torch.Tensor,
) -> torch.Tensor | None:
    if mask is None or mask.dtype != torch.bool:
        return mask
    additive = torch.zeros_like(mask, dtype=query.dtype)
    return additive.masked_fill(~mask, float("-inf"))


@contextmanager
def adaptive_pair_sdpa(
    pairs: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
    *,
    first_layer: int = 8,
    last_layer: int = 20,
) -> Iterator[BiasProbe]:
    """Bias paired latent rows by agreement in one Answerer attention pass."""
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    probe = BiasProbe()

    def adaptive_forward(
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
        layer = getattr(module, "layer_idx", -1)
        if not (first_layer <= layer <= last_layer):
            return sdpa_attention_forward(
                module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
            )
        existing = _additive_mask(attention_mask, query)
        pair_columns = pairs.to(query.device)
        pair_bias = query.new_zeros((1, 1, query.shape[-2], key.shape[-2]))
        scale = alpha * len(pair_columns)
        for index, pair in enumerate(pair_columns):
            boost = torch.log1p(weights[index].to(query) * scale)
            pair_bias[..., pair] = boost
        if existing is not None:
            pair_bias = pair_bias + existing
        output, _ = sdpa_attention_forward(
            module, query, key, value, pair_bias,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )
        probe.applied_calls += 1
        probe.key_count = int(key.shape[-2])
        return output, None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = adaptive_forward
    try:
        yield probe
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
