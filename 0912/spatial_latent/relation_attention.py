"""Cross-scale relation formation and reserved Answerer readout for SDPA."""

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
class RelationStep:
    """Sources whose joint contribution defines one cross-scale latent step."""

    columns: torch.Tensor
    include_previous: bool
    log_gain: float
    tail_offsets: tuple[int, ...] = ()
    first_layer: int = 8
    last_layer: int = 20


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RelationProbe(BiasProbe):
    """Runtime proof including the dynamically resolved preceding latent column."""

    previous_columns: tuple[int, ...] = ()


def _additive_mask(
    mask: torch.Tensor | None,
    query: torch.Tensor,
) -> torch.Tensor | None:
    """Convert a boolean permission mask while retaining additive masks unchanged."""
    if mask is None or mask.dtype != torch.bool:
        return mask
    additive = torch.zeros_like(mask, dtype=query.dtype)
    return additive.masked_fill(~mask, float("-inf"))


@contextmanager
def relation_step_sdpa(step: RelationStep) -> Iterator[RelationProbe]:
    """Boost scale-specific sources while preserving every original key and mask."""
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    probe = RelationProbe()

    def relation_forward(
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
        active = query.shape[-2] == 1 and step.first_layer <= layer <= step.last_layer
        mask = attention_mask
        if active:
            selected = step.columns.to(query.device)
            offsets = (2,) if step.include_previous else step.tail_offsets
            previous = tuple(key.shape[-2] - offset for offset in offsets)
            if previous:
                selected = torch.cat((selected, selected.new_tensor(previous)))
                probe.previous_columns = previous
            bias = query.new_zeros((1, 1, 1, key.shape[-2]))
            bias[..., selected] = step.log_gain
            existing = _additive_mask(mask, query)
            mask = bias if existing is None else bias + existing
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

    ALL_ATTENTION_FUNCTIONS["sdpa"] = relation_forward
    try:
        yield probe
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original


@contextmanager
def reserved_relation_sdpa(
    columns: torch.Tensor,
    alpha: float,
    *,
    first_layer: int = 8,
    last_layer: int = 20,
) -> Iterator[BiasProbe]:
    """Reserve a fixed share of terminal attention output for relation K/V rows."""
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    probe = BiasProbe()

    def reserved_forward(
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
        existing = _additive_mask(attention_mask, query)
        relation = columns.to(query.device)
        main_mask = query.new_zeros((1, 1, query.shape[-2], key.shape[-2]))
        main_mask[..., relation] = float("-inf")
        relation_mask = torch.full_like(main_mask, float("-inf"))
        relation_mask[..., relation] = 0
        if existing is not None:
            main_mask = main_mask + existing
            relation_mask = relation_mask + existing
        main_output, _ = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            main_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
        relation_output, _ = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            relation_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
        probe.applied_calls += 1
        probe.key_count = int(key.shape[-2])
        return main_output.mul(1.0 - alpha).add_(relation_output, alpha=alpha), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = reserved_forward
    try:
        yield probe
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
