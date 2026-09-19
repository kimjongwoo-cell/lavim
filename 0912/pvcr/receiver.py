"""Temporarily expose synthetic values at the existing sender cache addresses."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, DynamicLayer

from pvcr.errors import LayoutError


def layer_values(cache: Cache, layer_index: int) -> torch.Tensor:
    """Require populated full-attention cache layers at the adapter boundary."""
    layer = cache.layers[layer_index]
    if not isinstance(layer, DynamicLayer) or layer.values is None:
        raise LayoutError("PVCR requires initialized DynamicLayer values")
    return layer.values


@dataclass(frozen=True, slots=True)
class RelayBank:
    columns: torch.Tensor
    values: dict[int, torch.Tensor]


@dataclass
class CacheAudit:
    before_substitution: int
    after_substitution: int = -1
    before_restoration: int = -1
    after_restoration: int = -1


@contextmanager
def substitute_values(cache: Cache, bank: RelayBank) -> Iterator[CacheAudit]:
    """Preserve positions and restore original values even after cache growth/errors."""
    originals: dict[int, torch.Tensor] = {}
    audit = CacheAudit(int(cache.get_seq_length()))
    try:
        for index, synthetic in bank.values.items():
            values = layer_values(cache, index)
            columns = bank.columns.to(values.device)
            if synthetic.shape != values.index_select(-2, columns).shape:
                raise LayoutError("PVCR synthetic/cache shape mismatch")
            originals[index] = values.index_select(-2, columns).clone()
            values.index_copy_(-2, columns, synthetic.to(values))
        audit.after_substitution = int(cache.get_seq_length())
        if audit.after_substitution != audit.before_substitution:
            raise LayoutError("PVCR substitution changed cache length")
        yield audit
    finally:
        audit.before_restoration = int(cache.get_seq_length())
        for index, original in originals.items():
            values = layer_values(cache, index)
            values.index_copy_(-2, bank.columns.to(values.device), original.to(values))
        audit.after_restoration = int(cache.get_seq_length())
        if audit.after_restoration != audit.before_restoration:
            raise LayoutError("PVCR restoration changed cache length")


def receiver_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    columns: torch.Tensor,
    log_gain: float,
    causal: bool,
) -> torch.Tensor:
    """Preserve causal and supplied masks while biasing existing relay addresses."""
    queries, length = query.shape[-2], key.shape[-2]
    bias = query.new_zeros(1, 1, queries, length)
    bias[..., columns.to(query.device)] = log_gain
    if mask is not None:
        if mask.dtype == torch.bool:
            return bias.masked_fill(~mask[..., :length], float("-inf"))
        return bias + mask[..., :length]
    if causal and queries > 1:
        allowed = torch.arange(length, device=query.device)[None, :] <= (
            length - queries + torch.arange(queries, device=query.device)[:, None]
        )
        bias = bias.masked_fill(~allowed, float("-inf"))
    return bias
