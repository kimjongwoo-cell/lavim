"""Select visual evidence and its local tissue context from reasoner latent attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import torch


class LatentCacheLayer(Protocol):
    """Mutable dynamic-cache layer exposing its attention K/V tensors."""

    keys: torch.Tensor
    values: torch.Tensor


class LatentCache(Protocol):
    """Dynamic cache surface used to relay Reasoner K/V entries."""

    layers: Sequence[LatentCacheLayer]


def extract_terminal_latent_kv(
    cache: LatentCache,
    *,
    latent_steps: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Copy the final Reasoner latent K/V entries from every cache layer."""
    if latent_steps < 1:
        return {}
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_index, layer in enumerate(cache.layers):
        keys = layer.keys
        values = layer.values
        if keys.shape[-2] < latent_steps or values.shape[-2] < latent_steps:
            return {}
        result[layer_index] = (
            keys[..., -latent_steps:, :].clone(),
            values[..., -latent_steps:, :].clone(),
        )
    return result


def select_visual_grounded_latent_steps(
    latent_to_vision: Mapping[int, torch.Tensor],
    *,
    relative_threshold: float = 0.5,
) -> torch.Tensor:
    """Select latent steps that allocated substantial attention to visual KV."""
    if not latent_to_vision or not 0.0 < relative_threshold <= 1.0:
        return torch.empty(0, dtype=torch.long)
    per_layer_scores = tuple(
        attention.float().sum(dim=-1).mean(dim=1)
        for attention in latent_to_vision.values()
        if attention.ndim == 3
    )
    if not per_layer_scores:
        return torch.empty(0, dtype=torch.long)
    scores = torch.stack(per_layer_scores, dim=0).mean(dim=0)
    peak = scores.max()
    if not bool(peak > 0.0):
        return torch.empty(0, dtype=torch.long)
    return (scores >= peak * relative_threshold).nonzero(as_tuple=False).flatten()


def extract_selected_terminal_latent_kv(
    cache: LatentCache,
    *,
    latent_steps: int,
    selected_steps: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Copy selected visual-grounded Reasoner latent K/V entries from every layer."""
    if latent_steps < 1 or selected_steps.ndim != 1 or selected_steps.numel() == 0:
        return {}
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_index, layer in enumerate(cache.layers):
        if layer.keys.shape[-2] < latent_steps or layer.values.shape[-2] < latent_steps:
            return {}
        positions = selected_steps.to(device=layer.keys.device, dtype=torch.long)
        if bool((positions < 0).any()) or bool((positions >= latent_steps).any()):
            return {}
        offset_positions = positions + layer.keys.shape[-2] - latent_steps
        result[layer_index] = (
            layer.keys.index_select(-2, offset_positions).clone(),
            layer.values.index_select(-2, offset_positions).clone(),
        )
    return result


def latent_relay_length(relay: Mapping[int, tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Return the number of selected latent K/V rows in a relay payload."""
    first_layer = next(iter(relay.values()), None)
    return 0 if first_layer is None else int(first_layer[0].shape[-2])


def append_latent_kv_relay(
    cache: LatentCache,
    relay: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> LatentCache:
    """Append copied Reasoner latent K/V entries at the Answerer cache boundary."""
    if not relay:
        return cache
    if len(cache.layers) != len(relay):
        return cache
    for layer_index, layer in enumerate(cache.layers):
        keys, values = relay.get(layer_index, (None, None))
        if keys is None or values is None:
            return cache
        if keys.shape[:2] != layer.keys.shape[:2] or keys.shape[-1] != layer.keys.shape[-1]:
            return cache
        if values.shape[:2] != layer.values.shape[:2] or values.shape[-1] != layer.values.shape[-1]:
            return cache
    for layer_index, layer in enumerate(cache.layers):
        keys, values = relay[layer_index]
        layer.keys = torch.cat((layer.keys, keys.to(device=layer.keys.device)), dim=-2)
        layer.values = torch.cat((layer.values, values.to(device=layer.values.device)), dim=-2)
    return cache
