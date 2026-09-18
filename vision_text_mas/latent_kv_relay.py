"""Relay visual-grounded Reasoner latent K/V entries to the Answerer cache."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class CacheLayer(Protocol):
    keys: torch.Tensor
    values: torch.Tensor


class DynamicCache(Protocol):
    layers: Sequence[CacheLayer]


@dataclass(frozen=True, slots=True)
class RelationalVisualContext:
    """Visual cache coordinates and WSI patch hierarchy for relay scoring."""

    latent_steps: int
    vision_columns: torch.Tensor
    text_start: int
    patch_ids: torch.Tensor
    parent_patch_indices: tuple[int, ...]


def select_pathology_context_steps(
    cache: DynamicCache,
    context: RelationalVisualContext,
) -> torch.Tensor:
    """Select latent rows jointly aligned to linked parent/child visual patches.

    For each latent K row, score its least-supported member of each related
    patch pair against prompt-text K. A robust median-plus-MAD gate keeps every
    step with unusually strong relational visual alignment, without a top-k.
    """
    if context.latent_steps < 1 or context.vision_columns.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if context.patch_ids.numel() != context.vision_columns.numel():
        return torch.empty(0, dtype=torch.long)

    related_pairs: set[tuple[int, int]] = set()
    children_by_parent: dict[int, list[int]] = {}
    for child, parent in enumerate(context.parent_patch_indices):
        if parent < 0 or parent >= len(context.parent_patch_indices) or parent == child:
            continue
        related_pairs.add((min(child, parent), max(child, parent)))
        children_by_parent.setdefault(parent, []).append(child)
    for children in children_by_parent.values():
        for child_index, first in enumerate(children):
            for second in children[child_index + 1 :]:
                related_pairs.add((min(first, second), max(first, second)))
    if not related_pairs:
        return torch.empty(0, dtype=torch.long)

    layer_scores: list[torch.Tensor] = []
    for layer in cache.layers:
        keys = layer.keys.float()
        latent_start = keys.shape[-2] - context.latent_steps
        if latent_start <= context.text_start:
            continue
        columns = context.vision_columns.to(device=keys.device, dtype=torch.long)
        patch_ids = context.patch_ids.to(device=keys.device, dtype=torch.long)
        valid = (columns >= context.text_start) & (columns < latent_start)
        columns = columns[valid]
        patch_ids = patch_ids[valid]
        text_mask = torch.ones(
            latent_start - context.text_start, dtype=torch.bool, device=keys.device
        )
        text_mask[columns - context.text_start] = False
        text_columns = text_mask.nonzero(as_tuple=False).flatten() + context.text_start
        if columns.numel() == 0 or text_columns.numel() == 0:
            continue

        normalized = torch.nn.functional.normalize(keys, dim=-1)
        patch_centers: dict[int, torch.Tensor] = {}
        for patch_id in torch.unique(patch_ids).tolist():
            patch_columns = columns[patch_ids == patch_id]
            patch_centers[int(patch_id)] = torch.nn.functional.normalize(
                normalized.index_select(-2, patch_columns).mean(dim=-2), dim=-1
            )
        edges = tuple(
            (first, second)
            for first, second in related_pairs
            if first in patch_centers and second in patch_centers
        )
        if not edges:
            continue

        latent = normalized[..., latent_start:, :]
        text_center = torch.nn.functional.normalize(
            normalized.index_select(-2, text_columns).mean(dim=-2), dim=-1
        )
        text_similarity = (latent * text_center.unsqueeze(-2)).sum(dim=-1)
        pair_scores = torch.stack(
            [
                torch.minimum(
                    (latent * patch_centers[first].unsqueeze(-2)).sum(dim=-1),
                    (latent * patch_centers[second].unsqueeze(-2)).sum(dim=-1),
                )
                for first, second in edges
            ],
            dim=-1,
        ).amax(dim=-1)
        layer_scores.append((pair_scores - text_similarity).flatten(0, 1))

    if not layer_scores:
        return torch.empty(0, dtype=torch.long)
    visual_context_vote = (torch.cat(layer_scores, dim=0) > 0.0).float().mean(dim=0)
    positive = visual_context_vote[visual_context_vote > 0.0]
    if positive.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    median = positive.median()
    mad = (positive - median).abs().median()
    selected = (visual_context_vote >= median + mad) & (visual_context_vote > 0.0)
    return selected.nonzero(as_tuple=False).flatten().cpu()


def select_visual_grounded_steps(
    latent_to_vision: Mapping[int, torch.Tensor],
    threshold: float = 0.5,
    fallback_step_count: int = 0,
) -> torch.Tensor:
    """Return visual-grounded steps, or the consolidated final step without maps.

    The relay must run at the fixed 12,288-token context.  When attention maps
    are not materialized, the final latent state is the only non-redundant
    context-safe relay: later latent states converge while retaining the
    Reasoner's accumulated local-neighbour evidence.
    """
    scores = tuple(
        attention.float().sum(dim=-1).mean(dim=1)
        for attention in latent_to_vision.values()
        if attention.ndim == 3
    )
    if not scores:
        if fallback_step_count > 0:
            return torch.tensor([fallback_step_count - 1], dtype=torch.long)
        return torch.empty(0, dtype=torch.long)
    merged = torch.stack(scores).mean(dim=0)
    peak = merged.max()
    if not bool(peak > 0.0):
        return torch.empty(0, dtype=torch.long)
    return (merged >= peak * threshold).nonzero(as_tuple=False).flatten()


def select_visual_grounded_steps_from_masses(
    masses: Sequence[float],
    *,
    threshold: float,
    top_k: int,
) -> torch.Tensor:
    """Adaptively gate latent rows whose visual attention exceeds text/prompt context.

    Attention mass on the original visual columns is its share of the complete
    softmax row; all remaining mass is system/prompt/latent context.  A robust
    median-plus-MAD gate therefore relays only unusually visual-grounded rows,
    rather than forcing an arbitrary number of latent K/V entries into the
    Answerer's visual pool.
    """
    if not masses:
        return torch.empty(0, dtype=torch.long)
    scores = torch.tensor(masses, dtype=torch.float32)
    if not bool(scores.max() > 0.0):
        return torch.empty(0, dtype=torch.long)
    median = scores.median()
    mad = (scores - median).abs().median()
    cutoff = median + threshold * mad
    candidates = (scores > cutoff).nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        return candidates
    ranked = candidates[scores.index_select(0, candidates).argsort(descending=True)]
    return ranked[:top_k].sort().values


def select_visual_contrast_steps(
    cache: DynamicCache,
    *,
    latent_steps: int,
    vision_columns: torch.Tensor,
    text_start: int,
) -> torch.Tensor:
    """Select latent rows whose layer/head keys are closer to vision than text."""
    if latent_steps < 1 or vision_columns.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    layer_scores: list[torch.Tensor] = []
    for layer in cache.layers:
        keys = layer.keys.float()
        sequence_length = int(keys.shape[-2])
        latent_start = sequence_length - latent_steps
        columns = vision_columns.to(device=keys.device, dtype=torch.long)
        columns = columns[(columns >= text_start) & (columns < latent_start)]
        if columns.numel() == 0 or latent_start <= text_start:
            continue
        text_mask = torch.ones(latent_start - text_start, dtype=torch.bool, device=keys.device)
        text_mask[columns - text_start] = False
        text_columns = text_mask.nonzero(as_tuple=False).flatten() + text_start
        if text_columns.numel() == 0:
            continue
        normalized = torch.nn.functional.normalize(keys, dim=-1)
        visual_center = torch.nn.functional.normalize(
            normalized.index_select(-2, columns).mean(dim=-2), dim=-1
        )
        text_center = torch.nn.functional.normalize(
            normalized.index_select(-2, text_columns).mean(dim=-2), dim=-1
        )
        latent = normalized[..., latent_start:, :]
        visual_similarity = (latent * visual_center.unsqueeze(-2)).sum(dim=-1)
        text_similarity = (latent * text_center.unsqueeze(-2)).sum(dim=-1)
        layer_scores.append((visual_similarity - text_similarity).mean(dim=0))
    if not layer_scores:
        return torch.empty(0, dtype=torch.long)
    head_contrast = torch.cat(layer_scores, dim=0)
    visual_vote_ratio = (head_contrast > 0.0).float().mean(dim=0)
    positive = visual_vote_ratio[visual_vote_ratio > 0.0]
    if positive.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    median = positive.median()
    mad = (positive - median).abs().median()
    cutoff = median + mad
    selected = (visual_vote_ratio >= cutoff) & (visual_vote_ratio > 0.0)
    return selected.nonzero(as_tuple=False).flatten().cpu()


def extract_selected_latent_kv(
    cache: DynamicCache,
    latent_steps: int,
    selected_steps: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Copy selected final Reasoner latent entries from every layer."""
    if latent_steps < 1 or selected_steps.numel() == 0:
        return {}
    relay: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, layer in enumerate(cache.layers):
        positions = selected_steps.to(device=layer.keys.device, dtype=torch.long)
        if layer.keys.shape[-2] < latent_steps or bool((positions >= latent_steps).any()):
            return {}
        positions = positions + layer.keys.shape[-2] - latent_steps
        relay[index] = (
            layer.keys.index_select(-2, positions).clone(),
            layer.values.index_select(-2, positions).clone(),
        )
    return relay


def relay_length(relay: Mapping[int, tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Return the number of K/V rows in a relay payload."""
    first = next(iter(relay.values()), None)
    return 0 if first is None else int(first[0].shape[-2])


def append_relay(cache: DynamicCache, relay: Mapping[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Append the copied latent K/V entries to every Answerer cache layer."""
    if len(cache.layers) != len(relay):
        return
    for index, layer in enumerate(cache.layers):
        keys, values = relay.get(index, (None, None))
        if keys is None or values is None:
            return
        layer.keys = torch.cat((layer.keys, keys.to(layer.keys.device)), dim=-2)
        layer.values = torch.cat((layer.values, values.to(layer.values.device)), dim=-2)
