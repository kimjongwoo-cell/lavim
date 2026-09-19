"""Full-denominator, GQA-aware visual value contribution."""

from dataclasses import dataclass

import torch

from pvcr.errors import LayoutError
from pvcr.layout import Neighborhoods


@dataclass(frozen=True, slots=True)
class Contribution:
    value: torch.Tensor
    visual_mass: torch.Tensor
    support: torch.Tensor


def full_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    scale: float,
) -> torch.Tensor:
    """Return [batch, KV-head, key] probabilities from one latent query."""
    batch, query_heads, query_count, width = query.shape
    kv_heads, length = keys.shape[1:3]
    if query_count != 1 or query_heads % kv_heads:
        raise LayoutError("PVCR observes single-query GQA latent attention only")
    groups = query_heads // kv_heads
    grouped_query = query.float().reshape(batch, kv_heads, groups, width)
    logits = torch.einsum("bhgd,bhkd->bhgk", grouped_query, keys.float()) * scale
    logits = logits.reshape(batch, query_heads, 1, length)
    if mask is not None:
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask[..., :length], float("-inf"))
        else:
            logits = logits + mask[..., :length].float()
    probabilities = logits.softmax(-1).nan_to_num(0.0)
    return probabilities.reshape(batch, kv_heads, groups, length).mean(2)


def contribution(
    attention: torch.Tensor,
    values: torch.Tensor,
    layout: Neighborhoods,
    *,
    grouped: bool = True,
) -> Contribution:
    """Pool raw visual summands, using soft neighborhood weights when requested."""
    columns = layout.columns.to(values.device)
    visual_attention = attention.index_select(-1, columns)
    visual_values = values.index_select(-2, columns).float()
    visual_mass = visual_attention.sum(-1)
    if not grouped:
        pooled = torch.einsum("bhv,bhvd->bhd", visual_attention, visual_values)
        return Contribution(pooled.to(values.dtype), visual_mass, visual_mass)
    core = layout.core.to(device=values.device, dtype=torch.float32)
    context = layout.context.to(device=values.device, dtype=torch.float32)
    core_density = (visual_attention @ core.T) / core.sum(-1).clamp_min(1)
    context_density = (visual_attention @ context.T) / context.sum(-1).clamp_min(1)
    scores = (
        2
        * core_density
        * context_density
        / (core_density + context_density).clamp_min(1e-20)
    )
    support = scores.sum(-1)
    weights = scores / support.unsqueeze(-1).clamp_min(1e-20)
    membership = (core + context).clamp_max(1)
    token_weight = weights @ membership
    pooled = torch.einsum(
        "bhv,bhvd->bhd", visual_attention * token_weight, visual_values
    )
    return Contribution(pooled.to(values.dtype), visual_mass, support)
