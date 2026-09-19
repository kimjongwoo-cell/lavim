"""Adaptive visual-context selection within the latent head/step grid."""

import math

import torch
from pvcr.layout import Neighborhoods


def contrastive_query_margin(
    visual_queries: torch.Tensor,
    text_queries: torch.Tensor,
    latent_keys: torch.Tensor,
) -> torch.Tensor:
    """Contrast mean visual/text query affinity for each latent K/V head-step."""
    query_heads = visual_queries.shape[1]
    kv_heads = latent_keys.shape[1]
    if query_heads % kv_heads:
        raise ValueError("Query heads must divide evenly into KV-head groups")
    repeats = query_heads // kv_heads
    visual = torch.nn.functional.normalize(visual_queries.float(), dim=-1)
    text = torch.nn.functional.normalize(text_queries.float(), dim=-1)
    candidates = torch.nn.functional.normalize(latent_keys.float(), dim=-1)
    visual = visual.reshape(
        visual.shape[0], kv_heads, repeats, visual.shape[-2], visual.shape[-1]
    )
    text = text.reshape(
        text.shape[0], kv_heads, repeats, text.shape[-2], text.shape[-1]
    )
    visual_similarity = torch.einsum("bhrqd,bhsd->bhrqs", visual, candidates).mean(
        dim=(2, 3)
    )
    text_similarity = torch.einsum("bhrqd,bhsd->bhrqs", text, candidates).mean(
        dim=(2, 3)
    )
    return visual_similarity - text_similarity


def contrastive_context_margin(
    local_queries: torch.Tensor,
    context_queries: torch.Tensor,
    text_queries: torch.Tensor,
    latent_keys: torch.Tensor,
) -> torch.Tensor:
    """Score per-family latent rows against both local and surrounding queries."""
    query_heads = local_queries.shape[1]
    kv_heads = latent_keys.shape[1]
    if query_heads % kv_heads:
        raise ValueError("Query heads must divide evenly into KV-head groups")
    repeats = query_heads // kv_heads
    local = torch.nn.functional.normalize(local_queries.float(), dim=-1)
    context = torch.nn.functional.normalize(context_queries.float(), dim=-1)
    text = torch.nn.functional.normalize(text_queries.float(), dim=-1)
    candidates = torch.nn.functional.normalize(latent_keys.float(), dim=-1)
    local = local.reshape(
        local.shape[0], kv_heads, repeats, local.shape[-2], local.shape[-1]
    )
    context = context.reshape(
        context.shape[0], kv_heads, repeats, context.shape[-2], context.shape[-1]
    )
    text = text.reshape(
        text.shape[0], kv_heads, repeats, text.shape[-2], text.shape[-1]
    )
    local_similarity = torch.einsum(
        "bhrfd,bhsd->bhrfs", local, candidates
    ).mean(dim=2)
    context_similarity = torch.einsum(
        "bhrfd,bhsd->bhrfs", context, candidates
    ).mean(dim=2)
    text_similarity = torch.einsum("bhrqd,bhsd->bhrqs", text, candidates).mean(
        dim=(2, 3)
    )
    text_margin = text_similarity.unsqueeze(-2)
    return torch.minimum(
        local_similarity - text_margin, context_similarity - text_margin
    )


def select_visual_closer_pairs(margins: torch.Tensor) -> torch.Tensor:
    """Pass every latent K/V head-step whose nearest match is visual, not text."""
    return (margins > 0) & torch.isfinite(margins)


def select_contextual_pairs(margins: torch.Tensor) -> torch.Tensor:
    """Pass all positive head outliers above each step's median-plus-MAD margin."""
    work = margins.detach().cpu()
    median = work.median(dim=1, keepdim=True).values
    mad = (work - median).abs().median(dim=1, keepdim=True).values
    return (work > median + mad) & (work > 0) & torch.isfinite(work)


def select_attention_context_pairs(scores: torch.Tensor) -> torch.Tensor:
    """Relay each selected KV head-step through its strongest pathology family."""
    best_scores, families = scores.max(dim=2)
    selected = select_contextual_pairs(best_scores)
    family_mask = torch.nn.functional.one_hot(
        families, num_classes=scores.shape[2]
    ).permute(0, 1, 3, 2)
    return family_mask.to(selected.device, dtype=torch.bool) & selected.unsqueeze(2)


def select_heads(scores: torch.Tensor) -> torch.Tensor:
    """Select each head's above-median-plus-MAD steps, allowing an empty result."""
    # These tiny10-step statistics run on CPU to avoid CUDA median tie warnings.
    work = scores.detach().cpu()
    median = work.median(dim=-1, keepdim=True).values
    mad = (work - median).abs().median(dim=-1, keepdim=True).values
    return (work > median + mad) & (work > 0) & torch.isfinite(work)


def select_visual_fraction(
    scores: torch.Tensor, fraction: float = 0.20
) -> torch.Tensor:
    """Keep the strongest fixed fraction of visual-associated latent head-steps."""
    work = scores.detach().cpu()
    chosen = torch.zeros_like(work, dtype=torch.bool)
    for batch in range(work.shape[0]):
        candidates = (work[batch] > 0).nonzero()
        if candidates.numel() == 0:
            continue
        count = max(1, math.ceil(candidates.shape[0] * fraction))
        values = work[batch][candidates[:, 0], candidates[:, 1]]
        kept = candidates[values.argsort(descending=True)[:count]]
        chosen[batch, kept[:, 0], kept[:, 1]] = True
    return chosen


def select_relational_heads(scores: torch.Tensor) -> torch.Tensor:
    """Gate only family-step outliers across a head's complete spatial field."""
    work = scores.detach().cpu()
    flattened = work.flatten(start_dim=-2)
    median = flattened.median(dim=-1, keepdim=True).values.unsqueeze(-1)
    mad = (
        (flattened - median.squeeze(-1)).abs().median(dim=-1, keepdim=True).values
    ).unsqueeze(-1)
    return (work > median + mad) & (work > 0) & torch.isfinite(work)


def cap_relational_rows(
    scores: dict[int, torch.Tensor], selected: dict[int, torch.Tensor], capacity: int
) -> dict[int, torch.Tensor]:
    """Keep one adaptive spatial family per supported latent step."""
    masks = torch.stack(list(selected.values()))
    addresses = masks.any(dim=(0, 1, 2)).nonzero()
    if addresses.shape[0] <= capacity:
        return selected
    evidence = torch.stack(list(scores.values())).mean(dim=(0, 1, 2)).detach().cpu()
    allowed = torch.zeros_like(evidence, dtype=torch.bool)
    for step in range(evidence.shape[1]):
        families = addresses[addresses[:, 1] == step, 0]
        if families.numel():
            best = families[evidence[families, step].argmax()]
            allowed[best, step] = True
    if allowed.sum() > capacity:
        raise ValueError("A step-wise relay capacity must cover all latent steps")
    return {
        layer: mask & allowed.unsqueeze(0).unsqueeze(0)
        for layer, mask in selected.items()
    }


def select_relational_fraction(
    scores: dict[int, torch.Tensor],
    selected: dict[int, torch.Tensor],
    fraction: float = 0.10,
) -> dict[int, torch.Tensor]:
    """Keep the strongest fixed fraction of relation-qualified rows."""
    masks = torch.stack(list(selected.values()))
    addresses = masks.any(dim=(0, 1, 2)).nonzero()
    if addresses.numel() == 0:
        return selected
    count = max(1, math.ceil(addresses.shape[0] * fraction))
    evidence = torch.stack(list(scores.values())).mean(dim=(0, 1, 2)).detach().cpu()
    values = evidence[addresses[:, 0], addresses[:, 1]]
    kept = addresses[values.argsort(descending=True)[:count]]
    allowed = torch.zeros_like(evidence, dtype=torch.bool)
    allowed[kept[:, 0], kept[:, 1]] = True
    return {
        layer: mask & allowed.unsqueeze(0).unsqueeze(0)
        for layer, mask in selected.items()
    }


def grounding_scores(
    attention: torch.Tensor, layout: Neighborhoods, *, spatial: bool
) -> torch.Tensor:
    """Use visual attention only as a selector; never read or pool visual V."""
    visual = attention.index_select(-1, layout.columns.to(attention.device))
    mass = visual.sum(-1)
    background = (1 - mass).clamp_min(0) / max(
        1, attention.shape[-1] - visual.shape[-1]
    )
    if spatial:
        core = layout.core.to(attention)
        context = layout.context.to(attention)
        core_density = (visual @ core.T) / core.sum(-1).clamp_min(1)
        context_density = (visual @ context.T) / context.sum(-1).clamp_min(1)
        density = (
            2
            * core_density
            * context_density
            / (core_density + context_density).clamp_min(1e-20)
        ).amax(-1)
    else:
        return (mass - (1 - mass)).clamp_min(0)
    enrichment = (density / background.clamp_min(1e-20) - 1).clamp_min(0)
    return mass * enrichment


def relational_grounding_scores(
    attention: torch.Tensor, layout: Neighborhoods
) -> torch.Tensor:
    """Score every parent/child family that jointly grounds one latent row."""
    visual = attention.index_select(-1, layout.columns.to(attention.device))
    core = layout.core.to(attention)
    context = layout.context.to(attention)
    core_density = (visual @ core.T) / core.sum(-1).clamp_min(1)
    context_density = (visual @ context.T) / context.sum(-1).clamp_min(1)
    joint_density = (
        2
        * core_density
        * context_density
        / (core_density + context_density).clamp_min(1e-20)
    )
    background = (1 - visual.sum(-1, keepdim=True)).clamp_min(0) / max(
        1, attention.shape[-1] - visual.shape[-1]
    )
    membership = (core + context).clamp_max(1)
    family_mass = visual @ membership.T
    enrichment = (joint_density / background.clamp_min(1e-20) - 1).clamp_min(0)
    return family_mass * enrichment
