"""Native SDPA with observation and masks for appended latent-head memory."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Unpack

import torch
from pvcr.core import full_attention
from pvcr.errors import LayoutError
from pvcr.receiver import receiver_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from latent_visual_relay.cache import InsertedBlock
from latent_visual_relay.capture import Capture
from latent_visual_relay.selection import relational_grounding_scores


@dataclass(slots=True)
class AttentionState:
    """Track inserted memory masks and the currently observed sender."""

    capture: Capture | None = None
    blocks: list[InsertedBlock] = field(default_factory=list)
    receiver_calls: int = 0
    visual_columns: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    receiver_mass: dict[str, list[float]] = field(default_factory=dict)
    relay_log_gain: float = 0.0
    receiver_relation_gain: float = 0.0
    receiver_relation_calls: int = 0
    receiver_relation_bias_mass: float = 0.0


def memory_bias(
    blocks: list[InsertedBlock],
    query: torch.Tensor,
    layer: int,
    length: int,
    *,
    log_gain: float = 0.0,
) -> torch.Tensor:
    """Mask unselected heads, including every head in unobserved layers."""
    batch, query_heads = query.shape[:2]
    bias = query.new_zeros(batch, query_heads, 1, length)
    for block in blocks:
        allowed = block.allowed[layer].to(query.device)
        repeats = query_heads // allowed.shape[1]
        valid = allowed.repeat_interleave(repeats, dim=1).unsqueeze(-2)
        extra = query.new_full(valid.shape, log_gain).masked_fill(~valid, float("-inf"))
        bias.index_copy_(-1, block.columns.to(query.device), extra)
    return bias


def relation_family_read_bias(
    blocks: list[InsertedBlock],
    visual_attention: torch.Tensor,
    query: torch.Tensor,
    *,
    layer: int,
    length: int,
    log_gain: float,
) -> torch.Tensor:
    """Bias selected latent rows toward families jointly read at the receiver."""
    batch, query_heads = query.shape[:2]
    kv_heads = visual_attention.shape[1]
    if query_heads % kv_heads:
        raise LayoutError("Receiver query heads do not divide into KV-head groups")
    result = query.new_zeros(batch, query_heads, 1, length)
    repeats = query_heads // kv_heads
    for block in blocks:
        if block.layout is None or block.family_ids is None:
            continue
        if block.family_ids.numel() == 0:
            continue
        scores = relational_grounding_scores(visual_attention, block.layout)
        excess = (scores - scores.median(dim=-1, keepdim=True).values).clamp_min(0)
        strength = excess / excess.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        family_ids = block.family_ids.to(scores.device)
        if family_ids.max() >= strength.shape[-1] or family_ids.min() < 0:
            raise LayoutError("Inserted latent row refers to an unknown tissue family")
        rows = strength.index_select(-1, family_ids)
        rows = rows * block.allowed[layer].to(rows.device)
        rows = (
            rows.repeat_interleave(repeats, dim=1).unsqueeze(-2) * log_gain
        ).to(dtype=result.dtype, device=result.device)
        result.index_copy_(-1, block.columns.to(query.device), rows)
    return result


@contextmanager
def installed_attention(state: AttentionState) -> Iterator[None]:
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def forward(
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
        # Signature must match the installed Transformers attention ABI.
        layer = getattr(module, "layer_idx", -1)
        mask = attention_mask
        if layer >= 0 and key.shape[-2] > 12288:
            raise LayoutError("Attention would exceed fixed context12288")
        if layer >= 0 and state.blocks:
            causal = (
                is_causal
                if is_causal is not None
                else getattr(module, "is_causal", True)
            )
            mask = receiver_mask(
                query,
                key,
                mask,
                columns=torch.empty(0, dtype=torch.long),
                log_gain=0.0,
                causal=causal,
            )
            mask = mask + memory_bias(
                state.blocks,
                query,
                layer,
                key.shape[-2],
                log_gain=state.relay_log_gain,
            )
            state.receiver_calls += 1
            last_mask = mask[..., -1:, :] if mask is not None else None
            probabilities = full_attention(
                query[:, :, -1:, :],
                key,
                last_mask,
                scale=scaling if scaling is not None else query.shape[-1] ** -0.5,
            )
            if state.receiver_relation_gain > 0:
                relation_bias = relation_family_read_bias(
                    state.blocks,
                    probabilities,
                    query,
                    layer=layer,
                    length=key.shape[-2],
                    log_gain=state.receiver_relation_gain,
                )
                relation_mass = float(relation_bias.sum())
                if relation_mass > 0:
                    state.receiver_relation_calls += 1
                    state.receiver_relation_bias_mass += relation_mass
                last_bias = query.new_zeros(
                    query.shape[0],
                    query.shape[1],
                    query.shape[-2],
                    key.shape[-2],
                )
                last_bias[..., -1:, :] = relation_bias
                mask = mask + last_bias
                probabilities = full_attention(
                    query[:, :, -1:, :],
                    key,
                    mask[..., -1:, :],
                    scale=scaling if scaling is not None else query.shape[-1] ** -0.5,
                )
            if state.visual_columns.numel():
                visual = (
                    probabilities.index_select(
                        -1, state.visual_columns.to(query.device)
                    )
                    .sum(-1)
                    .mean()
                )
                state.receiver_mass.setdefault("original_visual", []).append(
                    float(visual)
                )
            for block in state.blocks:
                mass = (
                    probabilities.index_select(-1, block.columns.to(query.device))
                    .sum(-1)
                    .mean()
                )
                state.receiver_mass.setdefault(f"added_{block.sender}", []).append(
                    float(mass)
                )
        capture = state.capture
        if 8 <= layer <= 20 and capture is not None:
            if not capture.observing:
                capture.record_prompt_queries(layer, query, key.shape[-2])
            else:
                capture.query_heads = query.shape[1]
                probabilities = full_attention(
                    query,
                    key,
                    mask,
                    scale=scaling if scaling is not None else query.shape[-1] ** -0.5,
                )
                capture.record(layer, probabilities, key.shape[-2] - 1)
        return original(
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

    ALL_ATTENTION_FUNCTIONS["sdpa"] = forward
    try:
        yield None
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
