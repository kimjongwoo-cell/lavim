"""One SDPA call per layer, with optional sender observation and receiver bias."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Unpack

import torch
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from pvcr.capture import Capture
from pvcr.core import contribution, full_attention
from pvcr.errors import LayoutError
from pvcr.receiver import RelayBank, receiver_mask


@dataclass
class AttentionState:
    """Mutable role scopes shared by hooks within one isolated process."""

    capture: Capture | None = None
    receiver: RelayBank | None = None
    receiver_calls: int = 0
    log_gain: float = 0.6931471805599453


@contextmanager
def attention_adapter(state: AttentionState) -> Iterator[None]:
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
        layer = getattr(module, "layer_idx", -1)
        active = 8 <= layer <= 20
        mask = attention_mask
        if active and state.receiver is not None:
            mask = receiver_mask(
                query,
                key,
                mask,
                columns=state.receiver.columns,
                log_gain=state.log_gain,
                causal=is_causal
                if is_causal is not None
                else getattr(module, "is_causal", True),
            )
            state.receiver_calls += 1
        capture = state.capture
        if active and capture is not None and capture.observing:
            if capture.layout is None or dropout != 0:
                raise LayoutError(
                    "PVCR requires prepared deterministic sender attention"
                )
            probabilities = full_attention(
                query,
                key,
                mask,
                scale=scaling if scaling is not None else query.shape[-1] ** -0.5,
            )
            result = contribution(
                probabilities,
                value,
                capture.layout,
                grouped=capture.variant != "visual",
            )
            capture.record(
                layer,
                result,
                value[:, :, -1],
                key.shape[-2] - 1,
                query_heads=query.shape[1],
            )
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
