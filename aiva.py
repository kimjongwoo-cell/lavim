"""AIVA — Append-Invariant Visual Access (method draft 0909).

One principle: a memory group must not gain total attention evidence merely by
holding more tokens. Vanilla attention compares the native partition sums

    Z_V = sum_{i in V} exp(s_i),     Z_C = sum_{j in C_t} exp(s_j)

so a 2048-token visual block and a 300-token text context are compared by
cardinality as much as by compatibility. AIVA compares the group MEANS instead,

    Zbar_V = Z_V / |V|,             Zbar_C = Z_C / |C_t|

keeping vanilla attention inside each group. The resulting distribution

    alpha~_j = pi_{g(j)} * beta_{j|g(j)}

is exactly a softmax over count-corrected logits

    alpha~_j = softmax_j( s_j - log |G(j)| )

so the whole method is one additive per-group logit bias: no parameters, no
temperature, no threshold. Because softmax is shift invariant we implement the
equivalent single-sided form -- visual columns get +log(|C_t|/|V_t|), every
other column 0 -- which needs one bias row per query and nothing else.

|V_t| and |C_t| are the CAUSALLY VISIBLE counts at query position t, so the
correction tracks the context as it grows (append-invariance is the point).

Env:
    VLMAS_AIVA=1        enable (off = no hooks, byte-identical)
    VLMAS_AIVA_LAYERS   all | comma list (default all)
    VLMAS_AIVA_CONTROL=inverse   falsification arm: bias sign flipped, i.e.
                                 rewarding the larger group instead
"""
from __future__ import annotations

import contextlib
import math
import os

import torch


def enabled() -> bool:
    return os.environ.get("VLMAS_AIVA", "") == "1"


def _layers(backbone) -> list[int]:
    spec = os.environ.get("VLMAS_AIVA_LAYERS", "all").strip()
    total = len(backbone.lm.layers)
    if spec in ("", "all"):
        return list(range(total))
    picked = [int(p) for p in spec.replace(" ", "").split(",") if p]
    return [i for i in picked if 0 <= i < total]


def group_logit_bias(
    visual_columns: torch.Tensor,
    *,
    past_len: int,
    query_len: int,
    kv_len: int,
    device,
    dtype,
) -> torch.Tensor:
    """[1, 1, query_len, kv_len] additive bias implementing -log|G(j)|.

    Row r is the query at absolute position past_len + r; the visual columns
    visible to it get log(|C_t| / |V_t|) and everything else 0, which differs
    from subtracting log|G| on both groups only by a per-row constant that the
    softmax removes.
    """
    bias = torch.zeros((1, 1, query_len, kv_len), device=device, dtype=torch.float32)
    if visual_columns.numel() == 0:
        return bias.to(dtype)
    cols = visual_columns.to(device=device, dtype=torch.long)
    cols = cols[cols < kv_len]
    if cols.numel() == 0:
        return bias.to(dtype)
    sorted_cols, _ = torch.sort(cols)
    positions = torch.arange(past_len, past_len + query_len, device=device)
    # visible visual tokens per query row (all visual columns <= t)
    visible_visual = torch.searchsorted(
        sorted_cols, positions, right=True).clamp_min(1)
    visible_total = (positions + 1).clamp_min(1)
    visible_context = (visible_total - visible_visual).clamp_min(1)
    scale = torch.log(visible_context.float()) - torch.log(visible_visual.float())
    if os.environ.get("VLMAS_AIVA_CONTROL", "") == "inverse":
        scale = -scale
    bias[0, 0][:, sorted_cols] = scale.unsqueeze(1).to(bias.dtype)
    return bias.to(dtype)


def _causal_mask(query_len: int, kv_len: int, past_len: int, device, dtype):
    """Additive causal mask when the model handed the layer none (sdpa path)."""
    rows = torch.arange(past_len, past_len + query_len, device=device).unsqueeze(1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    blocked = cols > rows
    mask = torch.zeros((1, 1, query_len, kv_len), device=device, dtype=torch.float32)
    mask.masked_fill_(blocked.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask.to(dtype)


@contextlib.contextmanager
def apply_aiva(backbone):
    """Add the per-group logit bias to every attention call while open.

    The visual columns are read from ``backbone._aiva_visual_cols`` at CALL
    time, so the same open context stays correct across the consolidation
    boundary (which renumbers the retained columns).
    """
    if not enabled():
        yield False
        return
    layers = _layers(backbone)
    handles: list[object] = []
    stats = {"calls": 0}

    def pre_hook(module, args, kwargs):
        cols = getattr(backbone, "_aiva_visual_cols", None)
        if cols is None or int(cols.numel()) == 0:
            return None
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        if hidden is None:
            return None
        query_len = int(hidden.shape[1])
        past_len = 0
        if cache is not None and getattr(cache, "layers", None):
            keys = cache.layers[module.layer_idx].keys
            past_len = 0 if keys is None else int(keys.shape[2])
        kv_len = past_len + query_len
        mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        dtype = hidden.dtype
        bias = group_logit_bias(
            cols, past_len=past_len, query_len=query_len, kv_len=kv_len,
            device=hidden.device, dtype=dtype,
        )
        if mask is None:
            merged = _causal_mask(
                query_len, kv_len, past_len, hidden.device, dtype) + bias
        else:
            merged = mask[..., :kv_len].to(dtype) + bias
        stats["calls"] += 1
        if "attention_mask" in kwargs or len(args) < 3:
            kwargs["attention_mask"] = merged
            return args, kwargs
        new_args = list(args)
        new_args[2] = merged
        return tuple(new_args), kwargs

    for index in layers:
        handles.append(
            backbone.lm.layers[index].self_attn.register_forward_pre_hook(
                pre_hook, with_kwargs=True))
    try:
        yield True
    finally:
        for handle in handles:
            handle.remove()
        control = os.environ.get("VLMAS_AIVA_CONTROL", "") or "none"
        print(f"[AIVA] biased {stats['calls']} attention calls over "
              f"{len(layers)} layers (control={control})", flush=True)


def install_aiva(backbone) -> list:
    """Non-contextmanager form: install the hooks, return handles to remove.

    Used where the biased region spans a prefill and a latent-step loop that
    are not inside one indentation block.
    """
    if not enabled():
        return []
    manager = apply_aiva(backbone)
    entered = manager.__enter__()
    if not entered:
        manager.__exit__(None, None, None)
        return []
    return [manager]


def remove_aiva(handles: list) -> None:
    for manager in handles or []:
        manager.__exit__(None, None, None)


__all__ = ["apply_aiva", "enabled", "group_logit_bias",
           "install_aiva", "remove_aiva"]
