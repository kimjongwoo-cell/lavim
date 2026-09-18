"""Receiver-Conditioned Visual Paging (method draft 0910, Component 2).

Terminal visual access treated as a memory PLACEMENT problem, not attention
amplification. After Component 1 the retained visual memory is a set of
observation pages (one per acquired crop). At the terminal boundary:

  1. Receiver-side demand: run the existing answer-boundary token ONCE on the
     current cache to obtain the receiver query, then roll the single
     temporary KV entry back (length-record + truncate; no deepcopy). The
     query is captured PRE-RoPE (q_norm(q_proj(h))), which equals the spec's
     R^{-1}(xi_star) q_star — position-neutral by construction.
  2. Page demand E_p = mean over layers/heads of log-mean-exp of the scaled
     dot products between the position-neutral query and the page's
     position-neutral keys (cached keys un-rotated by their OWN stored MRoPE
     phase via the angle-subtraction identity with sin negated).
  3. Demand-to-recency: pages sorted ASCENDING by E_p — the most demanded
     observation lands nearest the answer boundary (rearrangement inequality
     under any monotone accessibility profile). No attention logit is touched.
  4. Placement: per-page rigid 3-D translation of the native (t,h,w)
     coordinates to sequential anchors (VLMAS_KV_RESTAGE_MROPE=1), so the
     demand order is realized POSITIONALLY; within-page spatial geometry is
     preserved exactly. Values are never modified.

Env: VLMAS_KV_RESTAGE_ORDER=demand (with VLMAS_KV_RESTAGE=1 VLMAS_KV_PARK=1).
Off = byte-identical; every other order mode untouched. VLMAS_KV_PAGING_LOG=1
prints per-page demand scores.
"""
from __future__ import annotations

import math
import os

import torch

from memory.aavm import reduce_query_heads
from memory.restage import rerotate_keys


def _pre_rope_queries(module, hidden: torch.Tensor) -> torch.Tensor:
    """q_norm(q_proj(h)) -> [B, H_q, L, D]; the model's own pre-RoPE queries."""
    shape = (*hidden.shape[:-1], -1, module.head_dim)
    return module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)


def _num_kv_heads(module) -> int:
    return int(module.k_proj.out_features) // int(module.head_dim)


@torch.no_grad()
def probe_receiver_query(backbone, cache, position_cursor: int) -> dict[int, torch.Tensor]:
    """Receiver query from ONE forward of the answer-boundary token, rolled back.

    Returns {layer: [H_kv, D]} pre-RoPE queries (GQA group-mean over the query
    heads each KV head serves). The probe appends exactly one KV column per
    layer, which is truncated immediately after (length-record + truncate —
    the trim-restore pattern; no deepcopy).
    """
    tokenizer = backbone.processor.tokenizer
    boundary = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if boundary is None or boundary < 0:
        boundary = int(tokenizer.eos_token_id)
    ids = torch.tensor([[boundary]], device=backbone.device, dtype=torch.long)

    lens = [int(layer.keys.shape[2]) for layer in cache.layers]
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(module, args, kwargs, _output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                return
            raw = _pre_rope_queries(module, hidden)      # [1, H_q, 1, D]
            u_q = raw[0].mean(dim=1).float()             # [H_q, D]
            captured[layer_index] = reduce_query_heads(
                u_q, _num_kv_heads(module)).cpu()        # [H_kv, D]

        return hook

    for index, layer in enumerate(backbone.lm.layers):
        handles.append(layer.self_attn.register_forward_hook(
            make_hook(index), with_kwargs=True))
    try:
        _ = backbone.lm(
            inputs_embeds=backbone.lm.embed_tokens(ids),
            position_ids=backbone._make_position_ids(1, past_len=position_cursor),
            past_key_values=cache,
            use_cache=True,
        )
    finally:
        for handle in handles:
            handle.remove()
        for layer, length in zip(cache.layers, lens):
            if int(layer.keys.shape[2]) > length:
                layer.keys = layer.keys[:, :, :length, :]
                layer.values = layer.values[:, :, :length, :]
    return captured


@torch.no_grad()
def page_demands(
    backbone,
    park_bank: dict,
    kept_counts: list[int],
    queries: dict[int, torch.Tensor],
) -> list[float]:
    """E_p per observation page: mean over (layer, head) of log-mean-exp.

    Bank keys are cached POST-RoPE at their stored coordinates; they are
    un-rotated by their own phase (rotation by -theta: same cos, negated sin)
    so content compatibility never mixes with cache distance.
    """
    old_pos = park_bank["pos"].to(device=backbone.device, dtype=torch.long)
    sample = park_bank["k"][0]
    cos_old, sin_old = backbone.lm.rotary_emb(sample, old_pos.unsqueeze(1))
    cos_old = cos_old.float()
    sin_old = sin_old.float()

    spans = []
    offset = 0
    for count in kept_counts:
        spans.append((offset, offset + count))
        offset += count

    totals = [0.0] * len(kept_counts)
    layers_used = 0
    for layer_index, keys in enumerate(park_bank["k"]):
        query = queries.get(layer_index)
        if query is None:
            continue
        layers_used += 1
        neutral = rerotate_keys(keys.float(), cos_old, -sin_old)[0]  # [H, n, D]
        q = query.to(device=neutral.device, dtype=neutral.dtype)     # [H, D]
        dim = q.shape[-1]
        logits = torch.einsum("hd,hnd->hn", q, neutral) / math.sqrt(dim)
        for page_index, (start, end) in enumerate(spans):
            if end <= start:
                continue
            block = logits[:, start:end]                              # [H, n_p]
            lme = torch.logsumexp(block, dim=1) - math.log(end - start)
            totals[page_index] += float(lme.mean().item())
    if layers_used == 0:
        return []
    return [total / layers_used for total in totals]


def demand_token_permutation(
    demands: list[float],
    kept_counts: list[int],
) -> tuple[list[int], list[int]]:
    """Ascending-demand page order -> token-level permutation.

    Lowest demand placed first (least accessible slot); highest demand lands
    at recency. Returns (token permutation, page counts in the new order).
    """
    order = sorted(range(len(demands)), key=lambda i: demands[i])
    starts = []
    offset = 0
    for count in kept_counts:
        starts.append(offset)
        offset += count
    perm: list[int] = []
    for page_index in order:
        perm.extend(range(starts[page_index],
                          starts[page_index] + kept_counts[page_index]))
    return perm, [kept_counts[i] for i in order]


def page_anchor_positions(
    old_pos: torch.Tensor,
    page_counts: list[int],
    position_cursor: int,
) -> tuple[torch.Tensor, int]:
    """Sequential per-page anchors realizing the demand order positionally.

    VLMAS_KV_RESTAGE_MROPE=1: each page's native (t,h,w) block is translated
    as a rigid body so its minimum lands on the running cursor — within-page
    geometry preserved exactly, pages strictly ordered on the position axis.
    Otherwise the default 1-D convention (sequential arange over the permuted
    token order) already realizes the order, matching restage's default path.
    """
    device = old_pos.device
    n = int(old_pos.shape[-1])
    if os.environ.get("VLMAS_KV_RESTAGE_MROPE", "").strip() != "1":
        seq = torch.arange(position_cursor, position_cursor + n, device=device)
        return seq.unsqueeze(0).expand(3, -1), position_cursor + n
    blocks = []
    cursor = int(position_cursor)
    offset = 0
    for count in page_counts:
        page = old_pos[:, offset:offset + count]
        base = page.min(dim=1, keepdim=True).values
        moved = page - base + cursor
        blocks.append(moved)
        cursor = int(moved.max().item()) + 1
        offset += count
    return torch.cat(blocks, dim=1), cursor


@torch.no_grad()
def demand_page_order(
    backbone,
    cache,
    park_bank: dict,
    kept_counts: list[int],
    position_cursor: int,
):
    """Full Component-2 decision. Returns (perm, page_counts_new, demands) or
    (None, None, None) when paging cannot apply (fail-safe to original order)."""
    if len(kept_counts) < 2:
        return None, None, None
    if sum(kept_counts) != int(park_bank["pos"].shape[-1]):
        return None, None, None
    queries = probe_receiver_query(backbone, cache, position_cursor)
    if not queries:
        return None, None, None
    demands = page_demands(backbone, park_bank, kept_counts, queries)
    if len(demands) != len(kept_counts):
        return None, None, None
    perm, page_counts_new = demand_token_permutation(demands, kept_counts)
    return perm, page_counts_new, demands


__all__ = [
    "demand_page_order",
    "demand_token_permutation",
    "page_anchor_positions",
    "page_demands",
    "probe_receiver_query",
]
