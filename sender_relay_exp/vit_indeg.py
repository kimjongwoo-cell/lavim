"""ViT-internal salience surrogate for backbones with no [CLS] token.

Why this exists
---------------
SCoRe (CVPR 2026) defines token salience as "the attention score of token v_i to the
[CLS] token", computed inside the Vision Encoder. The paper is explicit that the
alternative — cross-modal attention inside the LLM — is the thing it is arguing against:

    "using early-layer cross-modal attention as a pruning criterion is unreliable, as
     these signals are highly dispersed and inherently biased by positional encodings"

Qwen3-VL's vision tower has NO [CLS] token (no cls_token / class_embedding anywhere in
Qwen3VLVisionModel; it is patch_embed + pos_embed + 24 blocks + a 2x2 merger), so the
paper's definition cannot be transplanted literally. The natural generalisation, and the
only one that stays inside the vision encoder, is:

    [CLS] attention = "how much attention ONE distinguished query gives each patch"
    in-degree       = "how much attention ALL queries give each patch"

This module computes the second. Two properties make it usable here rather than a
hand-wave: Qwen3-VL's vision attention is FULL (no windowing anywhere in the module or
the config), so in-degree is a global quantity; and the tower's attention is unmasked and
non-causal, so every block's attention is a genuine row-stochastic matrix whose column
means form a distribution over patches.

It is a surrogate, not a reproduction. CLIP's [CLS] carries global semantics because
contrastive training put it there; nothing trained Qwen3-VL's patches to concentrate
attention the same way. Expect attention sinks — this codebase has already recorded
sink-dominated readouts elsewhere — and treat a sink-heavy result as a finding, not a bug.

The vision tower discards its attention weights (Qwen3VLVisionAttention.forward returns
only attn_output and never plumbs output_attentions), so `collect` re-derives them from
the module's own inputs with the module's own projections and rotary embedding.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- pure


def indegree(q: torch.Tensor, k: torch.Tensor, scaling: float) -> torch.Tensor:
    """Attention each KEY receives, averaged over heads and query rows.

    q, k: [heads, len, dim]. Returns [len], summing to 1 — a distribution over patches.
    """
    # promote to at least float32 (bf16 cannot hold a 1024-way distribution), but never
    # cast the result back down — the caller averages thousands of these
    dt = torch.promote_types(q.dtype, torch.float32)
    logits = (q.to(dt) @ k.to(dt).transpose(-1, -2)) * scaling
    return logits.softmax(dim=-1).mean(dim=(0, 1))


def pool_to_merged(scores: torch.Tensor, unit: int) -> torch.Tensor:
    """Patch scores -> merged-token scores, summing each group of `unit` patches.

    The processor lays patches out so that the `spatial_merge_size**2` patches feeding one
    merged token are consecutive — the same assumption the ViT-feature pooling in
    vistok_umap.py already makes (`vit.view(n_vis, 4, -1).mean(1)`). Summing, not
    averaging, keeps the total attention mass unchanged.
    """
    n = int(scores.numel())
    if unit <= 0 or n % unit:
        raise ValueError(f"{n} scores is not a multiple of merge unit {unit}")
    return scores.reshape(n // unit, unit).sum(dim=1)


def chunk_bounds(cu_seqlens) -> list[tuple[int, int]]:
    """cu_seqlens -> per-image [start, end) spans, dropping empty ones."""
    cu = [int(v) for v in cu_seqlens]
    return [(a, b) for a, b in zip(cu[:-1], cu[1:]) if b > a]


# ------------------------------------------------------------------------ runtime


def _attn_modules(visual):
    return [(i, blk.attn) for i, blk in enumerate(visual.blocks)]


@torch.no_grad()
def collect(model, pixel_values, grid_thw, merge_unit: int = 4, blocks=None):
    """Per-merged-token in-degree from the vision tower.

    Returns (pooled [n_merged] normalised to sum 1, per_block [n_blocks, n_merged] raw).
    `blocks` optionally restricts which vision blocks are averaged.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    visual = model.model.visual
    grabbed: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx, module):
        def hook(_m, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cu = kwargs.get("cu_seqlens")
            pos = kwargs.get("position_embeddings")
            if hidden is None or cu is None or pos is None:
                return None
            seq = hidden.shape[0]
            qkv = module.qkv(hidden).reshape(seq, 3, module.num_heads, -1).permute(1, 0, 2, 3)
            q, k, _v = qkv.unbind(0)                       # each [seq, heads, dim]
            cos, sin = pos
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q = q.transpose(0, 1)                           # [heads, seq, dim]
            k = k.transpose(0, 1)
            out = torch.zeros(seq, dtype=torch.float32, device=hidden.device)
            for a, b in chunk_bounds(cu):
                out[a:b] = indegree(q[:, a:b], k[:, a:b], module.scaling).float()
            grabbed[idx] = out.cpu()
            return None
        return hook

    for i, attn in _attn_modules(visual):
        if blocks is not None and i not in blocks:
            continue
        handles.append(attn.register_forward_pre_hook(make_hook(i, attn), with_kwargs=True))
    try:
        model.model.get_image_features(pixel_values, grid_thw, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    if not grabbed:
        raise RuntimeError("no vision attention captured — the hook never fired")
    order = sorted(grabbed)
    per_block = torch.stack([pool_to_merged(grabbed[i], merge_unit) for i in order])
    pooled = per_block.mean(dim=0)
    return pooled / pooled.sum(), per_block
