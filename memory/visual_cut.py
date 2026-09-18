"""Visual-contribution cut at the attention-output level (opt-in; off = no hooks).

Env
  VLMAS_VCUT=zero|mask|identity   zero: keep native alpha over V u C but drop the
                                  visual term  r_V = sum_{j in V} alpha_j v_j W_O
                                  (attention mass is spent, content is not read)
                                  mask: exclude visual columns from the softmax
                                  (visual KV access blocked, renormalised)
                                  identity: recompute without any cut (plumbing
                                  check: output must match native up to bf16)
  VLMAS_VCUT_STAGE=R|A|RA         R = Reasoner latent steps (one row per step),
                                  A = Answerer prompt prefill (all rows) + every
                                  generated token (+ VER score-only scoring)
  VLMAS_VCUT_LOG=1                per-context stats line
  VLMAS_VCUT_LAYERS=a-b           restrict the cut to decoder layers a..b
                                  (inclusive; default all) -- depth scan

Text / latent K/V columns are never touched; only the query rows of the chosen
stage lose (zero/mask) their visual read. Every LM layer is hooked.
"""
from __future__ import annotations

import contextlib
import os

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv


def mode() -> str:
    return os.environ.get("VLMAS_VCUT", "").strip()


def stages() -> str:
    return os.environ.get("VLMAS_VCUT_STAGE", "A").strip().upper()


def enabled(stage: str) -> bool:
    return mode() in ("zero", "mask", "identity") and stage.upper() in stages()


@contextlib.contextmanager
def install_visual_cut(backbone, visual_cols, stage: str):
    if not enabled(stage) or visual_cols is None or int(visual_cols.numel()) == 0:
        yield None
        return
    m = mode()
    log = os.environ.get("VLMAS_VCUT_LOG", "") == "1"
    layers = backbone.lm.layers
    stats = {"calls": 0, "rows": 0, "rho_sum": 0.0, "maxdiff": 0.0}
    handles = []

    def make_post(li):
        def post(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys = cache.layers[li].keys
            values = cache.layers[li].values
            L = int(keys.shape[2])
            T = int(hidden.shape[1])
            vis = visual_cols.to(device=keys.device, dtype=torch.long)
            vis = vis[vis < L]
            if int(vis.numel()) == 0 or int(vis.numel()) >= L:
                return output
            vmask = torch.zeros(L, dtype=torch.bool, device=keys.device)
            vmask[vis] = True
            Hd = module.head_dim
            G = module.num_key_value_groups
            cos, sin = pe
            q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)  # [1,Hq,T,D]
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            K = repeat_kv(keys, G).float()
            V = repeat_kv(values, G).float()
            scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling  # [1,Hq,T,L]
            if T > 1:  # causal mask for a multi-row (prompt prefill) call
                past = L - T
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + torch.arange(T, device=keys.device))[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            if m == "mask":
                scores = scores.masked_fill(vmask[None, None, None, :], float("-inf"))
                alpha = torch.softmax(scores, dim=-1)
                ctx = torch.matmul(alpha, V)
                rho = alpha.new_zeros(())
            else:
                alpha = torch.softmax(scores, dim=-1)
                rho = alpha[..., vmask].sum(-1)                       # [1,Hq,T]
                keep = (~vmask).float()[None, None, None, :] if m == "zero" else 1.0
                ctx = torch.matmul(alpha * keep, V)
            flat = ctx.transpose(1, 2).reshape(1, T, -1)
            new = module.o_proj(flat.to(module.o_proj.weight.dtype))
            stats["calls"] += 1
            stats["rows"] += T
            stats["rho_sum"] += float(rho.mean()) if m != "mask" else 0.0
            if log or m == "identity":
                stats["maxdiff"] = max(stats["maxdiff"], float((new.float() - attn_out.float()).abs().max()))
            new = new.to(attn_out.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return post

    band = os.environ.get("VLMAS_VCUT_LAYERS", "").strip()
    lo, hi = 0, len(layers) - 1
    if band:
        a, b = band.split("-")
        lo, hi = int(a), int(b)
    for li, layer in enumerate(layers):
        if lo <= li <= hi:
            handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()
        if stats["calls"]:
            print(f"[VCut:{stage}:{m}:L{lo}-{hi}] {stats['calls']} attn calls, {stats['rows']} rows, "
                  f"mean rho_V {stats['rho_sum']/stats['calls']:.4f}, "
                  f"max|out-native| {stats['maxdiff']:.3e}", flush=True)


__all__ = ["enabled", "mode", "stages", "install_visual_cut"]
