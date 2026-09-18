"""C2 13 Role-Endpoint Latent Handoff (RELH). Notion 3ddd771b-c9e2-8101.

Opt-in; off = no hooks, byte-identical. No selector, no duplication, no payload change:
the ONLY thing RELH changes is the positional phase at which the consumer reads ONE already
cached column per sender role - that role's terminal latent step (page §1, §3).

For a sender role r with T_r latent steps, the endpoint is the LAST latent column of that role
(architecture-defined role completion, not a score). At the consumer forward:
      K_bar = R(p_handoff) R(p_endpoint)^-1 K_endpoint        (exact RoPE rephase, §3)
      V_bar = V_endpoint                                      (payload untouched, §7.2)
  p_handoff = the coordinate just before the consumer's first logical position.
The cached tensors are NOT modified and NO column is appended, so cache cardinality is preserved
(§7.3, §7.4): the hook recomputes the consumer's attention with that one column's SCORE taken
from the rephased key, then reads the native values once.

Boundaries (§2): Navigator -> Reasoner and Reasoner -> Answerer. The Answerer has no latent
trajectory, so nothing is defined after it.

Env
  VLMAS_RELH=1|identity|answerer|reasoner
      1         both boundaries
      answerer  only Reasoner -> Answerer
      reasoner  only Navigator -> Reasoner
      identity  same recompute with p_handoff = p_endpoint (native output check)
  VLMAS_RELH_ROWS=gen|all   gen = last prompt row + generated rows (default)
"""
from __future__ import annotations

import contextlib
import os

import torch


def mode() -> str:
    v = os.environ.get("VLMAS_RELH", "").strip().lower()
    return v if v in ("1", "identity", "answerer", "reasoner") else ""


def enabled() -> bool:
    return mode() != ""


def config() -> dict:
    return {"mode": mode(), "rows": os.environ.get("VLMAS_RELH_ROWS", "gen").strip() or "gen"}


def active_at(stage: str) -> bool:
    """Is the rebinding applied when this consumer role runs?"""
    m = mode()
    if not m:
        return False
    if m in ("1", "identity"):
        return True
    return (m == "answerer" and stage == "answerer") or (m == "reasoner" and stage == "reasoner")


# --------------------------------------------------------------------------- bookkeeping


def record_endpoint(backbone, stage: str, cache_length: int, pos_cursor_after: int) -> None:
    """Remember the terminal latent column of a sender role (called right after its append).

    The latent steps occupy the cache tail, so the endpoint column is cache_length - 1 and its
    MRoPE coordinate is the last text position used, pos_cursor_after - 1 (all three axes equal).
    """
    if stage not in ("navigator", "reasoner"):
        return
    eps = dict(getattr(backbone, "_relh_endpoints", {}) or {})
    eps[stage] = {"col": int(cache_length) - 1, "pos": int(pos_cursor_after) - 1}
    backbone._relh_endpoints = eps


def endpoints_for(backbone, consumer: str) -> list[dict]:
    """Which sender endpoints this consumer should re-address (page §2)."""
    eps = getattr(backbone, "_relh_endpoints", {}) or {}
    sender = {"reasoner": "navigator", "answerer": "reasoner"}.get(consumer)
    out = []
    if sender and sender in eps:
        out.append({"sender": sender, **eps[sender]})
    return out


# --------------------------------------------------------------------------- runtime


@contextlib.contextmanager
def install_relh(backbone, endpoints, handoff_pos: int, *, rows="gen", identity=False):
    """Rebind each endpoint column's key view at every decoder layer; yields a stats dict."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
    from memory.restage import delta_cos_sin, rerotate_keys

    eps = [e for e in endpoints if int(e["col"]) >= 0]
    stats = {"calls": 0, "rows": 0, "endpoints": [(e["sender"], int(e["col"]), int(e["pos"]))
                                                  for e in eps],
             "handoff": int(handoff_pos), "shift": [], "w_native_sum": 0.0, "w_relh_sum": 0.0,
             "maxdiff_identity": 0.0, "skipped": 0}
    if not eps:
        yield stats
        return
    dev = backbone.device
    cols = torch.tensor([int(e["col"]) for e in eps], dtype=torch.long, device=dev)
    pos_list = [int(e["pos"]) for e in eps]
    old_pos = torch.tensor([pos_list, pos_list, pos_list], dtype=torch.long, device=dev)
    new_pos = old_pos.clone() if identity else torch.full_like(old_pos, int(handoff_pos))
    stats["shift"] = [int(new_pos[0, j]) - int(old_pos[0, j]) for j in range(len(eps))]
    sample = backbone.lm.layers[0].self_attn.q_proj.weight.new_zeros(1, 1, 1, 1)
    cos_o, sin_o = backbone.lm.rotary_emb(sample, old_pos.unsqueeze(1))
    cos_n, sin_n = backbone.lm.rotary_emb(sample, new_pos.unsqueeze(1))
    cos_d, sin_d = delta_cos_sin(cos_o.float(), sin_o.float(), cos_n.float(), sin_n.float())
    handles = []

    def make_post(li):
        def post(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys, values = cache.layers[li].keys, cache.layers[li].values
            L, T = int(keys.shape[2]), int(hidden.shape[1])
            if int(cols.max()) >= L:
                stats["skipped"] += 1
                return output
            sel = torch.arange(T, device=hidden.device) if rows == "all" else \
                torch.tensor([T - 1], device=hidden.device)
            Ts = int(sel.numel())
            Hd, Gq = module.head_dim, module.num_key_value_groups
            cos_pe, sin_pe = pe
            h_sel = hidden.index_select(1, sel)
            q = module.q_norm(module.q_proj(h_sel).view(1, Ts, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos_pe.index_select(1, sel),
                                        sin_pe.index_select(1, sel))
            up = (lambda t: t if t.dtype == torch.float64 else t.float())
            K = up(repeat_kv(keys, Gq))
            V = up(repeat_kv(values, Gq))
            qf = up(q)
            scores = torch.matmul(qf, K.transpose(-2, -1)) * module.scaling
            if T > 1:
                past = L - T
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + sel)[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            k_ep = keys.index_select(2, cols)
            k_bar = rerotate_keys(k_ep, cos_d.to(k_ep.dtype), sin_d.to(k_ep.dtype))
            K_bar = up(repeat_kv(k_bar, Gq))
            s_bar = torch.matmul(qf, K_bar.transpose(-2, -1)) * module.scaling   # [1,H,Ts,n_ep]
            native_alpha = torch.softmax(scores, dim=-1)
            new_scores = scores.clone()
            # the endpoint columns are causal for every consumer row (they precede it)
            new_scores[..., cols] = s_bar
            alpha = torch.softmax(new_scores, dim=-1)
            ctx = torch.matmul(alpha, V)
            flat = ctx.transpose(1, 2).reshape(1, Ts, -1)
            upd = module.o_proj(flat.to(module.o_proj.weight.dtype))
            with torch.no_grad():
                stats["w_native_sum"] += float(native_alpha[..., cols].sum(-1).mean())
                stats["w_relh_sum"] += float(alpha[..., cols].sum(-1).mean())
            if identity:
                stats["maxdiff_identity"] = max(stats["maxdiff_identity"], float(
                    (upd.float() - attn_out.index_select(1, sel).float()).abs().max()))
            new = attn_out.clone()
            new[:, sel, :] = upd.to(attn_out.dtype)
            stats["calls"] += 1
            stats["rows"] += Ts
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return post

    for li, layer in enumerate(backbone.lm.layers):
        handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()


def summary(stats: dict) -> str:
    c = max(stats["calls"], 1)
    return (f"calls={stats['calls']} rows={stats['rows']} endpoints={stats['endpoints']} "
            f"handoff={stats['handoff']} shift={stats['shift']} "
            f"w_native={stats['w_native_sum'] / c:.6f} w_relh={stats['w_relh_sum'] / c:.6f} "
            f"skipped={stats['skipped']} maxdiff_identity={stats['maxdiff_identity']:.3e}")


__all__ = ["mode", "enabled", "config", "active_at", "record_endpoint", "endpoints_for",
           "install_relh", "summary"]
