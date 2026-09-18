"""C2 12.7 Physical-Support Fisher Projection (PSFP). Notion 3ddd771b-c9e2-8141.

Opt-in; off = no hooks, byte-identical. Nothing is learned, no extra forward / crop encoding /
decode / latent step; hidden states, KV, RoPE positions and the Reasoner trajectory are untouched.
The ONLY thing PSFP changes is the terminal logit vector, and only when the receiver's decision
remainder directly opposes the physical-support visual direction.

Per decode step (and the last prompt row), at the fixed formation layers L24/L27/L30 (page §1):
  native, per head:  alpha = softmax over ALL columns; alpha_{j|g} = softmax inside support g
  residual space (o_proj = W_O, linear):
      m_V^(l) = W_O[concat_h sum_{j in V} alpha_j v_j]              native visual term   (§2)
      S^(l)   = W_O[concat_h sum_g sum_{j in g} alpha_{j|g} v_j]    support-conditioned  (§2)
  direct-logit map with the native final RMSNorm scale (§3):
      Phi(c) = W_U ( gamma * c / rms(h_T) )       (linear in c once rms(h_T) is fixed)
      e_V = sum_l Phi(m_V^(l)) ,  s_V = sum_l Phi(S^(l))
  remainder (§4):  r = z - e_V           (NOT a text-only logit vector; just z minus e_V)
  Fisher geometry of the native distribution p = softmax(z) (§5):
      <a,b>_F = sum_w p_w a_w b_w - (sum_w p_w a_w)(sum_w p_w b_w)
      <r,s_V>_F >= 0  ->  identity
      <r,s_V>_F <  0  ->  z~ = z - (<r,s_V>_F / (<s_V,s_V>_F + eps)) s_V
The correction has no gain constant, is invariant to positive rescaling of s_V, and is invariant
to a constant logit shift (§6). Support g = one retained physical observation (`_visual_meta.obs_id`).

Env
  VLMAS_PSFP=1|identity|diag   identity/diag = measure the conflict, leave the logits native
  VLMAS_PSFP_LAYERS=24,27,30
  VLMAS_PSFP_ROWS=gen|all
  VLMAS_PSFP_SUPPORT=obs|branch
"""
from __future__ import annotations

import contextlib
import os

import torch

EPS = 1e-12


def enabled() -> bool:
    return os.environ.get("VLMAS_PSFP", "").strip() in ("1", "identity", "diag")


def active() -> bool:
    """True only when the logits are actually corrected."""
    return os.environ.get("VLMAS_PSFP", "").strip() == "1"


def config() -> dict:
    return {
        "mode": os.environ.get("VLMAS_PSFP", "").strip(),
        "layers": os.environ.get("VLMAS_PSFP_LAYERS", "24,27,30").strip() or "24,27,30",
        "rows": os.environ.get("VLMAS_PSFP_ROWS", "gen").strip() or "gen",
        "support": os.environ.get("VLMAS_PSFP_SUPPORT", "obs").strip() or "obs",
    }


def layer_set(spec: str, n_layers: int) -> tuple[int, ...]:
    from memory.lrpsr import layer_set as _ls

    return _ls(spec, n_layers)


# --------------------------------------------------------------------------- pure math


def fisher_inner(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """<a,b>_F = a^T (diag(p) - p p^T) b, without building F. p, a, b are [..., V].

    Computed in the p-centred form sum_w p_w (a_w - E_p a)(b_w - E_p b): algebraically identical
    to the expanded form in the page, but <s,s>_F can never come out negative from rounding.
    """
    a_c = a - (p * a).sum(-1, keepdim=True)
    b_c = b - (p * b).sum(-1, keepdim=True)
    return (p * a_c * b_c).sum(-1)


def correct_logits(z: torch.Tensor, e_V: torch.Tensor, s_V: torch.Tensor):
    """Minimum-KL projection of the remainder r = z - e_V onto {<u, s_V>_F >= 0} (page §5).

    z, e_V, s_V are [V] (float32). Returns (z_tilde, diag).
    """
    p = torch.softmax(z, dim=-1)
    r = z - e_V
    rs = fisher_inner(p, r, s_V)
    ss = fisher_inner(p, s_V, s_V)
    # 0 when the remainder already supports s_V, and also when s_V carries no Fisher direction
    # (a constant logit vector), where the constraint is vacuous.
    k = torch.where(ss > EPS, torch.clamp(rs, max=0.0) / (ss + EPS), torch.zeros_like(ss))
    z_t = z - k * s_V
    delta = (z_t - z)
    return z_t, {
        "conflict": float(rs),
        "s_norm_F": float(ss),
        "activated": bool(rs < 0),
        "k": float(k),
        "delta_max": float(delta.abs().max()),
        "argmax_changed": bool(int(z_t.argmax()) != int(z.argmax())),
    }


# --------------------------------------------------------------------------- runtime


def engine_groups(engine, mode: str) -> dict:
    from memory import psar as _psar

    return _psar.engine_groups(engine, mode)


class _State:
    def __init__(self, n_groups: int):
        self.mV = None      # [d] residual-space native visual term, summed over layers
        self.S = None       # [d] residual-space support-sum term, summed over layers
        self.rms = None     # scalar, native final-norm scale of the scored row
        self.seen = 0
        self.stats = {"steps": 0, "activated": 0, "flips": 0, "conflict_sum": 0.0,
                      "delta_sum": 0.0, "k_sum": 0.0, "skipped": 0, "n_groups": n_groups}

    def reset(self):
        self.mV = None
        self.S = None
        self.seen = 0


@contextlib.contextmanager
def install_psfp(backbone, visual_cols, groups, *, layers=(24, 27, 30), rows="gen",
                 apply=True):
    """Read the formation-window support messages, then correct the terminal logits.

    Hooks: (1) post-hook on each formation layer's self_attn -> residual-space m_V and S of the
    scored row; (2) pre-hook on the final RMSNorm -> the native rms of that row; (3) post-hook on
    lm_head -> the Fisher projection on the scored row's logits. Off/`apply=False` leaves the
    logits native and only records the conflict.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    cols = visual_cols.to(device=backbone.device, dtype=torch.long)
    st = _State(len(groups))
    handles = []
    lset = tuple(layers)

    def make_attn_post(li):
        def post(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or cache is None:
                return output
            keys, values = cache.layers[li].keys, cache.layers[li].values
            L, T = int(keys.shape[2]), int(hidden.shape[1])
            vis = cols[cols < L]
            if int(vis.numel()) == 0 or int(vis.numel()) >= L:
                return output
            if li == lset[0]:
                st.reset()
            sel = torch.tensor([T - 1], device=hidden.device)
            Hd, Gq = module.head_dim, module.num_key_value_groups
            cos_pe, sin_pe = pe
            h_sel = hidden.index_select(1, sel)
            q = module.q_norm(module.q_proj(h_sel).view(1, 1, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos_pe.index_select(1, sel),
                                        sin_pe.index_select(1, sel))
            up = (lambda t: t if t.dtype == torch.float64 else t.float())
            K = up(repeat_kv(keys, Gq))
            V = up(repeat_kv(values, Gq))
            scores = torch.matmul(up(q), K.transpose(-2, -1)) * module.scaling       # [1,H,1,L]
            alpha = torch.softmax(scores, dim=-1)
            V_vis = V[:, :, vis, :]
            o_v = torch.matmul(alpha[..., vis], V_vis)

            def wo(x):
                y = module.o_proj(x.transpose(1, 2).reshape(1, 1, -1)
                                  .to(module.o_proj.weight.dtype))[0, 0]
                return y if y.dtype == torch.float64 else y.float()

            from memory.lrpsr import support_messages
            m_sum = sum(support_messages(scores[..., vis], V_vis, groups))
            mV = wo(o_v)
            S = wo(m_sum)
            st.mV = mV if st.mV is None else st.mV + mV
            st.S = S if st.S is None else st.S + S
            st.seen += 1
            return output
        return post

    def norm_pre(module, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        if h is None or h.dim() != 3:
            return None
        last = h[0, -1] if h.dtype == torch.float64 else h[0, -1].float()
        var = last.pow(2).mean()
        st.rms = (var + float(getattr(module, "variance_epsilon", 1e-6))).sqrt()
        st.gamma = module.weight
        return None

    def head_post(module, args, output):
        if st.seen != len(lset) or st.rms is None or st.mV is None:
            st.stats["skipped"] += 1
            st.reset()
            return output
        logits = output
        z = logits[0, -1] if logits.dtype == torch.float64 else logits[0, -1].float()
        gamma = st.gamma if st.gamma.dtype == torch.float64 else st.gamma.float()
        W = module.weight
        proj = torch.stack([st.mV, st.S], 0) * gamma / st.rms          # [2, d]
        both = proj.to(W.dtype) @ W.transpose(0, 1)                      # [2, V]
        both = both if both.dtype == torch.float64 else both.float()
        e_V, s_V = both[0], both[1]
        z_t, diag = correct_logits(z, e_V, s_V)
        st.stats["steps"] += 1
        st.stats["activated"] += int(diag["activated"])
        st.stats["flips"] += int(diag["argmax_changed"])
        st.stats["conflict_sum"] += diag["conflict"]
        st.stats["delta_sum"] += diag["delta_max"]
        st.stats["k_sum"] += diag["k"]
        st.reset()
        if not apply or not diag["activated"]:
            return output
        new = logits.clone()
        new[0, -1] = z_t.to(logits.dtype)
        return new

    for li, layer in enumerate(backbone.lm.layers):
        if li in lset:
            handles.append(layer.self_attn.register_forward_hook(make_attn_post(li),
                                                                 with_kwargs=True))
    handles.append(backbone.lm.norm.register_forward_pre_hook(norm_pre, with_kwargs=True))
    handles.append(backbone.model.lm_head.register_forward_hook(head_post))
    try:
        yield st.stats
    finally:
        for h in handles:
            h.remove()


def summary(stats: dict) -> str:
    s = max(stats["steps"], 1)
    a = max(stats["activated"], 1)
    return (f"steps={stats['steps']} supports={stats['n_groups']} "
            f"activated={stats['activated']} ({100.0 * stats['activated'] / s:.1f}%) "
            f"argmax_flips={stats['flips']} mean_conflict={stats['conflict_sum'] / s:.4e} "
            f"mean_k(active)={stats['k_sum'] / a:.4e} mean_delta_max={stats['delta_sum'] / s:.4f} "
            f"skipped={stats['skipped']}")


__all__ = ["enabled", "active", "config", "layer_set", "fisher_inner", "correct_logits",
           "engine_groups", "install_psfp", "summary"]
