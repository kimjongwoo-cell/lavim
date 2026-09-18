"""C2 12.6 Latent-Referenced Physical-Support Residual Readout (LR-PSR). Notion 3dcd771b-c9e2-81c6.

Opt-in; off = no hooks, byte-identical. Receiver-side readout inside ONE native Answerer forward:
no crop re-encoding, no extra forward, no Answerer latent step, no KV relocation, nothing learned.

At a fixed Answerer formation layer l (page §1: L24 / L27 / L30) and a selected query row:
  native, per head h:  alpha = softmax_j(q k_j / sqrt d)  over ALL columns
                       o_v = sum_{j in V} alpha_j v_j ,  o_n = o - o_v
  residual space (o_proj = W_O, linear):
      m_V  = W_O[concat_h o_v_h]                       native visual message      (§4)
      m_NV = W_O[concat_h o_n_h]                       non-visual sources         (§6)
      m_g  = W_O[concat_h sum_{j in g} softmax_{j in g}(q k_j / sqrt d) v_j]      (§3)
      c_t  = W_O[concat_h V_{r_t,h}]  for the Reasoner latent columns r_t          (§2)
  reference subspace: U_R = orth([c_1 ... c_T]) by numerical rank, P = U_R U_R^T, Q = I - P
      m_V,par = P m_V ,  m_V,perp = Q m_V ,  r_g = Q m_g ,  rbar = (1/G) sum_g r_g
      rhat    = ||m_V,perp|| * rbar / ||rbar||        (rbar = 0 -> rhat = m_V,perp)
      m~_V    = m_V,par + rhat                                                     (§4)
  output row = m_NV + m~_V                                                          (§6)

Conserved exactly (§5, unit-tested): P m~_V = P m_V, ||Q m~_V|| = ||Q m_V||, ||m~_V|| = ||m_V||.
The native attention contribution of question / latent / role / generated rows, the raw visual
K/V, RoPE positions, the C1 token budget and the Reasoner trajectory are untouched.

Support g = one retained physical observation (crop): `_visual_meta.obs_id` (page §11).

Env
  VLMAS_LRPSR=1|identity        identity = same recompute, native output (path check)
  VLMAS_LRPSR_LAYERS=24,27,30   fixed formation sites; a comma list (or a-b window)
  VLMAS_LRPSR_ROWS=gen|all      gen = last prompt row + generated rows (default)
  VLMAS_LRPSR_SUPPORT=obs|branch
"""
from __future__ import annotations

import contextlib
import os

import torch

EPS = 1e-12


def enabled() -> bool:
    return os.environ.get("VLMAS_LRPSR", "").strip() in ("1", "identity")


def config() -> dict:
    return {
        "mode": os.environ.get("VLMAS_LRPSR", "").strip(),
        "layers": os.environ.get("VLMAS_LRPSR_LAYERS", "24,27,30").strip() or "24,27,30",
        "rows": os.environ.get("VLMAS_LRPSR_ROWS", "gen").strip() or "gen",
        "support": os.environ.get("VLMAS_LRPSR_SUPPORT", "obs").strip() or "obs",
    }


def layer_set(spec: str, n_layers: int) -> tuple[int, ...]:
    """"24,27,30" -> (24, 27, 30); "24-30" -> every layer in the window."""
    if "-" in spec:
        a, b = (int(x) for x in spec.split("-"))
        idx = range(min(a, b), max(a, b) + 1)
    else:
        idx = (int(x) for x in spec.split(",") if x.strip())
    out = tuple(sorted({i for i in idx if 0 <= i < n_layers}))
    if not out:
        raise ValueError(f"empty LR-PSR layer set {spec!r} for {n_layers} layers")
    return out


# --------------------------------------------------------------------------- pure math


def orth(C: torch.Tensor, rtol: float | None = None) -> torch.Tensor:
    """Orthonormal basis of the column space of C [d, T] by numerical rank (no chosen rank)."""
    C = C.to(torch.float32)
    if C.numel() == 0:
        return C.new_zeros((C.shape[0], 0))
    U, S, _ = torch.linalg.svd(C, full_matrices=False)
    if S.numel() == 0:
        return C.new_zeros((C.shape[0], 0))
    tol = (max(C.shape) * torch.finfo(C.dtype).eps * S[0]) if rtol is None else rtol * S[0]
    rank = int((S > tol).sum())
    return U[:, :rank]


def project(U: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """P x with P = U U^T; x [..., d]. U [d, k] (k may be 0)."""
    if U.shape[-1] == 0:
        return torch.zeros_like(x)
    return (x @ U) @ U.transpose(-2, -1)


def support_messages(scores_vis: torch.Tensor, V_vis: torch.Tensor, groups) -> list[torch.Tensor]:
    """m_g per support, pre-W_O: [1, H, Ts, D] each. scores_vis [1,H,Ts,n_vis], V_vis [1,H,n_vis,D]."""
    out = []
    for g in groups:
        idx = torch.as_tensor(g, dtype=torch.long, device=scores_vis.device)
        a = torch.softmax(scores_vis.index_select(-1, idx), dim=-1)
        out.append(torch.matmul(a, V_vis.index_select(-2, idx)))
    return out


def readout(m_V: torch.Tensor, m_g: list[torch.Tensor] | torch.Tensor, U: torch.Tensor):
    """LR-PSR visual message (§4). m_V [..., d]; m_g list of [..., d] (or stacked [G, ..., d]).

    Returns (m_tilde, diag) with diag = {cos, norm_ratio, degenerate} for the log line.
    """
    m_par = project(U, m_V)
    m_perp = m_V - m_par
    R = torch.stack(list(m_g), dim=0) if isinstance(m_g, (list, tuple)) else m_g
    R_perp = R - project(U, R)
    rbar = R_perp.mean(0)
    nb = rbar.norm(dim=-1, keepdim=True)
    # Degenerate rows (the support complements cancel, or every support message lies inside the
    # reference span): the mean complement direction is numerical noise, so keep the native
    # complement instead of amplifying it. Scale-relative so it also fires in fp32.
    tol = (1e-6 * m_perp.norm(dim=-1, keepdim=True)).clamp_min(EPS)
    degenerate = bool((nb <= tol).any())
    rhat = torch.where(nb > tol, rbar * (m_perp.norm(dim=-1, keepdim=True) / nb.clamp_min(EPS)),
                       m_perp)
    cos = torch.nn.functional.cosine_similarity(rhat, m_perp, dim=-1).mean()
    ratio = (m_perp.norm(dim=-1) / m_V.norm(dim=-1).clamp_min(EPS)).mean()
    return m_par + rhat, {"cos": float(cos), "perp_share": float(ratio), "degenerate": degenerate}


# --------------------------------------------------------------------------- runtime


def engine_groups(engine, mode: str) -> dict:
    """{"cols", "latent_cols", "groups", "note"} from the backbone's visual provenance."""
    from memory import psar as _psar

    got = _psar.engine_groups(engine, mode)
    lat = getattr(engine, "_rpath_latent_cols", None)
    if got["groups"] is not None and (lat is None or int(lat.numel()) == 0):
        got = {**got, "groups": None, "note": "no Reasoner latent columns"}
    return {**got, "latent_cols": lat}


@contextlib.contextmanager
def install_lrpsr(backbone, visual_cols, latent_cols, groups, *, layers=(24, 27, 30),
                  rows="gen", identity=False):
    """Post-hooks on the given decoder layers; yields a stats dict."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    cols = visual_cols.to(device=backbone.device, dtype=torch.long)
    lat = latent_cols.to(device=backbone.device, dtype=torch.long)
    stats = {"calls": 0, "rows": 0, "rank_sum": 0, "cos_sum": 0.0, "perp_sum": 0.0,
             "mv_sum": 0.0, "degenerate": 0, "maxdiff_identity": 0.0, "n_groups": len(groups)}
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
            vis = cols[cols < L]
            lcols = lat[lat < L]
            if int(vis.numel()) == 0 or int(vis.numel()) >= L or int(lcols.numel()) == 0:
                return output
            sel = torch.arange(T, device=hidden.device) if rows == "all" else \
                torch.tensor([T - 1], device=hidden.device)
            Hd, Gq = module.head_dim, module.num_key_value_groups
            cos_pe, sin_pe = pe
            h_sel = hidden.index_select(1, sel)
            Ts = int(sel.numel())
            q = module.q_norm(module.q_proj(h_sel).view(1, Ts, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos_pe.index_select(1, sel),
                                        sin_pe.index_select(1, sel))
            K = repeat_kv(keys, Gq).float()
            V = repeat_kv(values, Gq).float()
            scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling
            if T > 1:
                past = L - T
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + sel)[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)
            o = torch.matmul(alpha, V)
            a_vis = alpha[..., vis]
            V_vis = V[:, :, vis, :]
            o_v = torch.matmul(a_vis, V_vis)
            o_n = o - o_v

            def wo(x):   # [1,H,Ts,D] -> [1,Ts,d_model] in residual space
                return module.o_proj(x.transpose(1, 2).reshape(1, x.shape[2], -1)
                                     .to(module.o_proj.weight.dtype)).float()

            if identity:
                upd = wo(o).to(attn_out.dtype)
            else:
                m_V = wo(o_v)
                m_NV = wo(o_n)
                m_g = [wo(m) for m in support_messages(scores[..., vis], V_vis, groups)]
                # Reasoner latent reference subspace (per layer, from the cached latent values)
                lat_v = V[:, :, lcols, :]                                    # [1,H,T_R,D]
                C = module.o_proj(
                    lat_v.permute(0, 2, 1, 3).reshape(1, int(lcols.numel()), -1)
                    .to(module.o_proj.weight.dtype)).float()[0].transpose(0, 1)   # [d, T_R]
                U = orth(C)
                m_tilde, diag = readout(m_V, m_g, U)
                stats["rank_sum"] += int(U.shape[-1])
                stats["cos_sum"] += diag["cos"]
                stats["perp_sum"] += diag["perp_share"]
                stats["degenerate"] += int(diag["degenerate"])
                upd = (m_NV + m_tilde).to(attn_out.dtype)
            stats["mv_sum"] += float(a_vis.sum(-1).mean())
            if identity:
                stats["maxdiff_identity"] = max(stats["maxdiff_identity"], float(
                    (upd.float() - attn_out.index_select(1, sel).float()).abs().max()))
            new = attn_out.clone()
            new[:, sel, :] = upd
            stats["calls"] += 1
            stats["rows"] += Ts
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return post

    for li, layer in enumerate(backbone.lm.layers):
        if li in tuple(layers):
            handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()


def summary(stats: dict) -> str:
    c = max(stats["calls"], 1)
    return (f"calls={stats['calls']} rows={stats['rows']} supports={stats['n_groups']} "
            f"mean_rank={stats['rank_sum'] / c:.1f} mean_cos(rhat,mV_perp)={stats['cos_sum'] / c:.4f} "
            f"mean_perp_share={stats['perp_sum'] / c:.4f} mean_mV={stats['mv_sum'] / c:.5f} "
            f"degenerate={stats['degenerate']} maxdiff_identity={stats['maxdiff_identity']:.3e}")


__all__ = ["enabled", "config", "layer_set", "orth", "project", "support_messages", "readout",
           "engine_groups", "install_lrpsr", "summary"]
