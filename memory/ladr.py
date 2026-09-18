"""C2 12.8 Latent-Anchored Dual-Frame Read (LADR). Notion 3ddd771b-c9e2-8171.

Opt-in; off = no hooks, byte-identical. The visual K/V payload, the C1 budget, the crop geometry
and the Reasoner trajectory are untouched: only the ADDRESS of the visual columns is doubled.

Every Answerer attention call (all layers, page §7), per head:
  s_acq,j = q k_j / sqrt(d)                                        native acquisition address
  K_role  = R(p_role - p_acq) K_j                                  role-local address (§3)
            p_role = p_acq - min(p_acq) + cursor_R  (one rigid 3-D MRoPE shift, so every
            within-crop and cross-crop relative coordinate is preserved, §2)
  s_role,j = q K_role,j / sqrt(d)
  s~_j = log((exp(s_acq,j) + exp(s_role,j)) / 2)                   frame marginalisation (§4)
  non-visual columns keep their native score; softmax over everything; V is read ONCE (§5.2).
Identity when the two frames coincide (§5.1): s_acq == s_role -> s~ == s_acq, exactly native.

No gain, no threshold, no temperature, no layer choice, no support ranking; no extra crop
encoding, model forward, latent step or decode. The rotation reuses memory/restage.py's
angle-subtraction helpers on the model's own cos/sin tables, so the MRoPE section layout is
never reconstructed here.

Env
  VLMAS_LADR=mc|1|identity
      mc       MC-LADR (page revision): the marginalised scores set only the visual-internal
               conditional distribution pi; each layer x head x row keeps its NATIVE total
               visual mass rho_V and every non-visual attention weight (page §4, §5.4):
                   alpha~_j = rho_V * pi_j (j in V),  alpha~_k = alpha_k native (k not in V)
      1        the first LADR formulation: the dual-frame scores re-enter the FULL softmax, so
               the visual-vs-non-visual balance can move too (kept for the runs already scored)
      identity same recompute with p_role = p_acq (native output check)
  VLMAS_LADR_ROWS=gen|all gen = last prompt row + generated rows (default)
"""
from __future__ import annotations

import contextlib
import os

import torch


def enabled() -> bool:
    return os.environ.get("VLMAS_LADR", "").strip() in ("mc", "1", "identity")


def config() -> dict:
    return {
        "mode": os.environ.get("VLMAS_LADR", "").strip(),
        "rows": os.environ.get("VLMAS_LADR_ROWS", "gen").strip() or "gen",
    }


# --------------------------------------------------------------------------- pure


def role_positions(old_pos: torch.Tensor, cursor: int) -> torch.Tensor:
    """Rigid 3-D MRoPE shift of the whole visual bank to the role boundary (page §2).

    old_pos [3, n] -> [3, n]; every difference p_j - p_k is preserved.
    """
    base = old_pos.min(dim=1, keepdim=True).values
    return old_pos - base + int(cursor)


def marginalize(s_acq: torch.Tensor, s_role: torch.Tensor) -> torch.Tensor:
    """log((e^a + e^b)/2), computed stably; equals a when a == b (page §4, §5.1)."""
    return torch.logaddexp(s_acq, s_role) - torch.log(torch.tensor(
        2.0, dtype=s_acq.dtype, device=s_acq.device))


def delta_tables(backbone, old_pos: torch.Tensor, cursor: int, sample: torch.Tensor,
                 identity: bool = False):
    """(cos_delta, sin_delta) [1, n, D] for p_role - p_acq on the model's own rotary tables."""
    from memory.restage import delta_cos_sin

    new_pos = old_pos if identity else role_positions(old_pos, cursor)
    cos_old, sin_old = backbone.lm.rotary_emb(sample, old_pos.unsqueeze(1))
    cos_new, sin_new = backbone.lm.rotary_emb(sample, new_pos.unsqueeze(1))
    return delta_cos_sin(cos_old.float(), sin_old.float(), cos_new.float(), sin_new.float())


# --------------------------------------------------------------------------- runtime


def engine_state(engine) -> dict:
    """{"cols", "positions", "cursor", "note"} for the terminal call, or groups=None reason."""
    bb = engine._backbone
    cols = getattr(engine, "_rpath_visual_cols", None)
    pos = getattr(bb, "_restage_vis_positions", None)
    if cols is None or int(cols.numel()) == 0:
        return {"cols": None, "note": "no visual columns"}
    if pos is None or int(pos.shape[-1]) != int(cols.numel()):
        have = None if pos is None else int(pos.shape[-1])
        return {"cols": None,
                "note": f"visual MRoPE positions missing/mismatched ({have} vs {int(cols.numel())})"}
    return {"cols": cols, "positions": pos, "note": f"n_vis={int(cols.numel())}"}


@contextlib.contextmanager
def install_ladr(backbone, visual_cols, positions, cursor, *, rows="gen", identity=False,
                 mass_conserved=False):
    """Dual-frame read on every decoder layer; yields a stats dict."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
    from memory.restage import rerotate_keys

    dev = backbone.device
    cols = visual_cols.to(device=dev, dtype=torch.long)
    old_pos = positions.to(device=dev, dtype=torch.long)
    stats = {"calls": 0, "rows": 0, "n_vis": int(cols.numel()), "cursor": int(cursor),
             "mass_conserved": bool(mass_conserved),
             "shift": None, "role_pref_sum": 0.0, "mv_sum": 0.0, "mv_native_sum": 0.0,
             "maxdiff_identity": 0.0}
    sample = backbone.lm.layers[0].self_attn.q_proj.weight.new_zeros(1, 1, 1, 1)
    cos_d, sin_d = delta_tables(backbone, old_pos, cursor, sample, identity=identity)
    new_pos = old_pos if identity else role_positions(old_pos, cursor)
    stats["shift"] = [int(x) for x in (new_pos[:, 0] - old_pos[:, 0]).tolist()]
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
            if int(cols.numel()) == 0 or int(cols.max()) >= L or int(cols.numel()) >= L:
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
            scores = torch.matmul(qf, K.transpose(-2, -1)) * module.scaling        # [1,H,Ts,L]
            if T > 1:
                past = L - T
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + sel)[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            # role-local keys: rotate the CACHED kv heads, then expand for GQA
            k_vis_raw = keys.index_select(2, cols)
            k_role = rerotate_keys(k_vis_raw, cos_d.to(k_vis_raw.dtype), sin_d.to(k_vis_raw.dtype))
            K_role = up(repeat_kv(k_role, Gq))
            s_role = torch.matmul(qf, K_role.transpose(-2, -1)) * module.scaling   # [1,H,Ts,n_vis]
            s_acq = scores[..., cols]
            s_tilde = marginalize(s_acq, s_role)
            native_alpha = torch.softmax(scores, dim=-1)
            if mass_conserved:
                # page §4: keep the native total visual mass and every non-visual weight; the
                # marginalised scores only set the visual-internal conditional distribution.
                rho = native_alpha[..., cols].sum(-1, keepdim=True)
                pi = torch.softmax(s_tilde, dim=-1)
                alpha = native_alpha.clone()
                alpha[..., cols] = rho * pi
            else:
                new_scores = scores.clone()
                new_scores[..., cols] = s_tilde
                alpha = torch.softmax(new_scores, dim=-1)
            ctx = torch.matmul(alpha, V)
            flat = ctx.transpose(1, 2).reshape(1, Ts, -1)
            upd = module.o_proj(flat.to(module.o_proj.weight.dtype))
            a_vis = alpha[..., cols]
            stats["mv_sum"] += float(a_vis.sum(-1).mean())
            with torch.no_grad():
                nat = native_alpha[..., cols]
                stats["mv_native_sum"] += float(nat.sum(-1).mean())
                stats["role_pref_sum"] += float((s_role > s_acq).float().mean())
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
    return (f"calls={stats['calls']} rows={stats['rows']} mc={int(stats.get('mass_conserved', False))} "
            f"n_vis={stats['n_vis']} "
            f"cursor={stats['cursor']} shift={stats['shift']} "
            f"mean_mV={stats['mv_sum'] / c:.5f} mean_mV_native={stats['mv_native_sum'] / c:.5f} "
            f"role_preferred={stats['role_pref_sum'] / c:.4f} "
            f"maxdiff_identity={stats['maxdiff_identity']:.3e}")


__all__ = ["enabled", "config", "role_positions", "marginalize", "delta_tables",
           "engine_state", "install_ladr", "summary"]
