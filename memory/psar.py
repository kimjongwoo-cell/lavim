"""Physical-Support Additive Readout (PSAR) -- C2 candidate (opt-in; off = no hooks).

Notion: C1 / C2 Method Evolution -> "Physical-Support Additive Readout (PSAR)"
(3dad771b-c9e2-81ae-9b08-e2b73e325fdf). Principle: within a physical support normalize,
across independent supports accumulate.

At an Answerer attention call in the formation window, for a query row t (native q, K, V):
  alpha_{t,j}      = softmax over ALL columns (native)
  o_V = sum_{j in V} alpha_j v_j ,  o_N = sum_{j notin V} alpha_j v_j     (o_native = o_N + o_V)
  alpha_{t,j|g}    = softmax_{j in g}(q^T k_j / sqrt d)                     [1. within-support read]
  m_g              = sum_{j in g} alpha_{t,j|g} v_j
  m_slide          = (1/|G_t|) sum_{g in G_t} m_g                           [2. across-support mean]
  MatchScale(m, o) = m * ||o|| / ||m||      (per head, per row, value space)
  target=visual (default):  o~ = o_N + (1-eta) o_V + eta MatchScale(m_slide, o_V)
  target=full:              o~ = (1-eta) o_native + eta MatchScale(m_slide, o_native)
then o_proj as usual. Q, K, V are the native states; nothing is learned.

Implementation choices (the spec leaves them open; reported in the log line):
  * support g = the C1 support of Question-Agnostic CSIS: the top-level 5x acquisition branch
    (memory.csis_select.support_branches after drop_self_parents). j -> g(j) is read from the
    backbone's visual provenance (_visual_meta, written at the C1 boundary in column order:
    abs column, selected pre-C1 index, obs_id, cell in crop, scale, xy, patch_id); it is used
    only if its columns equal the terminal's visual columns. A crop C1 emptied contributes
    nothing; G_t = supports with at least one retained column.
  * target. The spec writes MatchScale(m_slide, o_native) against "the native attention
    output". Replacing the whole output with a visual-only message would also erase the text
    context, and scaling the visual message up to the full output norm is exactly the visual
    amplification the page says PSAR is not. Default `visual` therefore swaps only the visual
    read and matches its norm to the native visual read; `full` is the literal reading.
  * eta. Default 1.0: with target=visual that is "native-scale conservation" with no free
    constant (the form the page prefers); VLMAS_PSAR_ETA sets a fixed mixing constant.
  * rows. "answer position": default `gen` = the last prompt row (it predicts the first answer
    token) and every generated row; `all` adds every prompt prefill row.
  * window. Formation window L24-30 (Notion "답은 어느 층에서 굳는가": answer-token rank falls
    from hundreds at L24 to 1-5 at L30; decision median L32-33). One fixed window, no search.

Invariant (unit-tested): with a single support containing every visual token, m_slide is the
renormalized native visual read, so target=visual, eta=1 reproduces the native output.

Env
  VLMAS_PSAR=1|identity          identity = same recompute, native output (path check)
  VLMAS_PSAR_LAYERS=24-30        inclusive decoder-layer window
  VLMAS_PSAR_ETA=1.0
  VLMAS_PSAR_TARGET=visual|full
  VLMAS_PSAR_ROWS=gen|all
  VLMAS_PSAR_SUPPORT=branch|obs|shuffled   branch = C1 support; obs = one crop per support;
                                           shuffled = branch sizes, membership permuted (falsifier)
"""
from __future__ import annotations

import contextlib
import os
import random

import torch

EPS = 1e-12


# --------------------------------------------------------------------------- config


def enabled() -> bool:
    return os.environ.get("VLMAS_PSAR", "").strip() in ("1", "identity")


def config() -> dict:
    return {
        "mode": os.environ.get("VLMAS_PSAR", "").strip(),
        "layers": os.environ.get("VLMAS_PSAR_LAYERS", "24-30").strip() or "24-30",
        "eta": float(os.environ.get("VLMAS_PSAR_ETA", "1.0").strip() or 1.0),
        "target": os.environ.get("VLMAS_PSAR_TARGET", "visual").strip() or "visual",
        "rows": os.environ.get("VLMAS_PSAR_ROWS", "gen").strip() or "gen",
        "support": os.environ.get("VLMAS_PSAR_SUPPORT", "branch").strip() or "branch",
    }


def layer_window(spec: str, n_layers: int) -> tuple[int, int]:
    a, b = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    lo, hi = max(0, min(a, b)), min(n_layers - 1, max(a, b))
    if lo > hi:
        raise ValueError(f"empty PSAR layer window {spec!r} for {n_layers} layers")
    return lo, hi


# --------------------------------------------------------------------------- pure


def support_groups(obs_ids, parents, mode: str = "branch", seed: int = 0) -> list[list[int]]:
    """Index lists into the retained visual columns, one per support with >= 1 column.

    obs_ids: observation (crop) index of every retained column, in column order -- the
    backbone's visual provenance, so no contiguity or per-crop count is assumed.
    parents: parent observation per observation (-1 root), over ALL observations.
    branch   g(j) = top-level 5x branch of obs_ids[j] (memory.csis_select, self-parent -> root)
    obs      g(j) = obs_ids[j]
    shuffled branch group sizes, membership permuted over the columns (falsifier)
    """
    from memory.csis_select import drop_self_parents, support_branches

    obs = [int(o) for o in (obs_ids.tolist() if hasattr(obs_ids, "tolist") else obs_ids)]
    if mode not in ("branch", "obs", "shuffled"):
        raise ValueError(f"support must be branch|obs|shuffled, got {mode!r}")
    if mode == "obs":
        key = obs
    else:
        par, _ = drop_self_parents(list(parents))
        br = support_branches(par)
        key = [br[o] if 0 <= o < len(br) else o for o in obs]
    groups: dict[int, list[int]] = {}
    for j, g in enumerate(key):
        groups.setdefault(g, []).append(j)
    out = [groups[g] for g in sorted(groups)]
    if mode == "shuffled":
        order = list(range(len(obs)))
        random.Random(int(seed)).shuffle(order)
        shuf, pos = [], 0
        for g in out:
            shuf.append(sorted(order[pos:pos + len(g)]))
            pos += len(g)
        out = shuf
    return out


def support_message(vis_scores: torch.Tensor, vis_values: torch.Tensor, groups) -> torch.Tensor:
    """m_slide = mean_g sum_{j in g} softmax_{j in g}(l_j) v_j.

    vis_scores [..., T, n_vis] logits of the visual columns; vis_values [..., n_vis, D]
    (e.g. [1, H, T, n_vis] and [1, H, n_vis, D]). Returns [..., T, D].
    """
    msgs = []
    for g in groups:
        idx = torch.as_tensor(g, dtype=torch.long, device=vis_scores.device)
        a = torch.softmax(vis_scores.index_select(-1, idx), dim=-1)                  # [..., T, |g|]
        msgs.append(torch.matmul(a, vis_values.index_select(-2, idx)))               # [..., T, D]
    return torch.stack(msgs, dim=0).mean(0)


def match_scale(m: torch.Tensor, ref: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return m * (ref.norm(dim=-1, keepdim=True) / m.norm(dim=-1, keepdim=True).clamp_min(eps))


def combine(o_n, o_v, m_slide, eta: float, target: str):
    if target == "visual":
        return o_n + (1.0 - eta) * o_v + eta * match_scale(m_slide, o_v)
    if target == "full":
        o = o_n + o_v
        return (1.0 - eta) * o + eta * match_scale(m_slide, o)
    raise ValueError(f"target must be visual|full, got {target!r}")


# ------------------------------------------------------------------------ runtime


def engine_groups(engine, mode: str) -> dict:
    """{"cols", "groups", "note"} from the backbone's visual provenance (_visual_meta).

    Requires the provenance columns to be exactly the columns the terminal reads
    (engine._rpath_visual_cols); otherwise groups is None and note says why -- PSAR never
    falls back to a guessed mapping.
    """
    bb = engine._backbone
    cols = getattr(engine, "_rpath_visual_cols", None)
    meta = getattr(bb, "_visual_meta", None)
    if cols is None or int(cols.numel()) == 0:
        return {"cols": None, "groups": None, "note": "no visual columns"}
    if meta is None:
        return {"cols": cols, "groups": None,
                "note": "no visual provenance (" + str(getattr(bb, "_visual_meta_note", "")) + ")"}
    if not torch.equal(meta["abs_cols"].cpu().long(), cols.cpu().long()):
        return {"cols": cols, "groups": None,
                "note": f"provenance columns ({int(meta['abs_cols'].numel())}) != terminal columns "
                        f"({int(cols.numel())}) -- moved after C1"}
    seed = 42 + int(getattr(engine, "_rpath_case_index", 0) or 0)
    groups = support_groups(meta["obs_id"], meta["parents"], mode, seed)
    return {"cols": cols, "groups": groups,
            "note": f"visual_meta c1={meta['c1']} kept_counts={list(meta['kept_counts'])}"}


@contextlib.contextmanager
def install_psar(backbone, visual_cols, groups, *, layers=(24, 30), eta=1.0, target="visual",
                 rows="gen", identity=False):
    """Hooks on decoder layers layers[0]..layers[1]; yields a stats dict."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    lo, hi = layers
    cols = visual_cols.to(device=backbone.device, dtype=torch.long)
    stats = {"calls": 0, "rows": 0, "mv_sum": 0.0, "scale_sum": 0.0, "cos_sum": 0.0,
             "maxdiff_identity": 0.0, "n": 0}
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
            vis = cols[cols < L]
            if int(vis.numel()) == 0 or int(vis.numel()) >= L:
                return output
            sel = torch.arange(T, device=hidden.device) if rows == "all" else \
                torch.tensor([T - 1], device=hidden.device)
            Hd = module.head_dim
            G = module.num_key_value_groups
            cos, sin = pe
            h_sel = hidden.index_select(1, sel)
            Ts = int(sel.numel())
            q = module.q_norm(module.q_proj(h_sel).view(1, Ts, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, sel),
                                        sin.index_select(1, sel))
            K = repeat_kv(keys, G).float()
            V = repeat_kv(values, G).float()
            scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling       # [1,H,Ts,L]
            if T > 1:
                past = L - T
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + sel)[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)
            o = torch.matmul(alpha, V)                                                    # [1,H,Ts,D]
            a_vis = alpha[..., vis]
            V_vis = V[:, :, vis, :]
            o_v = torch.matmul(a_vis, V_vis)
            o_n = o - o_v
            if identity:
                new_ctx = o
            else:
                m_slide = support_message(scores[..., vis], V_vis, groups)
                new_ctx = combine(o_n, o_v, m_slide, float(eta), target)
                ref = o_v if target == "visual" else o
                stats["scale_sum"] += float((ref.norm(dim=-1) /
                                             m_slide.norm(dim=-1).clamp_min(EPS)).mean())
                stats["cos_sum"] += float(torch.nn.functional.cosine_similarity(
                    m_slide, o_v, dim=-1).mean())
            stats["mv_sum"] += float(a_vis.sum(-1).mean())
            flat = new_ctx.transpose(1, 2).reshape(1, Ts, -1)
            upd = module.o_proj(flat.to(module.o_proj.weight.dtype))
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
        if lo <= li <= hi:
            handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()


def summary(stats: dict) -> str:
    c = max(stats["calls"], 1)
    return (f"calls={stats['calls']} rows={stats['rows']} mean_mV={stats['mv_sum'] / c:.5f} "
            f"mean_scale={stats['scale_sum'] / c:.5f} mean_cos(m_slide,o_V)={stats['cos_sum'] / c:.4f} "
            f"maxdiff_identity={stats['maxdiff_identity']:.3e}")


__all__ = ["enabled", "config", "layer_window", "support_groups", "support_message", "match_scale",
           "combine", "engine_groups", "install_psar", "summary"]
