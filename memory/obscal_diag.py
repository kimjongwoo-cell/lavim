"""C2 feasibility probe: WSI Observation-Level Receiver Calibration (opt-in).

Notion Experiment Log: "C2 방향 검증: WSI Observation-Level Receiver Calibration —
open-ended candidate-free feasibility". Nothing here runs unless VLMAS_OBSCAL=<jsonl>.

The page defers the minimum-change constraint/solver until a feasibility test passes, so
this module implements ONLY the measurement stage (no intervention, no decode change):

  receiver site l*   the last decoder layer, i.e. the residual the final norm consumes.
                     That choice keeps the suffix (RMSNorm -> tied unembedding) exact and
                     cheap, so the local sensitivity J_t is applied analytically instead
                     of being approximated. It measures that layer's visual write only.
  observation write  m_{o,t} = W_O ( concat_h sum_{j in S_o} alpha_{t,h,j} v_{h,j} )
                     with S_o = the Navigator observation (crop) a token came from.
  local influence    d_{o,t} = J_t m_{o,t},  J_t = d z_t / d h_t
                     RMSNorm JVP: dy = g * ( m/rms - h (h.m) / (n rms^3) ),  d = W_U dy
  competition        a_t = argmax_w z_t(w),  g_t(c) = z_t(a_t) - z_t(c),
                     v_{o,t}(c) = d_{o,t}(c) - d_{o,t}(a_t),  R_{o,t} = max_c v/(g+eps)
  provenance control the same per-token writes re-summed under a shuffled partition
                     (identical token set and observation sizes) — costs no extra forward.

Recorded per answer-content step: ||m_{o,t}||, the per-observation reach and its
numerator/denominator, the summed visual push, and the shuffled-partition counterparts.

Env
  VLMAS_OBSCAL=<path.jsonl>     output file (append; one row per case). Unset = no-op.
  VLMAS_OBSCAL_LAYER=<int>      receiver site (default: last layer)
  VLMAS_OBSCAL_MAXSTEPS=<int>   cap on probed answer steps (default 12)
"""
from __future__ import annotations

import contextlib
import json
import os

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

EPS = float(torch.finfo(torch.float64).tiny)


# --------------------------------------------------------------------------- pure


def rmsnorm_jvp(h, m, weight, eps: float = 1e-6):
    """Directional derivative of RMSNorm at h in the direction m (float64)."""
    h = h.to(torch.float64); m = m.to(torch.float64)
    w = weight.to(torch.float64)
    n = h.numel()
    ms = float((h * h).mean())
    rms = (ms + eps) ** 0.5
    dot = float(h @ m)
    return w * (m / rms - h * (dot / (n * rms ** 3)))


def reach_from(d, z, a):
    """(R, v_max, g at the best-reach token, n_push) for one influence vector."""
    d = d.to(torch.float64); z = z.to(torch.float64)
    g = z[a] - z
    v = d - d[a]
    mask = v > 0
    mask[a] = False
    if not bool(mask.any()):
        return {"R": None, "v_star": None, "g_star": None, "v_max": None, "n_push": 0}
    ratio = torch.where(mask, v / (g + EPS), torch.full_like(v, float("-inf")))
    c = int(torch.argmax(ratio))
    v_masked = torch.where(mask, v, torch.full_like(v, float("-inf")))
    cm = int(torch.argmax(v_masked))
    return {"R": float(ratio[c]), "v_star": float(v[c]), "g_star": float(g[c]),
            "v_max": float(v[cm]), "n_push": int(mask.sum())}


# ------------------------------------------------------------------------ runtime


@contextlib.contextmanager
def capture_observation_writes(backbone, group_cols, layer_index: int, store: dict):
    """Hook one decoder layer and record each observation's attention write for every row.

    store["m"] ends up as a [G, T, D] float32 CPU tensor: the o_proj output restricted to
    each observation's visual columns, i.e. exactly the term the page calls m_{o,t}.
    """
    layer = backbone.lm.layers[int(layer_index)]

    def hook(module, args, kwargs, output):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        if hidden is None or pe is None or cache is None:
            return output
        keys = cache.layers[int(layer_index)].keys
        values = cache.layers[int(layer_index)].values
        L = int(keys.shape[2]); T = int(hidden.shape[1])
        Hd = module.head_dim
        g_ = module.num_key_value_groups
        cos, sin = pe
        q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
        K = repeat_kv(keys, g_).float()
        V = repeat_kv(values, g_).float()
        scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling
        if T > 1:
            past = L - T
            col = torch.arange(L, device=keys.device)[None, None, None, :]
            row = (past + torch.arange(T, device=keys.device))[None, None, :, None]
            scores = scores.masked_fill(col > row, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        outs = []
        for gc in group_cols:
            gc = gc.to(keys.device); gc = gc[gc < L]
            ctx = torch.matmul(alpha[..., gc], V[:, :, gc, :])          # [1,Hq,T,D]
            flat = ctx.transpose(1, 2).reshape(1, T, -1)
            outs.append(module.o_proj(flat.to(module.o_proj.weight.dtype))[0].detach().float().cpu())
        store["m"] = torch.stack(outs)                                   # [G, T, D]
        store["T"] = T
        return output

    handle = layer.self_attn.register_forward_hook(hook, with_kwargs=True)
    try:
        yield store
    finally:
        handle.remove()


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, max_new_tokens: int = 512) -> None:
    out_path = os.environ.get("VLMAS_OBSCAL", "").strip()
    if not out_path:
        return
    from copy import deepcopy

    from memory.ofh import observation_groups, observation_pages, partition
    from memory.rmvr_diag import field_spans
    from memory.vcca_diag import _arm_pass, _unembedding
    from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json

    bb = engine._backbone
    got = observation_groups(engine, "true")
    if got is None:
        print(f"[ObsCal] case {case_index}: SKIP ({observation_pages(engine)[1]})", flush=True)
        return
    cols, groups_abs, page_src = got
    pages, _ = observation_pages(engine)
    unemb = _unembedding(bb)
    norm = getattr(getattr(bb, "lm", None), "norm", None)
    if unemb is None or norm is None:
        print(f"[ObsCal] case {case_index}: SKIP (no unembedding/final norm)", flush=True)
        return
    n_layers = len(bb.lm.layers)
    l_star = int(os.environ.get("VLMAS_OBSCAL_LAYER", str(n_layers - 1)) or (n_layers - 1))
    max_steps = int(os.environ.get("VLMAS_OBSCAL_MAXSTEPS", "12") or 12)

    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    gen_cache = deepcopy(cache)
    try:
        text = generate_terminal_json(
            backbone=bb, cache=gen_cache, position_cursor=position_cursor,
            system_prompt=system_prompt, user_prompt=user_prompt, json_prefix=json_prefix,
            max_new_tokens=int(max_new_tokens), temperature=0.0, top_p=1.0, do_sample=False)
    finally:
        del gen_cache
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    gen_ids, offsets = list(enc["input_ids"]), list(enc.get("offset_mapping") or [])
    if not gen_ids:
        print(f"[ObsCal] case {case_index}: SKIP (empty generation)", flush=True)
        return
    fields = field_spans(text, offsets, json_prefix) if offsets else {}
    ans_rows = fields.get("answer", {}).get("idx") or []
    rows_idx = (ans_rows or list(range(len(gen_ids))))[:max_steps]

    got_s = observation_groups(engine, "shuffled")
    groups_shuf = got_s[1] if got_s else []
    groups_all = list(groups_abs) + list(groups_shuf)      # true first, shuffled after
    G = len(groups_abs)
    store: dict = {}
    c = deepcopy(cache)
    try:
        with capture_observation_writes(bb, groups_all, l_star, store):
            out = _arm_pass(bb, c, prompt_ids, [gen_ids], position_cursor)[0]
    finally:
        del c
    if "m" not in store:
        print(f"[ObsCal] case {case_index}: SKIP (write hook did not fire)", flush=True)
        return
    m_all = store["m"]                       # [G, T_forward, D]
    pre = out["pre"]                         # [len(gen), D] pre-final-norm rows
    logits = out["logits"]                   # [len(gen), V]
    n_prompt = int(prompt_ids.shape[1])
    w_norm = norm.weight.detach().float().cpu()
    eps_n = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    W = unemb.detach().float()

    # shuffled provenance: same per-token writes, permuted membership (no extra forward)
    idx_true = partition(pages, "true", 42)
    idx_shuf = partition(pages, "shuffled", 42 + int(case_index))
    pos_of = {int(t): (o, k) for o, g in enumerate(idx_true) for k, t in enumerate(g)}

    steps = []
    for t in rows_idx:
        row_f = n_prompt - 1 + int(t)         # row of the forward that predicts token t
        h = pre[int(t)].to(torch.float64)
        z = logits[int(t)].to(torch.float64)
        a = int(torch.argmax(z))

        def per_partition(lo, hi):
            recs, d_sum = [], None
            for o in range(lo, hi):
                m = m_all[o, row_f]
                dy = rmsnorm_jvp(h, m, w_norm, eps_n)
                d = (W.to(torch.float64) @ dy.to(W.device, torch.float64)).cpu()
                d_sum = d if d_sum is None else d_sum + d
                r = reach_from(d, z, a)
                r.update({"o": o - lo, "m_norm": float(m.norm()), "d_a": float(d[a])})
                recs.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
            tot = reach_from(d_sum, z, a) if d_sum is not None else {}
            return recs, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in tot.items()}

        per_obs, tot = per_partition(0, G)
        step = {
            "t": int(t), "y": int(gen_ids[int(t)]), "a": a,
            "obs": per_obs, "total": tot,
            "m_norm_sum": round(float(sum(o["m_norm"] for o in per_obs)), 6),
        }
        if groups_shuf:
            # identical token set, permuted membership: the summed write (and therefore
            # `total`) must match the true partition — that is the implementation gate —
            # while the per-observation split should differ if provenance carries anything
            per_shuf, tot_shuf = per_partition(G, G + len(groups_shuf))
            step["obs_shuffled"] = per_shuf
            step["total_shuffled"] = tot_shuf
        steps.append(step)
    rec = {
        "case": int(case_index), "layer": l_star, "n_layers": n_layers,
        "pages": pages, "page_src": page_src, "n_vis": int(cols.numel()),
        "gen_text": text[:300], "rows": [int(r) for r in rows_idx],
        "answer_rows": [int(r) for r in ans_rows],
        "shuffled_sizes": [len(g) for g in idx_shuf],
        "steps": steps,
        "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
    }
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else float("nan"))
    rt = [s["total"]["R"] for s in steps if s["total"]["R"] is not None]
    mx = [max(o["m_norm"] for o in s["obs"]) for s in steps]
    print(f"[ObsCal] case {case_index} l*={l_star} obs={len(groups_abs)} steps={len(steps)} "
          f"R_total_med={med(rt):.4f} m_norm_max_med={med(mx):.3f} src={page_src} "
          f"attn={rec['attn']}", flush=True)


__all__ = ["run", "rmsnorm_jvp", "reach_from", "capture_observation_writes"]
