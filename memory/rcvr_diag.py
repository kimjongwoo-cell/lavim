"""C2 probe: Receiver-Conditioned Visual KV Re-Read — §7 kill test 1 + reallocation ceiling.

Notion Method Evolution: "Receiver-Conditioned Visual KV Re-Read (RCVR)" (Candidate).
Nothing here runs unless VLMAS_RCVR=<out.jsonl> is set.

RCVR proposes that the receiver should first consume its native NON-VISUAL contribution,
form an updated diagnostic state, and only then re-read the persistent visual KV with the
resulting query — keeping the native visual mass m_V fixed and changing only the
allocation WITHIN the visual columns.

This module does not implement that forward. It measures the two things that decide
whether implementing it is worth it, both of which are exactly computable from one native
attention pass:

  1. §7 kill test 1 — is the updated query even different?
        h_hat = h + r_C,   r_C = W_O concat_h[ sum_{j not visual} alpha_j v_j ]
        q     = q_norm(W_Q(Norm(h))),   q_hat = q_norm(W_Q(Norm(h_hat)))
        gap   = 1 - cos(q, q_hat)
     If the gap is ~0 the mechanism is an identity and RCVR is dead on arrival.

  2. Reallocation ceiling — the LARGEST change any within-visual re-weighting could make:
        v = sum_j beta_j v_j       (native visual read, beta = alpha_V / m_V)
        ceiling = max_j || v_j - v ||        (attained by moving all mass onto one column)
     This bounds not just RCVR but every "re-weight the visual columns" method. Compared
     against ||v|| and against the whole layer output it says how much headroom exists at
     all. If the visual value rows are nearly collinear, no reallocation can matter.

Both quantities are ROPE-FREE. Rotary embedding at a fixed position is one orthogonal map
applied to q and q_hat alike, so it preserves their cosine; and the ceiling only involves
value rows, which are never rotated. That is why this probe needs no positional machinery
and cannot be wrong about it.

Env
  VLMAS_RCVR=<path.jsonl>   output file (append; one row per case). Unset = no-op.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

import torch

EPS = 1e-12


# --------------------------------------------------------------------------- pure


def visual_split(alpha, visual_mask):
    """(mass on the visual columns, mass on everything else)."""
    a = alpha.to(torch.float64)
    m = visual_mask.to(torch.bool)
    m_v = float(a[m].sum())
    return m_v, float(a.sum()) - m_v


def context_contribution(alpha, values, visual_mask):
    """sum_{j not visual} alpha_j v_j — the native NON-visual head output (= m_C * c).

    Deliberately not renormalised: RCVR adds the native contribution to the residual,
    which carries its native mass.
    """
    a = alpha.to(torch.float64).clone()
    a[visual_mask.to(torch.bool)] = 0.0
    return a @ values.to(torch.float64)


def visual_read(alpha, values, visual_mask):
    """(beta over the visual columns, v = beta-weighted mean of visual values, m_V)."""
    a = alpha.to(torch.float64)
    m = visual_mask.to(torch.bool)
    m_v = float(a[m].sum())
    # guard only when there is genuinely no visual mass — adding EPS unconditionally would
    # leave beta summing to 1 - 2.5e-12 and quietly bias every downstream ratio
    beta = a[m] / (m_v if m_v > EPS else 1.0)
    return beta, beta @ values.to(torch.float64)[m], m_v


def realloc_ceiling(beta, values):
    """(largest shift ANY re-weighting could make to the weighted mean, ||mean||).

    sup over distributions beta' of ||beta' V - beta V|| is attained at a vertex of the
    simplex, i.e. by putting all the mass on the single farthest row.
    """
    v = beta.to(torch.float64) @ values.to(torch.float64)
    shifts = (values.to(torch.float64) - v).norm(dim=-1)
    return float(shifts.max()), float(v.norm())


def cos_gap(q, q_hat):
    """1 - cos(q, q_hat); 0 when either side is degenerate."""
    a = q.to(torch.float64).reshape(-1)
    b = q_hat.to(torch.float64).reshape(-1)
    na, nb = float(a.norm()), float(b.norm())
    if na < EPS or nb < EPS:
        return 0.0
    return 1.0 - float(a @ b) / (na * nb)


# ------------------------------------------------------------------------ runtime


def _layers(backbone):
    lm = getattr(backbone, "lm", None)
    return None if lm is None else getattr(lm, "layers", None)


def _cache_values(cache, layer_idx):
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return layers[layer_idx].values
    return cache.value_cache[layer_idx]


def _head_query(layer, hidden, n_heads, head_dim):
    """q for every head at one position, exactly as the layer builds it (pre-RoPE)."""
    x = layer.input_layernorm(hidden)
    q = layer.self_attn.q_proj(x).view(n_heads, head_dim)
    return layer.self_attn.q_norm(q)


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        candidates=None, case_index: int = -1) -> None:
    out_path = os.environ.get("VLMAS_RCVR", "").strip()
    if not out_path:
        return
    from vision_text_mas.latent_terminal import _assistant_prompt

    bb = engine._backbone
    cols = getattr(engine, "_rpath_visual_cols", None)
    layers = _layers(bb)
    if cols is None or int(cols.numel()) == 0 or layers is None:
        print(f"[RCVR] case {case_index}: SKIP (cols="
              f"{0 if cols is None else int(cols.numel())} layers={layers is not None})", flush=True)
        return

    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    if ids.shape[1] < 2:
        print(f"[RCVR] case {case_index}: SKIP (prompt too short)", flush=True)
        return

    work = deepcopy(cache)
    try:
        # fill the cache with everything but the last prompt token
        _forward(bb, work, ids[:, :-1], position_cursor, want_attn=False, store=None)
        # the receiver position: the row that produces the first answer token
        hidden_in: dict = {}
        handles = [
            lay.register_forward_pre_hook(
                (lambda idx: (lambda _m, args: hidden_in.__setitem__(idx, args[0][0, -1].detach())))(i),
                with_kwargs=False)
            for i, lay in enumerate(layers)
        ]
        try:
            attns = _forward(bb, work, ids[:, -1:], position_cursor + ids.shape[1] - 1
                             if position_cursor is not None else None,
                             want_attn=True, store=None)
        finally:
            for h in handles:
                h.remove()
        if attns is None:
            print(f"[RCVR] case {case_index}: SKIP (no attentions returned)", flush=True)
            return

        kv_len = int(attns[0].shape[-1])
        vmask = torch.zeros(kv_len, dtype=torch.bool)
        keep = cols[cols < kv_len].to(torch.long).cpu()
        vmask[keep] = True
        if int(vmask.sum()) == 0:
            print(f"[RCVR] case {case_index}: SKIP (visual cols outside cache)", flush=True)
            return

        cfg = bb.lm.config if hasattr(bb.lm, "config") else bb.model.config
        n_heads = getattr(cfg, "num_attention_heads")
        head_dim = getattr(cfg, "head_dim", getattr(cfg, "hidden_size") // n_heads)

        rec = {"case": int(case_index), "n_layers": len(layers), "n_heads": n_heads,
               "kv_len": kv_len, "n_visual": int(vmask.sum()),
               "n_cands": len(candidates or ()),
               "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
               "layers": []}

        for li, lay in enumerate(layers):
            alpha = attns[li][0, :, -1, :].detach().to(torch.float64).cpu()   # [heads, kv]
            vals = _cache_values(work, li)[0].detach().to(torch.float64).cpu()  # [kv_heads, kv, dim]
            n_kv = vals.shape[0]
            groups = n_heads // n_kv
            h_in = hidden_in.get(li)
            if h_in is None:
                continue

            ctx = torch.zeros(n_heads * head_dim, dtype=torch.float64)
            gaps, mvs, ceil_rel, ceil_abs, vnorms = [], [], [], [], []
            per_head_v = []
            for h in range(n_heads):
                v_rows = vals[h // groups]
                ctx[h * head_dim:(h + 1) * head_dim] = context_contribution(alpha[h], v_rows, vmask)
                beta, v, m_v = visual_read(alpha[h], v_rows, vmask)
                shift, vn = realloc_ceiling(beta, v_rows[vmask])
                mvs.append(m_v)
                ceil_abs.append(shift)
                vnorms.append(vn)
                ceil_rel.append(shift / (vn + EPS))
                per_head_v.append(v)

            o_w = lay.self_attn.o_proj.weight.detach().to(torch.float64).cpu()
            r_c = o_w @ ctx                                        # [hidden]
            h_native = h_in.to(torch.float64).cpu()
            q_now = _head_query(lay, h_native.to(h_in.dtype).to(bb.device), n_heads, head_dim)
            q_hat = _head_query(lay, (h_native + r_c).to(h_in.dtype).to(bb.device), n_heads, head_dim)
            q_now, q_hat = q_now.detach().cpu(), q_hat.detach().cpu()
            gaps = [cos_gap(q_now[h], q_hat[h]) for h in range(n_heads)]

            rec["layers"].append({
                "l": li,
                "cos_gap": [round(v, 6) for v in gaps],
                "m_v": [round(v, 6) for v in mvs],
                "ceil_rel": [round(v, 5) for v in ceil_rel],
                "ceil_abs": [round(v, 5) for v in ceil_abs],
                "v_norm": [round(v, 5) for v in vnorms],
                "r_c_norm": round(float(r_c.norm()), 5),
                "h_norm": round(float(h_native.norm()), 5),
            })
    finally:
        del work
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    all_gap = [v for L in rec["layers"] for v in L["cos_gap"]]
    all_mv = [v for L in rec["layers"] for v in L["m_v"]]
    all_cr = [v for L in rec["layers"] for v in L["ceil_rel"]]
    print(f"[RCVR] case {case_index} layers={len(rec['layers'])} heads={n_heads} "
          f"kv={kv_len} vis={int(vmask.sum())} cos_gap_med={med(all_gap):.5f} "
          f"m_v_med={med(all_mv):.5f} ceil_rel_med={med(all_cr):.3f} attn={rec['attn']}", flush=True)


@torch.no_grad()
def _forward(backbone, cache, input_ids, position_cursor, want_attn, store):
    """One forward on `cache` (mutates it). Returns the attention tuple when asked."""
    attention_mask = torch.ones_like(input_ids, device=backbone.device)
    past_len = backbone._kv_len(cache)
    position_ids = None
    if backbone.mrope_pos and hasattr(backbone, "_text_positions"):
        position_ids = backbone._text_positions(
            input_ids.shape[1],
            start=position_cursor if position_cursor is not None else past_len)
    if past_len > 0:
        past_mask = torch.ones((attention_mask.shape[0], past_len),
                               dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        if backbone.mrope_pos and position_cursor is not None:
            backbone.vlm.rope_deltas = torch.tensor(
                [[position_cursor - past_len]], dtype=torch.long, device=backbone.device)
    out = backbone.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_values=cache, use_cache=True,
                         output_attentions=bool(want_attn))
    return getattr(out, "attentions", None) if want_attn else None
