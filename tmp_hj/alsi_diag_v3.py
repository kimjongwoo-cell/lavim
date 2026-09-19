"""C2 structural diagnosis: Answerer Latent Source Interference (opt-in).

Notion Experiment Log: "[C2 구조 진단] Answerer Latent Source Interference — visual vs
Reasoner/question/prefix source decomposition". Nothing here runs unless VLMAS_ALSI=<jsonl>.
Measurement only: no decode change, no intervention, no candidate/GT signal.

  sources S = {V, R, Q, Y}                                                    [section 3]
      V  persistent visual KV        (engine._rpath_visual_cols)
      R  Reasoner latent block       (engine._rpath_latent_cols)
      Q  static text before the answer (everything else < generation onset)
      Y  Answerer prefix generated so far (columns >= generation onset)
  source message  m_{t,s} = W_O [ (+)_h sum_{j in I_s} alpha_{t,h,j} v_{h,j} ]
  exactness       m_attn = sum_s m_s must hold to numerical precision   -> eps_dec [8A]
  latent effect   d_{t,s} = J_{l->L} m_{t,s}, propagated through the NATIVE suffix and
                  the final norm, by symmetric finite differences at two scales   [4, 8B]
                      dbar_s ~ ( f(h + e m_s) - f(h - e m_s) ) / (2 e)
                  At l = L-1 the analytic RMSNorm JVP is also computed, so the FD path is
                  cross-checked against an exact answer on every case.
  interference    Gamma_{V,s} = cos(dbar_V, dbar_s)                              [5]
  cancellation    kappa_V = -min(0, dbar_V . dbar_N) / ||dbar_V||^2,  N = R+Q+Y   [5]
  unique frac     gram = the six off-diagonal <dbar_a, dbar_b>; with d_norm this gives the
                  full {V,R,Q,Y} Gram, hence rho_uniq = 1 - g^T A^+ g / ||dbar_V||^2 with
                  A = Gram(R,Q,Y) -- the visual effect not spanned by non-visual sources.
  null            dbar_V of a DIFFERENT case (rolling bank) against this case's dbar_s,
                  so the norms stay realistic and only source-content pairing breaks [8C]

Layers are fixed by construction, never chosen from the results: l in {L/2, 3L/4, L-1}.

Env
  VLMAS_ALSI=<path.jsonl>      output file (append; one row per case). Unset = no-op.
  VLMAS_ALSI_LAYERS=a,b,c      override the layer checkpoints (default: L/2,3L/4,L-1)
  VLMAS_ALSI_MAXSTEPS=<int>    cap on probed answer steps (default 6)
  VLMAS_ALSI_EPS=e1,e2,e3      FD steps as fractions of ||h||, suffix run in float32
                               (default 0.1,0.05,0.02; the middle scale is the estimate)
  VLMAS_ALSI_ARM=<str>         free-form label for the addressing arm (recorded only)
"""
from __future__ import annotations

import contextlib
import json
import os
from copy import deepcopy

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

SOURCES = ("V", "R", "Q", "Y")


# --------------------------------------------------------------------------- pure


def cosine(a, b, eps: float = 1e-12) -> float:
    a = a.to(torch.float64); b = b.to(torch.float64)
    na, nb = float(a.norm()), float(b.norm())
    return float(a @ b) / (na * nb + eps)


def cancellation(d_v, d_n, eps: float = 1e-12) -> float:
    """kappa = -min(0, d_V . d_N) / ||d_V||^2 : the part of the visual direction that the
    other sources actively push back on (0 when they do not oppose it)."""
    d_v = d_v.to(torch.float64); d_n = d_n.to(torch.float64)
    return float(-min(0.0, float(d_v @ d_n)) / (float(d_v @ d_v) + eps))


def source_columns(n_total: int, vis_cols, lat_cols, gen_start: int, row_abs: int):
    """Partition [0, row_abs] into V / R / Q / Y index tensors (causal: nothing above row)."""
    hi = min(int(row_abs) + 1, int(n_total))
    mark = torch.zeros(hi, dtype=torch.uint8)
    if vis_cols is not None:
        v = vis_cols[vis_cols < hi]
        mark[v.to(torch.long)] = 1
    if lat_cols is not None:
        r = lat_cols[lat_cols < hi]
        mark[r.to(torch.long)] = 2
    idx = torch.arange(hi, dtype=torch.long)
    gen = idx >= int(gen_start)
    out = {
        "V": idx[(mark == 1) & ~gen],
        "R": idx[(mark == 2) & ~gen],
        "Q": idx[(mark == 0) & ~gen],
        "Y": idx[gen],
    }
    return out


def rmsnorm_jvp(h, m, weight, eps: float = 1e-6):
    """Directional derivative of RMSNorm at h in the direction m (float64)."""
    from memory.obscal_diag import rmsnorm_jvp as _j
    return _j(h, m, weight, eps)


# --------------------------------------------------------------------------- capture


@contextlib.contextmanager
def capture_source_writes(backbone, layer_index: int, col_groups, store: dict):
    """Hook one decoder layer; record each source's attention write for every row.

    store gets "m" [G, T, D] (per source, o_proj applied), "m_all" [T, D] (all columns),
    "h_out" [T, D] (the layer's output residual) and "pe" (this row's position embeddings).
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
        L = int(keys.shape[2]); T = int(hidden.shape[1]); Hd = module.head_dim
        cos, sin = pe
        q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
        K = repeat_kv(keys, module.num_key_value_groups).float()
        V = repeat_kv(values, module.num_key_value_groups).float()
        scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling
        if T > 1:
            past = L - T
            col = torch.arange(L, device=keys.device)[None, None, None, :]
            row = (past + torch.arange(T, device=keys.device))[None, None, :, None]
            scores = scores.masked_fill(col > row, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)

        # o_proj in float32 on purpose: in the model's bf16 the four per-source
        # projections and the projection of their sum disagree by ~4e-3 (bf16 eps is
        # 7.8e-3), which would swamp the section-8A exactness check with rounding instead
        # of the indexing errors it is meant to catch.
        w_o = module.o_proj.weight.detach().float()
        b_o = getattr(module.o_proj, "bias", None)
        b_o = b_o.detach().float() if b_o is not None else None

        def project(cols):
            if cols is None:
                ctx = torch.matmul(alpha, V)
            else:
                cols = cols.to(keys.device); cols = cols[cols < L]
                if int(cols.numel()) == 0:
                    return torch.zeros(T, int(w_o.shape[0]), dtype=torch.float32)
                ctx = torch.matmul(alpha[..., cols], V[:, :, cols, :])
            flat = ctx.transpose(1, 2).reshape(1, T, -1).float()
            out = flat[0] @ w_o.T
            if b_o is not None and cols is None:
                out = out + b_o          # bias belongs to the total, not to any source
            return out.detach().float().cpu()

        store["m"] = torch.stack([project(c) for c in col_groups])       # [G, T, D]
        store["m_all"] = project(None)                                   # [T, D]
        store["pe"] = (cos.detach(), sin.detach())
        store["T"] = T
        return output

    def out_hook(module, args, kwargs, output):
        h = output[0] if isinstance(output, tuple) else output
        store["h_out"] = h.detach()[0].float().cpu()
        return output

    h1 = layer.self_attn.register_forward_hook(hook, with_kwargs=True)
    h2 = layer.register_forward_hook(out_hook, with_kwargs=True)
    try:
        yield store
    finally:
        h1.remove(); h2.remove()


@contextlib.contextmanager
def suffix_float32(backbone, cache, layer_index: int):
    """Run the native suffix (layers > layer_index and the final norm) in float32.

    In the model's bf16 the finite difference had no usable step at all: a large relative
    step left the linear regime (v1, e=1.0/0.5 absolute: jvp_agree L18 0.80, L27 0.56) and a
    small one drowned in bf16 rounding (v2, e=0.02/0.01 relative: L18 1.16, L27 0.93). Only
    the suffix modules are upcast; bf16 -> float32 -> bf16 round-trips exactly, so the model
    comes back bit-for-bit. `cache` must be a private copy — its suffix layers are upcast and
    left that way (the probe's `sc_full` is a deepcopy made for this purpose).
    """
    lm = backbone.lm
    lo = int(layer_index) + 1
    mods = [lm.layers[i] for i in range(lo, len(lm.layers))] + [lm.norm]
    orig = [next(m.parameters()).dtype for m in mods]
    try:
        for m in mods:
            m.float()
        for i in range(lo, len(lm.layers)):
            cl = cache.layers[i]
            if getattr(cl, "keys", None) is not None and cl.keys.dtype != torch.float32:
                cl.keys = cl.keys.float()
                cl.values = cl.values.float()
        yield
    finally:
        for m, dt in zip(mods, orig):
            m.to(dt)


@torch.no_grad()
def suffix_forward(backbone, cache, h_row, pe_row, layer_index: int, row_pos: int):
    """f(h): run decoder layers (layer_index, L) on ONE row with `cache` cropped to
    row_pos, then the final norm. The cache grows by one row and is cropped back.
    Runs in whatever dtype the suffix modules currently hold (float32 under
    `suffix_float32`)."""
    lm = backbone.lm
    lo = int(layer_index) + 1
    head = lm.layers[lo] if lo < len(lm.layers) else lm.norm
    dt = next(head.parameters()).dtype
    x = h_row.to(device=backbone.device, dtype=dt)
    x = x.view(1, 1, -1)
    cos, sin = pe_row
    pe = (cos[..., -1:, :].to(device=x.device, dtype=dt),
          sin[..., -1:, :].to(device=x.device, dtype=dt))
    for li in range(int(layer_index) + 1, len(lm.layers)):
        x = lm.layers[li](hidden_states=x, position_embeddings=pe,
                          attention_mask=None, past_key_values=cache, use_cache=True)
        x = x[0] if isinstance(x, tuple) else x
    out = lm.norm(x)[0, -1].detach().float().cpu()
    cache.crop(int(row_pos))
    return out


@torch.no_grad()
def propagate(backbone, cache, h_out_row, pe_row, m_rows, layer_index: int, row_pos: int,
              eps_scales):
    """dbar_s for every source by symmetric finite differences at several scales.

    Returns {scale: [G, D]} plus the base output; a scale-to-scale agreement is the
    section-8B linearity check (no hyperparameter is tuned on it)."""
    # The step is RELATIVE to the residual it perturbs: ||step * m|| = e * ||h||. An
    # absolute e=1 left the linear regime (the two scales disagreed by ~55%), because a
    # source message can be as large as the residual itself. Dividing by the same step
    # keeps the estimate of d/da f(h + a m) at a=0 correct whatever the step is.
    hn = float(torch.as_tensor(h_out_row).norm())
    out = {}
    for e in eps_scales:
        rows = []
        for m in m_rows:
            mn = float(m.norm())
            step = (e * hn / mn) if (mn > 1e-12 and hn > 0) else 1.0
            plus = suffix_forward(backbone, cache, h_out_row + step * m, pe_row, layer_index, row_pos)
            minus = suffix_forward(backbone, cache, h_out_row - step * m, pe_row, layer_index, row_pos)
            rows.append(((plus - minus) / (2.0 * step)).double())
        out[e] = torch.stack(rows)
    return out


# --------------------------------------------------------------------------- driver


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, max_new_tokens: int = 512) -> None:
    out_path = os.environ.get("VLMAS_ALSI", "").strip()
    if not out_path:
        return
    from memory.rmvr_diag import field_spans
    from memory.vcca_diag import _arm_pass
    from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json

    bb = engine._backbone
    vis_cols = getattr(engine, "_rpath_visual_cols", None)
    lat_cols = getattr(engine, "_rpath_latent_cols", None)
    if vis_cols is None:
        print(f"[ALSI] case {case_index}: SKIP (no visual bookkeeping)", flush=True)
        return
    vis_cols = vis_cols.detach().cpu().to(torch.long)
    lat_cols = (lat_cols.detach().cpu().to(torch.long) if lat_cols is not None
                else torch.zeros(0, dtype=torch.long))

    n_layers = len(bb.lm.layers)
    env_l = os.environ.get("VLMAS_ALSI_LAYERS", "").strip()
    layers = ([int(v) for v in env_l.split(",") if v.strip()] if env_l
              else [n_layers // 2, (3 * n_layers) // 4, n_layers - 1])
    layers = sorted({max(0, min(n_layers - 1, int(l))) for l in layers})
    max_steps = int(os.environ.get("VLMAS_ALSI_MAXSTEPS", "6") or 6)
    eps_scales = [float(v) for v in
                  (os.environ.get("VLMAS_ALSI_EPS", "0.1,0.05,0.02").split(","))if v.strip()]
    arm = os.environ.get("VLMAS_ALSI_ARM", "").strip() or (
        os.environ.get("VLMAS_KV_ROUTE_MODE", "").strip() or "native")

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
        print(f"[ALSI] case {case_index}: SKIP (empty generation)", flush=True)
        return
    fields = field_spans(text, offsets, json_prefix) if offsets else {}
    ans_rows = fields.get("answer", {}).get("idx") or []
    rows_idx = (ans_rows or list(range(len(gen_ids))))[:max_steps]

    base_len = int(bb._kv_len(cache))
    n_prompt = int(prompt_ids.shape[1])
    gen_start = base_len + n_prompt                     # first generated column
    norm = bb.lm.norm
    w_norm = norm.weight.detach().float().cpu()
    eps_n = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))

    rec = {"case": int(case_index), "arm": arm, "layers": layers, "n_layers": n_layers,
           "n_vis": int(vis_cols.numel()), "n_lat": int(lat_cols.numel()),
           "gen_start": gen_start, "base_len": base_len, "n_prompt": n_prompt,
           "gen_text": text, "rows": rows_idx, "answer_rows": ans_rows,
           "eps": eps_scales, "steps": [],
           "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager") or "eager"}

    bank = getattr(engine, "_alsi_bank", None)          # previous case's dbar_V, per layer
    new_bank: dict = {}

    for l in layers:
        # one forward per layer: teacher-forced prompt+generation, source writes captured
        groups = source_columns(gen_start + len(gen_ids), vis_cols, lat_cols,
                                gen_start, gen_start + len(gen_ids) - 1)
        store: dict = {}
        c = deepcopy(cache)
        try:
            with capture_source_writes(bb, l, [groups[s] for s in SOURCES], store):
                _arm_pass(bb, c, prompt_ids, [gen_ids], position_cursor)
        finally:
            del c
        if "m" not in store or "h_out" not in store:
            print(f"[ALSI] case {case_index}: SKIP (hook did not fire at L{l})", flush=True)
            return
        m = store["m"]                                   # [4, T, D]
        m_all = store["m_all"]                           # [T, D]
        h_out = store["h_out"]                           # [T, D]
        T = int(store["T"])
        # _arm_pass rows predict the NEXT token: the state that decides generated token j
        # is row n_prompt-1+j, one before the token's own row.
        first = T - len(gen_ids) - 1

        # one teacher-forced cache per layer, then walk the probed rows from the last to
        # the first so a single in-place crop is enough (crop only ever shortens).
        sc_full = deepcopy(cache)
        _rebuild_to(bb, sc_full, prompt_ids, [gen_ids], position_cursor)
        layer_steps = []
        for t in sorted(rows_idx, reverse=True):
            fr = first + int(t)
            if fr < 0 or fr >= T:
                continue
            row_abs = gen_start - 1 + int(t)             # the query position that decides t
            m_rows = [m[i, fr].double() for i in range(len(SOURCES))]
            dec_err = float((m_all[fr].double() - sum(m_rows)).norm()
                            / (m_all[fr].double().norm() + 1e-12))
            sc_full.crop(row_abs)
            with suffix_float32(bb, sc_full, l):
                d = propagate(bb, sc_full, h_out[fr], store["pe"],
                              [x.float() for x in m_rows], l, row_abs, eps_scales)
            # The MIDDLE scale is the estimate; the outer scales bracket it from both sides,
            # so both agreements being small (a plateau) is the section-8B linearity evidence.
            mid = eps_scales[len(eps_scales) // 2]
            d0 = d[mid]
            fid = (float((d[eps_scales[0]] - d0).norm() / (d0.norm() + 1e-12))
                   if len(eps_scales) > 1 else None)
            fid_lo = (float((d[eps_scales[-1]] - d0).norm() / (d0.norm() + 1e-12))
                      if len(eps_scales) > 2 else None)
            d_v = d0[0]
            d_n = d0[1] + d0[2] + d0[3]
            step = {
                "t": int(t), "layer": int(l), "y": int(gen_ids[t]),
                "eps_dec": round(dec_err, 8), "jvp_agree": (round(fid, 6) if fid is not None else None),
                "jvp_agree_lo": (round(fid_lo, 6) if fid_lo is not None else None),
                "eps_mid": mid, "suffix_dtype": "float32",
                "m_norm": {s: round(float(m_rows[i].norm()), 6) for i, s in enumerate(SOURCES)},
                "d_norm": {s: round(float(d0[i].norm()), 6) for i, s in enumerate(SOURCES)},
                "gamma": {s: round(cosine(d_v, d0[i]), 6) for i, s in enumerate(SOURCES) if s != "V"},
                "gamma_N": round(cosine(d_v, d_n), 6),
                "kappa_V": round(cancellation(d_v, d_n), 6),
                "kappa": {s: round(cancellation(d_v, d0[i]), 6)
                          for i, s in enumerate(SOURCES) if s != "V"},
                # full {V,R,Q,Y} Gram: the diagonal is d_norm^2, so only the six
                # off-diagonal inner products are stored. rho_uniq needs the R/Q/Y
                # block, which cos(d_V, d_s) alone cannot reconstruct.
                "gram": {f"{a}{b}": round(float(d0[i] @ d0[j]), 6)
                         for i, a in enumerate(SOURCES)
                         for j, b in enumerate(SOURCES) if i < j},
            }
            if l == n_layers - 1:
                # exact analytic check: the suffix is only the final norm here
                an = rmsnorm_jvp(h_out[fr].double(), m_rows[0], w_norm, eps_n)
                step["norm_jvp_cos"] = round(cosine(an, d_v), 6)
                step["norm_jvp_ratio"] = round(float(an.norm() / (d_v.norm() + 1e-12)), 6)
            prev = (bank or {}).get(l)
            if prev is not None and prev.shape == d_v.shape:
                step["gamma_N_null"] = round(cosine(prev, d_n), 6)
                step["kappa_V_null"] = round(cancellation(prev, d_n), 6)
                # shuffled pairing for rho_uniq: another case's d_V against THIS case's
                # R/Q/Y subspace, so only the source pairing breaks.
                step["gram_null"] = {"VV": round(float(prev @ prev), 6),
                                     **{f"V{b}": round(float(prev @ d0[j]), 6)
                                        for j, b in enumerate(SOURCES) if b != "V"}}
            new_bank.setdefault(l, d_v.clone())
            layer_steps.append(step)
        del sc_full
        rec["steps"].extend(sorted(layer_steps, key=lambda s: s["t"]))

    engine._alsi_bank = new_bank or bank
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    k = [s["kappa_V"] for s in rec["steps"]]
    e = [s["eps_dec"] for s in rec["steps"]]
    print(f"[ALSI] case {case_index} arm={arm} layers={layers} steps={len(rec['steps'])} "
          f"eps_dec_max={max(e) if e else float('nan'):.2e} "
          f"kappa_V_med={sorted(k)[len(k) // 2] if k else float('nan'):.4f}", flush=True)


def _rebuild_to(backbone, cache, prompt_ids, gen_list, position_cursor):
    """Re-run the teacher-forced pass on `cache` so it holds prompt+generation K/V."""
    from memory.vcca_diag import _arm_pass
    _arm_pass(backbone, cache, prompt_ids, gen_list, position_cursor)
    return cache


__all__ = ["cosine", "cancellation", "source_columns", "capture_source_writes",
           "suffix_forward", "propagate", "run", "SOURCES"]
