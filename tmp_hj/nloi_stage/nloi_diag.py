"""C2 structural diagnosis: Non-local Observation Integration (opt-in).

Notion Experiment Log: "[C2 구조 진단] Non-local Observation Integration — 멀리 떨어진 WSI
observations의 joint integration 검정". Nothing here runs unless VLMAS_NLOI=<jsonl>.

Question: is the Answerer's final normalized state explained by the independent effects of
single observations, or do separate observations (including far-apart ones) interact?

  unit       O_o = one Navigator crop = its block of persistent visual KV columns
  removal    value-contribution cut at the Answerer, on every LM layer and every Answerer
             query row (prompt prefill + teacher-forced answer prefix). The native attention
             weights over the whole cache are kept and only sum_{j in cut} alpha_j v_j is
             dropped (memory/visual_cut 'zero' restricted to a column subset). The Reasoner
             rows already in the cache are the fixed native prefix. One attention layer's
             output is exactly additive in the removed columns, so any non-additivity at the
             final state comes from downstream integration (later layers / MLP / norm), not
             from the removal operator.
  reference  h_F = final-norm output under the same hook with nothing cut (float32 recompute
             path). The bf16 native forward is recorded only as a path-drift gauge.
  per pair   Delta_i = h_F - h_{-i};  Psi_ij = (h_F - h_{-ij}) - (Delta_i + Delta_j)
             C_ij = ||h_F - h_{-ij}|| - max(||Delta_i||, ||Delta_j||)
  controls   true crop grouping vs token-shuffled grouping (same sizes, memory/ofh.partition),
             all-visual cut as magnitude reference, per-crop native visual mass, crop level-0
             boxes / magnification / parent for the distance analysis (distance bins and the
             distance-label permutation are offline).
  rows       the rows predicting the first VLMAS_NLOI_ROWS answer-content tokens of the native
             greedy answer (row 0 = decision row). Candidates and the gold answer are unused.

Env
  VLMAS_NLOI=<path.jsonl>     output file (append; one row per case). Unset = no-op.
  VLMAS_NLOI_ROWS=<int>       probed answer rows (default 4)
  VLMAS_NLOI_SHUFFLED=0|1     token-shuffled grouping arm (default 1)
  VLMAS_NLOI_DUMP=<dir>       also torch.save every condition's final states (float32)
  VLMAS_NLOI_ARM=<str>        free-form addressing label (recorded only)
  VLMAS_NLOI_FROM_LAYER=<int> cut only on decoder layers >= this index (default 0 = Stage A
                              full path; Notion §6 Stage B uses 18/24/27/30/33). Layers below
                              still run the identity recompute so every arm shares one path.
"""
from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import time

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv


# --------------------------------------------------------------------------- pure


def pair_metrics(h_f, h_i, h_j, h_ij) -> dict:
    """Section 3 quantities for one pair at one row (float64 vectors)."""
    d_i = h_f - h_i
    d_j = h_f - h_j
    d_ij = h_f - h_ij
    psi = d_ij - (d_i + d_j)
    n_i, n_j, n_ij, n_psi = (float(v.norm()) for v in (d_i, d_j, d_ij, psi))
    n_sum = float((d_i + d_j).norm())
    mx = max(n_i, n_j)
    denom = n_i * n_j
    return {
        "d_i": n_i, "d_j": n_j, "d_ij": n_ij, "d_sum": n_sum, "psi": n_psi,
        "psi_rel": n_psi / (n_i + n_j) if (n_i + n_j) > 0 else None,
        "C": n_ij - mx,
        "C_rel": (n_ij - mx) / mx if mx > 0 else None,
        "cos_ij": float(d_i @ d_j) / denom if denom > 0 else None,
    }


def box_center(box):
    if box is None:
        return None
    x, y, w, h = (float(v) for v in box)
    return (x + 0.5 * w, y + 0.5 * h)


def overlap_frac(a, b) -> float:
    """Shared area / smaller area of two (x, y, w, h) boxes; 0 when unknown."""
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return (iw * ih) / max(min(aw * ah, bw * bh), 1e-9)


def pair_geometry(i: int, j: int, boxes, mags, parents) -> dict:
    ci = box_center(boxes[i]) if boxes and i < len(boxes) else None
    cj = box_center(boxes[j]) if boxes and j < len(boxes) else None
    dist = math.dist(ci, cj) if ci is not None and cj is not None else None
    mi = int(mags[i]) if mags and i < len(mags) else None
    mj = int(mags[j]) if mags and j < len(mags) else None
    pi = int(parents[i]) if parents and i < len(parents) else -1
    pj = int(parents[j]) if parents and j < len(parents) else -1
    return {
        "dist": dist, "mag": [mi, mj], "same_mag": (mi == mj) if mi is not None else None,
        "parent_child": bool(pi == j or pj == i),
        "siblings": bool(pi >= 0 and pi == pj),
        "overlap": overlap_frac(boxes[i], boxes[j]) if boxes and max(i, j) < len(boxes) else None,
    }


def all_pairs(n: int):
    return list(itertools.combinations(range(int(n)), 2))


def js_rows(logp_a, logp_b):
    """Jensen-Shannon divergence per row of two [R, V] log-prob tensors (natural log)."""
    pa, pb = logp_a.exp(), logp_b.exp()
    m = (0.5 * (pa + pb)).clamp_min(1e-300).log()
    kl_a = (pa * (logp_a - m)).sum(-1)
    kl_b = (pb * (logp_b - m)).sum(-1)
    return (0.5 * (kl_a + kl_b)).clamp_min(0.0)


# ------------------------------------------------------------------------ runtime


@contextlib.contextmanager
def install_value_cut(backbone, cut_cols, mass_groups=None, mass_rows=None, from_layer: int = 0):
    """Recompute every LM layer's attention output with the value term of `cut_cols` dropped.

    cut_cols: LongTensor of absolute cache columns (empty = identity recompute).
    from_layer: the cut is applied on layers >= from_layer only (identity recompute below).
    mass_groups / mass_rows: optional per-group native attention mass capture at the given
    forward rows (head-mean, summed over layers; divide by rec["n_layers_mass"]).
    Yields rec {"calls", "mass" [G, R] or None, "n_layers_mass"}.
    """
    layers = backbone.lm.layers
    rec = {"calls": 0, "mass": None, "n_layers_mass": 0}
    handles = []
    keep_cache: dict = {}

    def make_hook(li):
        def hook(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys = cache.layers[li].keys
            values = cache.layers[li].values
            L = int(keys.shape[2]); T = int(hidden.shape[1])
            Hd = module.head_dim
            g_ = module.num_key_value_groups
            cos, sin = pe
            q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            K = repeat_kv(keys, g_).float()
            V = repeat_kv(values, g_).float()
            scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling       # [1,Hq,T,L]
            if T > 1:
                past = L - T
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + torch.arange(T, device=keys.device))[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)
            if mass_groups is not None and mass_rows is not None:
                rows = mass_rows.to(keys.device)
                m = []
                for gc in mass_groups:
                    gc = gc.to(keys.device); gc = gc[gc < L]
                    m.append(alpha[0][:, rows][..., gc].sum(-1).mean(0))               # [R]
                cur = torch.stack(m).double().cpu()                                    # [G, R]
                rec["mass"] = cur if rec["mass"] is None else rec["mass"] + cur
                rec["n_layers_mass"] += 1
            key = (L, str(keys.device), li >= int(from_layer))
            keep = keep_cache.get(key)
            if keep is None:
                keep = torch.ones(L, dtype=alpha.dtype, device=keys.device)
                if li >= int(from_layer) and cut_cols is not None and int(cut_cols.numel()):
                    cc = cut_cols.to(device=keys.device, dtype=torch.long)
                    keep[cc[cc < L]] = 0.0
                keep_cache[key] = keep
            ctx = torch.matmul(alpha * keep, V)
            flat = ctx.transpose(1, 2).reshape(1, T, -1)
            new = module.o_proj(flat.to(module.o_proj.weight.dtype)).to(attn_out.dtype)
            rec["calls"] += 1
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return hook

    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def final_rows(backbone, cache, input_ids, position_cursor, rows):
    """One forward of `input_ids` on a COPY of `cache`; final-norm output at `rows` (float64)."""
    from copy import deepcopy
    c = deepcopy(cache)
    store: dict = {}
    rows_dev = rows.to(backbone.device)

    def hook(_module, _args, out):
        store["h"] = out[0].index_select(0, rows_dev).detach().to(torch.float64)

    handle = backbone.lm.norm.register_forward_hook(hook)
    try:
        attention_mask = torch.ones_like(input_ids, device=backbone.device)
        past_len = backbone._kv_len(c)
        position_ids = None
        if backbone.mrope_pos and hasattr(backbone, "_text_positions"):
            position_ids = backbone._text_positions(
                input_ids.shape[1], start=position_cursor if position_cursor is not None else past_len)
        if past_len > 0:
            past_mask = torch.ones((attention_mask.shape[0], past_len),
                                   dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
            if backbone.mrope_pos:
                backbone.vlm.rope_deltas = torch.tensor(
                    [[position_cursor - past_len]], dtype=torch.long, device=backbone.device)
        backbone.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                       past_key_values=c, use_cache=True)
    finally:
        handle.remove()
        del c
    return store["h"]


def _r(x, nd=6):
    return None if x is None else round(float(x), nd)


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, max_new_tokens: int = 512) -> None:
    out_path = os.environ.get("VLMAS_NLOI", "").strip()
    if not out_path:
        return
    from copy import deepcopy

    from memory.ofh import observation_groups, observation_pages, partition
    from memory.rmvr_diag import field_spans
    from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json

    t0 = time.time()
    bb = engine._backbone
    got = observation_groups(engine, "true")
    if got is None:
        print(f"[NLOI] case {case_index}: SKIP ({observation_pages(engine)[1]})", flush=True)
        return
    cols, groups_true, page_src = got
    pages, _ = observation_pages(engine)
    P = len(groups_true)
    if P < 2:
        print(f"[NLOI] case {case_index}: SKIP (observations={P})", flush=True)
        return
    boxes = list(getattr(bb, "_prune_image_boxes", ()) or ())
    mags = [int(x) for x in (getattr(bb, "_prune_image_magnifications", ()) or ())]
    parents = [int(x) for x in (getattr(bb, "_prune_image_parent_indices", ()) or ())]
    geom_ok = len(boxes) == P and len(mags) == P
    n_rows = max(1, int(os.environ.get("VLMAS_NLOI_ROWS", "4") or 4))
    do_shuf = os.environ.get("VLMAS_NLOI_SHUFFLED", "1").strip() != "0"
    dump_dir = os.environ.get("VLMAS_NLOI_DUMP", "").strip()
    from_layer = int(os.environ.get("VLMAS_NLOI_FROM_LAYER", "0") or 0)

    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    n_prompt = int(prompt_ids.shape[1])
    # v2: keep the token ids the model actually generated. generate_terminal_json returns
    # decode(...).strip(); re-tokenizing that text drops the leading-space quote token
    # (' "' + 'Yes' becomes '"Yes'), so the teacher-forced prefix would not be the native one.
    gen_cache = deepcopy(cache)
    model = bb.model
    had_attr = "generate" in vars(model)
    orig_generate = model.generate
    captured: dict = {}

    def _capture_generate(*args, **kwargs):
        out = orig_generate(*args, **kwargs)
        fed = kwargs.get("input_ids", args[0] if args else None)
        captured["ids"] = out[0, int(fed.shape[1]):].detach().cpu().tolist()
        return out

    model.generate = _capture_generate
    try:
        text = generate_terminal_json(
            backbone=bb, cache=gen_cache, position_cursor=position_cursor,
            system_prompt=system_prompt, user_prompt=user_prompt, json_prefix=json_prefix,
            max_new_tokens=int(max_new_tokens), temperature=0.0, top_p=1.0, do_sample=False)
    finally:
        if had_attr:
            model.generate = orig_generate
        else:
            del model.generate
        del gen_cache
    special = set(tok.all_special_ids)
    gen_ids = [int(t) for t in captured.get("ids", []) if int(t) not in special]
    if not gen_ids:
        print(f"[NLOI] case {case_index}: SKIP (empty generation)", flush=True)
        return
    raw, offsets, prev = "", [], 0
    for k in range(len(gen_ids)):
        cur = tok.decode(gen_ids[:k + 1], skip_special_tokens=True)
        offsets.append((prev, len(cur)))
        prev = len(cur)
        raw = cur
    lead = len(raw) - len(raw.lstrip())
    offsets = [(max(s - lead, 0), max(e - lead, 0)) for s, e in offsets]
    text_match = raw.lstrip() == text
    fields = field_spans(raw.lstrip(), offsets, json_prefix)
    ans_rows = fields.get("answer", {}).get("idx") or []
    tok_rows = (ans_rows or list(range(len(gen_ids))))[:n_rows]
    t_max = max(tok_rows)
    # the row predicting token t needs the input up to token t-1 only
    ids = torch.cat([prompt_ids, torch.tensor([gen_ids[:t_max]], dtype=prompt_ids.dtype,
                                              device=bb.device)], dim=1)
    fwd_rows = torch.tensor([n_prompt - 1 + int(t) for t in tok_rows], dtype=torch.long)
    R = int(fwd_rows.numel())

    empty = torch.zeros(0, dtype=torch.long)

    def cut_run(cut, mass_groups=None):
        with install_value_cut(bb, cut, mass_groups=mass_groups,
                               mass_rows=fwd_rows if mass_groups is not None else None,
                               from_layer=from_layer) as rec:
            h = final_rows(bb, cache, ids, position_cursor, fwd_rows)
        return h, rec

    h_native = final_rows(bb, cache, ids, position_cursor, fwd_rows)
    h_f, rec_f = cut_run(empty, mass_groups=groups_true)
    h_zero, _ = cut_run(cols)
    if rec_f["calls"] == 0:
        print(f"[NLOI] case {case_index}: SKIP (attention hook did not fire)", flush=True)
        return
    mass = (rec_f["mass"] / max(rec_f["n_layers_mass"], 1)) if rec_f["mass"] is not None else None

    lm_head = bb.model.lm_head
    wd = lm_head.weight.dtype

    def logp(h):
        return torch.log_softmax(lm_head(h.to(wd)).to(torch.float64), dim=-1)

    lp_f = logp(h_f)
    a_f = lp_f.argmax(-1)

    def out_stats(h):
        lp = logp(h)
        return {"js": [_r(v, 8) for v in js_rows(lp_f, lp).tolist()],
                "flip": [bool(int(x) != int(y)) for x, y in zip(lp.argmax(-1).tolist(), a_f.tolist())],
                "dlp_a": [_r(float(lp_f[r, a_f[r]] - lp[r, a_f[r]]), 6) for r in range(R)]}

    pairs = all_pairs(P)
    dumps = {"native": h_native.float().cpu(), "ident": h_f.float().cpu(), "zero_all": h_zero.float().cpu()}

    def arm(groups, tag):
        singles = []
        for o in range(P):
            h, _ = cut_run(groups[o])
            singles.append(h)
            dumps[f"{tag}_s{o}"] = h.float().cpu()
        prs = []
        for (i, j) in pairs:
            h_ij, _ = cut_run(torch.cat([groups[i], groups[j]]))
            dumps[f"{tag}_p{i}_{j}"] = h_ij.float().cpu()
            per_row = []
            for r in range(R):
                m = pair_metrics(h_f[r], singles[i][r], singles[j][r], h_ij[r])
                per_row.append({k: _r(v) for k, v in m.items()})
            ent = {"i": i, "j": j, "rows": per_row, "out": out_stats(h_ij)}
            prs.append(ent)
        sing = []
        for o in range(P):
            sing.append({"o": o, "d": [_r((h_f[r] - singles[o][r]).norm()) for r in range(R)],
                         "out": out_stats(singles[o])})
        # all-observation superadditivity: sum of single effects vs the joint all-visual cut
        d_all = h_f - h_zero
        s_sum = sum((h_f - s) for s in singles)
        superadd = [{"d_all": _r(d_all[r].norm()), "d_sum_singles": _r(s_sum[r].norm()),
                     "psi_all": _r((d_all[r] - s_sum[r]).norm())} for r in range(R)]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"singles": sing, "pairs": prs, "all": superadd}

    res_true = arm(groups_true, "true")
    geo = [pair_geometry(i, j, boxes, mags, parents) if geom_ok else None for (i, j) in pairs]
    for ent, g in zip(res_true["pairs"], geo):
        ent["geom"] = g
    res_shuf = None
    shuf_idx = None
    if do_shuf:
        got_s = observation_groups(engine, "shuffled")
        if got_s is not None:
            res_shuf = arm(got_s[1], "shuf")
            shuf_idx = partition(pages, "shuffled", 42 + int(getattr(engine, "_rpath_case_index", 0) or 0))

    rec_out = {
        "case": int(case_index), "arm": os.environ.get("VLMAS_NLOI_ARM", "").strip() or (
            "canonical" if os.environ.get("VLMAS_KV_ROUTE_MODE", "").strip() == "canonical" else "native"),
        "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"), "from_layer": from_layer,
        "pages": [int(p) for p in pages], "page_src": page_src, "n_vis": int(cols.numel()),
        "boxes": [list(b) if b is not None else None for b in boxes] if geom_ok else None,
        "mags": mags if geom_ok else None, "parents": parents if len(parents) == P else None,
        "n_prompt": n_prompt, "tok_rows": [int(t) for t in tok_rows], "answer_rows": [int(t) for t in ans_rows],
        "gen_text": text[:300],
        "gen_tokens": [int(t) for t in gen_ids[:t_max + 1]],
        "argmax_ident": [int(x) for x in a_f.tolist()],
        "argmax_eq_native_tok": [bool(int(a) == int(gen_ids[int(t)])) for a, t in zip(a_f.tolist(), tok_rows)],
        "text_match": bool(text_match),
        "path_drift": [_r((h_f[r] - h_native[r]).norm()) for r in range(R)],
        "h_norm": [_r(h_f[r].norm()) for r in range(R)],
        "zero_all": {"d": [_r((h_f[r] - h_zero[r]).norm()) for r in range(R)], "out": out_stats(h_zero)},
        "mass_true": None if mass is None else [[_r(v, 8) for v in row] for row in mass.tolist()],
        "true": res_true, "shuffled": res_shuf,
        "shuffled_sizes": None if shuf_idx is None else [len(g) for g in shuf_idx],
        "n_forwards": 3 + (P + len(pairs)) * (2 if res_shuf is not None else 1),
        "sec": round(time.time() - t0, 1), "nloi": "v2",
    }
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec_out) + "\n")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        torch.save({"case": int(case_index), "rows": fwd_rows, "h": dumps},
                   os.path.join(dump_dir, f"nloi_case{int(case_index):03d}.pt"))

    med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else float("nan"))
    pt = [p["rows"][0]["psi_rel"] for p in res_true["pairs"] if p["rows"][0]["psi_rel"] is not None]
    ct = [p["rows"][0]["C_rel"] for p in res_true["pairs"] if p["rows"][0]["C_rel"] is not None]
    msg = (f"[NLOI] case {case_index} obs={P} pairs={len(pairs)} rows={R} geom={int(geom_ok)} "
           f"psi_rel_med={med(pt):.4f} C_rel_med={med(ct):.4f}")
    if res_shuf is not None:
        ps = [p["rows"][0]["psi_rel"] for p in res_shuf["pairs"] if p["rows"][0]["psi_rel"] is not None]
        cs = [p["rows"][0]["C_rel"] for p in res_shuf["pairs"] if p["rows"][0]["C_rel"] is not None]
        msg += f" shuf_psi_rel_med={med(ps):.4f} shuf_C_rel_med={med(cs):.4f}"
    msg += (f" d_all={rec_out['zero_all']['d'][0]} drift={rec_out['path_drift'][0]} "
            f"tok_eq={sum(rec_out['argmax_eq_native_tok'])}/{R} text_match={int(text_match)} "
            f"from_layer={from_layer} src={page_src} arm={rec_out['arm']} attn={rec_out['attn']} sec={rec_out['sec']} nloi=v2")
    print(msg, flush=True)


__all__ = ["run", "pair_metrics", "pair_geometry", "all_pairs", "box_center", "overlap_frac",
           "js_rows", "install_value_cut", "final_rows"]
