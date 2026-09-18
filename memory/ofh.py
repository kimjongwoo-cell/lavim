"""C2 structure test: Observation-Factorized Latent Handoff (opt-in).

Notion Experiment Log: "C2 구조 검증: Observation-Factorized Latent Handoff — WSI
observation-level receiver invariance". Nothing here runs unless VLMAS_OFH (decode-time
arm) or VLMAS_OFH_DIAG (density-sensitivity probe) is set.

The readout rule itself is the LME group competition already implemented in
memory/group_attn.py mode "c2":
    s_o = LSE_{j in S_o}(l_j) - log|S_o|,  beta_o = softmax_o(s_o),
    pi_{j|o} = softmax_{j in S_o}(l_j),    alpha~_j = m_V^native * beta_o * pi_{j|o}
so the native total visual mass and the non-visual distribution are untouched. What this
module adds is the PARTITION and its falsifiers:
    true      S_o = the Navigator observation (crop) a token came from
    shuffled  the same observation sizes, token->observation membership permuted inside
              the case (kills physical observation identity, keeps the size histogram)
    random    a random partition with the same size histogram (kills identity and the
              crop-contiguity of the membership)
and the Stage-A density falsifiers (Notion section 5-6):
    duplicate   one observation's visual K/V columns appended once more (2x tokens,
                identical content) -> s_o is algebraically invariant under "c2"
    subsample   deterministic thinning (every other token) of one observation

Env
  VLMAS_OFH=true|shuffled|random      decode-time arm (Answerer prefill + generation)
  VLMAS_OFH_SEED=<int>                permutation seed (default 42 + case index)
  VLMAS_OFH_DIAG=<path.jsonl>         density-sensitivity probe (post-decode, no effect
                                      on the recorded answer)
  VLMAS_OFH_DIAG_TARGETS=<int>        observations probed per case (default 2: the
                                      largest and the smallest native visual mass)
"""
from __future__ import annotations

import contextlib
import json
import os
import random
from copy import deepcopy

import torch


# --------------------------------------------------------------------------- pure


def crop_spans(pages):
    """[(start, end)] token index span of every observation (crop)."""
    out, base = [], 0
    for p in pages:
        out.append((base, base + int(p)))
        base += int(p)
    return out


def partition(pages, mode: str, seed: int):
    """List of index lists (into the visual token order) per observation.

    true      contiguous crop spans
    shuffled  same sizes, membership permuted across the whole visual set
    random    same size histogram, random partition (identical to shuffled for equal
              sizes, kept separate so the log records which falsifier ran)
    """
    spans = crop_spans(pages)
    if mode == "true":
        return [list(range(a, b)) for a, b in spans]
    n = spans[-1][1] if spans else 0
    rng = random.Random(int(seed))
    order = list(range(n))
    rng.shuffle(order)
    out, pos = [], 0
    for a, b in spans:
        k = b - a
        out.append(sorted(order[pos:pos + k]))
        pos += k
    return out


def duplicate_plan(groups, target: int, n_vis: int):
    """Columns to append (the target observation's tokens) and the new group lists."""
    extra = list(groups[target])
    new = [list(g) for g in groups]
    new[target] = new[target] + list(range(n_vis, n_vis + len(extra)))
    return extra, new


def subsample_plan(groups, target: int, keep_every: int = 2):
    """Deterministic thinning of one observation: keep every k-th token (>=1 token)."""
    thinned = list(groups[target])[::keep_every] or list(groups[target])[:1]
    new = [list(g) for g in groups]
    new[target] = thinned
    dropped = [j for j in groups[target] if j not in set(thinned)]
    return dropped, new


def js_divergence(p, q):
    """Jensen-Shannon divergence of two probability vectors (natural log)."""
    p = p.double(); q = q.double()
    m = 0.5 * (p + q)
    kl = lambda a, b: torch.sum(torch.where(a > 0, a * (torch.log(a + 1e-300) - torch.log(b + 1e-300)),
                                            torch.zeros_like(a)))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


# ------------------------------------------------------------------------ runtime


def observation_pages(engine):
    """(token count per observation, source) without requiring the routing bookkeeping.

    `_route_pages` only exists when VLMAS_KV_ROUTE / VLMAS_RETR is set, and turning those
    on would change the run's native condition, so fall back to the reassembly metadata
    and finally to an equal split (exact here: every crop is encoded to the same number
    of merged tokens). The source is logged so a non-uniform layout can be spotted.
    """
    bb = engine._backbone
    cols = getattr(engine, "_rpath_visual_cols", None)
    if cols is None or int(cols.numel()) == 0:
        return None, "no_cols"
    n = int(cols.numel())
    pages = getattr(bb, "_route_pages", None)
    if pages and sum(int(p) for p in pages) == n:
        return [int(p) for p in pages], "route_pages"
    meta = getattr(bb, "_reassembly_meta", None)
    if meta and meta[0] and sum(int(p) for p in meta[0]) == n:
        return [int(p) for p in meta[0]], "reassembly_meta"
    mags = tuple(getattr(bb, "_prune_image_magnifications", ()) or ())
    if mags and n % len(mags) == 0:
        return [n // len(mags)] * len(mags), "equal_split"
    return None, "unavailable"


def observation_groups(engine, mode: str = "true", seed: int | None = None):
    """(all visual columns, [absolute columns per observation], page source) or None."""
    bb = engine._backbone
    cols = getattr(engine, "_rpath_visual_cols", None)
    pages, src = observation_pages(engine)
    if cols is None or not pages:
        return None
    if seed is None:
        seed = 42 + int(getattr(engine, "_rpath_case_index", 0) or 0)
    idx = partition([int(p) for p in pages], mode, seed)
    groups = [cols.index_select(0, torch.tensor(g, dtype=torch.long, device=cols.device))
              for g in idx]
    return cols, groups, src


@torch.no_grad()
def _next_token_probs(backbone, cache, prompt_ids, gen_ids, position_cursor, rows):
    """Teacher-force prompt+gen on a cache copy; softmax over the whole vocabulary at `rows`."""
    from memory.vcca_diag import _arm_pass
    out = _arm_pass(backbone, cache, prompt_ids, [list(gen_ids)], position_cursor)[0]
    logits = out["logits"]
    return {int(r): torch.softmax(logits[int(r)].double(), dim=-1) for r in rows}


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, max_new_tokens: int = 512) -> None:
    """Stage-A probe: how much does within-observation token density move the output,
    natively and under the observation-factorized readout?"""
    out_path = os.environ.get("VLMAS_OFH_DIAG", "").strip()
    if not out_path:
        return
    from memory import group_attn as _ga
    from memory.rmvr_diag import field_spans
    from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json

    bb = engine._backbone
    got = observation_groups(engine, "true")
    if got is None:
        print(f"[OFH] case {case_index}: SKIP (visual bookkeeping missing: "
              f"{observation_pages(engine)[1]})", flush=True)
        return
    cols, groups_abs, page_src = got
    pages, _ = observation_pages(engine)
    n_vis = int(cols.numel())
    idx_groups = partition(pages, "true", 42)

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
        print(f"[OFH] case {case_index}: SKIP (empty generation)", flush=True)
        return
    fields = field_spans(text, offsets, json_prefix) if offsets else {}
    rows = fields.get("answer", {}).get("idx") or list(range(min(len(gen_ids), 8)))

    # native visual mass per observation at the first probed row (for target choice)
    c = deepcopy(cache)
    try:
        with _ga.install_group_attention(bb, groups_abs, cols, "capture") as rec:
            _next_token_probs(bb, c, prompt_ids, gen_ids, position_cursor, rows[:1])
    finally:
        del c
    masses = _ga.summarize_masses(rec, int(prompt_ids.shape[1]) - 1 + int(rows[0]))
    m_g = masses.get("m_g") or [0.0] * len(groups_abs)
    n_targets = max(1, int(os.environ.get("VLMAS_OFH_DIAG_TARGETS", "2") or 2))
    order = sorted(range(len(m_g)), key=lambda i: (-m_g[i], i))
    targets = [order[0]] + ([order[-1]] if n_targets > 1 and len(order) > 1 else [])

    def probs(mode, groups_idx, dup_cols=None, drop_cols=None):
        """mode: native | ofh ; groups_idx: index lists (into the CURRENT visual order);
        dup_cols: visual indices to append once more; drop_cols: visual indices to remove
        from the cache entirely (the density change has to be real for the native arm,
        which runs without any hook)."""
        c = deepcopy(cache)
        try:
            if drop_cols is not None and len(drop_cols):
                drop_abs = {int(cols[j]) for j in drop_cols}
                keep = torch.tensor([i for i in range(int(bb._kv_len(cache))) if i not in drop_abs],
                                    dtype=torch.long)
                for layer in c.layers:
                    if layer.keys is None:
                        continue
                    k = keep.to(layer.keys.device)
                    layer.keys = layer.keys.index_select(2, k)
                    layer.values = layer.values.index_select(2, k)
                remap = {int(o): i for i, o in enumerate(keep.tolist())}
                kept_vis = [int(cols[j]) for j in range(int(cols.numel())) if int(cols[j]) not in drop_abs]
                old_to_new_vis = {int(v): i for i, v in enumerate(kept_vis)}
                cols_now = torch.tensor([remap[v] for v in kept_vis], dtype=torch.long, device=cols.device)
                groups_idx = [[old_to_new_vis[int(cols[j])] for j in g if int(cols[j]) not in drop_abs]
                              for g in groups_idx]
                gabs = [torch.tensor([int(cols_now[j]) for j in g], dtype=torch.long, device=cols.device)
                        for g in groups_idx if g]
                ctx = (_ga.install_group_attention(bb, gabs, cols_now, "c2")
                       if mode == "ofh" else contextlib.nullcontext())
                with ctx:
                    return _next_token_probs(bb, c, prompt_ids, gen_ids, position_cursor, rows)
            if dup_cols is not None and len(dup_cols):
                add = cols.index_select(0, torch.tensor(dup_cols, dtype=torch.long, device=cols.device))
                for layer in c.layers:
                    if layer.keys is None:
                        continue
                    i = add.to(layer.keys.device)
                    layer.keys = torch.cat([layer.keys, layer.keys.index_select(2, i)], dim=2)
                    layer.values = torch.cat([layer.values, layer.values.index_select(2, i)], dim=2)
                base_len = int(bb._kv_len(cache))
                all_cols = torch.cat([cols, torch.arange(base_len, base_len + len(dup_cols), device=cols.device)])
            else:
                all_cols = cols
            gabs = []
            for g in groups_idx:
                gabs.append(torch.tensor([int(all_cols[j]) for j in g], dtype=torch.long, device=cols.device))
            ctx = (_ga.install_group_attention(bb, gabs, all_cols, "c2")
                   if mode == "ofh" else contextlib.nullcontext())
            with ctx:
                return _next_token_probs(bb, c, prompt_ids, gen_ids, position_cursor, rows)
        finally:
            del c

    base = {m: probs(m, idx_groups) for m in ("native", "ofh")}
    recs = []
    for t in targets:
        dup_cols, dup_groups = duplicate_plan(idx_groups, t, n_vis)
        sub_dropped, sub_groups = subsample_plan(idx_groups, t)
        for mode in ("native", "ofh"):
            p_dup = probs(mode, dup_groups, dup_cols=dup_cols)
            p_sub = probs(mode, idx_groups, drop_cols=sub_dropped)
            recs.append({
                "target": int(t), "mode": mode,
                "n_tok": len(idx_groups[t]), "m_g": round(float(m_g[t]), 6),
                "js_dup": [round(js_divergence(base[mode][r], p_dup[r]), 8) for r in rows],
                "js_sub": [round(js_divergence(base[mode][r], p_sub[r]), 8) for r in rows],
                "n_dropped": len(sub_dropped),
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else float("nan"))
    rec_out = {
        "case": int(case_index), "pages": pages, "page_src": page_src, "n_vis": n_vis,
        "rows": [int(r) for r in rows], "gen_text": text[:300],
        "m_g": [round(float(v), 6) for v in m_g], "targets": [int(t) for t in targets],
        "probes": recs,
        "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
    }
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec_out) + "\n")
    parts = []
    for r in recs:
        parts.append(f"{r['mode']}[o{r['target']} dup={med(r['js_dup']):.2e} sub={med(r['js_sub']):.2e}]")
    print(f"[OFH] case {case_index} rows={len(rows)} obs={len(idx_groups)} src={page_src} " + " ".join(parts)
          + f" attn={rec_out['attn']}", flush=True)


__all__ = ["run", "observation_groups", "partition", "crop_spans", "duplicate_plan",
           "subsample_plan", "js_divergence"]
