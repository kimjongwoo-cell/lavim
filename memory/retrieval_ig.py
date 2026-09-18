"""Kill test: latent-query score s_g  vs  causal usefulness I_g per cross-scale group.

VLMAS_RETR=igsweep  (Answerer stage; uses the visual bookkeeping that
                     retrieval_group uses; NO cut on the normal decode)
For each case, after the normal Answerer decode, on the pre-Answerer cache copy:
  S      per-token latent-query score (retrieval_group.score_tokens: q^l from the
         last Reasoner latent step, cached keys de-rotated to pre-RoPE, z-scored
         per layer, mean over layers)
  groups 5x root + its 20x children (retrieval_group.groups_from_tree)
  logp   teacher-forced log p(candidate) for every answer candidate under
           full     : no cut
           g<root>  : that group's visual columns masked out of the Answerer
                      softmax (visual_cut 'mask', stage A)
           novis    : all visual columns masked (reference floor)
  I_g = m_full - m_{remove g} is computed post hoc from the gold answer
        (m = log p(gold) - max_{y != gold} log p(y)).
VLMAS_RETR_OUT=<jsonl>  one row per case: S (raw, per visual token), pages /
                         parents / groups / token starts, per-condition log-probs.
Off path: nothing here runs unless VLMAS_RETR=igsweep.
"""
from __future__ import annotations

import contextlib
import json
import os
from copy import deepcopy

import torch


@contextlib.contextmanager
def _force_mask(backbone, cols):
    """Temporarily force VLMAS_VCUT=mask / STAGE=A so install_visual_cut applies
    to this scoring block only (the env is restored afterwards)."""
    from memory import visual_cut as _vcut
    keys = ("VLMAS_VCUT", "VLMAS_VCUT_STAGE", "VLMAS_VCUT_LAYERS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ["VLMAS_VCUT"] = "mask"
    os.environ["VLMAS_VCUT_STAGE"] = "A"
    os.environ.pop("VLMAS_VCUT_LAYERS", None)
    try:
        with _vcut.install_visual_cut(backbone, cols, stage="A") as st:
            yield st
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        candidates, case_index: int) -> None:
    from memory import retrieval_group as _rg
    from vision_text_mas.latent_terminal import score_terminal_continuation

    bb = engine._backbone
    cols_all = getattr(engine, "_rpath_visual_cols", None)
    rc = getattr(bb, "_route_vis_cols", None)
    pos = getattr(bb, "_route_vis_pos", None)
    pages = getattr(bb, "_route_pages", None)
    parents = tuple(getattr(bb, "_prune_image_parent_indices", ()) or ())
    hid = getattr(bb, "_retr_hidden", None)
    cands = [str(c).strip() for c in (candidates or ()) if str(c).strip()]
    ok = (cols_all is not None and rc is not None and pos is not None and pages
          and int(rc.numel()) == int(cols_all.numel()) and cands and hid is not None)
    if not ok:
        print(f"[RetrIG] case {case_index}: SKIP (cols={cols_all is not None} rc={rc is not None} "
              f"pos={pos is not None} pages={bool(pages)} hid={hid is not None} cands={len(cands)})", flush=True)
        return
    cols_all = cols_all.to(bb.device)
    rc = rc.to(bb.device)
    pos = pos.to(bb.device)
    S = _rg.score_tokens(bb, cache, rc, pos, hid)
    par, roots, groups, crop_of, starts = _rg.groups_from_tree(pages, parents)

    def tokens_of(group):
        idx = []
        for p in group:
            idx += list(range(starts[p], starts[p] + int(pages[p])))
        return idx

    conds = [("full", None)]
    for r in roots:
        conds.append((f"g{r}", torch.tensor(tokens_of(groups[r]), dtype=torch.long, device=bb.device)))
    conds.append(("novis", torch.arange(int(cols_all.numel()), device=bb.device)))

    def score_all(ctx_cols):
        rows = {}
        cm = contextlib.nullcontext() if ctx_cols is None else _force_mask(bb, cols_all.index_select(0, ctx_cols))
        with cm:
            for y in cands:
                c = deepcopy(cache)
                try:
                    tot, n = score_terminal_continuation(
                        backbone=bb, cache=c, position_cursor=position_cursor,
                        system_prompt=system_prompt, user_prompt=user_prompt,
                        json_prefix=json_prefix, continuation=f'"{y}"')
                    rows[y] = round(float(tot), 4)
                finally:
                    del c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows

    aq_only = os.environ.get("VLMAS_RETR_AQ_ONLY", "").strip() == "1"
    if aq_only:
        conds = []
    # Duplication test (VLMAS_RETR_DUP=1): append an exact copy of one group's
    # visual K/V columns (post-RoPE keys copied as-is -> identical logits, so
    # the group is doubled in cardinality with no content change) and re-score
    # every candidate. VLMAS_RETR_DUP_ONLY=1 keeps only full + dup<root>.
    dup = os.environ.get("VLMAS_RETR_DUP", "").strip() == "1"
    if dup and os.environ.get("VLMAS_RETR_DUP_ONLY", "").strip() == "1":
        conds = [("full", None)]
    # Duplication test v2 (VLMAS_RETR_DUP2=1): per candidate, full(capture) /
    # dup g (native, capture) / mfix_dup g (m_V fixed to the full forward's
    # value, visual-internal renormalisation) / c2_full / c2_dup g (LME group
    # competition, native m_V preserved). Masses (m_V, m_g, rho_g) at the last
    # prompt row are logged per condition (candidate-independent row).
    dup2 = os.environ.get("VLMAS_RETR_DUP2", "").strip() == "1"
    if dup2:
        conds = []
    vsim = os.environ.get("VLMAS_RETR_VSIM", "").strip() == "1"
    if vsim and os.environ.get("VLMAS_RETR_VSIM_ONLY", "").strip() == "1":
        conds = []
    # C2 structural diagnosis (VLMAS_RETR_RVU=1, memory/rvu_ablation.py):
    # mass-fixed single-crop removal, repeated vs unique morphology.
    rvu = os.environ.get("VLMAS_RETR_RVU", "").strip() == "1"
    if rvu:
        conds = []

    def _dup_cache(abs_cols):
        c = deepcopy(cache)
        for layer in c.layers:
            if layer.keys is None:
                continue
            idx = abs_cols.to(layer.keys.device)
            layer.keys = torch.cat([layer.keys, layer.keys.index_select(2, idx)], dim=2)
            layer.values = torch.cat([layer.values, layer.values.index_select(2, idx)], dim=2)
        return c

    def score_all_dup(abs_cols):
        rows = {}
        for y in cands:
            c = _dup_cache(abs_cols)
            try:
                tot, n = score_terminal_continuation(
                    backbone=bb, cache=c, position_cursor=position_cursor,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    json_prefix=json_prefix, continuation=f'"{y}"')
                rows[y] = round(float(tot), 4)
            finally:
                del c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows
    result = {
        "case": int(case_index), "n_vis": int(cols_all.numel()),
        "pages": [int(x) for x in pages], "parents": [int(x) for x in par],
        "roots": [int(r) for r in roots],
        "groups": {str(r): [int(p) for p in groups[r]] for r in roots},
        "starts": [int(x) for x in starts],
        "S": [round(float(x), 4) for x in S.tolist()],
        "cands": cands, "logp": {},
    }
    for name, ctx in conds:
        result["logp"][name] = score_all(ctx)
    if rvu:
        from memory import rvu_ablation as _rvu
        _rvu.run(engine, cache=cache, position_cursor=position_cursor,
                 system_prompt=system_prompt, user_prompt=user_prompt, json_prefix=json_prefix,
                 cands=cands, cols_all=cols_all, pages=pages, starts=starts,
                 case_index=case_index, result=result)
    if dup2:
        from memory import group_attn as _ga
        from vision_text_mas.latent_terminal import _assistant_prompt
        prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                                   user_prompt=user_prompt, json_prefix=json_prefix)
        n_prompt = int(bb.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])
        L0 = int(bb._kv_len(cache))
        gabs = [cols_all.index_select(0, torch.tensor(tokens_of(groups[r]), dtype=torch.long, device=bb.device)) for r in roots]
        names = ["full", "c2_full"] + [f"{k}{r}" for r in roots for k in ("dup", "mfix_dup", "c2_dup")]
        lp = {nm: {} for nm in names}
        masses = {}

        def _score(c, y, ctx):
            with ctx as rr:
                tot, _n = score_terminal_continuation(
                    backbone=bb, cache=c, position_cursor=position_cursor,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    json_prefix=json_prefix, continuation=f'"{y}"')
            return round(float(tot), 4), rr

        for yi, y in enumerate(cands):
            c = deepcopy(cache)
            try:
                lp["full"][y], r0 = _score(c, y, _ga.install_group_attention(bb, gabs, cols_all, "capture"))
            finally:
                del c
            ref = r0["mv"]
            if yi == 0:
                masses["full"] = _ga.summarize_masses(r0, n_prompt - 1)
            c = deepcopy(cache)
            try:
                lp["c2_full"][y], r1 = _score(c, y, _ga.install_group_attention(bb, gabs, cols_all, "c2"))
            finally:
                del c
            if yi == 0:
                masses["c2_full"] = _ga.summarize_masses(r1, n_prompt - 1)
            for gi, r in enumerate(roots):
                ng = int(gabs[gi].numel())
                app = torch.arange(L0, L0 + ng, device=bb.device)
                g_dup = [g if k != gi else torch.cat([g, app]) for k, g in enumerate(gabs)]
                vis_dup = torch.cat([cols_all, app])
                for tag, mode in (("dup", "capture"), ("mfix_dup", "massfix"), ("c2_dup", "c2")):
                    c = _dup_cache(gabs[gi])
                    try:
                        lp[f"{tag}{r}"][y], rr = _score(
                            c, y, _ga.install_group_attention(bb, g_dup, vis_dup, mode, mv_ref=ref))
                    finally:
                        del c
                    if yi == 0:
                        masses[f"{tag}{r}"] = _ga.summarize_masses(rr, n_prompt - 1)
                        masses[f"{tag}{r}"]["calls"] = rr["calls"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result["logp"].update(lp)
        result["masses"] = masses
        result["n_prompt"] = n_prompt
        mf = masses.get("full", {}); print(
            f"[RetrIG:DUP2] case {case_index} m_V full={mf.get('m_V')} rho_g={mf.get('rho_g')} "
            f"dup m_V={[masses.get(f'dup{r}', {}).get('m_V') for r in roots]} "
            f"mfix m_V={[masses.get(f'mfix_dup{r}', {}).get('m_V') for r in roots]} "
            f"c2 m_V={masses.get('c2_full', {}).get('m_V')} conds={len(names)} cands={len(cands)}", flush=True)
    if dup:
        for r in roots:
            abs_cols = cols_all.index_select(0, torch.tensor(tokens_of(groups[r]), dtype=torch.long, device=bb.device))
            result["logp"][f"dup{r}"] = score_all_dup(abs_cols)
        # identity check: duplicating zero columns must reproduce "full" exactly
        result["logp"]["dup_none"] = score_all_dup(torch.zeros(0, dtype=torch.long, device=bb.device))
        f0 = result["logp"].get("full", {}); dn = result["logp"]["dup_none"]
        mx = max((abs(f0[y] - dn[y]) for y in f0 if y in dn), default=float("nan"))
        print(f"[RetrIG:DUP] case {case_index} dup_none-vs-full max|dlogp|={mx:.4f} "
              f"dup_groups={[f'dup{r}' for r in roots]} n_dup_cols={[len(tokens_of(groups[r])) for r in roots]}", flush=True)
    if vsim:
        from memory import vsim_dump as _vs
        _vs.run(engine, cache=cache, case_index=case_index)
    if os.environ.get("VLMAS_RETR_AQ", "").strip() == "1":
        # Answerer-side group scores (C2 kill test): one teacher-forced forward
        # of prompt + first candidate on a cache copy, hooks read the logits.
        from memory import answerer_group_probe as _agp
        from vision_text_mas.latent_terminal import _assistant_prompt
        prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                                   user_prompt=user_prompt, json_prefix=json_prefix)
        n_prompt = int(bb.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])
        gcols = [cols_all.index_select(0, torch.tensor(tokens_of(groups[r]), dtype=torch.long, device=bb.device)) for r in roots]
        c = deepcopy(cache)
        try:
            with _agp.capture_group_scores(bb, gcols, cols_all, n_prompt) as rec:
                score_terminal_continuation(
                    backbone=bb, cache=c, position_cursor=position_cursor,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    json_prefix=json_prefix, continuation=f'"{cands[0]}"')
        finally:
            del c
        result["aq"] = _agp.summarize(rec)
        result["aq"]["n_prompt"] = n_prompt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        la = result["aq"].get("last", {})
        print(f"[RetrIG:AQ] case {case_index} calls={rec['calls']} T={rec['T']} n_prompt={n_prompt} "
              f"M_V(last)={la.get('M_V')} mass={la.get('mass')} lme={la.get('lme')} sizes={rec['sizes']}", flush=True)
    out = os.environ.get("VLMAS_RETR_OUT", "").strip()
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(result) + "\n")
    sg = {str(r): round(float(S[torch.tensor(tokens_of(groups[r]), device=S.device)].mean()), 3) for r in roots}
    print(f"[RetrIG] case {case_index} roots={roots} groups={result['groups']} s_g(mean z)={sg} "
          f"conds={[c for c, _ in conds]} cands={len(cands)}", flush=True)


def group_abs_cols(engine):
    """(cols_all, [abs cols per 5x-root group]) from the visual bookkeeping, or None."""
    from memory import retrieval_group as _rg
    bb = engine._backbone
    cols_all = getattr(engine, "_rpath_visual_cols", None)
    pages = getattr(bb, "_route_pages", None)
    parents = tuple(getattr(bb, "_prune_image_parent_indices", ()) or ())
    if cols_all is None or not pages or sum(int(x) for x in pages) != int(cols_all.numel()):
        return None
    par, roots, groups, crop_of, starts = _rg.groups_from_tree(pages, parents)
    out = []
    for r in roots:
        idx = []
        for p in groups[r]:
            idx += list(range(starts[p], starts[p] + int(pages[p])))
        out.append(cols_all.index_select(0, torch.tensor(idx, dtype=torch.long, device=cols_all.device)))
    return cols_all, out


__all__ = ["run", "group_abs_cols"]
