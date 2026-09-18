"""C2 structural diagnosis: mass-fixed repeated-vs-unique morphology ablation (opt-in).

VLMAS_RETR=igsweep VLMAS_RETR_RVU=1   (runs inside memory/retrieval_ig.run on the
pre-Answerer cache copy; nothing here runs otherwise)

Unit g = one Navigator crop = its visual block in the persistent KV.
  e_g  = Norm(mean_{j in g} E_j), E = frozen vision-encoder output after the
         merger (bb._retr_img_feat, same order as the visual columns)
  R_g  = max_{h != g, m(h) = m(g), no physical overlap} cos(e_g, e_h)
         overlap = |box_g n box_h| / min(|box_g|, |box_h|) > VLMAS_RVU_OVERLAP
         (default 0: any shared level-0 area excludes the pair)
  per magnification with >= 3 comparable crops (crops whose R_g is defined):
         rep = argmax R_g, uniq = argmin R_g (ties: mean cos to the comparable
         same-magnification crops, then lower index), rand = Random(seed) draw
         among the remaining comparable crops of that magnification.
  R_g never sees the Answerer output, the gold answer or any utility.
Conditions (teacher-forced log p of every candidate; Answerer rows; all layers):
  full      native forward, capture hook: per layer/head/row m_V reference and
            per-crop native mass m_g at the first-answer-token row
  fix_none  dropfix with nothing dropped: the recompute-path reference of the
            drop arms (every case unless VLMAS_RVU_IDENTITY_CASES limits it;
            the float32 recompute differs from the native bf16 forward)
  zeroA     visual value term removed at the Answerer (memory/visual_cut 'zero',
            stage A) -- the reference of the GTEx common direction mu_V
  drop<p>   crop p masked out of the softmax, remaining visual renormalised to
            the full row's m_V and non-visual to 1 - m_V (every crop p)
The rep/uniq/rand labels only select among the per-crop drop conditions.
"""
from __future__ import annotations

import contextlib
import os
import random
from copy import deepcopy

import torch
import torch.nn.functional as F

_N_CASES = 0


def crop_embeddings(feat: torch.Tensor, pages, starts) -> torch.Tensor:
    """[P, D] L2-normalised crop means of the encoder embeddings."""
    rows = [feat[starts[p]:starts[p] + int(pages[p])].float().mean(0) for p in range(len(pages))]
    return F.normalize(torch.stack(rows), dim=-1)


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


def repetition_scores(emb: torch.Tensor, mags, boxes, thr: float = 0.0) -> dict:
    """Same-magnification, non-overlapping nearest-neighbour cosine per crop."""
    P = int(emb.shape[0])
    sim = (emb @ emb.T).tolist()
    ov = [[overlap_frac(boxes[i], boxes[j]) if i != j else 1.0 for j in range(P)] for i in range(P)]
    valid = [[i != j and int(mags[i]) == int(mags[j]) and ov[i][j] <= thr for j in range(P)]
             for i in range(P)]
    R, meanc = [], []
    for i in range(P):
        nb = [sim[i][j] for j in range(P) if valid[i][j]]
        R.append(max(nb) if nb else None)
        meanc.append(sum(nb) / len(nb) if nb else None)
    return {"sim": sim, "overlap": ov, "valid": valid, "R": R, "meanc": meanc}


def select_arms(R, meanc, mags, seed: int) -> dict:
    """rep / uniq / rand per magnification (keys are str(magnification))."""
    out = {}
    for m in sorted({int(x) for x in mags}):
        comp = [p for p in range(len(mags)) if int(mags[p]) == m and R[p] is not None]
        d = {"comparable": comp, "n_same_mag": sum(int(x) == m for x in mags),
             "primary": len(comp) >= 3, "rep": None, "uniq": None, "rand": None}
        if len(comp) >= 2:
            d["rep"] = max(comp, key=lambda p: (R[p], meanc[p], -p))
            d["uniq"] = min((p for p in comp if p != d["rep"]), key=lambda p: (R[p], meanc[p], p))
            rest = [p for p in comp if p not in (d["rep"], d["uniq"])]
            if rest:
                d["rand"] = random.Random(int(seed) * 100 + m).choice(rest)
        out[str(m)] = d
    return out


@torch.no_grad()
def crop_value_similarity(backbone, cache, cols, pages, starts) -> list:
    """Layer-mean cosine between crop-mean cached values (secondary R^V)."""
    P = len(pages)
    acc = torch.zeros(P, P)
    L = len(backbone.lm.layers)
    for li in range(L):
        v = cache.layers[li].values.index_select(2, cols)[0].float()          # [Hkv, n, Hd]
        n = v.shape[1]
        vt = v.permute(1, 0, 2).reshape(n, -1)
        m = F.normalize(torch.stack([vt[starts[p]:starts[p] + int(pages[p])].mean(0) for p in range(P)]), dim=-1)
        acc += (m @ m.T).cpu()
    return (acc / L).tolist()


@contextlib.contextmanager
def _force_zero_a(backbone, cols):
    """VLMAS_VCUT=zero / STAGE=A for this scoring block only (env restored)."""
    from memory import visual_cut as _vcut
    keys = ("VLMAS_VCUT", "VLMAS_VCUT_STAGE", "VLMAS_VCUT_LAYERS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ["VLMAS_VCUT"] = "zero"
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


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        cands, cols_all, pages, starts, case_index: int, result: dict) -> None:
    global _N_CASES
    from memory import group_attn as _ga
    from vision_text_mas.latent_terminal import _assistant_prompt, score_terminal_continuation

    bb = engine._backbone
    P = len(pages)
    mags = [int(x) for x in (getattr(bb, "_prune_image_magnifications", ()) or ())]
    boxes = list(getattr(bb, "_prune_image_boxes", ()) or ())
    feat = getattr(bb, "_retr_img_feat", None)
    if len(mags) != P or len(boxes) != P or feat is None or int(feat.shape[0]) != int(cols_all.numel()):
        print(f"[RVU] case {case_index}: SKIP (mags={len(mags)} boxes={len(boxes)} P={P} "
              f"feat={None if feat is None else tuple(feat.shape)} n_vis={int(cols_all.numel())})", flush=True)
        return
    thr = float(os.environ.get("VLMAS_RVU_OVERLAP", "0") or 0)
    n_ident = int(os.environ.get("VLMAS_RVU_IDENTITY_CASES", "1000000") or 1000000)
    do_ident = _N_CASES < n_ident
    _N_CASES += 1

    emb = crop_embeddings(feat, pages, starts)
    rep = repetition_scores(emb, mags, boxes, thr)
    sel = select_arms(rep["R"], rep["meanc"], mags, seed=42 + int(case_index))
    simV = crop_value_similarity(bb, cache, cols_all, pages, starts)
    repV = {"R": [], "meanc": []}
    for i in range(P):
        nb = [simV[i][j] for j in range(P) if rep["valid"][i][j]]
        repV["R"].append(max(nb) if nb else None)
        repV["meanc"].append(sum(nb) / len(nb) if nb else None)

    crop_cols = [cols_all[starts[p]:starts[p] + int(pages[p])] for p in range(P)]
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    n_prompt = int(bb.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])

    def _score(y, ctx):
        c = deepcopy(cache)
        try:
            with ctx as rr:
                tot, _n = score_terminal_continuation(
                    backbone=bb, cache=c, position_cursor=position_cursor,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    json_prefix=json_prefix, continuation=f'"{y}"')
        finally:
            del c
        return round(float(tot), 4), rr

    names = ["full", "zeroA"] + (["fix_none"] if do_ident else []) + [f"drop{p}" for p in range(P)]
    lp = {nm: {} for nm in names}
    diag = {"dev_mv": 0.0, "dev_row": 0.0, "fallback": 0, "calls_min": None, "ident_max": None}
    masses = {}
    for yi, y in enumerate(cands):
        lp["full"][y], r0 = _score(y, _ga.install_group_attention(bb, crop_cols, cols_all, "capture"))
        ref = r0["mv"]
        if yi == 0:
            masses["full"] = _ga.summarize_masses(r0, n_prompt - 1)
        lp["zeroA"][y], _ = _score(y, _force_zero_a(bb, cols_all))
        drops = ([("fix_none", None)] if do_ident else []) + [(f"drop{p}", crop_cols[p]) for p in range(P)]
        for nm, dcols in drops:
            lp[nm][y], rr = _score(y, _ga.install_group_attention(
                bb, crop_cols, cols_all, "dropfix", mv_ref=ref, drop_cols=dcols))
            diag["dev_mv"] = max(diag["dev_mv"], rr["dev_mv"])
            diag["dev_row"] = max(diag["dev_row"], rr["dev_row"])
            diag["fallback"] += rr["fallback"]
            diag["calls_min"] = rr["calls"] if diag["calls_min"] is None else min(diag["calls_min"], rr["calls"])
            if yi == 0:
                masses[nm] = _ga.summarize_post_masses(rr, n_prompt - 1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if do_ident:
        diag["ident_max"] = round(max(abs(lp["fix_none"][y] - lp["full"][y]) for y in cands), 4)
    diag["dev_mv"] = float(f"{diag['dev_mv']:.3e}")
    diag["dev_row"] = float(f"{diag['dev_row']:.3e}")

    result["logp"].update(lp)
    result["n_prompt"] = n_prompt
    result["rvu"] = {
        "mags": mags, "boxes": [list(b) if b is not None else None for b in boxes],
        "tokens": [int(x) for x in pages], "overlap_thr": thr,
        "simE": [[_r(x, 5) for x in row] for row in rep["sim"]],
        "simV": [[_r(x, 5) for x in row] for row in simV],
        "overlap": [[_r(x, 4) for x in row] for row in rep["overlap"]],
        "R_E": [_r(x, 5) for x in rep["R"]], "meanc_E": [_r(x, 5) for x in rep["meanc"]],
        "R_V": [_r(x, 5) for x in repV["R"]], "meanc_V": [_r(x, 5) for x in repV["meanc"]],
        "sel": sel, "masses": masses, "diag": diag, "identity": do_ident,
    }
    mf = masses.get("full", {})
    sel_s = {m: (d["rep"], d["uniq"], d["rand"], "P" if d["primary"] else "-") for m, d in sel.items()}
    print(f"[RVU] case {case_index} mags={mags} R_E={[_r(x, 3) for x in rep['R']]} sel(rep,uniq,rand)={sel_s} "
          f"m_V={mf.get('m_V')} m_g={mf.get('m_g')} dev_mv={diag['dev_mv']} dev_row={diag['dev_row']} "
          f"fallback={diag['fallback']} calls_min={diag['calls_min']} ident_max={diag['ident_max']} "
          f"conds={len(names)} cands={len(cands)} rvu=v2", flush=True)


__all__ = ["run", "crop_embeddings", "overlap_frac", "repetition_scores", "select_arms"]
