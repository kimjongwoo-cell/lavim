"""C2 diagnosis: Visual-to-Candidate Contrast Alignment (opt-in).

Notion Experiment Log: "C2 구조 진단: Visual-to-Candidate Contrast Alignment (5셋×20)".
Nothing here runs unless VLMAS_VCCA=<out.jsonl> is set.

The predecessor diagnosis (cross-dataset common-mode) looked at ONE position, the first
divergence, and lost most of GTEx/TCGA to gold-token collisions. This one builds the
candidate token trie and analyses EVERY branching node, plus a sequence-level control
that does not depend on any single-token definition.

Per case, on the pre-Answerer cache copy (same entry point as the igsweep kill test):
  arms      = full (native) and zeroA (memory/visual_cut 'zero', stage A: the Answerer
              keeps its native attention mass over the visual columns but the visual
              value term is dropped);
  one forward per candidate per arm over prompt + that candidate's tokens, which yields
  every prefix row of that candidate at once, so it serves the node analysis (Stage B/C)
  and the teacher-forced sequence score (Stage D) from the same pass;
  per branching node n with next-token set T_n (|T_n| = k >= 2):
    U_n      = orthonormal basis of the CENTERED unembedding rows {w_t - mean : t in T_n}
               — the exact directions that can change candidate-relative logits;
    rho_pre  = ||U_n^T d_pre||^2 / ||d_pre||^2,  d_pre  = h_full_pre  - h_zeroA_pre;
    rho_post = ||U_n^T d_post||^2 / ||d_post||^2 (post final norm);
    Delta z(t) = w_t^T d_post,  R_disc = ||Delta z - mean||^2 / ||Delta z||^2;
    Control A = the same rho for the FULL state itself (is the visual delta specifically
                contrast-poor, or is the whole Answerer state?);
    Control D = candidate logits recomputed from the captured post-norm state vs the
                native forward logits (gate; if it fails the row is not interpretable).
  Stage D: Delta S(c) = log p_full(c) - log p_zeroA(c), R_seq on the centered vector.

All ratios are computed here in float64 (the raw 2560-d states are far too large to log
for every node); the JSONL keeps the scalars, the per-node Delta z, and the raw states of
the first node only, which is what the offline gates in §9 need.

Env
  VLMAS_VCCA=<path.jsonl>   output file (append; one row per case). Unset = no-op.
"""
from __future__ import annotations

import contextlib
import json
import os
from copy import deepcopy

import torch

EPS = float(torch.finfo(torch.float64).tiny)


# --------------------------------------------------------------------------- pure


def branching_nodes(candidate_ids):
    """Trie nodes where the candidates propose >= 2 DIFFERENT next tokens.

    A node is a prefix shared by some candidates. Candidates that END at the prefix
    contribute no next token (they have no unembedding row to contrast) and are listed
    under "ended" instead. Nodes come back ordered by increasing prefix length.
    """
    seqs = [tuple(int(t) for t in s) for s in candidate_ids]
    by_prefix: dict = {}
    for i, s in enumerate(seqs):
        for length in range(len(s) + 1):
            entry = by_prefix.setdefault(s[:length], {"members": [], "next": {}, "ended": []})
            entry["members"].append(i)
            if length < len(s):
                entry["next"][i] = s[length]
            else:
                entry["ended"].append(i)
    out = []
    for prefix in sorted(by_prefix, key=lambda p: (len(p), p)):
        entry = by_prefix[prefix]
        distinct = sorted(set(entry["next"].values()))
        if len(distinct) >= 2:
            out.append({"prefix": prefix, "members": entry["members"],
                        "next": entry["next"], "distinct": distinct,
                        "ended": entry["ended"]})
    return out


def contrast_basis(rows):
    """Orthonormal basis (D, r) of the span of the CENTERED candidate unembedding rows.

    Centering is what makes this the *contrast* subspace: a direction that moves every
    candidate logit by the same amount cannot change which candidate wins, so it is
    projected out. Rank is decided by the standard numpy/LAPACK rule
    (sigma > sigma_max * max(shape) * eps), not a tuned threshold.
    """
    c = rows.to(torch.float64)
    c = c - c.mean(dim=0, keepdim=True)
    _u, s, vh = torch.linalg.svd(c, full_matrices=False)
    if s.numel() == 0:
        return torch.zeros(c.shape[1], 0, dtype=torch.float64)
    tol = float(s.max()) * max(c.shape) * torch.finfo(torch.float64).eps
    rank = int((s > tol).sum())
    return vh[:rank].transpose(0, 1).contiguous()


def accessible_fraction(basis, delta):
    """Fraction of ||delta||^2 that lies inside the contrast subspace."""
    d = delta.to(torch.float64).reshape(-1)
    den = float(d @ d)
    if basis.shape[1] == 0:
        return 0.0
    proj = basis.to(torch.float64).transpose(0, 1) @ d
    return float(proj @ proj) / (den + EPS)


def disc_ratio(delta_z):
    """Share of the candidate logit shift that actually separates the candidates."""
    z = torch.as_tensor(list(delta_z), dtype=torch.float64)
    if z.numel() == 0:
        return 0.0
    centered = z - z.mean()
    return float(centered @ centered) / (float(z @ z) + EPS)


def seq_ratio(delta_s):
    """Stage D counterpart of disc_ratio, on whole-candidate log-likelihood shifts."""
    return disc_ratio(delta_s)


# ------------------------------------------------------------------------ runtime


def _unembedding(backbone):
    """The matrix whose rows are token readout directions (this model ties embeddings)."""
    lm = getattr(backbone, "lm", None)
    head = getattr(lm, "lm_head", None) or getattr(getattr(backbone, "model", None), "lm_head", None)
    if head is not None and getattr(head, "weight", None) is not None:
        return head.weight
    model = getattr(backbone, "model", None)
    lang = getattr(model, "language_model", None)
    emb = getattr(lang, "embed_tokens", None) if lang is not None else None
    if emb is None and lm is not None:
        emb = getattr(lm, "embed_tokens", None)
    return None if emb is None else emb.weight


@contextlib.contextmanager
def _zero_visual(backbone, cols):
    """visual_cut 'zero' at the Answerer stage for this block only (env restored)."""
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


@contextlib.contextmanager
def _capture_prenorm_rows(backbone, store):
    """Capture ALL rows of the final norm's input (the pre-unembedding residual)."""
    norm = getattr(getattr(backbone, "lm", None), "norm", None)
    if norm is None:
        yield False
        return

    def hook(_module, args, _kwargs=None):
        hidden = args[0] if args else None
        if hidden is not None:
            store["h"] = hidden[0].detach().float().cpu()

    handle = norm.register_forward_pre_hook(hook, with_kwargs=False)
    try:
        yield True
    finally:
        handle.remove()


@torch.no_grad()
def _forward_rows(backbone, cache, input_ids, position_cursor):
    """One forward of `input_ids` on `cache` (mutates it).

    Returns (pre-norm rows, post-norm rows, logits rows), each [seq, *].
    """
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
        if backbone.mrope_pos:
            backbone.vlm.rope_deltas = torch.tensor(
                [[position_cursor - past_len]], dtype=torch.long, device=backbone.device)
    store: dict = {}
    with _capture_prenorm_rows(backbone, store):
        out = backbone.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=cache, use_cache=True, output_hidden_states=True)
    post = out.hidden_states[-1][0].detach().float().cpu()
    logits = out.logits[0].detach().float().cpu()
    return store.get("h"), post, logits


@torch.no_grad()
def _arm_pass(backbone, cache, prompt_ids, cand_ids, position_cursor):
    """Per candidate: the prefix rows that predict its tokens, and its token log-probs.

    Row j of `pre`/`post` is the state that predicts candidate token j, i.e. the trie
    node whose prefix is the candidate's first j tokens.
    """
    n_prompt = int(prompt_ids.shape[1])
    per_cand = []
    for ids_list in cand_ids:
        seq = torch.tensor([list(ids_list)], dtype=prompt_ids.dtype, device=backbone.device)
        ids = torch.cat([prompt_ids, seq], dim=1)
        c = deepcopy(cache)
        try:
            pre, post, logits = _forward_rows(backbone, c, ids, position_cursor)
        finally:
            del c
        lo, hi = n_prompt - 1, n_prompt - 1 + len(ids_list)
        logprob = torch.log_softmax(logits[lo:hi].to(torch.float64), dim=-1)
        token_lp = [float(logprob[j, int(t)]) for j, t in enumerate(ids_list)]
        per_cand.append({
            "pre": None if pre is None else pre[lo:hi].to(torch.float64),
            "post": post[lo:hi].to(torch.float64),
            "logits": logits[lo:hi].to(torch.float64),
            "logp": float(sum(token_lp)),
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_cand


def _node_record(node, full, zeroa, unemb):
    """Everything §5–§8 asks for at one branching node."""
    member = node["members"][0]
    row = len(node["prefix"])
    toks = node["distinct"]
    rows_w = unemb[toks].to(torch.float64)
    basis = contrast_basis(rows_w)

    pre_f = None if full[member]["pre"] is None else full[member]["pre"][row]
    pre_z = None if zeroa[member]["pre"] is None else zeroa[member]["pre"][row]
    post_f = full[member]["post"][row]
    post_z = zeroa[member]["post"][row]
    d_post = post_f - post_z

    dz = [float(rows_w[i] @ d_post) for i in range(len(toks))]
    native = [float(full[member]["logits"][row, int(t)]) for t in toks]
    recomputed = [float(rows_w[i] @ post_f) for i in range(len(toks))]
    gate_abs = max(abs(a - b) for a, b in zip(native, recomputed))
    scale = max(abs(v) for v in native) or 1.0

    rec = {
        "m": row,
        "prefix": [int(t) for t in node["prefix"][:8]],
        "k": len(toks),
        "toks": [int(t) for t in toks],
        "rank": int(basis.shape[1]),
        "n_members": len(node["members"]),
        "n_ended": len(node["ended"]),
        "rho_post": accessible_fraction(basis, d_post),
        "rho_post_full": accessible_fraction(basis, post_f),
        "d_post_norm": float(d_post.norm()),
        "h_post_full_norm": float(post_f.norm()),
        "dz": [round(v, 6) for v in dz],
        "dz_bar": round(sum(dz) / len(dz), 6),
        "R_disc": disc_ratio(dz),
        "gate_abs": round(gate_abs, 6),
        "gate_rel": round(gate_abs / scale, 6),
    }
    if pre_f is not None and pre_z is not None:
        d_pre = pre_f - pre_z
        rec.update({
            "rho_pre": accessible_fraction(basis, d_pre),
            "rho_pre_full": accessible_fraction(basis, pre_f),
            "d_pre_norm": float(d_pre.norm()),
            "h_pre_full_norm": float(pre_f.norm()),
        })
    return rec


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        candidates, case_index: int) -> None:
    out_path = os.environ.get("VLMAS_VCCA", "").strip()
    if not out_path:
        return
    from vision_text_mas.latent_terminal import _assistant_prompt

    bb = engine._backbone
    cands = [str(c).strip() for c in (candidates or ()) if str(c).strip()]
    cols = getattr(engine, "_rpath_visual_cols", None)
    if len(cands) < 2 or cols is None or int(cols.numel()) == 0:
        print(f"[VCCA] case {case_index}: SKIP (cands={len(cands)} "
              f"cols={0 if cols is None else int(cols.numel())})", flush=True)
        return
    unemb = _unembedding(bb)
    if unemb is None:
        print(f"[VCCA] case {case_index}: SKIP (no unembedding matrix found)", flush=True)
        return
    unemb = unemb.detach().cpu()

    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    cand_ids = [list(tok(f'"{y}"', add_special_tokens=False)["input_ids"]) for y in cands]
    nodes = branching_nodes(cand_ids)
    if not nodes:
        print(f"[VCCA] case {case_index}: SKIP (no branching node)", flush=True)
        return

    arms = {}
    for arm in ("full", "zeroa"):
        ctx = _zero_visual(bb, cols.to(bb.device)) if arm == "zeroa" else contextlib.nullcontext()
        with ctx:
            arms[arm] = _arm_pass(bb, cache, prompt_ids, cand_ids, position_cursor)
    full, zeroa = arms["full"], arms["zeroa"]

    node_recs = [_node_record(n, full, zeroa, unemb) for n in nodes]
    logp_f = [c["logp"] for c in full]
    logp_z = [c["logp"] for c in zeroa]
    d_s = [a - b for a, b in zip(logp_f, logp_z)]

    rec = {
        "case": int(case_index),
        "cands": cands,
        "cand_ids": [s[:8] for s in cand_ids],
        "n_prompt": int(prompt_ids.shape[1]),
        "n_nodes": len(node_recs),
        "nodes": node_recs,
        "logp_full": [round(v, 6) for v in logp_f],
        "logp_zeroa": [round(v, 6) for v in logp_z],
        "dS": [round(v, 6) for v in d_s],
        "dS_bar": round(sum(d_s) / len(d_s), 6),
        "R_seq": seq_ratio(d_s),
        "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
    }
    # the first node's raw states, so the offline pass can redo the geometry from scratch
    first = nodes[0]
    member, row = first["members"][0], len(first["prefix"])
    rec["first_node_raw"] = {
        "m": row,
        "post_full": [round(float(v), 5) for v in full[member]["post"][row].tolist()],
        "post_zeroa": [round(float(v), 5) for v in zeroa[member]["post"][row].tolist()],
    }
    if full[member]["pre"] is not None:
        rec["first_node_raw"]["pre_full"] = [round(float(v), 5) for v in full[member]["pre"][row].tolist()]
        rec["first_node_raw"]["pre_zeroa"] = [round(float(v), 5) for v in zeroa[member]["pre"][row].tolist()]

    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    rp = [n.get("rho_post") for n in node_recs]
    rd = [n["R_disc"] for n in node_recs]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    print(f"[VCCA] case {case_index} cands={len(cands)} nodes={len(node_recs)} "
          f"rho_post_med={med(rp):.4f} R_disc_med={med(rd):.4f} R_seq={rec['R_seq']:.4f} "
          f"gate_max={max(n['gate_rel'] for n in node_recs):.4f} attn={rec['attn']}", flush=True)
