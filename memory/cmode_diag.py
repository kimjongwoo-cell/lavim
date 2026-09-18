"""C2 diagnosis: cross-dataset common-mode / residual decomposition (opt-in).

Notion Experiment Log: "C2 구조 진단: Cross-Dataset Common-Mode / Residual Decomposition
(5셋×20)". Nothing here runs unless VLMAS_CMODE=<out.jsonl> is set.

Per case, on the pre-Answerer cache copy (same entry point as the igsweep kill test):
  decision row = the position where the tokenized candidate continuations '"<cand>"'
                 FIRST diverge (the fixed JSON prefix and any shared answer prefix sit
                 before it), so datasets with different candidate sets are compared at
                 the same functional position;
  arms         = full (native) and zeroA (memory/visual_cut 'zero', stage A: the
                 Answerer keeps its native attention mass over the visual columns but
                 the visual value term is dropped);
  captured     = the pre-final-norm residual at that row (h, the state the native final
                 norm + unembedding consume), the post-norm hidden, and the row logits
                 restricted to the candidate tokens, for both arms.
Everything else (Stage A common-mode concentration, Stage B leave-one-dataset-out
transfer, Stage C counterfactual removal, PC2 control) is offline on the JSONL.

Env
  VLMAS_CMODE=<path.jsonl>   output file (append; one row per case). Unset = no-op.
"""
from __future__ import annotations

import contextlib
import json
import os
from copy import deepcopy

import torch


def divergence_position(candidate_ids):
    """First index where the candidate token id lists differ; None if they never do."""
    seqs = [list(s) for s in candidate_ids]
    if len(seqs) < 2:
        return None
    limit = min(len(s) for s in seqs)
    for k in range(limit):
        first = seqs[0][k]
        if any(s[k] != first for s in seqs[1:]):
            return k
    if all(len(s) == len(seqs[0]) for s in seqs):
        return None                      # identical candidates: no decision row
    return limit                         # one candidate ends where another continues


def decision_row_tokens(candidate_ids):
    """(k, [token id of each candidate at k, or None if that candidate ended])."""
    k = divergence_position(candidate_ids)
    if k is None:
        return None, []
    return k, [(list(s)[k] if len(s) > k else None) for s in candidate_ids]


def row_record(case_index, cands, candidate_ids, hidden_full, hidden_zeroa,
               logits_full, logits_zeroa, extra=None):
    k, toks = decision_row_tokens(candidate_ids)
    seqs = [list(s) for s in candidate_ids]
    rec = {
        "case": int(case_index),
        "cands": list(cands),
        "k": k,
        "tok": toks,
        "cand_ids": [s[:8] for s in seqs],
        "n_distinct": len({t for t in toks if t is not None}),
        "h_full": [float(v) for v in hidden_full],
        "h_zeroa": [float(v) for v in hidden_zeroa],
        "d": [float(a) - float(b) for a, b in zip(hidden_full, hidden_zeroa)],
        "logit_full": [float(v) for v in logits_full],
        "logit_zeroa": [float(v) for v in logits_zeroa],
    }
    rec.update(dict(extra or {}))
    return rec


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
def _capture_prenorm(backbone, store):
    """Capture the input of the model's final norm (the pre-unembedding residual)."""
    norm = getattr(getattr(backbone, "lm", None), "norm", None)
    if norm is None:
        yield False
        return

    def hook(_module, args, _kwargs=None):
        hidden = args[0] if args else None
        if hidden is not None:
            store["h"] = hidden[0, -1].detach().float().cpu()

    handle = norm.register_forward_pre_hook(hook, with_kwargs=False)
    try:
        yield True
    finally:
        handle.remove()


@torch.no_grad()
def _forward_last_row(backbone, cache, input_ids, position_cursor):
    """One forward of `input_ids` on `cache` (mutates it); returns (pre-norm h, post-norm h, logits)."""
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
    with _capture_prenorm(backbone, store):
        out = backbone.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=cache, use_cache=True, output_hidden_states=True)
    post = out.hidden_states[-1][0, -1].detach().float().cpu()
    logits = out.logits[0, -1].detach().float().cpu()
    return store.get("h"), post, logits


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        candidates, case_index: int) -> None:
    out_path = os.environ.get("VLMAS_CMODE", "").strip()
    if not out_path:
        return
    from vision_text_mas.latent_terminal import _assistant_prompt

    bb = engine._backbone
    cands = [str(c).strip() for c in (candidates or ()) if str(c).strip()]
    cols = getattr(engine, "_rpath_visual_cols", None)
    if len(cands) < 2 or cols is None or int(cols.numel()) == 0:
        print(f"[CMode] case {case_index}: SKIP (cands={len(cands)} "
              f"cols={0 if cols is None else int(cols.numel())})", flush=True)
        return
    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    cand_ids = [list(tok(f'"{y}"', add_special_tokens=False)["input_ids"]) for y in cands]
    k, toks = decision_row_tokens(cand_ids)
    if k is None:
        print(f"[CMode] case {case_index}: SKIP (candidates never diverge)", flush=True)
        return
    ids = prompt_ids
    if k > 0:
        shared = torch.tensor([cand_ids[0][:k]], dtype=prompt_ids.dtype, device=bb.device)
        ids = torch.cat([prompt_ids, shared], dim=1)
    arms = {}
    for arm in ("full", "zeroa"):
        c = deepcopy(cache)
        ctx = _zero_visual(bb, cols.to(bb.device)) if arm == "zeroa" else contextlib.nullcontext()
        try:
            with ctx:
                arms[arm] = _forward_last_row(bb, c, ids, position_cursor)
        finally:
            del c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pre_f, post_f, log_f = arms["full"]
    pre_z, post_z, log_z = arms["zeroa"]
    if pre_f is None or pre_z is None:
        print(f"[CMode] case {case_index}: SKIP (final-norm hook found no module)", flush=True)
        return
    keep = [t for t in toks if t is not None]
    rec = row_record(
        case_index, cands, cand_ids,
        pre_f.tolist(), pre_z.tolist(),
        [float(log_f[t]) for t in keep], [float(log_z[t]) for t in keep],
        extra={
            "post_full": [round(float(v), 5) for v in post_f.tolist()],
            "post_zeroa": [round(float(v), 5) for v in post_z.tolist()],
            "tok_kept": keep,
            "n_prompt": int(prompt_ids.shape[1]),
            "n_ctx": int(ids.shape[1]),
            "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
        })
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    dn = float(torch.tensor(rec["d"]).norm())
    print(f"[CMode] case {case_index} k={k} cands={len(cands)} |d|={dn:.4f} "
          f"|h_full|={float(pre_f.norm()):.4f} |h_zeroa|={float(pre_z.norm()):.4f} "
          f"logit_full={[round(v, 3) for v in rec['logit_full'][:6]]} attn={rec['attn']}", flush=True)
