"""How much of the answer margin is the JSON prefix? (opt-in)

Nothing here runs unless VLMAS_GPFX=<jsonl>. Measurement only.

RMVR showed the Answerer's decision is dominated by the denominator: the margin g between
the emitted token and its best challenger is 3.6-22.8 nats while the whole visual push is
0.4-0.9. But the Answerer never decodes from a blank slate — `json_prefix` ('{"answer": ')
is written into the assistant turn for it, so generation starts INSIDE the answer value.
This asks how much of g that prefill alone is worth.

The token sequence is held fixed: the case is generated once, normally, and the SAME
generated ids are then teacher-forced under three prompt conditions that differ only in
what precedes them.

    full      assistant turn (system + question) + json_prefix   <- the real condition
    noprefix  assistant turn, json_prefix removed
    bare      json_prefix only, no system/question turn
    naked     neither: the latent handoff cache alone

For every answer-span row we record, against the token `a` the real run emitted:
    g      z(a) - max_{c != a} z(c)      the margin, same definition and units as RMVR
    keep   argmax z == a                 does this condition still produce that token
    lp     log p(a)
g_full - g_noprefix is the prefix's contribution; g_full - g_bare is the question's.

No candidates, no ground truth, no decode change; one extra teacher-forced pass per arm.

Env
  VLMAS_GPFX=<path.jsonl>     output file (append; one row per case). Unset = no-op.
  VLMAS_GPFX_MAXSTEPS=<int>   cap on probed answer rows (default 12)
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

import torch

ARMS = ("full", "noprefix", "bare", "naked")


def margins(logits, ids):
    """Per row: g, keep, lp against the token the real run emitted."""
    out = []
    z_all = torch.as_tensor(logits).to(torch.float64)
    for j, t in enumerate(ids):
        z = z_all[j]
        t = int(t)
        za = float(z[t])
        other = z.clone()
        other[t] = float("-inf")
        best = float(other.max())
        lp = float(torch.log_softmax(z, dim=-1)[t])
        out.append({"g": round(za - best, 4), "keep": bool(za >= best),
                    "lp": round(lp, 4), "argmax": int(z.argmax())})
    return out


def arm_prompt(processor, system_prompt, user_prompt, json_prefix, arm: str):
    """The four prompt conditions, built with the pipeline's own assembler."""
    from vision_text_mas.latent_terminal import _assistant_prompt

    if arm == "full":
        return _assistant_prompt(processor, system_prompt=system_prompt,
                                 user_prompt=user_prompt, json_prefix=json_prefix)
    if arm == "noprefix":
        return _assistant_prompt(processor, system_prompt=system_prompt,
                                 user_prompt=user_prompt, json_prefix=None)
    if arm == "bare":
        return _assistant_prompt(processor, system_prompt="", user_prompt="",
                                 json_prefix=json_prefix)
    if arm == "naked":
        return ""
    raise ValueError(arm)


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, max_new_tokens: int = 512) -> None:
    out_path = os.environ.get("VLMAS_GPFX", "").strip()
    if not out_path:
        return
    from memory.rmvr_diag import field_spans
    from memory.vcca_diag import _arm_pass
    from vision_text_mas.latent_terminal import generate_terminal_json

    bb = engine._backbone
    tok = bb.processor.tokenizer
    max_steps = int(os.environ.get("VLMAS_GPFX_MAXSTEPS", "12") or 12)

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
        print(f"[GPfx] case {case_index}: SKIP (empty generation)", flush=True)
        return
    fields = field_spans(text, offsets, json_prefix) if offsets else {}
    ans_rows = fields.get("answer", {}).get("idx") or []
    rows_idx = (ans_rows or list(range(len(gen_ids))))[:max_steps]

    rec = {"case": int(case_index), "gen_text": text, "rows": rows_idx,
           "answer_rows": ans_rows, "json_prefix": json_prefix,
           "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager") or "eager",
           "arms": {}}
    for arm in ARMS:
        prompt = arm_prompt(bb.processor, system_prompt, user_prompt, json_prefix, arm)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if int(ids.shape[1]) == 0:
            # "naked": _arm_pass needs at least one prompt row, so start the teacher
            # forcing one token earlier and drop that row afterwards.
            ids = torch.tensor([[gen_ids[0]]], dtype=torch.long)
            seq, shift = gen_ids[1:], 1
        else:
            seq, shift = gen_ids, 0
        ids = ids.to(bb.device)
        c = deepcopy(cache)
        try:
            out = _arm_pass(bb, c, ids, [seq], position_cursor)[0]
        finally:
            del c
        rows = [{"t": int(t), **m} for t, m in
                zip([r - shift for r in rows_idx if r - shift >= 0],
                    [margins(out["logits"], seq)[r - shift] for r in rows_idx
                     if 0 <= r - shift < len(seq)])]
        rec["arms"][arm] = {"n_prompt": int(ids.shape[1]), "rows": rows,
                            "g_med": (sorted(x["g"] for x in rows)[len(rows) // 2]
                                      if rows else None),
                            "keep": sum(1 for x in rows if x["keep"]), "n": len(rows)}
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    line = " ".join(f"{a}:g={rec['arms'][a]['g_med']} keep={rec['arms'][a]['keep']}/"
                    f"{rec['arms'][a]['n']}" for a in ARMS)
    print(f"[GPfx] case {case_index} rows={len(rows_idx)} {line}", flush=True)


__all__ = ["margins", "arm_prompt", "run", "ARMS"]
