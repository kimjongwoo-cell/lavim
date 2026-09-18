"""Where is the answer decided? Layer-wise logit-lens readout at the Answerer (opt-in).

Nothing here runs unless VLMAS_LENS=<jsonl>. Measurement only: one teacher-forced pass,
no extra forward, no intervention, no candidate/GT signal.

For every probed answer row t and every decoder layer l we read the residual stream out
of layer l through the model's own final norm and (tied) unembedding:

    z_t^(l) = W_U * Norm_L( h_t^(l) )

and record, against the token the full model actually emits (a = argmax z_t^(L-1)):

    lock_l      argmax z_t^(l) == a                 (is the answer already decided here?)
    rank_l      rank of a in z_t^(l)
    g_l         z_t^(l)(a) - max_{c != a} z_t^(l)(c)   (the margin, in the same units as
                                                        RMVR's g, so the two are comparable)
    p_l         softmax(z_t^(l))(a)

The pre-registered readouts are the FIRST layer from which lock stays true to the end
("decision layer") and the layer at which g first reaches half of its final value.

Caveat this measurement cannot escape: a logit lens reads the residual with the FINAL
layer's norm/unembedding, so early layers are read out of distribution. Use lock/rank
(order statistics) as primary and treat g_l/p_l as descriptive.

Env
  VLMAS_LENS=<path.jsonl>     output file (append; one row per case). Unset = no-op.
  VLMAS_LENS_MAXSTEPS=<int>   cap on probed answer rows (default 8)
  VLMAS_LENS_TOPK=<int>       how many challengers to keep per layer (default 3)
"""
from __future__ import annotations

import contextlib
import json
import os
from copy import deepcopy

import torch


@contextlib.contextmanager
def capture_layer_residuals(backbone, store: dict):
    """Record every decoder layer's output residual for every row of the forward."""
    handles = []

    def make(li):
        def hook(module, args, kwargs, output):
            h = output[0] if isinstance(output, tuple) else output
            store[li] = h.detach()[0].float().cpu()
            return output
        return hook

    for li, layer in enumerate(backbone.lm.layers):
        handles.append(layer.register_forward_hook(make(li), with_kwargs=True))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def lens_readout(h_row, norm, unemb, answer_id: int, topk: int = 3):
    """One layer, one row -> (lock, rank, g, p, top challengers)."""
    w = norm.weight.detach().float().cpu()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    h = h_row.double()
    y = (w.double() * h / torch.sqrt((h * h).mean() + eps))
    z = (unemb.double() @ y)
    a = int(z.argmax())
    za = float(z[answer_id])
    masked = z.clone()
    masked[answer_id] = float("-inf")
    best_other = float(masked.max())
    rank = int((z > za).sum()) + 1
    p = float(torch.softmax(z, dim=-1)[answer_id])
    tv, ti = torch.topk(z, k=min(int(topk), int(z.numel())))
    return {
        "lock": bool(a == answer_id),
        "argmax": a,
        "rank": rank,
        "g": round(za - best_other, 4),
        "p": round(p, 6),
        "top": [int(v) for v in ti.tolist()],
    }


def decision_layer(locks) -> int | None:
    """First layer from which `lock` is true all the way to the end."""
    first = None
    for i in range(len(locks) - 1, -1, -1):
        if locks[i]:
            first = i
        else:
            break
    return first


def half_margin_layer(gs) -> int | None:
    """First layer whose margin reaches half of the final margin (final > 0 only)."""
    if not gs or gs[-1] <= 0:
        return None
    half = gs[-1] / 2.0
    for i, g in enumerate(gs):
        if g >= half:
            return i
    return None


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        case_index: int, max_new_tokens: int = 512) -> None:
    out_path = os.environ.get("VLMAS_LENS", "").strip()
    if not out_path:
        return
    from memory.rmvr_diag import field_spans
    from memory.vcca_diag import _arm_pass, _unembedding
    from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json

    bb = engine._backbone
    unemb = _unembedding(bb)
    norm = getattr(getattr(bb, "lm", None), "norm", None)
    if unemb is None or norm is None:
        print(f"[Lens] case {case_index}: SKIP (no unembedding/final norm)", flush=True)
        return
    max_steps = int(os.environ.get("VLMAS_LENS_MAXSTEPS", "8") or 8)
    topk = int(os.environ.get("VLMAS_LENS_TOPK", "3") or 3)

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
        print(f"[Lens] case {case_index}: SKIP (empty generation)", flush=True)
        return
    fields = field_spans(text, offsets, json_prefix) if offsets else {}
    ans_rows = fields.get("answer", {}).get("idx") or []
    rows_idx = (ans_rows or list(range(len(gen_ids))))[:max_steps]

    store: dict = {}
    c = deepcopy(cache)
    try:
        with capture_layer_residuals(bb, store):
            out = _arm_pass(bb, c, prompt_ids, [gen_ids], position_cursor)[0]
    finally:
        del c
    if not store:
        print(f"[Lens] case {case_index}: SKIP (residual hook did not fire)", flush=True)
        return
    logits = out["logits"]                      # [len(gen), V] final readout
    n_layers = len(bb.lm.layers)
    T = int(store[0].shape[0])
    # _arm_pass hands back the row that PREDICTS generated token j (n_prompt-1+j), so the
    # residual to read for token t sits one row earlier than the token's own row.
    first = T - len(gen_ids) - 1
    W = unemb.detach().float().cpu()

    rec = {"case": int(case_index), "n_layers": n_layers, "gen_text": text,
           "rows": rows_idx, "answer_rows": ans_rows, "steps": [],
           "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager") or "eager",
           "arm": os.environ.get("VLMAS_KV_ROUTE_MODE", "").strip() or "native"}
    for t in rows_idx:
        fr = first + int(t)
        if fr < 0 or fr >= T:
            continue
        a = int(torch.as_tensor(logits[int(t)]).argmax())
        per = [lens_readout(store[li][fr], norm, W, a, topk) for li in range(n_layers)]
        locks = [p["lock"] for p in per]
        gs = [p["g"] for p in per]
        rec["steps"].append({
            "t": int(t), "y": int(gen_ids[t]), "a": a,
            "tok": tok.decode([a]),
            "decision_layer": decision_layer(locks),
            "half_margin_layer": half_margin_layer(gs),
            "first_lock": (locks.index(True) if any(locks) else None),
            "n_lock": int(sum(locks)),
            "g_final": gs[-1], "layers": per,
        })
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dl = [s["decision_layer"] for s in rec["steps"] if s["decision_layer"] is not None]
    print(f"[Lens] case {case_index} steps={len(rec['steps'])} L={n_layers} "
          f"decision_layer_med={sorted(dl)[len(dl) // 2] if dl else None}", flush=True)


__all__ = ["capture_layer_residuals", "lens_readout", "decision_layer",
           "half_margin_layer", "run"]
