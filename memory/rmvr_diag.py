"""C2 diagnosis: Open/Closed Receiver-Margin Visual Reach (opt-in).

Notion Experiment Log: "C2 구조 진단: Open/Closed Receiver-Margin Visual Reach (5셋×20)".
Nothing here runs unless VLMAS_RMVR=<out.jsonl> is set.

The predecessor (Visual-to-Candidate Contrast Alignment) showed the visual causal delta
IS aligned with the candidate-contrast directions, yet the teacher-forced winner changed
in only 10/100 cases. The question here is whether the DIFFERENTIAL visual push clears
the competition margin the non-visual receiver has already built.

PRIMARY (§0 of the revised page, candidate-free, open/closed alike). The native answer
is decoded greedily from the pre-Answerer cache, then the SAME generated prefix is
teacher-forced into both arms:
  arms  = full (native) and zeroA (memory/visual_cut 'zero', stage A: the Answerer keeps
          its attention mass over the visual columns but the visual value term is dropped)
  per generated position t, over the WHOLE vocabulary V:
      a_t = argmax_w z0_t(w),   dz_t(w) = zF_t(w) - z0_t(w)
      g_t(c) = z0_t(a_t) - z0_t(c) >= 0,   v_t(c) = dz_t(c) - dz_t(a_t)
      R_t = max over {c != a_t, v_t(c) > 0} of v_t(c) / (g_t(c) + eps),  lam_t = 1/R_t
      flip_t = argmax zF_t != a_t          (exact reconstruction gate: flip <=> R_t > 1)
  Numerator and denominator are logged separately (v_max and the g at that token, plus
  the g of the best-reach challenger), because "R < 1 is common" alone is just a
  restatement of "the winner rarely changes".
  Positions are split into the answer-value span (primary) and the JSON formatting
  tokens (control) using the tokenizer offsets of the generated text.

CONTROL (§4–§9 of the page, retained as retrospective closed-set validation): the same
quantities restricted to the candidate token trie, plus the GT repair/break thresholds.
`eps` is the float64 tiny constant; it only stabilises exact ties and is not tuned.

Env
  VLMAS_RMVR=<path.jsonl>       output file (append; one row per case). Unset = no-op.
  VLMAS_RMVR_NO_CONTROL=1       skip the closed-set candidate control (primary only).
"""
from __future__ import annotations

import contextlib
import json
import os
import re

import torch

EPS = float(torch.finfo(torch.float64).tiny)


# --------------------------------------------------------------------------- pure


def node_reach(z0, zF):
    """Receiver margin / visual push / reach over one explicit token set (list form).

    z0, zF: native logits of the zeroA and full arms for the same tokens, same order.
    Returns the zeroA winner index, per-token dz/g/v/R, the reach R_n (max over the
    challengers the visual delta actually pushes), the analytic gain threshold
    lam_n = 1/R_n, the predicted flip (R_n > 1) and the observed flip.
    """
    z0 = [float(v) for v in z0]
    zF = [float(v) for v in zF]
    dz = [f - z for f, z in zip(zF, z0)]
    a = max(range(len(z0)), key=lambda i: (z0[i], -i))
    g = [z0[a] - z for z in z0]
    v = [d - dz[a] for d in dz]
    R = [(v[i] / (g[i] + EPS)) if i != a else None for i in range(len(z0))]
    pos = [i for i in range(len(z0)) if i != a and v[i] > 0]
    R_n = max((R[i] for i in pos), default=None)
    aF = max(range(len(zF)), key=lambda i: (zF[i], -i))
    return {
        "a": a, "aF": aF, "dz": dz, "g": g, "v": v, "R": R,
        "R_n": R_n, "lam_n": (1.0 / R_n) if R_n not in (None, 0.0) else None,
        "n_push": len(pos),
        "pred_flip": bool(R_n is not None and R_n > 1.0),
        "flip": bool(aF != a),
    }


@torch.no_grad()
def vocab_reach(z0, zF):
    """Same quantities over the whole vocabulary (tensor form), scalars only.

    Also returns the numerator/denominator split the revised page asks for: the largest
    positive visual push v_max with the receiver gap at that token, and the gap at the
    best-reach challenger.
    """
    z0 = z0.to(torch.float64).reshape(-1)
    zF = zF.to(torch.float64).reshape(-1)
    a = int(torch.argmax(z0))
    aF = int(torch.argmax(zF))
    dz = zF - z0
    g = z0[a] - z0
    v = dz - dz[a]
    mask = v > 0
    mask[a] = False
    out = {
        "a": a, "aF": aF, "flip": bool(aF != a),
        "n_push": int(mask.sum()),
        "dz_a": float(dz[a]), "z0_a": float(z0[a]),
        "R": None, "lam": None, "c_star": None, "v_star": None, "g_star": None,
        "v_max": None, "g_at_v_max": None, "c_v_max": None,
        "pred_flip": False,
    }
    if not bool(mask.any()):
        return out
    ratio = torch.where(mask, v / (g + EPS), torch.full_like(v, float("-inf")))
    c_star = int(torch.argmax(ratio))
    v_masked = torch.where(mask, v, torch.full_like(v, float("-inf")))
    c_vmax = int(torch.argmax(v_masked))
    R = float(ratio[c_star])
    out.update({
        "R": R, "lam": (1.0 / R) if R != 0.0 else None,
        "c_star": c_star, "v_star": float(v[c_star]), "g_star": float(g[c_star]),
        "v_max": float(v[c_vmax]), "g_at_v_max": float(g[c_vmax]), "c_v_max": c_vmax,
        "pred_flip": bool(R > 1.0),
    })
    return out


def gt_thresholds(z0, zF, gold):
    """§9 retrospective thresholds over one token set, from the same native logits.

    gold = index of the gold candidate's token (None if the gold is not a member here).
    lam_repair: smallest visual gain that lets the gold catch the zeroA winner.
    lam_break : smallest gain at which some non-gold token overtakes a gold winner.
    """
    if gold is None:
        return {"lam_repair": None, "lam_break": None, "gold_is_winner": None, "v_gold": None}
    r = node_reach(z0, zF)
    a, dz = r["a"], r["dz"]
    if gold != a:
        v_y = dz[gold] - dz[a]
        return {"lam_repair": (r["g"][gold] / v_y) if v_y > 0 else None,
                "lam_break": None, "gold_is_winner": False, "v_gold": v_y}
    worse = [(z0[gold] - z0[c]) / (dz[c] - dz[gold])
             for c in range(len(z0)) if c != gold and dz[c] > dz[gold]]
    return {"lam_repair": None, "lam_break": (min(worse) if worse else None),
            "gold_is_winner": True, "v_gold": 0.0}


def summarize(reaches):
    """§0/§7 summary over a list of per-step (or per-node) reach dicts."""
    vals = [r["R"] if "R" in r else r["R_n"] for r in reaches]
    vals = [v for v in vals if v is not None]
    n = len(reaches)
    med = None
    if vals:
        s = sorted(vals)
        med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    reach_hits = sum(1 for v in vals if v >= 1.0)
    return {
        "n": n, "n_push": len(vals),
        "R_max": (max(vals) if vals else None), "R_med": med,
        "F_reach": (reach_hits / n) if n else None,
        "flips": sum(1 for r in reaches if r.get("flip")),
        "pred_flips": sum(1 for r in reaches if r.get("pred_flip")),
        "gate_ok": all(bool(r.get("pred_flip")) == bool(r.get("flip")) for r in reaches),
    }


def _close_quote(text, start):
    """Index of the first unescaped double quote at or after `start`."""
    end = start
    while end < len(text):
        if text[end] == '"' and (end == 0 or text[end - 1] != "\\"):
            break
        end += 1
    return end


def field_spans(text, offsets, json_prefix=None):
    """Token indices of every JSON string VALUE in a generated structured_json readout.

    offsets: the tokenizer's (start, end) char offsets of the generated tokens.
    json_prefix: the prompt's JSON prefix. With `structured_json` it already opens the
    answer string (e.g. '{"answer": "'), so the generation STARTS inside that value and
    the field name has to come from the prefix, not from the generated text.
    Returns {field: {"span": (start, end), "idx": [token indices]}}.
    """
    out = {}
    opened = re.search(r'"([A-Za-z_][\w ]*)"\s*:\s*("?)\s*$', json_prefix or "")
    if opened:
        # the prefix may already contain the value's opening quote, or the model emits it
        start = 0 if opened.group(2) == '"' else (1 if text[:1] == '"' else None)
        if start is not None:
            end = _close_quote(text, start)
            out[opened.group(1)] = {
                "span": (start, end),
                "idx": [i for i, (s, e) in enumerate(offsets) if e > start and s < end]}
    for m in re.finditer(r'"([A-Za-z_][\w ]*)"\s*:\s*"', text):
        start = m.end()
        end = start
        while end < len(text):
            if text[end] == '"' and text[end - 1] != "\\":
                break
            end += 1
        idx = [i for i, (s, e) in enumerate(offsets) if e > start and s < end]
        out.setdefault(m.group(1), {"span": (start, end), "idx": idx})
    return out


def answer_span(text, offsets):
    """(token indices, char span) of the answer value; ([], None) when there is none."""
    f = field_spans(text, offsets).get("answer")
    return (f["idx"], f["span"]) if f else ([], None)


# ------------------------------------------------------------------------ runtime


@contextlib.contextmanager
def _force_identity(backbone, cols):
    """visual_cut 'identity' for this block only: the same float32 attention recompute the
    zeroA arm goes through, with nothing cut. Env is restored afterwards."""
    from memory import visual_cut as _vcut
    keys = ("VLMAS_VCUT", "VLMAS_VCUT_STAGE", "VLMAS_VCUT_LAYERS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ["VLMAS_VCUT"] = "identity"
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
        candidates, case_index: int, normal_output=None, max_new_tokens: int = 512) -> None:
    out_path = os.environ.get("VLMAS_RMVR", "").strip()
    if not out_path:
        return
    from copy import deepcopy

    from memory.vcca_diag import _arm_pass, _zero_visual, branching_nodes
    from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json

    bb = engine._backbone
    cols = getattr(engine, "_rpath_visual_cols", None)
    if cols is None or int(cols.numel()) == 0:
        print(f"[RMVR] case {case_index}: SKIP (no visual columns)", flush=True)
        return
    cols = cols.to(bb.device)
    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)

    # ---- native generation (greedy, deterministic) from the pre-Answerer cache
    gen_cache = deepcopy(cache)
    try:
        text = generate_terminal_json(
            backbone=bb, cache=gen_cache, position_cursor=position_cursor,
            system_prompt=system_prompt, user_prompt=user_prompt, json_prefix=json_prefix,
            max_new_tokens=int(max_new_tokens), temperature=0.0, top_p=1.0, do_sample=False)
    finally:
        del gen_cache
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    gen_ids = list(enc["input_ids"])
    offsets = list(enc.get("offset_mapping") or [])
    if not gen_ids:
        print(f"[RMVR] case {case_index}: SKIP (empty native generation)", flush=True)
        return
    fields = field_spans(text, offsets, json_prefix) if offsets else {}
    span = fields.get("answer", {}).get("span")
    span_set = set(fields.get("answer", {}).get("idx", []))
    label = {}
    for name, fld in fields.items():
        for i in fld["idx"]:
            label.setdefault(i, name)

    # ---- both arms over the same generated prefix
    arms = {}
    for arm in ("full", "zeroa"):
        ctx = _zero_visual(bb, cols) if arm == "zeroa" else contextlib.nullcontext()
        with ctx:
            arms[arm] = _arm_pass(bb, cache, prompt_ids, [gen_ids], position_cursor)
    lf, lz = arms["full"][0]["logits"], arms["zeroa"][0]["logits"]
    # identity control: the zeroA arm runs through visual_cut's float32 recompute while
    # the native full arm does not, so part of the measured push can be the path change.
    # VLMAS_RMVR_IDENT=1 adds the same recompute WITHOUT any cut, which gives both the
    # path-only pseudo-push (native vs identity) and a path-consistent visual push
    # (identity vs zeroA).
    li = None
    if os.environ.get("VLMAS_RMVR_IDENT", "").strip() == "1":
        with _force_identity(bb, cols):
            li = _arm_pass(bb, cache, prompt_ids, [gen_ids], position_cursor)[0]["logits"]

    steps = []
    for t in range(len(gen_ids)):
        r = vocab_reach(lz[t], lf[t])
        if li is not None:
            path = vocab_reach(lf[t], li[t])        # no visual change: pure recompute path
            clean = vocab_reach(lz[t], li[t])       # both arms recomputed
            r.update({
                "v_max_path": path["v_max"], "R_path": path["R"], "flip_path": path["flip"],
                "v_max_clean": clean["v_max"], "R_clean": clean["R"],
                "flip_clean": clean["flip"], "g_star_clean": clean["g_star"],
            })
        r = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
        r.update({"t": t, "y": int(gen_ids[t]), "field": label.get(t, "format"),
                  "in_answer": bool(t in span_set)})
        steps.append(r)
    prim = [s for s in steps if s["field"] == "answer"]
    fallback = not prim
    if fallback:
        prim = steps
    content = {name: summarize([s for s in steps if s["field"] == name])
               for name in sorted({s["field"] for s in steps} - {"answer", "format"})}
    fmt_steps = [s for s in steps if s["field"] == "format"]
    rec = {
        "case": int(case_index),
        "n_prompt": int(prompt_ids.shape[1]),
        "gen_text": text,
        "gen_ids": [int(t) for t in gen_ids],
        "offsets": [[int(s), int(e)] for s, e in offsets],
        "json_prefix": (None if json_prefix is None else str(json_prefix)[-40:]),
        "answer_span": span,
        "answer_steps": sorted(span_set),
        "native_output": (None if normal_output is None else str(normal_output)[:400]),
        "steps": steps,
        "primary": summarize(prim),
        "primary_is_fallback": bool(fallback),
        "content": content,
        "formatting": summarize(fmt_steps) if fmt_steps else None,
        "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
    }

    # ---- closed-set control (candidate trie + GT thresholds), when choices exist
    cands = [str(c).strip() for c in (candidates or ()) if str(c).strip()]
    if len(cands) >= 2 and os.environ.get("VLMAS_RMVR_NO_CONTROL", "").strip() != "1":
        cand_ids = [list(tok(f'"{y}"', add_special_tokens=False)["input_ids"]) for y in cands]
        nodes = branching_nodes(cand_ids)
        if nodes:
            carms = {}
            for arm in ("full", "zeroa"):
                ctx = _zero_visual(bb, cols) if arm == "zeroa" else contextlib.nullcontext()
                with ctx:
                    carms[arm] = _arm_pass(bb, cache, prompt_ids, cand_ids, position_cursor)
            cfull, czeroa = carms["full"], carms["zeroa"]
            node_recs, reaches = [], []
            for node in nodes:
                member = node["members"][0]
                row = len(node["prefix"])
                toks = [int(t) for t in node["distinct"]]
                z0 = [float(czeroa[member]["logits"][row, t]) for t in toks]
                zF = [float(cfull[member]["logits"][row, t]) for t in toks]
                rr = node_reach(z0, zF)
                reaches.append(rr)
                node_recs.append({
                    "m": row, "k": len(toks), "toks": toks,
                    "z0": [round(v, 6) for v in z0], "zF": [round(v, 6) for v in zF],
                    "next": [[int(i), int(t)] for i, t in sorted(node["next"].items())],
                    "members": [int(i) for i in node["members"]],
                    "n_ended": len(node["ended"]),
                    "a": rr["a"], "aF": rr["aF"], "n_push": rr["n_push"],
                    "R_n": rr["R_n"], "lam_n": rr["lam_n"],
                    "pred_flip": rr["pred_flip"], "flip": rr["flip"],
                })
            rec["control"] = {
                "cands": cands,
                "cand_ids": [s[:8] for s in cand_ids],
                "nodes": node_recs,
                "logp_full": [round(c["logp"], 6) for c in cfull],
                "logp_zeroa": [round(c["logp"], 6) for c in czeroa],
                "summary": summarize([{**r, "R": r["R_n"]} for r in reaches]),
            }

    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    p = rec["primary"]
    f = lambda v: "-" if v is None else f"{v:.4f}"
    ctrl = rec.get("control", {}).get("summary")
    cont = " ".join(f"{name}[n={s['n']} R_med={f(s['R_med'])} push={s['n_push']} flips={s['flips']}]"
                    for name, s in rec["content"].items())
    print(f"[RMVR] case {case_index} gen_tok={len(gen_ids)} ans_steps={len(span_set)} "
          f"content={cont or '-'} "
          f"R_med={f(p['R_med'])} R_max={f(p['R_max'])} F_reach={f(p['F_reach'])} "
          f"push={p['n_push']}/{p['n']} flips={p['flips']}/{p['pred_flips']} gate_ok={p['gate_ok']} "
          f"ctrl_nodes={0 if not ctrl else ctrl['n']} ctrl_gate={'-' if not ctrl else ctrl['gate_ok']} "
          f"attn={rec['attn']}", flush=True)


__all__ = ["run", "node_reach", "vocab_reach", "gt_thresholds", "summarize", "answer_span"]
