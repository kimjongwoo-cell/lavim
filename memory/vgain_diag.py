"""C2 causal test: Answerer Visual-Path Gain Intervention — native addressing fixed (opt-in).

Notion Experiment Log: "[C2 causal test] Answerer Visual-Path Gain Intervention — native
addressing fixed" (3dad771b-c9e2-81e7-aaee-d38619006f85). Nothing here runs unless
VLMAS_VGAIN=<out.jsonl> is set.

Intervention (§2). At every Answerer attention call (all decoder layers, all heads, the
prompt prefill rows and every generated row) the native attention output
    o = o_N + o_V,   o_V = sum_{j in V} alpha_j V_j,   o_N = sum_{j notin V} alpha_j V_j
is replaced by
    o~ = gamma_N * o_N + gamma_V * o_V.
alpha is the native softmax(QK^T) of that very call; Q, K, RoPE, the visual column set and
its order are never touched (§3). Implemented as the float32 recompute forward hook of
memory/visual_cut.py with the value rows weighted, ctx = alpha @ (w * V), w_j = gamma_V on
visual columns and gamma_N elsewhere. gamma_V = 0, gamma_N = 1 is visual_cut 'zero'.
Rows after the first layer of course see the downstream consequence of the earlier layers'
gain (their Q/K are computed from the intervened residual stream); what is fixed is that
no intervention acts on Q/K/logits directly.

Arms, run on copies of the pre-Answerer cache (same entry point as the other C2 probes):
  ident        gamma_V = gamma_N = 1 through the recompute path (Control A reference; the
               bf16 native forward differs from the float32 recompute at ulp level, so
               every gamma arm is compared against ident, not against the native decode)
  V<g>         matched visual gain, g in grid, g != 1
  W<g>         Control C: visual V rows replaced by a donor slide's visual V rows (K kept
               native, so the addressing is the matched one), g in grid, g != 0  (W0 == V0)
  N<g>         Control D: gamma on the nonvisual term, g in grid, g != 1
Per arm:
  gen   greedy Answerer decode with exactly the native call's generation kwargs, parsed by
        the pipeline's own structured_json chain (_extract_json_object -> _extract_json_answer
        -> _extract_answer_line -> _normalize_choice), no terminal repair.
  cand  teacher-forced candidate continuations (lead + choice + tail, tail = '",' / '"}' as the
        native decode closed the value) after the prompt, in both quote tokenizations and,
        since the decode text is strip()ped, with and without a space before the quote
        (candidate_leads / candidate_variants; 09-14 fix: a joined-string BPE '"S', a bare
        closing '"' and a dropped space scored every candidate -22..-40 and put the wrong one
        on top): logits of every distinct next token at
        every branching node of the variant trie, each variant's summed log-prob (logp_var)
        and per candidate their logsumexp (logp). The prompt is prefilled once per arm and
        each variant is scored on top of it (cache cropped back between variants).
Gates logged per case:
  gate_split  split prompt/candidate pass vs one joint pass (ident, first candidate)
  gate_A      native (no hook) vs ident candidate logits, first candidate
  gate_B      visual_cut 'zero' vs V0 candidate logits, first candidate
  parse_ok    (offline) native_answer here == answer.answer in the case's result.json

Donor (Control C): the most recent earlier case of the same process whose slide differs;
the first case of a run has none (W arms skipped, logged). Donor rows are matched by visual
order; a visual-token count mismatch skips W for that case.

Env
  VLMAS_VGAIN=<path.jsonl>        output (append; one row per case). Unset = no-op.
  VLMAS_VGAIN_GRID=0,0.5,1,1.5,2,4
  VLMAS_VGAIN_ARMS=V,W,N          which families besides ident
  VLMAS_VGAIN_NO_GEN=1            skip the decodes (candidate logits only)
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import time
from copy import deepcopy

import torch

_DONORS: list[dict] = []   # [{"slide": str, "case": int, "v": [layer tensors, cpu]}], newest last


# --------------------------------------------------------------------------- pure


def parse_grid(text: str | None) -> tuple[float, ...]:
    raw = (text or "0,0.5,1,1.5,2,4").replace(" ", "")
    out: list[float] = []
    for tok in raw.split(","):
        if tok:
            val = float(tok)
            if val not in out:
                out.append(val)
    return tuple(out)


def fmt_gamma(g: float) -> str:
    return f"{g:g}"


def arm_plan(grid, families) -> list[dict]:
    """[{name, src, gv, gn}] in run order; ident first."""
    fams = [f.strip().upper() for f in families if f.strip()]
    plan = [{"name": "ident", "src": "ident", "gv": 1.0, "gn": 1.0}]
    for fam in ("V", "W", "N"):
        if fam not in fams:
            continue
        for g in grid:
            if fam in ("V", "N") and g == 1.0:
                continue          # identical to ident
            if fam == "W" and g == 0.0:
                continue          # identical to V0 (content removed either way)
            gv, gn = (1.0, g) if fam == "N" else (g, 1.0)
            plan.append({"name": f"{fam}{fmt_gamma(g)}", "src": fam, "gv": gv, "gn": gn})
    return plan


def value_weights(length: int, vis: torch.Tensor, gv: float, gn: float,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    w = torch.full((length,), float(gn), dtype=dtype, device=device)
    w[vis] = float(gv)
    return w


def pick_donor(bank: list[dict], slide: str | None):
    """Newest bank entry whose slide differs from `slide` (None when there is none)."""
    for entry in reversed(bank):
        if slide is None or entry["slide"] != slide:
            return entry
    return None


def push_donor(bank: list[dict], entry: dict, keep: int = 2) -> None:
    """Newest last; one entry per slide; at most `keep` slides."""
    bank[:] = [e for e in bank if e["slide"] != entry["slide"]]
    bank.append(entry)
    del bank[:-keep]


def candidate_lead(native_text: str | None) -> str:
    """Whitespace + opening quote the native decode put before the answer value."""
    m = re.match(r'\s*"', native_text or "")
    return m.group(0) if m else ' "'


def candidate_leads(native_text: str | None) -> list[str]:
    """Opening-quote forms to score. The terminal decode returns .strip()ped text
    (latent_terminal.py), so a bare leading '"' may have been ' "' in the decode: score both."""
    lead = candidate_lead(native_text)
    return [lead] if lead != '"' else ['"', ' "']


def candidate_tail(native_text: str | None) -> str:
    """Closing quote plus the delimiter the native decode emitted right after it ('",' / '"}')."""
    m = re.match(r'\s*"[^"]*("[,}]?)', native_text or "")
    return m.group(1) if m else '"'


def candidate_variants(tok, lead: str, choice: str, tail: str) -> list[list[int]]:
    """Token sequences a decode can emit for lead + choice + tail (unique, separate-quote form first).

    The prompt ends in '":', so the decode's first token is either the quote alone ('"', then
    'S','ple','en') or a quote merged into the value ('"S'); the text alone cannot tell which,
    and BPE of the joined string picks the merged one. Both are scored and summed per candidate.
    The tail ('",' / '"}') is taken from the native decode, not a bare '"'.
    """
    sep = (list(tok(lead, add_special_tokens=False)["input_ids"])
           + list(tok(f"{choice}{tail}", add_special_tokens=False)["input_ids"]))
    joint = list(tok(f"{lead}{choice}{tail}", add_special_tokens=False)["input_ids"])
    return [sep] if joint == sep else [sep, joint]


def logsumexp_by(values, groups, n: int) -> list[float | None]:
    """Per-group log(sum(exp(values))) for groups 0..n-1."""
    out: list[float | None] = []
    for g in range(n):
        vs = [float(v) for v, gg in zip(values, groups) if gg == g]
        out.append(round(float(torch.logsumexp(torch.tensor(vs, dtype=torch.float64), 0)), 6) if vs else None)
    return out


def parse_answer(text: str | None, choices) -> str:
    """The pipeline's structured_json answer chain (latent_answerer), without repair."""
    from vision_text_mas.latent_answerer import (
        STRUCTURED_JSON_TERMINAL_PREFIX,
        _extract_answer_line,
        _extract_json_answer,
        _extract_json_object,
        _normalize_choice,
    )
    raw = (text or "").strip()
    if not raw.lstrip().startswith("{"):
        raw = STRUCTURED_JSON_TERMINAL_PREFIX + raw
    parsed = _extract_json_object(raw).get("answer")
    ans = ((parsed.strip() if isinstance(parsed, str) else "")
           or _extract_json_answer(raw) or _extract_answer_line(raw))
    return _normalize_choice(ans, tuple(choices or ()))


# ------------------------------------------------------------------------ runtime


@contextlib.contextmanager
def install_gain(backbone, visual_cols, gv: float = 1.0, gn: float = 1.0, donor=None):
    """Weighted-value attention recompute on every LM layer (see module docstring).

    donor: per-layer visual value rows [1, H_kv, n_vis, D] (visual order) or None.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    layers = backbone.lm.layers
    cols = visual_cols.to(device=backbone.device, dtype=torch.long)
    donor_dev = None if donor is None else [d.to(backbone.device) for d in donor]
    stats = {"calls": 0, "rows": 0, "rho_sum": 0.0, "maxdiff": 0.0}
    handles = []

    def make_post(li):
        def post(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys = cache.layers[li].keys
            values = cache.layers[li].values
            L = int(keys.shape[2])
            T = int(hidden.shape[1])
            vis = cols[cols < L]
            if int(vis.numel()) == 0 or int(vis.numel()) >= L:
                return output
            Hd = module.head_dim
            G = module.num_key_value_groups
            cos, sin = pe
            q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            K = repeat_kv(keys, G).float()
            V = repeat_kv(values, G).float()
            if donor_dev is not None:
                V[:, :, vis, :] = repeat_kv(donor_dev[li][:, :, :int(vis.numel()), :], G).float()
            scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling
            if T > 1:
                past = L - T
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + torch.arange(T, device=keys.device))[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)
            del scores
            stats["rho_sum"] += float(alpha[:, :, -1, vis].sum(-1).mean())   # last row only
            if gv != 1.0 or gn != 1.0:
                V = V * value_weights(L, vis, gv, gn, device=V.device)[None, None, :, None]
            ctx = torch.matmul(alpha, V)
            flat = ctx.transpose(1, 2).reshape(1, T, -1)
            new = module.o_proj(flat.to(module.o_proj.weight.dtype))
            stats["calls"] += 1
            stats["rows"] += T
            if gv == 1.0 and gn == 1.0 and donor_dev is None:
                stats["maxdiff"] = max(stats["maxdiff"], float((new.float() - attn_out.float()).abs().max()))
            new = new.to(attn_out.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return post

    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()
        del donor_dev


@torch.no_grad()
def _forward_logits(backbone, cache, input_ids, position_cursor):
    """One forward of `input_ids` on `cache` (mutates it); logits [seq, vocab] on device."""
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
    out = backbone.model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=cache, use_cache=True)
    return out.logits[0]


@torch.no_grad()
def candidate_pass(backbone, cache, prompt_ids, cand_ids, position_cursor, nodes, only_first=False):
    """Prefill prompt[:-1] once, then score each candidate as [last prompt token] + tokens.

    Returns {"z": [[logit per distinct token] per node], "logp": [per candidate],
             "first_rows": float32 cpu logits rows of candidate 0}.
    """
    n = int(prompt_ids.shape[1])
    targets = cand_ids[:1] if only_first else cand_ids
    c = deepcopy(cache)
    try:
        if n > 1:
            _forward_logits(backbone, c, prompt_ids[:, :-1], position_cursor)
        base = backbone._kv_len(c)
        rows_by_cand, logp = [], []
        for ids in targets:
            seq = torch.cat([prompt_ids[:, -1:],
                             torch.tensor([list(ids)], dtype=prompt_ids.dtype, device=prompt_ids.device)], dim=1)
            logits = _forward_logits(backbone, c, seq, position_cursor + n - 1)[:len(ids)].float()
            lp = torch.log_softmax(logits, dim=-1)
            logp.append(float(sum(float(lp[j, int(t)]) for j, t in enumerate(ids))))
            rows_by_cand.append(logits)
            c.crop(base)
        z = []
        if not only_first:
            for node in nodes:
                member, row = node["members"][0], len(node["prefix"])
                z.append([round(float(rows_by_cand[member][row, int(t)]), 5) for t in node["distinct"]])
        first = rows_by_cand[0].detach().cpu()
        del rows_by_cand
    finally:
        del c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"z": z, "logp": [round(v, 6) for v in logp], "first_rows": first}


@torch.no_grad()
def joint_first(backbone, cache, prompt_ids, ids, position_cursor):
    """Candidate 0 scored in one joint forward (split-pass gate)."""
    n = int(prompt_ids.shape[1])
    seq = torch.cat([prompt_ids, torch.tensor([list(ids)], dtype=prompt_ids.dtype,
                                              device=prompt_ids.device)], dim=1)
    c = deepcopy(cache)
    try:
        logits = _forward_logits(backbone, c, seq, position_cursor)[n - 1:n - 1 + len(ids)].float().cpu()
    finally:
        del c
    return logits


def _maxabs(a, b):
    return round(float((a.float() - b.float()).abs().max()), 6)


@contextlib.contextmanager
def _visual_cut_zero(backbone, cols):
    from memory.vcca_diag import _zero_visual
    with _zero_visual(backbone, cols) as st:
        yield st


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        candidates, case_index: int, normal_output=None, gen_kwargs=None) -> None:
    out_path = os.environ.get("VLMAS_VGAIN", "").strip()
    if not out_path:
        return
    from memory.vcca_diag import branching_nodes
    from vision_text_mas.latent_terminal import _assistant_prompt, generate_terminal_json

    t0 = time.time()
    bb = engine._backbone
    cols = getattr(engine, "_rpath_visual_cols", None)
    slide = getattr(engine, "_rpath_slide_id", None)
    if cols is None or int(cols.numel()) == 0:
        print(f"[VGain] case {case_index}: SKIP (no visual columns)", flush=True)
        return
    cols = cols.to(bb.device)
    n_vis = int(cols.numel())
    choices = tuple(str(c).strip() for c in (candidates or ()) if str(c).strip())
    grid = parse_grid(os.environ.get("VLMAS_VGAIN_GRID"))
    plan = arm_plan(grid, os.environ.get("VLMAS_VGAIN_ARMS", "V,W,N").split(","))
    do_gen = os.environ.get("VLMAS_VGAIN_NO_GEN", "").strip() != "1" and gen_kwargs is not None

    # this case's visual V rows (visual order) -> donor bank after the arms ran
    mine = [layer.values[:, :, cols, :].detach().to("cpu", copy=True) for layer in cache.layers]
    donor = pick_donor(_DONORS, slide)
    donor_note = None
    if donor is None:
        donor_note = "no_donor"
    elif int(donor["v"][0].shape[2]) != n_vis:
        donor_note = f"n_vis_mismatch({int(donor['v'][0].shape[2])}!={n_vis})"

    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt,
                               user_prompt=user_prompt, json_prefix=json_prefix)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    lead = candidate_lead(normal_output)
    tail = candidate_tail(normal_output)
    leads = candidate_leads(normal_output)
    cand_ids, var_of = [], []          # token variants (see candidate_variants) -> candidate index
    for ci, y in enumerate(choices):
        for ld in leads:
            for ids in candidate_variants(tok, ld, y, tail):
                if ids not in cand_ids:
                    cand_ids.append(ids)
                    var_of.append(ci)
    nodes = branching_nodes(cand_ids) if len(cand_ids) >= 2 else []
    native_answer = parse_answer(normal_output, choices)
    lead_tok_ok = None
    if normal_output and native_answer:
        have = list(tok(str(normal_output), add_special_tokens=False)["input_ids"])
        want = [w for ld in leads for w in candidate_variants(tok, ld, native_answer, tail)]
        lead_tok_ok = any(have[:len(w)] == w for w in want)   # retokenized decode matches one variant

    rec = {
        "case": int(case_index), "slide": slide, "n_vis": n_vis, "n_prompt": int(prompt_ids.shape[1]),
        "cands": list(choices), "lead": lead, "leads": leads, "tail": tail, "lead_tok_ok": lead_tok_ok, "var_of": var_of,
        "nodes": [{"m": len(nd["prefix"]), "members": nd["members"][:4],
                   "toks": [int(t) for t in nd["distinct"]],
                   "tok_of_cand": {str(k): int(v) for k, v in nd["next"].items()}} for nd in nodes],
        "native_text": (None if normal_output is None else str(normal_output)[:600]),
        "native_answer": native_answer,
        "grid": list(grid), "donor": (None if donor is None else {"slide": donor["slide"], "case": donor["case"]}),
        "donor_note": donor_note, "arms": {},
        "attn": os.environ.get("VLMAS_ATTN_IMPLEMENTATION", "eager"),
    }

    # ---- native (no hook) candidate logits: Control A reference
    if cand_ids:
        nat = candidate_pass(bb, cache, prompt_ids, cand_ids, position_cursor, nodes)
        rec["arms"]["native"] = {"src": "native", "gv": 1.0, "gn": 1.0, "z": nat["z"],
                                 "logp": logsumexp_by(nat["logp"], var_of, len(choices)), "logp_var": nat["logp"],
                                 "answer": native_answer}
        nat_first = nat["first_rows"]

    for arm in plan:
        name, src = arm["name"], arm["src"]
        if src == "W" and donor_note is not None:
            rec["arms"][name] = {"src": src, "gv": arm["gv"], "gn": arm["gn"], "skip": donor_note}
            continue
        dv = donor["v"] if src == "W" else None
        ta = time.time()
        entry = {"src": src, "gv": arm["gv"], "gn": arm["gn"]}
        if do_gen:
            gc = deepcopy(cache)
            try:
                with install_gain(bb, cols, arm["gv"], arm["gn"], dv) as st:
                    text = generate_terminal_json(
                        backbone=bb, cache=gc, position_cursor=position_cursor,
                        system_prompt=system_prompt, user_prompt=user_prompt, **gen_kwargs)
            finally:
                del gc
            n_gen = len(tok(text, add_special_tokens=False)["input_ids"])
            entry.update({"text": text[:600], "n_gen": n_gen, "answer": parse_answer(text, choices),
                          "rho_v": round(st["rho_sum"] / max(st["calls"], 1), 6),
                          "calls": st["calls"]})
            if name == "ident":
                entry["maxdiff_attn"] = round(st["maxdiff"], 6)
        if cand_ids:
            with install_gain(bb, cols, arm["gv"], arm["gn"], dv):
                cp = candidate_pass(bb, cache, prompt_ids, cand_ids, position_cursor, nodes)
            entry.update({"z": cp["z"], "logp": logsumexp_by(cp["logp"], var_of, len(choices)), "logp_var": cp["logp"]})
            if name == "ident":
                rec["gate_A"] = _maxabs(cp["first_rows"], nat_first)
                rec["gate_A_argmax_same"] = bool(
                    (cp["first_rows"].argmax(-1) == nat_first.argmax(-1)).all())
                with install_gain(bb, cols, 1.0, 1.0, None):
                    jf = joint_first(bb, cache, prompt_ids, cand_ids[0], position_cursor)
                rec["gate_split"] = _maxabs(cp["first_rows"], jf)
                rec["gate_split_argmax_same"] = bool((cp["first_rows"].argmax(-1) == jf.argmax(-1)).all())
            if name == "V0":
                with _visual_cut_zero(bb, cols):
                    zc = candidate_pass(bb, cache, prompt_ids, cand_ids, position_cursor, nodes,
                                        only_first=True)
                rec["gate_B"] = _maxabs(cp["first_rows"], zc["first_rows"])
        entry["sec"] = round(time.time() - ta, 2)
        rec["arms"][name] = entry
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    push_donor(_DONORS, {"slide": slide, "case": int(case_index), "v": mine})
    rec["sec_total"] = round(time.time() - t0, 1)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    ans = {k: v.get("answer") for k, v in rec["arms"].items() if "answer" in v}
    changed = sum(1 for k, v in ans.items() if k not in ("native", "ident") and v != ans.get("ident"))
    print(f"[VGain] case {case_index} slide={slide} n_vis={n_vis} cands={len(choices)} nodes={len(nodes)} "
          f"native={native_answer!r} ident={ans.get('ident')!r} changed_vs_ident={changed}/"
          f"{sum(1 for k in ans if k not in ('native', 'ident'))} "
          f"gate_A={rec.get('gate_A')} gate_split={rec.get('gate_split')} gate_B={rec.get('gate_B')} "
          f"lead_tok_ok={lead_tok_ok} donor={donor_note or 'ok'} sec={rec['sec_total']} "
          f"attn={rec['attn']}", flush=True)


__all__ = ["run", "install_gain", "arm_plan", "parse_grid", "pick_donor", "push_donor",
           "parse_answer", "candidate_lead", "candidate_leads", "candidate_tail", "candidate_variants", "logsumexp_by", "value_weights"]
