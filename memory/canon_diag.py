"""Model-level diagnosis of Observation-Canonical Visual Addressing (opt-in, VLMAS_CANON_DIAG=<jsonl>).

Why: nav3 + canonical (+ re-read) dropped ExpertVQA 39.84 -> 32.81 and SlideBench 43.65 -> 38.58 with
identical Navigator patches; lost items are mostly one-step shifts on graded / count / location options.
This probe asks what canonical changes INSIDE the Answerer, on copies of the same pre-Answerer cache.

Address arms (keys of the visual columns re-rotated before the Answerer prompt; values never touched):
  native       as run
  canonical    route_address.apply_canonical (crop-local (h,w), t dropped, every crop superimposed at
               anchor b - c right before the answer boundary)
  canonical_t  same but crops on distinct t slots (not superimposed)
  shift        native 3-D geometry kept (global offsets, inter-crop layout); the whole visual block is
               translated by one scalar so its largest coordinate sits at b - 1 (pure "move closer")
  gap<G>       (opt-in, e.g. gap2048) every cached column up to and including the last visual column
               re-rotated by -G: for the Answerer rows the visual block and everything before it sit G
               positions farther away, the Reasoner tail / latent columns keep their distance. Exactly the
               geometry of VLMAS_POS_GAP=<G>, which computes it consistently from the Reasoner prefill on.
Per arm:
  * teacher-forced log-prob of every answer choice (memory.vgain_diag candidate machinery) -> argmax answer
  * at the decision row (last prompt token + the answer-value lead, i.e. the row that emits the first
    content token), for every decoder layer, from a float32 recompute of that row's attention:
      rho_v / rho_prompt / rho_lat / rho_rest  head-mean attention mass on visual cols, the Answerer prompt
                                                (+lead), the Reasoner latent cols, everything else
      crop_ent   normalized entropy of the visual mass over crops (1 = spread evenly)
      cell_corr  Spearman(token visual mass, h + w) inside crops — canonical puts cell (15,15) of every crop
                 nearest to the query, so a recency-driven read shows up as a positive correlation
      dla_v      direct logit attribution of the layer's VISUAL read to logit[A] - logit[B]
                 (final-RMSNorm linearized at the row), A/B = native answer vs canonical answer when they
                 differ, else native top-1 vs top-2, using the first differing token at the decision row
      dla_attn   same for the whole attention output of the layer
Nothing here runs unless VLMAS_CANON_DIAG is set. Requires visual bookkeeping (VLMAS_KV_ROUTE=1 with
VLMAS_KV_ROUTE_MODE=bookkeep keeps the pipeline decode native).

Env
  VLMAS_CANON_DIAG=<path.jsonl>
  VLMAS_CANON_DIAG_ARMS=native,canonical,canonical_t,shift
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from copy import deepcopy

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

ARMS = ("native", "canonical", "canonical_t", "shift")


# --------------------------------------------------------------------------- pure


def average_ranks(v: torch.Tensor) -> torch.Tensor:
    """Ranks with ties given their average rank (h + w has many ties)."""
    v = v.double().flatten()
    order = v.argsort()
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(v.numel(), dtype=v.dtype)
    uniq, inverse = torch.unique(v, return_inverse=True)
    sums = torch.zeros(uniq.numel(), dtype=v.dtype).index_add_(0, inverse, ranks)
    counts = torch.zeros(uniq.numel(), dtype=v.dtype).index_add_(0, inverse, torch.ones_like(v))
    return (sums / counts)[inverse]


def spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.double().flatten().cpu()
    y = y.double().flatten().cpu()
    if x.numel() < 3:
        return None
    rx = average_ranks(x); ry = average_ranks(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = float(rx.norm() * ry.norm())
    return None if den == 0.0 else float((rx * ry).sum()) / den


def normalized_entropy(masses: torch.Tensor) -> float | None:
    m = masses.double().clamp_min(0.0)
    tot = float(m.sum())
    if tot <= 0.0 or m.numel() < 2:
        return None
    p = m / tot
    h = float(-(p * p.clamp_min(1e-300).log()).sum())
    return h / math.log(m.numel())


def page_cells(pos: torch.Tensor, pages) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop id and (h + w) of every visual token from native MRoPE positions [3, n] and page sizes."""
    crop = torch.empty(pos.shape[-1], dtype=torch.long)
    hw = torch.empty(pos.shape[-1], dtype=torch.long)
    start = 0
    for k, size in enumerate(pages):
        page = pos[:, start:start + size].detach().cpu().long()
        u = page - page.min(dim=1, keepdim=True).values
        crop[start:start + size] = k
        hw[start:start + size] = u[1] + u[2]
        start += size
    return crop, hw


def shift_positions(pos: torch.Tensor, b_pos: int) -> torch.Tensor:
    """Translate every axis by one scalar so the largest coordinate lands on b - 1."""
    delta = int(b_pos) - 1 - int(pos.max())
    return pos + delta


def parse_gap_arm(arm: str):
    """'gap2048' -> 2048; anything else -> None."""
    if isinstance(arm, str) and arm.startswith("gap") and arm[3:].isdigit() and int(arm[3:]) > 0:
        return int(arm[3:])
    return None


def first_diff_token(a_ids, b_ids):
    """Index and tokens of the first position where two token lists differ."""
    for j, (x, y) in enumerate(zip(a_ids, b_ids)):
        if int(x) != int(y):
            return j, int(x), int(y)
    return None


# --------------------------------------------------------------------------- arms


@torch.no_grad()
def _rerotate(bb, cache, cols, pos_old, pos_new):
    from memory.restage import delta_cos_sin, rerotate_keys
    sample = cache.layers[0].keys
    cos_old, sin_old = bb.lm.rotary_emb(sample, pos_old.unsqueeze(1))
    cos_new, sin_new = bb.lm.rotary_emb(sample, pos_new.unsqueeze(1))
    cos_d, sin_d = delta_cos_sin(cos_old.float(), sin_old.float(), cos_new.float(), sin_new.float())
    for layer in cache.layers:
        block = layer.keys.index_select(2, cols).float()
        layer.keys.index_copy_(2, cols, rerotate_keys(block, cos_d, sin_d).to(layer.keys.dtype))


@contextlib.contextmanager
def _env(key, value):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@torch.no_grad()
def apply_arm(bb, cache, arm: str, b_pos: int) -> str:
    """Re-address the visual keys of `cache` in place; returns a short note."""
    if arm == "native":
        return "native"
    cols = bb._route_vis_cols.to(device=bb.device, dtype=torch.long)
    pos = bb._route_vis_pos.to(device=bb.device, dtype=torch.long)
    pages = list(bb._route_pages)
    if arm in ("canonical", "canonical_t"):
        from memory import route_address as ra
        with _env("VLMAS_KV_ROUTE_MODE", arm):
            n = ra.apply_canonical(bb, cache, cols, pos, pages, b_pos)
        return f"{arm} n={n}"
    if arm == "shift":
        new = shift_positions(pos, b_pos)
        _rerotate(bb, cache, cols, pos, new)
        return f"shift delta={int(new.max()) - int(pos.max())}"
    gap = parse_gap_arm(arm)
    if gap is not None:
        n = int(cols.max()) + 1
        idx = torch.arange(n, device=bb.device, dtype=torch.long)
        zero = torch.zeros(3, n, device=bb.device, dtype=torch.long)
        _rerotate(bb, cache, idx, zero, zero - gap)   # only new - old matters for the delta rotation
        return f"gap{gap} posthoc cols=0..{n - 1}"
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------- capture


@torch.no_grad()
def capture_decision_row(bb, cache, prompt_ids, lead_ids, position_cursor, groups, pair_tokens):
    """Prefill prompt[:-1], then forward [last prompt token + lead]; per-layer stats of the LAST row."""
    from memory.vgain_diag import _forward_logits
    layers = bb.lm.layers
    n = int(prompt_ids.shape[1])
    c = deepcopy(cache)
    store: dict = {"layers": {}}
    handles = []
    try:
        if n > 1:
            _forward_logits(bb, c, prompt_ids[:, :-1], position_cursor)
        prompt_start = int(groups["prompt_start"])

        def make_post(li):
            def post(module, args, kwargs, output):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                kv = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
                if hidden is None or pe is None or kv is None:
                    return output
                T = int(hidden.shape[1]); row = T - 1
                keys = kv.layers[li].keys; values = kv.layers[li].values
                L = int(keys.shape[2]); Hd = module.head_dim; G = module.num_key_value_groups
                cos, sin = pe
                cos_r = cos[..., row:row + 1, :] if cos.dim() >= 3 else cos
                sin_r = sin[..., row:row + 1, :] if sin.dim() >= 3 else sin
                q = module.q_norm(module.q_proj(hidden[:, row:row + 1, :]).view(1, 1, -1, Hd)).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos_r, sin_r)
                K = repeat_kv(keys, G).float(); V = repeat_kv(values, G).float()
                alpha = torch.softmax(torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling, dim=-1)  # [1,H,1,L]
                a = alpha[0, :, 0, :].mean(0)                                    # head-mean [L]
                vis = groups["vis"].to(a.device); lat = groups["lat"].to(a.device)
                rho_v = float(a[vis].sum()); rho_lat = float(a[lat].sum()) if lat.numel() else 0.0
                rho_p = float(a[prompt_start:].sum()); rho_rest = max(0.0, 1.0 - rho_v - rho_lat - rho_p)
                mv = a[vis].detach().cpu()
                crop_mass = torch.zeros(int(groups["crop"].max()) + 1, dtype=torch.float64).index_add_(0, groups["crop"], mv.double())
                vmask = torch.zeros(L, dtype=torch.bool, device=a.device); vmask[vis] = True
                oV = torch.matmul(alpha * vmask.float(), V).transpose(1, 2).reshape(1, 1, -1)
                oV = module.o_proj(oV.to(module.o_proj.weight.dtype))[0, 0].float()
                out = output[0] if isinstance(output, tuple) else output
                store["layers"][li] = {
                    "rho_v": rho_v, "rho_prompt": rho_p, "rho_lat": rho_lat, "rho_rest": rho_rest,
                    "crop_ent": normalized_entropy(crop_mass), "crop_max": float(crop_mass.max() / max(crop_mass.sum(), 1e-12)),
                    "cell_corr": spearman(mv, groups["hw"].double()),
                    "oV": oV.detach(), "oA": out[0, row].float().detach(),
                }
                return output
            return post

        def norm_pre(module, args, kwargs=None):
            x = args[0] if args else (kwargs or {}).get("hidden_states")
            if x is not None:
                store["final"] = x[0, -1].float().detach()

        for li, layer in enumerate(layers):
            handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
        handles.append(bb.lm.norm.register_forward_pre_hook(norm_pre))
        seq = torch.cat([prompt_ids[:, -1:], lead_ids], dim=1)
        logits = _forward_logits(bb, c, seq, position_cursor + n - 1)[-1].float()
    finally:
        for h in handles:
            h.remove()
        del c
    # DLA toward A - B through the final RMSNorm linearized at this row
    res = {"top_token": int(logits.argmax())}
    final = store.get("final")
    W = bb.model.get_output_embeddings().weight
    if final is not None and pair_tokens is not None:
        ta, tb = pair_tokens
        d = (W[ta] - W[tb]).float()
        w = bb.lm.norm.weight.float()
        rms = torch.sqrt((final * final).mean() + 1e-6)
        scale = (w * d) / rms
        res["logit_diff"] = float(logits[ta] - logits[tb])
        for li, st in store["layers"].items():
            st["dla_v"] = float((st["oV"] * scale).sum())
            st["dla_attn"] = float((st["oA"] * scale).sum())
    per_layer = {}
    keys = ("rho_v", "rho_prompt", "rho_lat", "rho_rest", "crop_ent", "crop_max", "cell_corr", "dla_v", "dla_attn")
    for key in keys:
        per_layer[key] = [None if store["layers"].get(li, {}).get(key) is None else round(float(store["layers"][li][key]), 5)
                          for li in range(len(layers))]
    res["layers"] = per_layer
    return res


# --------------------------------------------------------------------------- run


@torch.no_grad()
def run(engine, *, cache, position_cursor, system_prompt, user_prompt, json_prefix,
        candidates, case_index: int, normal_output=None) -> None:
    out_path = os.environ.get("VLMAS_CANON_DIAG", "").strip()
    if not out_path:
        return
    from memory.vcca_diag import branching_nodes
    from memory.vgain_diag import (candidate_lead, candidate_leads, candidate_pass, candidate_tail,
                                   candidate_variants, logsumexp_by, parse_answer)
    from vision_text_mas.latent_terminal import _assistant_prompt

    t0 = time.time()
    bb = engine._backbone
    if getattr(bb, "_route_vis_cols", None) is None or getattr(bb, "_route_vis_pos", None) is None \
            or not getattr(bb, "_route_pages", None):
        print(f"[CanonDiag] case {case_index}: SKIP (no visual bookkeeping; set VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=bookkeep)", flush=True)
        return
    arms = [a.strip() for a in os.environ.get("VLMAS_CANON_DIAG_ARMS", ",".join(ARMS)).split(",") if a.strip()]
    choices = tuple(str(c).strip() for c in (candidates or ()) if str(c).strip())
    tok = bb.processor.tokenizer
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt, user_prompt=user_prompt, json_prefix=json_prefix)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    n_prompt = int(prompt_ids.shape[1])
    b_pos = int(position_cursor) + n_prompt - 1
    lead = candidate_lead(normal_output); tail = candidate_tail(normal_output); leads = candidate_leads(normal_output)
    lead_ids = tok(lead, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    cand_ids, var_of = [], []
    for ci, y in enumerate(choices):
        for ld in leads:
            for ids in candidate_variants(tok, ld, y, tail):
                if ids not in cand_ids:
                    cand_ids.append(ids); var_of.append(ci)
    nodes = branching_nodes(cand_ids) if len(cand_ids) >= 2 else []

    base_len = int(bb._kv_len(cache))
    crop, hw = page_cells(bb._route_vis_pos, bb._route_pages)
    lat = getattr(engine, "_rpath_latent_cols", None)
    groups = {"vis": bb._route_vis_cols.long(), "lat": (lat.long() if lat is not None else torch.empty(0, dtype=torch.long)),
              "prompt_start": base_len, "crop": crop, "hw": hw}

    rec = {"case": int(case_index), "slide": getattr(engine, "_rpath_slide_id", None), "n_vis": int(groups["vis"].numel()),
           "pages": list(bb._route_pages), "n_prompt": n_prompt, "b_pos": b_pos, "cands": list(choices), "lead": lead,
           "native_text": None if normal_output is None else str(normal_output)[:300],
           "native_answer": parse_answer(normal_output, choices), "arms": {},
           "pos_gap": int(os.environ.get("VLMAS_POS_GAP", "0") or 0)}
    # 1) candidate log-probs per arm
    scored = {}
    for arm in arms:
        c = deepcopy(cache)
        try:
            note = apply_arm(bb, c, arm, b_pos)
            cp = candidate_pass(bb, c, prompt_ids, cand_ids, position_cursor, nodes) if cand_ids else {"logp": []}
        finally:
            del c
        lp = logsumexp_by(cp["logp"], var_of, len(choices)) if cand_ids else []
        order = sorted(range(len(lp)), key=lambda i: -(lp[i] if lp[i] is not None else -1e9))
        scored[arm] = {"note": note, "logp": lp, "answer": choices[order[0]] if order else None,
                       "runner_up": choices[order[1]] if len(order) > 1 else None}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # 2) DLA pair: native answer vs canonical answer (or native top-2)
    A = scored.get("native", {}).get("answer")
    B = scored.get("canonical", {}).get("answer")
    if B is None or B == A:
        B = scored.get("native", {}).get("runner_up")
    pair_tokens = None; pair_note = None
    if A and B and A != B:
        a_ids = candidate_variants(tok, lead, A, tail)[0]; b_ids = candidate_variants(tok, lead, B, tail)[0]
        fd = first_diff_token(a_ids, b_ids)
        n_lead = int(lead_ids.shape[1])
        if fd is not None and fd[0] == n_lead:
            pair_tokens = (fd[1], fd[2])
        else:
            pair_note = f"first differing token at {None if fd is None else fd[0]} != decision row {n_lead}"
    rec["pair"] = {"A": A, "B": B, "tokens": pair_tokens, "note": pair_note}
    # 3) decision-row capture per arm
    for arm in arms:
        c = deepcopy(cache)
        try:
            apply_arm(bb, c, arm, b_pos)
            cap = capture_decision_row(bb, c, prompt_ids, lead_ids, position_cursor, groups, pair_tokens)
        finally:
            del c
        rec["arms"][arm] = {**scored[arm], **cap}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rec["sec"] = round(time.time() - t0, 1)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    def mean_l(arm, key, lo=18, hi=35):
        v = [x for x in rec["arms"][arm]["layers"][key][lo:hi + 1] if x is not None]
        return None if not v else round(sum(v) / len(v), 4)
    print("[CanonDiag] case {} ".format(case_index) + " | ".join(
        f"{a}: ans={rec['arms'][a]['answer']!r} rho_v(L18-35)={mean_l(a, 'rho_v')} cell_corr={mean_l(a, 'cell_corr')} "
        f"crop_ent={mean_l(a, 'crop_ent')} dla_v={mean_l(a, 'dla_v')}" for a in arms)
        + f" | pair={A!r}/{B!r} sec={rec['sec']}", flush=True)


__all__ = ["run", "apply_arm", "capture_decision_row", "spearman", "average_ranks", "normalized_entropy", "page_cells",
           "shift_positions", "first_diff_token", "ARMS"]
