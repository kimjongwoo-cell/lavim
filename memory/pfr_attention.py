"""C2 -- Provenance-Factorized Re-Read (opt-in, VLMAS_ANSWERER_PFR=1|identity|record|shuffled).

Answerer-side attention: split the cache by provenance into M_V (crop K/V
columns from the Reasoner prefill) and M_C (everything else). For a query row
q_t the native attention alpha over V u C is computed as usual; we keep

    rho_V = sum_{j in V} alpha_tj              (NEVER changed)
    m_C   = sum_{j in C} alpha_tj v_j          (context message, as native)

then form the provisional state x_C = x_t + W_O m_C, re-query
q_C = W_Q LN(x_C) (same layer's input LN / q_norm / rotary), read the visual
memory again with beta = softmax_{j in V}(q_C k_j / sqrt d), and emit

    m~_V = rho_V * sum_{j in V} beta_tj v_j
    m~_t = m_C + m~_V   ->  o_proj(m~_t) replaces the attention output.

Applied to every generated token and to the LAST prompt-prefill row (the one
that emits the first answer token); all other rows untouched. Off = no hooks.

Modes (Notion Experiment Log "[C2 구조 검증] Context-Updated Answerer Query Visual Re-Read
-- matched-context specificity", arms A-D):
  record    A  native output untouched; statistics and the per-call r_C = W_O m_C are recorded
  identity  B  same-query re-read (q_t instead of q_C): beta == alpha_V / rho_V
  1         C  matched context-updated query (the original PFR)
  shuffled  D  q from x_t + r_C of a DONOR case: own m_C and rho_V are kept, only the query is
               built from the donor's context update. Donor = fixed in-dataset derangement
               (VLMAS_ANSWERER_PFR_PERM json {case: donor}); r_C comes from the donor's arm-A
               record dump (VLMAS_ANSWERER_PFR_DONOR_DIR). Call alignment:
                 prefill call k  -> donor prefill call k (clamped)
                 generated call g -> donor gen call round(u (T_d - 1)), u = g / max(T_i - 1, 1)
               with T_i the TARGET's native generated-call count (its own arm-A dump) and u
               clamped to [0, 1]. No "repeat the donor's last r_C" tail.

Per-call statistics (whenever a layer gate or a dump is set, or mode is record|shuffled), all
computed inside the same call so they are step-aligned:
  js_nat_id = JS(beta_nat, beta_id)   js_nat_m = JS(beta_nat, beta_matched)
  js_nat_s  = JS(beta_nat, beta_shuf) js_m_s   = JS(beta_matched, beta_shuf)
(mean over heads, natural log), rho, ||r_C|| matched / shuffled and their ratio, cos(q, q_C),
and maxdiff = max |o_proj(re-read row) - native bf16 output row| for the arm's own row.

Env
  VLMAS_ANSWERER_PFR=1|identity|record|shuffled
  VLMAS_ANSWERER_PFR_LAYER=27          comma list / a-b ranges; unset = every layer (old behaviour)
  VLMAS_ANSWERER_PFR_DUMP=<dir>        case_<idx>.pt (calls: T, kind, r_C fp16, stats) + stats.jsonl
                                       (requires exactly one active layer)
  VLMAS_ANSWERER_PFR_DONOR_DIR=<dir>   shuffled: arm-A dump dir of the same dataset
  VLMAS_ANSWERER_PFR_PERM=<json>       shuffled: {"<case>": <donor case>}
  VLMAS_ANSWERER_PFR_LOG=1             old per-install stats line for modes 1|identity
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import random

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

MODES = ("1", "identity", "record", "shuffled")


def mode() -> str:
    return os.environ.get("VLMAS_ANSWERER_PFR", "").strip()


def enabled() -> bool:
    return mode() in MODES


# --------------------------------------------------------------------------- pure


def parse_layers(text: str | None, n_layers: int) -> tuple[int, ...]:
    """'27' | '24,27' | '24-30' -> sorted unique layer ids; empty -> every layer."""
    text = (text or "").strip()
    if not text:
        return tuple(range(n_layers))
    out: list[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = (int(v) for v in tok.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(tok))
    bad = [v for v in out if not 0 <= v < n_layers]
    if bad or not out:
        raise ValueError(f"VLMAS_ANSWERER_PFR_LAYER out of range 0..{n_layers - 1}: {text!r}")
    return tuple(sorted(set(out)))


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence over the last dim (natural log). Rows are distributions."""
    p = p.double().clamp_min(0.0)
    q = q.double().clamp_min(0.0)
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * (a.clamp_min(eps).log() - b.clamp_min(eps).log())).sum(-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def fixed_derangement(cases, seed: int = 0) -> dict[int, int]:
    """One fixed permutation with no fixed point over the given case ids (seeded)."""
    ids = sorted({int(c) for c in cases})
    if len(ids) < 2:
        raise ValueError("a derangement needs at least two cases")
    rng = random.Random(seed)
    while True:
        perm = ids[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm)):
            return dict(zip(ids, perm))


def donor_call_index(k: int, is_prefill: bool, n_target_gen: int,
                     n_donor_prefill: int, n_donor_gen: int) -> int | None:
    """Target call -> donor call of the same kind (see module docstring)."""
    if is_prefill:
        return None if n_donor_prefill == 0 else min(int(k), n_donor_prefill - 1)
    if n_donor_gen == 0:
        return None
    u = min(max(int(k) / max(int(n_target_gen) - 1, 1), 0.0), 1.0)
    return int(math.floor(u * (n_donor_gen - 1) + 0.5))


# --------------------------------------------------------------------------- hook


def _load_donor(case_index: int):
    donor_dir = os.environ.get("VLMAS_ANSWERER_PFR_DONOR_DIR", "").strip()
    perm_path = os.environ.get("VLMAS_ANSWERER_PFR_PERM", "").strip()
    if not donor_dir or not perm_path:
        raise RuntimeError("shuffled mode needs VLMAS_ANSWERER_PFR_DONOR_DIR and VLMAS_ANSWERER_PFR_PERM")
    with open(perm_path) as fh:
        perm = {int(k): int(v) for k, v in json.load(fh).items()}
    if case_index not in perm:
        raise RuntimeError(f"case {case_index} missing from {perm_path}")
    d_case = perm[case_index]
    if d_case == case_index:
        raise RuntimeError(f"permutation maps case {case_index} to itself")
    donor = torch.load(os.path.join(donor_dir, f"case_{d_case:04d}.pt"), map_location="cpu")
    own = torch.load(os.path.join(donor_dir, f"case_{case_index:04d}.pt"), map_location="cpu")
    pre = [c["r_C"] for c in donor["calls"] if c["kind"] == "prefill"]
    gen = [c["r_C"] for c in donor["calls"] if c["kind"] == "gen"]
    n_target_gen = sum(1 for c in own["calls"] if c["kind"] == "gen")
    return {"case": d_case, "layer": int(donor["layer"]), "pre": pre, "gen": gen,
            "n_target_gen": n_target_gen}


@contextlib.contextmanager
def install_pfr(backbone, visual_cols, case_index: int = -1):
    """Hook the selected decoder layers of the LM for the duration of the block."""
    if not enabled() or visual_cols is None or int(visual_cols.numel()) == 0:
        yield None
        return
    md = mode()
    identity = md == "identity"
    record = md == "record"
    shuffled = md == "shuffled"
    log = os.environ.get("VLMAS_ANSWERER_PFR_LOG", "") == "1"
    layers = backbone.lm.layers
    layer_env = os.environ.get("VLMAS_ANSWERER_PFR_LAYER", "").strip()
    active = parse_layers(layer_env, len(layers))
    dump_dir = os.environ.get("VLMAS_ANSWERER_PFR_DUMP", "").strip()
    detail = bool(layer_env) or bool(dump_dir) or record or shuffled
    if (dump_dir or shuffled) and len(active) != 1:
        raise RuntimeError("PFR dump / shuffled mode needs exactly one layer (VLMAS_ANSWERER_PFR_LAYER)")
    donor = _load_donor(int(case_index)) if shuffled else None
    if donor is not None and donor["layer"] != active[0]:
        raise RuntimeError(f"donor dump layer {donor['layer']} != active layer {active[0]}")
    stash: dict[int, torch.Tensor] = {}
    stats = {"calls": 0, "rho_sum": 0.0, "maxdiff": 0.0, "kl_sum": 0.0}
    calls: list[dict] = []
    counters = {"prefill": 0, "gen": 0}
    handles = []

    def make_pre(li):
        def pre(module, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is not None:
                stash[li] = x  # residual stream x_t entering the layer
        return pre

    def make_post(li):
        def post(module, args, kwargs, output):
            x_t_full = stash.get(li)
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if x_t_full is None or hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            T = int(hidden.shape[1])
            keys = cache.layers[li].keys
            values = cache.layers[li].values
            L = int(keys.shape[2])
            vis = visual_cols.to(device=keys.device, dtype=torch.long)
            vis = vis[vis < L]
            if int(vis.numel()) == 0 or int(vis.numel()) >= L:
                return output
            vmask = torch.zeros(L, dtype=torch.bool, device=keys.device)
            vmask[vis] = True
            layer = layers[li]
            row = T - 1  # single decode token, or the last prompt-prefill row
            h_row = hidden[:, row:row + 1, :]
            x_row = x_t_full[:, row:row + 1, :]
            cos, sin = pe
            cos_r = cos[..., row:row + 1, :] if cos.dim() >= 3 else cos
            sin_r = sin[..., row:row + 1, :] if sin.dim() >= 3 else sin
            Hd = module.head_dim
            G = module.num_key_value_groups

            def make_query(h):
                q = module.q_norm(module.q_proj(h).view(1, 1, -1, Hd)).transpose(1, 2)  # [1,Hq,1,D]
                q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos_r, sin_r)
                return q

            K = repeat_kv(keys, G).float()      # [1,Hq,L,D]
            V = repeat_kv(values, G).float()
            q = make_query(h_row).float()
            scores = torch.matmul(q, K.transpose(-2, -1)) * module.scaling  # [1,Hq,1,L]
            alpha = torch.softmax(scores, dim=-1)
            rho = alpha[..., vmask].sum(dim=-1, keepdim=True)              # [1,Hq,1,1]
            m_C = torch.matmul(alpha * (~vmask).float(), V)                 # [1,Hq,1,D]
            K_V = K[..., vmask, :].transpose(-2, -1)
            V_V = V[..., vmask, :]
            q_C = None
            r_C = None
            if not identity or detail:
                # provisional state after the context read, re-query with the
                # same layer's LN / W_Q / q_norm / rotary
                mc_flat = m_C.transpose(1, 2).reshape(1, 1, -1).to(module.o_proj.weight.dtype)
                r_C = module.o_proj(mc_flat)
                x_C = x_row + r_C
                q_C = make_query(layer.input_layernorm(x_C)).float()
            kind = "prefill" if T > 1 else "gen"
            k = counters[kind]
            counters[kind] += 1
            q_S = None
            r_S = None
            if shuffled:
                idx = donor_call_index(k, kind == "prefill", donor["n_target_gen"],
                                       len(donor["pre"]), len(donor["gen"]))
                if idx is None:
                    raise RuntimeError(f"donor case {donor['case']} has no {kind} calls")
                bank = donor["pre"] if kind == "prefill" else donor["gen"]
                r_S = bank[idx].to(device=x_row.device, dtype=x_row.dtype).view(1, 1, -1)
                q_S = make_query(layer.input_layernorm(x_row + r_S)).float()
            if identity:
                q_use = q
            elif shuffled:
                q_use = q_S
            else:
                q_use = q_C          # mode "1"; unused for record
            new_row = None
            beta = None
            if not record:
                s_v = torch.matmul(q_use, K_V) * module.scaling
                beta = torch.softmax(s_v, dim=-1)                           # [1,Hq,1,|V|]
                m_V = rho * torch.matmul(beta, V_V)
                m = (m_C + m_V).transpose(1, 2).reshape(1, 1, -1)
                new_row = module.o_proj(m.to(module.o_proj.weight.dtype))
            native_row = attn_out[:, row:row + 1, :].float()
            if (log or identity) and not record:
                stats["calls"] += 1
                stats["rho_sum"] += float(rho.mean())
                stats["maxdiff"] = max(stats["maxdiff"], float((new_row.float() - native_row).abs().max()))
                a_v = alpha[..., vmask] / rho.clamp_min(1e-9)
                stats["kl_sum"] += float((beta * (beta.clamp_min(1e-12).log() - a_v.clamp_min(1e-12).log())).sum(-1).mean())
            if detail:
                b_nat = alpha[..., vmask] / rho.clamp_min(1e-12)
                b_id = torch.softmax(torch.matmul(q, K_V) * module.scaling, dim=-1)
                b_m = torch.softmax(torch.matmul(q_C, K_V) * module.scaling, dim=-1)
                rec = {"T": T, "kind": kind, "k": k, "rho": float(rho.mean()),
                       "js_nat_id": float(js_divergence(b_nat, b_id).mean()),
                       "js_nat_m": float(js_divergence(b_nat, b_m).mean()),
                       "rnorm_m": float(r_C.float().norm()),
                       "cos_q_qC": float(torch.nn.functional.cosine_similarity(
                           q.flatten(2), q_C.flatten(2), dim=-1).mean())}
                if shuffled:
                    b_s = torch.softmax(torch.matmul(q_S, K_V) * module.scaling, dim=-1)
                    rec["js_nat_s"] = float(js_divergence(b_nat, b_s).mean())
                    rec["js_m_s"] = float(js_divergence(b_m, b_s).mean())
                    rec["rnorm_s"] = float(r_S.float().norm())
                    rec["rnorm_ratio"] = rec["rnorm_s"] / max(rec["rnorm_m"], 1e-12)
                if new_row is not None:
                    rec["maxdiff"] = float((new_row.float() - native_row).abs().max())
                calls.append({"layer": li, "T": T, "kind": kind,
                              "r_C": r_C.detach().float().reshape(-1).half().cpu(), "stats": rec})
            if record:
                return output
            out = attn_out.clone()
            out[:, row:row + 1, :] = new_row.to(out.dtype)
            return (out, *output[1:]) if isinstance(output, tuple) else out
        return post

    for li in active:
        layer = layers[li]
        handles.append(layer.register_forward_pre_hook(make_pre(li), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True))
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()
        if stats["calls"]:
            print(f"[PFR:{mode()}] {stats['calls']} attn calls, mean rho_V {stats['rho_sum']/stats['calls']:.4f}, "
                  f"max|out-native| {stats['maxdiff']:.3e}, mean KL(beta||alpha_V) {stats['kl_sum']/stats['calls']:.4f}", flush=True)
        if detail and calls:
            recs = [c["stats"] for c in calls]

            def agg(key, fn):
                vals = [r[key] for r in recs if key in r]
                return None if not vals else round(fn(vals), 6)

            first = recs[0]
            summary = {
                "case": int(case_index), "mode": md, "layers": list(active), "calls": len(recs),
                "n_vis": int(visual_cols.numel()),
                "prefill_calls": counters["prefill"], "gen_calls": counters["gen"],
                "donor": None if donor is None else donor["case"],
                "rho_mean": agg("rho", lambda v: sum(v) / len(v)),
                "js_nat_id_max": agg("js_nat_id", max),
                "js_nat_m_mean": agg("js_nat_m", lambda v: sum(v) / len(v)),
                "js_nat_s_mean": agg("js_nat_s", lambda v: sum(v) / len(v)),
                "js_m_s_mean": agg("js_m_s", lambda v: sum(v) / len(v)),
                "rnorm_ratio_mean": agg("rnorm_ratio", lambda v: sum(v) / len(v)),
                "maxdiff_max": agg("maxdiff", max),
                "decision_row": {key: first.get(key) for key in
                                 ("kind", "js_nat_id", "js_nat_m", "js_nat_s", "js_m_s", "rnorm_ratio", "cos_q_qC")},
            }
            print("[PFRctx] " + json.dumps(summary), flush=True)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                torch.save({"case": int(case_index), "layer": int(active[0]), "mode": md,
                            "donor": summary["donor"],
                            "calls": [{"T": c["T"], "kind": c["kind"], "r_C": c["r_C"]} for c in calls],
                            "stats": recs},
                           os.path.join(dump_dir, f"case_{int(case_index):04d}.pt"))
                with open(os.path.join(dump_dir, "stats.jsonl"), "a") as fh:
                    fh.write(json.dumps(summary) + "\n")


__all__ = ["enabled", "mode", "install_pfr", "parse_layers", "js_divergence",
           "fixed_derangement", "donor_call_index", "MODES"]
