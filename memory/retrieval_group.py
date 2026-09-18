"""Reasoning-conditioned retrieval of a cross-scale pathology context (kill test).

VLMAS_RETR=context|token|random  (Answerer stage; needs VLMAS_VCUT=mask STAGE=A)
  query  q^l = q_norm(W_Q^l LN^l(h_R^l)) from the LAST Reasoner latent step
         (per-layer residual input captured by the latent loop, pre-RoPE)
  keys   cached visual keys de-rotated to pre-RoPE with their stored 3-D
         MRoPE positions (k_pre = R(-pos) k)
  score  s_j = mean_l z_l( mean_h q^l_h . k^l_{h,j} * scaling )
  context: j* = argmax s_j -> its crop p* -> group = 5x root(p*) + all 20x
           children of that root; keep the whole group, mask the rest
  token:   keep the top-B tokens by s_j, B = |group|
  random:  a different root's group (same crop count when possible), seed 42+case
  dump:    no cut; write the full score vector S per case (bias estimation)
VLMAS_RETR_BIAS=<jsonl of dump>  subtract the leave-one-out mean score
         profile (same absolute visual index, other cases) before argmax --
         removes the case-independent "early token / first crop" component
VLMAS_RETR_OUT=<jsonl>  per-case selection log.
"""
from __future__ import annotations

import json
import os
import random

import torch

from memory.restage import rerotate_keys
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv


def mode() -> str:
    return os.environ.get("VLMAS_RETR", "").strip()


@torch.no_grad()
def score_tokens(backbone, cache, cols, pos, hidden_per_layer):
    """Return S [n] (mean over layers of per-layer z-scored head-mean scores)."""
    layers = backbone.lm.layers
    sample = cache.layers[0].keys
    if pos is not None:
        cos_old, sin_old = backbone.lm.rotary_emb(sample, pos.unsqueeze(1))
        cos_old, sin_old = cos_old.float(), -sin_old.float()   # rotate by -pos
    S = None
    for li, layer in enumerate(layers):
        attn = layer.self_attn
        h = hidden_per_layer[li].to(backbone.device).view(1, 1, -1).to(attn.q_proj.weight.dtype)
        q = attn.q_norm(attn.q_proj(layer.input_layernorm(h)).view(1, 1, -1, attn.head_dim)).transpose(1, 2).float()  # [1,Hq,1,D]
        k = cache.layers[li].keys.index_select(2, cols).float()                         # [1,Hkv,n,D]
        if pos is not None:
            k = rerotate_keys(k, cos_old, sin_old)
        K = repeat_kv(k, attn.num_key_value_groups)                                    # [1,Hq,n,D]
        s = (torch.matmul(q, K.transpose(-2, -1)) * attn.scaling)[0, :, 0, :].mean(0)  # [n]
        z = (s - s.mean()) / (s.std() + 1e-6)
        S = z if S is None else S + z
    return S / len(layers)


def groups_from_tree(pages, parents):
    n = len(pages)
    par = list(parents) if len(parents) == n else [-1] * n
    roots = [p for p in range(n) if par[p] < 0 or par[p] >= n]
    groups = {r: [r] + [c for c in range(n) if par[c] == r] for r in roots}
    crop_of = []
    for p, cnt in enumerate(pages):
        crop_of += [p] * int(cnt)
    starts = [sum(int(x) for x in pages[:p]) for p in range(n)]
    return par, roots, groups, crop_of, starts


def select(backbone, cache, cols, pos, pages, parents, case_index: int):
    m = mode()
    hid = getattr(backbone, "_retr_hidden", None)
    if hid is None or len(hid) != len(backbone.lm.layers):
        raise RuntimeError("retrieval: no per-layer latent hidden captured")
    S = score_tokens(backbone, cache, cols, pos, hid)
    S_raw = S.clone()
    out = os.environ.get("VLMAS_RETR_OUT", "").strip()
    if m == "dump":
        if out:
            with open(out, "a") as f:
                f.write(json.dumps({"case": int(case_index), "S": [round(float(x), 4) for x in S.tolist()]}) + "\n")
        print(f"[Retr:dump] case {case_index} n={int(S.numel())} (no cut)", flush=True)
        return torch.arange(int(cols.numel()), dtype=torch.long), {"mode": "dump"}
    bias_path = os.environ.get("VLMAS_RETR_BIAS", "").strip()
    bias_note = ""
    if bias_path and os.path.exists(bias_path):
        others = [json.loads(l) for l in open(bias_path)]
        others = [o for o in others if int(o["case"]) != int(case_index) and len(o["S"]) == int(S.numel())]
        if others:
            prof = torch.tensor(others[0]["S"]).new_zeros(int(S.numel()))
            for o in others:
                prof += torch.tensor(o["S"])
            prof = (prof / len(others)).to(S.device)
            S = S - prof
            bias_note = f" bias_LOO(n={len(others)})"
    par, roots, groups, crop_of, starts = groups_from_tree(pages, parents)
    j_star = int(S.argmax()); p_star = crop_of[j_star]
    root = p_star if (par[p_star] < 0 or par[p_star] >= len(pages)) else par[p_star]
    ctx_group = groups.get(root, [root])
    def tokens_of(group):
        idx = []
        for p in group:
            idx += list(range(starts[p], starts[p] + int(pages[p])))
        return idx
    ctx_tokens = tokens_of(ctx_group); B = len(ctx_tokens)
    if m == "context":
        keep = ctx_tokens; chosen = ctx_group
    elif m == "token":
        keep = sorted(torch.topk(S, k=min(B, int(S.numel()))).indices.tolist()); chosen = sorted({crop_of[i] for i in keep})
    elif m == "random":
        # budget-matched: another root's group, then top up (random tokens
        # outside that group) or subsample to exactly B tokens
        rng = random.Random(42 + int(case_index))
        others = [r for r in roots if r != root]
        same = [r for r in others if len(groups[r]) == len(ctx_group)]
        pool = same or others or [root]
        r = rng.choice(pool); chosen = groups[r]; keep = tokens_of(chosen)
        if len(keep) > B:
            keep = sorted(rng.sample(keep, B))
        elif len(keep) < B:
            rest = [i for i in range(int(cols.numel())) if i not in set(keep)]
            keep = sorted(keep + rng.sample(rest, B - len(keep)))
    else:
        raise ValueError(m)
    keep_t = torch.tensor(sorted(keep), dtype=torch.long)
    info = {"case": int(case_index), "mode": m, "j_star": j_star, "p_star": p_star, "root": root,
            "ctx_group": ctx_group, "chosen": list(chosen), "keep": len(keep), "n_vis": int(cols.numel()),
            "pages": [int(x) for x in pages], "parents": par,
            "S_top5": [(int(i), round(float(S[i]), 3)) for i in torch.topk(S, k=min(5, int(S.numel()))).indices],
            "S_crop_mean": [round(float(S[starts[p]:starts[p] + int(pages[p])].mean()), 3) for p in range(len(pages))]}
    info["S_raw_crop_mean"] = [round(float(S_raw[starts[p]:starts[p] + int(pages[p])].mean()), 3) for p in range(len(pages))]
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(info) + "\n")
    print(f"[Retr:{m}] case {case_index} j*={j_star} crop={p_star} root={root} ctx_group={ctx_group} "
          f"chosen={list(chosen)} keep={len(keep)}/{int(cols.numel())} S_crop={info['S_crop_mean']}{bias_note}", flush=True)
    return keep_t, info


__all__ = ["mode", "select", "score_tokens", "groups_from_tree"]
