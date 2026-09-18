"""Answerer-side group scores for the C2 kill test (opt-in; off = no hooks).

For the Answerer query rows we record, per layer and per cross-scale group g
(5x root + its 20x children), three numbers computed from the attention
logits the model already forms:

    mass_g  = sum_{j in g} alpha_{t,j}            native group attention mass
    lse_g   = LSE_{j in g} l_{t,j}                 log-sum-exp of group logits
    lme_g   = lse_g - log|g|                       size-normalised (log-mean-exp)

with l_{t,j} = q_t . k_j * scaling (post-RoPE, same as the model), alpha the
full causal softmax over every visible column (text + latent + visual), heads
averaged.  Rows: "last" = the last prompt row (the query that emits the first
answer token) and "cont" = the teacher-forced continuation rows.

Used by retrieval_ig (VLMAS_RETR=igsweep VLMAS_RETR_AQ=1): the values are
compared post hoc with the leave-one-group-out utility I_g of the same case.
"""
from __future__ import annotations

import contextlib
import math

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv


@contextlib.contextmanager
def capture_group_scores(backbone, group_cols: list[torch.Tensor], visual_cols: torch.Tensor,
                         n_prompt: int):
    """group_cols: list of LongTensors of ABSOLUTE cache column indices, one per
    group; visual_cols: all visual columns (absolute). n_prompt: number of prompt
    rows in the single prompt+continuation forward that runs inside the context.
    Yields a dict filled per layer: rec["last"|"cont"][name] -> list over layers."""
    layers = backbone.lm.layers
    rec = {"last": {"mass": [], "lse": [], "lme": [], "M_V": []},
           "cont": {"mass": [], "lse": [], "lme": [], "M_V": []},
           "sizes": [int(g.numel()) for g in group_cols], "calls": 0, "T": None}
    handles = []

    def make_hook(li):
        def hook(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or cache is None:
                return output
            keys = cache.layers[li].keys
            L = int(keys.shape[2]); T = int(hidden.shape[1])
            if T < n_prompt:          # not the prompt+continuation forward
                return output
            rec["T"] = T
            past = L - T
            rows_last = [n_prompt - 1]
            rows_cont = list(range(n_prompt, T - 1))
            rows = rows_last + rows_cont
            Hd = module.head_dim; G = module.num_key_value_groups
            cos, sin = pe
            q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            q = q[:, :, rows].float()                                              # [1,Hq,R,D]
            K = repeat_kv(keys, G).float()
            scores = torch.matmul(q, K.transpose(-2, -1)) * module.scaling         # [1,Hq,R,L]
            col = torch.arange(L, device=keys.device)[None, None, None, :]
            row = (past + torch.tensor(rows, device=keys.device))[None, None, :, None]
            scores = scores.masked_fill(col > row, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)
            vis = visual_cols.to(keys.device)
            for name, sl in (("last", slice(0, 1)), ("cont", slice(1, None))):
                if name == "cont" and not rows_cont:
                    continue
                a = alpha[:, :, sl]; s = scores[:, :, sl]                          # [1,Hq,r,L]
                mass, lse, lme = [], [], []
                for g in group_cols:
                    gc = g.to(keys.device)
                    mass.append(float(a[..., gc].sum(-1).mean()))
                    l = torch.logsumexp(s[..., gc], dim=-1)                          # [1,Hq,r]
                    lse.append(float(l.mean()))
                    lme.append(float((l - math.log(int(gc.numel()))).mean()))
                rec[name]["mass"].append(mass); rec[name]["lse"].append(lse); rec[name]["lme"].append(lme)
                rec[name]["M_V"].append(float(a[..., vis].sum(-1).mean()))
            rec["calls"] += 1
            return output
        return hook

    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()


def summarize(rec: dict) -> dict:
    """Layer-mean per group for each row set + per-layer arrays (kept small)."""
    out = {"sizes": rec["sizes"], "calls": rec["calls"], "T": rec["T"]}
    for name in ("last", "cont"):
        r = rec[name]
        if not r["mass"]:
            continue
        nl = len(r["mass"]); ng = len(r["mass"][0])
        out[name] = {
            "n_layers": nl,
            "M_V": round(sum(r["M_V"]) / nl, 5),
            "mass": [round(sum(r["mass"][l][g] for l in range(nl)) / nl, 6) for g in range(ng)],
            "lse": [round(sum(r["lse"][l][g] for l in range(nl)) / nl, 4) for g in range(ng)],
            "lme": [round(sum(r["lme"][l][g] for l in range(nl)) / nl, 4) for g in range(ng)],
            "mass_layers": [[round(v, 6) for v in row] for row in r["mass"]],
            "lme_layers": [[round(v, 4) for v in row] for row in r["lme"]],
        }
    return out


__all__ = ["capture_group_scores", "summarize"]
