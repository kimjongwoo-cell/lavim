"""Group-structured Answerer attention hooks (opt-in; off = no hooks).

Three modes, all applied to EVERY query row of the Answerer forward(s) they wrap
(prompt prefill rows, teacher-forced continuation rows, generated tokens), on all
LM layers, recomputing the attention output from the model's own q/K/V exactly as
memory/visual_cut.py does (bf16 ulp-level differences vs native):

  capture   no change; records per layer the native visual mass m_V[h, t] and the
            per-group mass m_g[h, t] (used as reference by massfix and for logging)
  massfix   duplication control: keep the row's visual mass at a REFERENCE value
            m_V^ref[layer][h, t] (captured from the un-duplicated forward of the
            same prompt+continuation), renormalising only inside the visual set:
                a~_j = m_V^ref * a_j / sum_V a_k        (j in V)
                a~_j = (1 - m_V^ref) * a_j / sum_C a_k  (j not in V)
            so the row still sums to 1 and only the visual-internal shares can
            carry the duplication effect.
  c2        Cardinality-Invariant Visual Evidence Competition (user spec):
                s_g = LSE_{j in g} l_j - log|g|,  beta_g = softmax_g(s_g)
                a_{j|g} = softmax_{j in g}(l_j)
                a~_j = m_V^native * beta_{g(j)} * a_{j|g(j)}   (j in V)
            non-visual a unchanged; m_V^native is this row's own native visual
            mass, so the total visual mass is preserved and only the inter-group
            split changes. No hyper-parameters.

Groups are given as ABSOLUTE cache column indices (duplicated columns appended
by the duplication test are mapped to their source group by the caller).
"""
from __future__ import annotations

import contextlib
import math

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv


def dropfix_alpha(scores, vmask, dmask, ref):
    """Mass-fixed removal on one attention block (mode 'dropfix').
    scores [..., L] logits (causal -inf already applied); vmask / dmask bool [L]
    (dmask a subset of vmask); ref [...] target visual mass per row. The dropped
    columns leave the softmax, the remaining visual columns are rescaled to sum
    to ref and the non-visual columns to 1 - ref (row sum 1, relative shares
    inside each set unchanged). Returns (new_alpha, visual mass after the fix)."""
    a_d = torch.softmax(scores.masked_fill(dmask, float("-inf")), dim=-1)
    rem = vmask & ~dmask
    mv_d = a_d[..., rem].sum(-1)
    sv = (ref / mv_d.clamp_min(1e-12))[..., None]
    sc = ((1.0 - ref) / (1.0 - mv_d).clamp_min(1e-12))[..., None]
    new_alpha = torch.where(rem, a_d * sv, a_d * sc)
    return new_alpha, new_alpha[..., rem].sum(-1)


def summarize_post_masses(rec, row: int) -> dict:
    """summarize_masses over the post-fix masses recorded by mode 'dropfix'."""
    return summarize_masses({"mv": rec.get("mv_post", {}), "mg": rec.get("mg_post", {})}, row)


@contextlib.contextmanager
def install_group_attention(backbone, group_cols, visual_cols, mode: str, mv_ref=None,
                            record_rows=None, drop_cols=None):
    """group_cols: list of LongTensor (absolute cache columns) per group.
    visual_cols: LongTensor of all visual columns (absolute; should equal the union).
    mode: capture | massfix | c2 | dropfix.  mv_ref: dict layer -> tensor [Hq, T]
    (massfix, dropfix).  drop_cols: absolute visual columns removed (dropfix).
    Yields rec: {"mv": {layer: [Hq,T] tensor}, "mg": {layer: [G,Hq,T]}, "calls": n}
    (native masses); dropfix also fills "mv_post"/"mg_post" (after the fix),
    "dev_mv" (max |m_V after fix - m_V^ref|), "dev_row" (max |row sum - 1|) and
    "fallback" (calls left native because the reference rows did not match)."""
    layers = backbone.lm.layers
    rec = {"mv": {}, "mg": {}, "calls": 0, "T": None, "mode": mode,
           "mv_post": {}, "mg_post": {}, "dev_mv": 0.0, "dev_row": 0.0, "fallback": 0}
    handles = []
    G = len(group_cols)

    def make_hook(li):
        def hook(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or pe is None or cache is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys = cache.layers[li].keys
            values = cache.layers[li].values
            L = int(keys.shape[2]); T = int(hidden.shape[1])
            vis = visual_cols.to(device=keys.device, dtype=torch.long)
            vis = vis[vis < L]
            if int(vis.numel()) == 0:
                return output
            Hd = module.head_dim
            g_ = module.num_key_value_groups
            cos, sin = pe
            q = module.q_norm(module.q_proj(hidden).view(1, T, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            K = repeat_kv(keys, g_).float()
            V = repeat_kv(values, g_).float()
            scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling      # [1,Hq,T,L]
            past = L - T
            if T > 1:
                col = torch.arange(L, device=keys.device)[None, None, None, :]
                row = (past + torch.arange(T, device=keys.device))[None, None, :, None]
                scores = scores.masked_fill(col > row, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)                                       # [1,Hq,T,L]
            vmask = torch.zeros(L, dtype=torch.bool, device=keys.device); vmask[vis] = True
            mv = alpha[..., vmask].sum(-1)                                              # [1,Hq,T]
            gm = []
            for gc in group_cols:
                gc = gc.to(keys.device); gc = gc[gc < L]
                gm.append(alpha[..., gc].sum(-1))
            rec["mv"][li] = mv[0].detach().cpu()
            rec["mg"][li] = torch.stack([x[0] for x in gm]).detach().cpu() if gm else None
            rec["calls"] += 1; rec["T"] = T
            if mode == "capture":
                return output
            new_alpha = alpha.clone()
            if mode == "massfix":
                ref = mv_ref[li].to(keys.device).float()                                 # [Hq,Tref]
                if ref.shape[-1] != T:
                    return output   # row mismatch: leave native (logged by caller via calls)
                ref = ref[None]                                                          # [1,Hq,T]
                mv_c = mv.clamp_min(1e-12); mc_c = (1.0 - mv).clamp_min(1e-12)
                sv = (ref / mv_c)[..., None]; sc = ((1.0 - ref) / mc_c)[..., None]
                new_alpha = torch.where(vmask[None, None, None, :], alpha * sv, alpha * sc)
            elif mode == "c2":
                # group log-mean-exp competition, native m_V preserved
                s_list, within = [], []
                for gc in group_cols:
                    gc = gc.to(keys.device); gc = gc[gc < L]
                    lg = scores[..., gc]                                                  # [1,Hq,T,|g|]
                    lse = torch.logsumexp(lg, dim=-1)                                     # [1,Hq,T]
                    s_list.append(lse - math.log(max(int(gc.numel()), 1)))
                    within.append((gc, torch.softmax(lg, dim=-1)))
                s = torch.stack(s_list, dim=-1)                                           # [1,Hq,T,G]
                beta = torch.softmax(s, dim=-1)
                new_alpha = alpha.clone()
                for gi, (gc, aw) in enumerate(within):
                    new_alpha[..., gc] = mv[..., None] * beta[..., gi:gi + 1] * aw
            elif mode == "dropfix":
                # mass-fixed removal: drop_cols leave the softmax, the remaining
                # visual mass is restored to the reference row's m_V^ref
                ref = mv_ref[li].to(keys.device).float()                                 # [Hq,Tref]
                if ref.shape[-1] != T:
                    rec["fallback"] += 1
                    return output
                dmask = torch.zeros(L, dtype=torch.bool, device=keys.device)
                if drop_cols is not None and int(drop_cols.numel()):
                    dc = drop_cols.to(device=keys.device, dtype=torch.long)
                    dmask[dc[dc < L]] = True
                new_alpha, mv_post = dropfix_alpha(scores, vmask, dmask, ref[None])
                rec["dev_mv"] = max(rec["dev_mv"], float((mv_post - ref[None]).abs().max()))
                rec["dev_row"] = max(rec["dev_row"], float((new_alpha.sum(-1) - 1.0).abs().max()))
                mgp = []
                for gc in group_cols:
                    gc = gc.to(keys.device); gc = gc[gc < L]
                    mgp.append(new_alpha[..., gc].sum(-1)[0])
                rec["mv_post"][li] = mv_post[0].detach().cpu()
                rec["mg_post"][li] = torch.stack(mgp).detach().cpu() if mgp else None
            else:
                raise ValueError(mode)
            ctx = torch.matmul(new_alpha, V)
            flat = ctx.transpose(1, 2).reshape(1, T, -1)
            new = module.o_proj(flat.to(module.o_proj.weight.dtype)).to(attn_out.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return hook

    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()


def summarize_masses(rec, row: int) -> dict:
    """Layer-mean, head-mean native masses at one query row: m_V, m_g[], rho_g[]."""
    mvs, mgs = [], []
    for li in sorted(rec["mv"]):
        mv = rec["mv"][li]; mg = rec["mg"][li]
        if mv is None or mg is None or row >= mv.shape[-1]:
            continue
        mvs.append(float(mv[:, row].mean()))
        mgs.append([float(mg[gi, :, row].mean()) for gi in range(mg.shape[0])])
    if not mvs:
        return {}
    nl = len(mvs); G = len(mgs[0])
    m_v = sum(mvs) / nl
    m_g = [sum(mgs[l][gi] for l in range(nl)) / nl for gi in range(G)]
    return {"m_V": round(m_v, 6), "m_g": [round(x, 6) for x in m_g],
            "rho_g": [round(x / max(m_v, 1e-12), 4) for x in m_g], "n_layers": nl}


__all__ = ["install_group_attention", "summarize_masses", "summarize_post_masses", "dropfix_alpha"]
