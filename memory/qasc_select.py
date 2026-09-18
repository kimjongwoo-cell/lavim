"""C1 #14 — Question-Agnostic Salience–Coverage Log-Det Selection with Cross-Scale Redundancy.

Notion C1/C2 Method Evolution page of the same name (Development order 14). Reasoner-stage visual
KV selection that replaces the Pruning-B score/keep rule; retained merged tokens alone enter the
decoder prefill (same slicing path as Pruning-B, so positions stay consistent).

    salience   a_i  = mean over heads and queries of attention RECEIVED by patch i at vision block
                      l_v inside its own crop, averaged over the 2x2 pre-merge patches of token i
    feature    z_i  = normalized merger output (the visual embedding the decoder consumes)
    cross      n_i  = 1 - max_{j in C(i)} (1 + cos(z_i, z_j)) / 2, where C(i) are the tokens of
                      LOWER-magnification crops whose grid cell contains token i's slide-space centre;
                      n_i = 1 when C(i) is empty
    utility    F(S) = log det(I + sum_{j in S} a_j n_j z_j z_j^T),  |S| <= B, pool = all crops jointly
    greedy     Delta(i|S) = log(1 + a_i n_i z_i^T M_S^{-1} z_i), exact fast-greedy (incremental Cholesky
               on the N x N dual kernel I + B B^T with rows b_i = sqrt(a_i n_i) z_i)

Env (all opt-in; unset VLMAS_C1_QASC = untouched):
    VLMAS_C1_QASC=1         enable at the Reasoner prefill
    VLMAS_QASC_KEEP=0.25    budget B = ceil(keep * N)  (Pruning-B's matched 25%)
    VLMAS_QASC_VLAYER=-1    vision block for a_i (negative = from the end; default last block)
    VLMAS_QASC_CROSS=1      0 disables the cross-scale factor (n_i = 1) as an ablation
Choices not fixed by the Notion page and set here: l_v = last vision block, B = 25%, C(i) = the one
lower-magnification cell under token i's centre per overlapping crop, no per-crop minimum.
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- pure


def received_attention(q: torch.Tensor, k: torch.Tensor, scaling: float) -> torch.Tensor:
    """q, k [heads, len, dim] -> attention received per key, mean over heads and query rows (sums to 1)."""
    logits = (q.float() @ k.float().transpose(-1, -2)) * scaling
    return logits.softmax(dim=-1).mean(dim=(0, 1))


def merged_mean(scores: torch.Tensor, unit: int) -> torch.Tensor:
    n = int(scores.numel())
    if unit <= 0 or n % unit:
        raise ValueError(f"{n} patch scores not divisible by merge unit {unit}")
    return scores.reshape(n // unit, unit).mean(dim=1)


def token_centers(box, gh: int, gw: int) -> torch.Tensor:
    """(x, y, w, h) slide box + merged grid -> [gh*gw, 2] slide-space (x, y) centres, raster order."""
    x, y, w, h = (float(v) for v in box)
    rows = (torch.arange(gh, dtype=torch.float64) + 0.5) / gh * h + y
    cols = (torch.arange(gw, dtype=torch.float64) + 0.5) / gw * w + x
    cy, cx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)


def cross_scale_factor(z: torch.Tensor, counts, grids, mags, boxes):
    """n_i per token (all images concatenated) and the number of tokens that had a relation."""
    N = int(z.shape[0])
    n = torch.ones(N, dtype=torch.float32, device=z.device)
    starts, off = [], 0
    for c in counts:
        starts.append(off)
        off += int(c)
    related = 0
    for p, (gh_p, gw_p) in enumerate(grids):
        if boxes[p] is None or not mags[p]:
            continue
        lower = [q for q in range(len(grids)) if q != p and boxes[q] is not None and mags[q] and mags[q] < mags[p]]
        if not lower:
            continue
        centres = token_centers(boxes[p], gh_p, gw_p)
        zp = z[starts[p]:starts[p] + counts[p]]
        best = torch.full((int(counts[p]),), -1.0, dtype=torch.float32, device=z.device)
        for q in lower:
            xq, yq, wq, hq = (float(v) for v in boxes[q])
            gh_q, gw_q = grids[q]
            inside = (centres[:, 0] >= xq) & (centres[:, 0] < xq + wq) & (centres[:, 1] >= yq) & (centres[:, 1] < yq + hq)
            if not bool(inside.any()):
                continue
            idx = inside.nonzero(as_tuple=True)[0]
            r = ((centres[idx, 1] - yq) / hq * gh_q).floor().clamp(0, gh_q - 1).long()
            c = ((centres[idx, 0] - xq) / wq * gw_q).floor().clamp(0, gw_q - 1).long()
            j = starts[q] + r * gw_q + c
            s = (1.0 + (zp[idx.to(z.device)] * z[j.to(z.device)]).sum(dim=-1)) / 2.0
            best[idx.to(z.device)] = torch.maximum(best[idx.to(z.device)], s.float())
        has = best >= 0
        related += int(has.sum())
        seg = n[starts[p]:starts[p] + counts[p]]
        seg[has] = (1.0 - best[has]).clamp_min(0.0)
    return n, related


@torch.no_grad()
def logdet_greedy(b: torch.Tensor, budget: int):
    """Greedy max of log det(I + B_S B_S^T) over rows of b [N, d]. Returns (order LongTensor, gains list).

    Gain of i given S = log(d_i^2) with d_i^2 = 1 + |b_i|^2 - |c_i|^2 (incremental Cholesky of I + B B^T),
    identical to log(1 + b_i^T (I + B_S^T B_S)^{-1} b_i).
    """
    b = b.double() if b.device.type == "cpu" else b.float()
    N = int(b.shape[0])
    budget = max(0, min(int(budget), N))
    d2 = 1.0 + (b * b).sum(dim=1)
    C = torch.zeros(N, budget, dtype=b.dtype, device=b.device)
    chosen = torch.zeros(N, dtype=torch.bool, device=b.device)
    order, gains = [], []
    for t in range(budget):
        score = d2.masked_fill(chosen, float("-inf"))
        j = int(torch.argmax(score))
        dj2 = float(d2[j])
        order.append(j)
        gains.append(math.log(max(dj2, 1e-300)))
        chosen[j] = True
        if t == budget - 1:
            break
        dj = math.sqrt(max(dj2, 1e-300))
        kj = b @ b[j]                                   # [N] off-diagonal kernel column
        e = (kj - C[:, :t] @ C[j, :t]) / dj
        e[chosen] = 0.0
        C[:, t] = e
        d2 = d2 - e * e
    return torch.tensor(order, dtype=torch.long), gains


# --------------------------------------------------------------------------- runtime


@torch.no_grad()
def select_vision_features(bb, pixel_values, grid_thw):
    """Full vision tower + C1 #14 selection. Returns (retained merged features [B, d], keep mask [N])."""
    if os.environ.get("VLMAS_C1_SELECT", "").strip() == "nova":
        # C1 #21 NOVA (memory/nova_select.py): spatial-null mass e_i = 1 - b_i times the full-vision
        # morphology direction, support-wise log-det, no quota, at this same prefill site
        from memory import nova_select as _nova
        return _nova.select_vision_features(bb, pixel_values, grid_thw)
    if os.environ.get("VLMAS_C1_SELECT", "").strip() in ("ssda", "ntrs"):
        # C1 #14.8 SSDA / #14.6 NTRS (memory/ssda_select.py) at the same prefill site
        from memory import ssda_select as _ssda
        return _ssda.select_vision_features(bb, pixel_values, grid_thw)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    t0 = time.time()
    visual = bb.visual
    n_blocks = len(visual.blocks)
    layer = int(os.environ.get("VLMAS_QASC_VLAYER", "-1"))
    layer = layer % n_blocks
    attn = visual.blocks[layer].attn
    grabbed = {}

    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cu = kwargs.get("cu_seqlens")
        pos = kwargs.get("position_embeddings")
        if hidden is None or cu is None or pos is None:
            return None
        seq = hidden.shape[0]
        qkv = module.qkv(hidden).reshape(seq, 3, module.num_heads, -1).permute(1, 0, 2, 3)
        q, k, _ = qkv.unbind(0)
        cos, sin = pos
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        out = torch.zeros(seq, dtype=torch.float32, device=hidden.device)
        cuts = [int(v) for v in cu]
        for a, e in zip(cuts[:-1], cuts[1:]):
            if e > a:
                out[a:e] = received_attention(q[:, a:e], k[:, a:e], module.scaling)
        grabbed["a"] = out
        return None

    handle = attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        feats = bb.vlm.get_image_features(pixel_values, grid_thw, return_dict=True).pooler_output
    finally:
        handle.remove()
    feats = torch.cat(list(feats), dim=0) if isinstance(feats, (list, tuple)) else feats
    if "a" not in grabbed:
        raise RuntimeError("QASC: vision attention hook did not fire")
    unit = int(visual.spatial_merge_unit)
    merge = int(round(math.sqrt(unit)))
    a = merged_mean(grabbed["a"], unit)
    N = int(feats.shape[0])
    if int(a.numel()) != N:
        raise RuntimeError(f"QASC: salience {int(a.numel())} != features {N}")
    thw = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
    counts = [t * h * w // unit for t, h, w in thw]
    grids = [(h // merge, w // merge) for _, h, w in thw]
    z = F.normalize(feats.float(), dim=-1)
    mags = tuple(int(v) for v in (getattr(bb, "_prune_image_magnifications", ()) or ()))
    boxes = tuple(getattr(bb, "_prune_image_boxes", ()) or ())
    if len(mags) != len(counts):
        mags = (0,) * len(counts)
    if len(boxes) != len(counts):
        boxes = (None,) * len(counts)
    if os.environ.get("VLMAS_QASC_CROSS", "1") == "0":
        n_cross, related = torch.ones(N, dtype=torch.float32, device=z.device), 0
    else:
        n_cross, related = cross_scale_factor(z, counts, grids, mags, boxes)
    a_tilde = (a.to(z.device) * n_cross).clamp_min(0.0)
    keep_ratio = float(os.environ.get("VLMAS_QASC_KEEP", "0.25"))
    budget = max(1, math.ceil(N * keep_ratio))
    order, gains = logdet_greedy(a_tilde.sqrt().unsqueeze(1) * z, budget)
    keep = torch.zeros(N, dtype=torch.bool, device=feats.device)
    keep[order.to(feats.device)] = True
    image_of = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long))
    kept_cpu = keep.detach().cpu()
    bb._prefill_prune_survivor_scores = a_tilde.detach().float().cpu()[kept_cpu]
    bb._prefill_prune_survivor_group_ids = torch.nonzero(kept_cpu, as_tuple=True)[0]
    bb._prefill_prune_survivor_image_ids = image_of[kept_cpu]
    per_image = [int(kept_cpu[image_of == i].sum()) for i in range(len(counts))]
    sel = order.to(a_tilde.device)
    at_sel = a_tilde[sel].double()
    cover = [(math.exp(g) - 1.0) / float(x) for g, x in zip(gains, at_sel.tolist()) if float(x) > 0]
    print(
        f"[QASC] kept {int(keep.sum())}/{N} layer={layer} per_image={per_image} "
        f"mags={list(mags)} cross_related={related} n_cross_min={float(n_cross.min()):.3f} "
        f"sum_a_sel={float(at_sel.sum()):.4f} a_max/mean={float(a_tilde.max() / a_tilde.mean().clamp_min(1e-12)):.2f} "
        f"cover_first={(f'{cover[0]:.4f}' if cover else 'na')} cover_last={(f'{cover[-1]:.4f}' if cover else 'na')} "
        f"empty_crops={sum(1 for v in per_image if v == 0)} sec={time.time() - t0:.2f}",
        flush=True,
    )
    return feats[keep], keep


__all__ = ["received_attention", "merged_mean", "token_centers", "cross_scale_factor", "logdet_greedy",
           "select_vision_features"]
