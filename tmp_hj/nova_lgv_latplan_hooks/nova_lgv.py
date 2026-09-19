"""C1 NOVA Final (Early-LGV L2) — Notion "NOVA Final — Early-Layer Local Structural Salience + Support-wise LogDet".

Replaces only memory.secld_select.evidence (the e_i generator). Everything else is NOVA as run on the dashboard
(memory/nova_select.py + secld_select.select: q_i = sqrt(e_i) z_i with z_i = merger output, support-wise log-det,
exact greedy, one global budget, no quota).

  z~_i = normalize( mean_{r in P(i)} h_r^(2) )       vision block index 2 output, raw 2x2 patch mean (page §3.1)
  s_i  = (1/|N4(i)|) sum_{j in N4(i)} ||z~_i - z~_j||^2   4-neighbour, in-crop, border tokens use existing neighbours
  e_i  = s_i / mean_{j in query} s_j   per-query mean normalisation (page §11.5 decision, 09-19); raw s kept only as ablation
h^(2) is captured by a forward hook on visual.blocks[2] during the same full-vision forward that NOVA already runs
(registered by the import hook around nova_select.select_vision_features). No extra forward.
Log: [NOVALGV] n= s_mean(raw)= s_q10= s_q50= s_q90= e_q10= e_q90= crops= norm=mean sec=
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F

STORE = {"h2": None, "v6": None}
import os as _os
SIGNAL = _os.environ.get("VLMAS_NOVA_SIGNAL", "lgv").strip() or "lgv"   # lgv | lgvclip | vn6g2 | vn6n1


def capture(bb) -> None:
    """Register (once) a forward hook on the vision block 2 of backbone bb."""
    vis = bb.visual
    if getattr(vis, "_nova_lgv_hooked", False):
        return
    def hook(mod, args, out):
        STORE["h2"] = (out[0] if isinstance(out, tuple) else out).detach()
    vis.blocks[2].register_forward_hook(hook)
    def vhook(mod, args, kwargs, out):
        hs = kwargs.get("hidden_states", args[0] if args else None); L = hs.shape[0]
        v = mod.qkv(hs).reshape(L, 3, mod.num_heads, -1)[:, 2]          # attention value per head
        STORE["v6"] = v.float().norm(dim=-1).mean(1).detach()            # head-mean ||v_p||
    vis.blocks[6].attn.register_forward_hook(vhook, with_kwargs=True)
    vis._nova_lgv_hooked = True
    print(f"[NOVALGV] hooked visual.blocks[2] + blocks[6].attn (signal={SIGNAL})", flush=True)


def _lgv_grid(z: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    g = z.view(gh, gw, -1)
    s = torch.zeros(gh, gw, device=z.device, dtype=torch.float32)
    c = torch.zeros(gh, gw, device=z.device, dtype=torch.float32)
    # right / left / down / up neighbours, only where they exist
    d = ((g[:, 1:] - g[:, :-1]) ** 2).sum(-1)
    s[:, :-1] += d; c[:, :-1] += 1; s[:, 1:] += d; c[:, 1:] += 1
    d = ((g[1:] - g[:-1]) ** 2).sum(-1)
    s[:-1] += d; c[:-1] += 1; s[1:] += d; c[1:] += 1
    return (s / c.clamp_min(1)).reshape(-1)


@torch.no_grad()
def evidence_lgv(pixel_values, grid_thw, cfg=None, *, merge: int = 2, patch: int = 16, temporal: int = 2) -> dict:
    t0 = time.time()
    h = STORE["h2"]
    if h is None:
        raise RuntimeError("NOVALGV: block-2 hidden not captured")
    thw = [tuple(int(v) for v in row) for row in torch.as_tensor(grid_thw).tolist()]
    unit = merge * merge
    P = sum(t * hh * ww for t, hh, ww in thw)
    if int(h.shape[0]) != P:
        raise RuntimeError(f"NOVALGV: captured {int(h.shape[0])} patches != grid {P}")
    pooled = F.normalize(h.float().view(P // unit, unit, -1).mean(1), dim=-1)
    out, grids, off = [], [], 0
    for t, hh, ww in thw:
        gh, gw = hh // merge, ww // merge
        for _ in range(t):
            n = gh * gw
            out.append(_lgv_grid(pooled[off:off + n], gh, gw)); grids.append((gh, gw)); off += n
    s = torch.cat(out)
    if SIGNAL in ("vn6g2", "vn6n1"):
        # NOVA rho form with rho replaced by (1 - value norm): v^ = query min-max of block-6 head-mean ||v||,
        # b = in-crop-normalised Gaussian(sigma 2) of (1 - v^)  (memory/nova_rho_select.local_occupancy), e = 1 - b
        from memory import nova_rho_select as _rho
        v = STORE["v6"]
        if v is None or int(v.shape[0]) != P:
            raise RuntimeError("NOVALGV vn6g2: block-6 value norm not captured")
        vt = v.float().view(P // unit, unit).mean(1)
        vh = (vt - vt.min()) / (vt.max() - vt.min()).clamp_min(1e-12)
        if SIGNAL == "vn6g2":
            b = _rho.local_occupancy((1.0 - vh).to(torch.float32), grids, 2.0)
        else:  # vn6n1: first-order lattice adjacency incl. self, b_i = mean_{j in {i} U N4(i)} (1 - v^_j)
            nb, o2 = [], 0
            for gh, gw in grids:
                n = gh * gw; x = (1.0 - vh[o2:o2 + n]).view(1, 1, gh, gw)
                k = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=x.device, dtype=x.dtype)[None, None]
                num = F.conv2d(F.pad(x, (1, 1, 1, 1)), k); den = F.conv2d(F.pad(torch.ones_like(x), (1, 1, 1, 1)), k)
                nb.append((num / den).reshape(-1)); o2 += n
            b = torch.cat(nb)
        e = (1.0 - b).clamp(0.0, 1.0)
        s = vh
    else:
        e = s / s.mean().clamp_min(1e-12)
        if SIGNAL == "lgvclip":
            e = e.clamp(max=1.0)
    zeros = torch.zeros_like(s)
    q = torch.quantile(s, torch.tensor([0.1, 0.5, 0.9], device=s.device))
    qe = torch.quantile(e, torch.tensor([0.1, 0.9], device=e.device))
    print(f"[NOVALGV] n={int(s.numel())} s_mean={float(s.mean()):.4f} s_q10={float(q[0]):.4f} s_q50={float(q[1]):.4f} "
          f"s_q90={float(q[2]):.4f} e_q10={float(qe[0]):.3f} e_q90={float(qe[1]):.3f} crops={len(grids)} norm=mean signal={SIGNAL} e_mean={float(e.mean()):.3f} "
          f"sec={time.time() - t0:.3f}", flush=True)
    STORE["h2"] = None; STORE["v6"] = None
    return {"r": zeros, "m": zeros, "b": zeros, "e": e, "grids": grids}


# ---------------------------------------------------------------------------------------------------------------
# C1 merge (09-19): keep NOVA's selection, then fold every dropped token into its most similar kept token of the
# SAME crop (cosine of normalised merger outputs z). f~_k = ||f_k|| * normalize( (f_k + sum_{p->k} f_p) / (1+n_k) ):
# ToMe-style size-weighted mean, rescaled to the kept token's own norm (stays in the merger's output distribution).
# Token count / positions / KV size unchanged (25%). Env VLMAS_NOVA_MERGE=1.
@torch.no_grad()
def merge_dropped(feats: torch.Tensor, keep: torch.Tensor, counts) -> tuple:
    f = feats.float(); z = F.normalize(f, dim=-1)
    out = f[keep].clone()
    kept_idx = torch.nonzero(keep, as_tuple=True)[0]
    pos_in_out = torch.full((f.shape[0],), -1, dtype=torch.long, device=f.device)
    pos_in_out[kept_idx] = torch.arange(kept_idx.numel(), device=f.device)
    acc = out.clone(); cnt = torch.ones(kept_idx.numel(), device=f.device)
    off, merged, sims = 0, 0, []
    for c in counts:
        sl = torch.arange(off, off + int(c), device=f.device); off += int(c)
        k = sl[keep[sl]]; p = sl[~keep[sl]]
        if k.numel() == 0 or p.numel() == 0:
            continue
        s = z[p] @ z[k].T                                  # [P_g, K_g]
        best = s.argmax(1); sims.append(s.max(1).values)
        tgt = pos_in_out[k[best]]
        acc.index_add_(0, tgt, f[p]); cnt.index_add_(0, tgt, torch.ones_like(tgt, dtype=cnt.dtype))
        merged += int(p.numel())
    mean = acc / cnt[:, None]
    res = F.normalize(mean, dim=-1) * out.norm(dim=-1, keepdim=True)
    sim = torch.cat(sims) if sims else torch.zeros(1, device=f.device)
    print(f"[NOVAMERGE] kept={int(kept_idx.numel())} merged={merged} per_kept_mean={float(cnt.mean()) - 1:.2f} "
          f"per_kept_max={int(cnt.max()) - 1} assign_cos_mean={float(sim.mean()):.3f} assign_cos_q10={float(torch.quantile(sim, 0.1)):.3f} "
          f"shift_cos={float(F.cosine_similarity(res, out, dim=-1).mean()):.3f}", flush=True)
    return res.to(feats.dtype)
