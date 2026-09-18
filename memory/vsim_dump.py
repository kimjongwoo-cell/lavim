"""Persistent-cache group similarity dump (C1 kill test: does V-similarity predict
pruning replaceability?).  VLMAS_RETR=igsweep VLMAS_RETR_VSIM=1 [VLMAS_RETR_VSIM_ONLY=1]

Per case, on the pre-Answerer cache, for the 5x-root groups g (root + its 20x
children) we mean-pool the cached states over the group's visual columns at
every LM layer, L2-normalise, and take the layer-average cosine between groups:

    sim_X(g,h) = (1/L) sum_l cos( mean_{i in g} X_i^l , mean_{j in h} X_j^l )

for X in {V (values), K (keys as stored, post-RoPE), Kd (keys de-rotated to
pre-RoPE with their stored 3-D positions, content-side), E (vision-encoder
output after the merger, i.e. the embeddings scattered into the Reasoner
prefill; single "layer")}.  R_g = max_{h != g} sim_X(g,h) is compared post hoc
with the group-removal utility I_g of the same case (canonical igsweep).
Off path: nothing runs unless VLMAS_RETR_VSIM=1.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from memory.restage import rerotate_keys


@torch.no_grad()
def _group_means(x, gidx):
    """x: [n, D] float; gidx: list of LongTensor (local indices) -> [G, D] L2-normalised."""
    m = torch.stack([x.index_select(0, g).mean(0) for g in gidx])
    return F.normalize(m, dim=-1)


@torch.no_grad()
def run(engine, *, cache, case_index: int) -> None:
    from memory import retrieval_group as _rg
    bb = engine._backbone
    cols_all = getattr(engine, "_rpath_visual_cols", None)
    rc = getattr(bb, "_route_vis_cols", None)
    pos = getattr(bb, "_route_vis_pos", None)
    pages = getattr(bb, "_route_pages", None)
    parents = tuple(getattr(bb, "_prune_image_parent_indices", ()) or ())
    feat = getattr(bb, "_retr_img_feat", None)
    if cols_all is None or not pages or sum(int(p) for p in pages) != int(cols_all.numel()):
        print(f"[VSim] case {case_index}: SKIP (cols={cols_all is not None} pages={bool(pages)})", flush=True)
        return
    cols = cols_all.to(bb.device)
    par, roots, groups, crop_of, starts = _rg.groups_from_tree(pages, parents)
    gidx = []
    for r in roots:
        idx = []
        for p in groups[r]:
            idx += list(range(starts[p], starts[p] + int(pages[p])))
        gidx.append(torch.tensor(idx, dtype=torch.long, device=bb.device))
    G = len(roots)
    layers = bb.lm.layers
    simV = torch.zeros(G, G); simK = torch.zeros(G, G); simKd = torch.zeros(G, G)
    per_layer_V = []
    cos_old = sin_old = None
    if pos is not None and rc is not None and int(rc.numel()) == int(cols.numel()):
        sample = cache.layers[0].keys
        cos_old, sin_old = bb.lm.rotary_emb(sample, pos.to(bb.device).unsqueeze(1))
        cos_old, sin_old = cos_old.float(), -sin_old.float()
    for li in range(len(layers)):
        v = cache.layers[li].values.index_select(2, cols)[0].float()          # [Hkv, n, Hd]
        k = cache.layers[li].keys.index_select(2, cols)[0].float()
        n = v.shape[1]
        vt = v.permute(1, 0, 2).reshape(n, -1)                                 # [n, Hkv*Hd]
        kt = k.permute(1, 0, 2).reshape(n, -1)
        mv = _group_means(vt, gidx); mk = _group_means(kt, gidx)
        sv = mv @ mv.T; sk = mk @ mk.T
        simV += sv.cpu(); simK += sk.cpu(); per_layer_V.append([round(float(x), 4) for x in sv.flatten().tolist()])
        if cos_old is not None:
            kd = rerotate_keys(k.unsqueeze(0), cos_old, sin_old)[0]             # pre-RoPE
            kdt = kd.permute(1, 0, 2).reshape(n, -1)
            mkd = _group_means(kdt, gidx); simKd += (mkd @ mkd.T).cpu()
    L = len(layers)
    simV /= L; simK /= L; simKd /= L
    simE = None
    if feat is not None and int(feat.shape[0]) == int(cols.numel()):
        me = _group_means(feat.to(bb.device).float(), gidx); simE = (me @ me.T).cpu()
    out = {"case": int(case_index), "roots": [int(r) for r in roots],
           "sizes": [int(g.numel()) for g in gidx], "n_layers": L,
           "simV": [[round(float(x), 5) for x in row] for row in simV.tolist()],
           "simK": [[round(float(x), 5) for x in row] for row in simK.tolist()],
           "simKd": [[round(float(x), 5) for x in row] for row in simKd.tolist()] if cos_old is not None else None,
           "simE": [[round(float(x), 5) for x in row] for row in simE.tolist()] if simE is not None else None,
           "simV_layers": per_layer_V}
    path = os.environ.get("VLMAS_RETR_VSIM_OUT", "").strip() or os.environ.get("VLMAS_RETR_OUT", "").strip()
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(out) + "\n")
    rv = [max(float(simV[g, h]) for h in range(G) if h != g) for g in range(G)]
    print(f"[VSim] case {case_index} roots={roots} sizes={out['sizes']} R_V={[round(x, 3) for x in rv]} "
          f"simE={'yes' if simE is not None else 'no'} simKd={'yes' if cos_old is not None else 'no'}", flush=True)


__all__ = ["run"]
