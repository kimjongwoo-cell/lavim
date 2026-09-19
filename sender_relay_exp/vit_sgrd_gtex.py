#!/usr/bin/env python3
"""C1 #14.9 Spatial-Graph Rate-Distortion selection on nav3 GTEx crops, with and without removing the positional
component of the merger features (smoke, CPU).

Positional feature template (analogue of the #14.6 salience null template):
    n_{m,p}  = unit merger feature of content-free image m (white/black/gray/H&E pink/noise) at grid position p
    pos_p    = mean_m ( n_{m,p} - mean_p' n_{m,p'} )           (content constant of each image removed)
    per slide: x_i = z_i - mean_{crop} z,  beta = max(0, sum_i <x_i, pos_p(i)> / sum_i ||pos_p(i)||^2)
    z~_i     = normalize(z_i - beta * pos_p(i))
#14.9 (Notion 14.9): 4-neighbour grid graph per crop, edge cost 1 - <z_u, z_v>, all-pairs geodesic (Floyd-Warshall),
initial distortion = eccentricity, demand weight = #14.6 residual salience p, one global greedy on distortion reduction.

Outputs runs/vit_sgrd_gtex/: summary.json, <key>_spatial.png (smoke cases), <key>_sgrd.npz (selections), null_feats.npz
usage: vit_sgrd_gtex.py <out_dir> [--figs gtex_00,gtex_04,gtex_08,gtex_12] [--keep 0.25]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sender_relay_exp"))

import numpy as np  # noqa: E402

ARR = REPO / "sender_relay_exp" / "runs" / "vit_ntrs_gtex" / "arrays"
AUDIT = REPO / "sender_relay_exp" / "runs" / "vit_sal_audit" / "full"
DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
MODEL = Path("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")


# --------------------------------------------------------------------------- pure


def grid_edges(h: int = 16, w: int = 16):
    idx = np.arange(h * w).reshape(h, w)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return a, b


def geodesic(z: np.ndarray, h: int = 16, w: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """All-pairs shortest path on the 4-neighbour grid, edge cost 1 - <z_u, z_v> (Floyd-Warshall). Returns (D, costs)."""
    n = h * w
    a, b = grid_edges(h, w)
    cost = np.clip(1.0 - (z[a] * z[b]).sum(axis=1), 1e-9, None)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    D[a, b] = cost
    D[b, a] = cost
    for k in range(n):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D, cost


def positional_template(null_feats: np.ndarray) -> np.ndarray:
    """[m, P, d] content-free merger features -> [P, d] positional component."""
    f = null_feats / np.linalg.norm(null_feats, axis=-1, keepdims=True)
    return (f - f.mean(axis=1, keepdims=True)).mean(axis=0)


def remove_position(z: np.ndarray, counts, pos: np.ndarray) -> tuple[np.ndarray, float]:
    offs = np.cumsum([0] + list(counts))
    num = den = 0.0
    P = np.concatenate([pos for _ in counts])
    for g in range(len(counts)):
        x = z[offs[g]:offs[g + 1]] - z[offs[g]:offs[g + 1]].mean(axis=0, keepdims=True)
        num += float((x * pos).sum())
        den += float((pos * pos).sum())
    beta = max(0.0, num / den) if den > 0 else 0.0
    r = z - beta * P
    return r / np.linalg.norm(r, axis=1, keepdims=True), beta


def rd_greedy(dists, p, counts, budget):
    offs = np.cumsum([0] + list(counts))
    cur = [d.max(axis=1) for d in dists]
    D0 = [float((p[offs[g]:offs[g + 1]] * cur[g]).sum()) for g in range(len(counts))]
    gain = [(p[offs[g]:offs[g + 1]][:, None] * np.clip(cur[g][:, None] - dists[g], 0, None)).sum(axis=0) for g in range(len(counts))]
    chosen = np.zeros(int(offs[-1]), dtype=bool)
    order, first = [], {}
    for _ in range(budget):
        flat = np.concatenate(gain)
        flat[chosen] = -np.inf
        j = int(np.argmax(flat))
        g = int(np.searchsorted(offs, j, side="right") - 1)
        if g not in first:
            first[g] = float(flat[j] / max(D0[g], 1e-12))
        order.append(j)
        chosen[j] = True
        cur[g] = np.minimum(cur[g], dists[g][:, j - offs[g]])
        pg = p[offs[g]:offs[g + 1]]
        gain[g] = (pg[:, None] * np.clip(cur[g][:, None] - dists[g], 0, None)).sum(axis=0)
    return np.sort(np.array(order)), first, D0


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


# --------------------------------------------------------------------------- main


def null_features(out: Path) -> np.ndarray:
    f = out / "null_feats.npz"
    if f.exists():
        return np.load(f)["feats"]
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    rng = np.random.default_rng(0)
    imgs = [Image.new("RGB", (512, 512), c) for c in ((255, 255, 255), (0, 0, 0), (128, 128, 128), (230, 180, 210))]
    imgs.append(Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)))
    enc = proc.image_processor(images=imgs, return_tensors="pt")
    with torch.no_grad():
        feats = model.model.get_image_features(enc["pixel_values"].float(), enc["image_grid_thw"], return_dict=True).pooler_output
    feats = (torch.cat(list(feats), 0) if isinstance(feats, (list, tuple)) else feats).float().numpy().reshape(5, 256, -1)
    np.savez(f, feats=feats, names=np.array(["white", "black", "gray", "he_pink", "noise"]))
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--figs", default="gtex_00,gtex_04,gtex_08,gtex_12")
    ap.add_argument("--keep", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nf = null_features(out)
    pos = positional_template(nf)
    fn = nf / np.linalg.norm(nf, axis=-1, keepdims=True)
    print(f"[null] feats {nf.shape} {time.time() - t0:.0f}s; white cos adjacent median "
          f"{np.median((fn[0][grid_edges()[0]] * fn[0][grid_edges()[1]]).sum(1)):.3f}; "
          f"||pos|| median {np.median(np.linalg.norm(pos, axis=1)):.3f}", flush=True)

    b16 = None
    from memory import ssda_select as ss
    b16 = ss.load_null_b16()
    hot8 = np.argsort(-b16)[:8]
    hot_cells = [divmod(int(i), 16) for i in sorted(hot8.tolist())]
    figs = set(k for k in args.figs.split(",") if k)
    ea, eb = grid_edges()
    summary = {}
    edge = {"orig": {"bg": [], "tis": []}, "resid": {"bg": [], "tis": []}}
    for f in sorted(ARR.glob("*_arrays.npz")):
        key = f.name.replace("_arrays.npz", "")
        a = np.load(f)
        X = a["X"].astype(np.float64)
        z = X / np.linalg.norm(X, axis=1, keepdims=True)
        counts = a["counts"].tolist()
        mags = a["mags"].tolist()
        tissue = a["tissue"]
        p = a["w_page"]
        offs = np.cumsum([0] + counts)
        B = int(round(args.keep * int(offs[-1])))
        zr, beta = remove_position(z, counts, pos)
        names = [str(n) for n in a["sel_names"]]
        old = {n: np.asarray(a[f"sel_{i}"]) for i, n in enumerate(names)}
        fl = next(v for n, v in old.items() if n.startswith("#14.6 residual + support"))
        crop_t = [float(tissue[offs[g]:offs[g + 1]].mean()) for g in range(len(counts))]
        local = np.concatenate([np.arange(c) for c in counts])
        rec = {"beta": beta, "mags": mags, "crop_tissue": crop_t}
        sels = {}
        for tag, zz in (("orig", z), ("resid", zr)):
            dists, rough = [], []
            for g in range(len(counts)):
                zg = zz[offs[g]:offs[g + 1]]
                d, cost = geodesic(zg)
                dists.append(d)
                rough.append(float(cost.mean()))
                tg = tissue[offs[g]:offs[g + 1]]
                edge[tag]["bg"] += cost[(tg[ea] < 0.1) & (tg[eb] < 0.1)].tolist()
                edge[tag]["tis"] += cost[(tg[ea] > 0.5) & (tg[eb] > 0.5)].tolist()
            S, first, D0 = rd_greedy(dists, p, counts, B)
            pc = [int(((S >= offs[g]) & (S < offs[g + 1])).sum()) for g in range(len(counts))]
            sels[tag] = S
            rec[tag] = {"per_crop": pc, "sd": float(np.std(pc)), "rho_count_tissue": spearman(pc, crop_t),
                        "rho_count_rough": spearman(pc, rough), "rho_D0_background": spearman(D0, [1 - t for t in crop_t]),
                        "first_pick_frac": float(np.median(list(first.values()))), "overlap_14.6": len(set(S) & set(fl)) / B,
                        "dmin_orig_space": float((1 - (z @ z[S].T).max(axis=1)).mean()),
                        "hot8_share": float(np.isin(local[S], hot8).mean()), "tissue_sel": float(tissue[S].mean()),
                        "x5_share": float(sum(c for c, m in zip(pc, mags) if m == 5) / B)}
        rec["overlap_orig_resid"] = len(set(sels["orig"]) & set(sels["resid"])) / B
        summary[key] = rec
        np.savez(out / f"{key}_sgrd.npz", sel_orig=sels["orig"], sel_resid=sels["resid"], beta=beta)
        print(f"[{key}] beta={beta:.3f} orig sd={rec['orig']['sd']:.1f} rho_t={rec['orig']['rho_count_tissue']} | "
              f"resid sd={rec['resid']['sd']:.1f} rho_t={rec['resid']['rho_count_tissue']} tissue_sel={rec['resid']['tissue_sel']:.3f} "
              f"ov(orig,resid)={rec['overlap_orig_resid']:.2f}", flush=True)
        if key in figs:
            from vit_ntrs_gtex import draw
            from vision_text_mas.cli import _open_slide
            from vision_text_mas.dataset import load_cases
            from vision_text_mas.geometry import Box
            from vision_text_mas.navigation_render import render_box
            row = {r["key"]: r for r in json.loads((AUDIT / "manifest.json").read_text())}[key]
            result = json.loads(Path(row["result"]).read_text())
            case = load_cases(DATA / f"{row['ds']}.json", slide_root=DATA / "slides", indices=(int(row["dataset_index"]),))[0]
            slide = _open_slide(case)
            images = [render_box(slide, Box(**pp["box"]), max_side=256) for pp in result["patches"]]
            rows = {"Stratified (C1)": old["Stratified (C1)"], "Random": old["Random"],
                    f"#14.6 FL · sd {np.std([int(((fl >= offs[g]) & (fl < offs[g + 1])).sum()) for g in range(len(counts))]):.1f}": fl,
                    f"#14.9 RD (raw feature) · sd {rec['orig']['sd']:.1f}": sels["orig"],
                    f"#14.9 RD (position-removed, β={beta:.2f}) · sd {rec['resid']['sd']:.1f}": sels["resid"]}
            draw(key, images, mags, [(16, 16)] * len(counts), rows, hot_cells, out / f"{key}_spatial.png", dpi=80)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print("\n== adjacent edge cost (median): background-background vs tissue-tissue")
    for tag in ("orig", "resid"):
        print(f"  {tag:6s} bg {np.median(edge[tag]['bg']):.3f} (n={len(edge[tag]['bg'])})  tissue {np.median(edge[tag]['tis']):.3f} (n={len(edge[tag]['tis'])})")
    print("== #14.9 over cases")
    for tag in ("orig", "resid"):
        v = [s[tag] for s in summary.values()]
        pcs = [x for s in v for x in s["per_crop"]]
        med = lambda k: float(np.median([s[k] for s in v if s[k] is not None]))
        print(f"  {tag:6s} crop sd {np.mean([s['sd'] for s in v]):.1f} range {min(pcs)}-{max(pcs)} rho(count,tissue) {med('rho_count_tissue'):+.2f} "
              f"rho(count,rough) {med('rho_count_rough'):+.2f} rho(D0,bg) {med('rho_D0_background'):+.2f} first {med('first_pick_frac'):.2f} "
              f"ov14.6 {np.mean([s['overlap_14.6'] for s in v]):.3f} dmin {np.mean([s['dmin_orig_space'] for s in v]):.4f} "
              f"hot8 {np.mean([s['hot8_share'] for s in v]):.3f} tissue_sel {np.mean([s['tissue_sel'] for s in v]):.3f} x5 {np.mean([s['x5_share'] for s in v]):.3f}")
    print(f"  beta median {np.median([s['beta'] for s in summary.values()]):.3f}; overlap orig-resid {np.mean([s['overlap_orig_resid'] for s in summary.values()]):.3f}")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
