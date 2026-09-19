#!/usr/bin/env python3
"""C1 #14.6 (Null-Template Residual Salience + Physical-Support Constrained Morphology Coverage) kept-token maps
and allocation diagnostics on the nav3 GTEx crops (12 crops = 4 x5 + 8 x20 per case), CPU fp32.

Reuses the salience audit (runs/vit_sal_audit/full: per_block received attention, tissue, boxes of the crops nav3
actually picked) and the content-free block-23 maps (runs/vit_nullcal/null_maps.npz). Re-renders the crops once to
get the merged vision embeddings z (vision tower forward only; no LLM).

Rows: Stratified (C1) | Random | Top-k ViT block-23 raw a | #14 QASC | #14.6 residual top-k | #14.6 full.
Why-diagnostics (allocation across crops under the single global budget):
  V0 page          w = support softmax of (x - gamma b), s = (1 + cos) / 2
  V1 s = max(cos,0) same w
  V2 coverage only w = 1 / N_g,   s = (1 + cos) / 2
  V3 slide-level w w = exp(x - gamma b) / sum over ALL tokens (no per-support normalisation), s = (1 + cos) / 2
  V4 raw salience  w = a / sum_g a (no residual),  s = (1 + cos) / 2

usage: vit_ntrs_gtex.py <out_dir> [--ds gtex] [--keep 0.25] [--threads 16]
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
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vit_sal_stats import crop_spans, spearman  # noqa: E402

DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
MODEL = Path("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
AUDIT = REPO / "sender_relay_exp" / "runs" / "vit_sal_audit" / "full"
NULL = REPO / "sender_relay_exp" / "runs" / "vit_nullcal" / "null_maps.npz"
EPS = 1e-3
BLOCK = 23


# --------------------------------------------------------------------------- pure


def centred_log(maps, eps: float = EPS) -> np.ndarray:
    lg = np.log(np.asarray(maps, dtype=float) + eps)
    return lg - lg.mean(axis=-1, keepdims=True)


def fit_gamma(x_crops, b: np.ndarray) -> float:
    return max(0.0, float(sum(float(x @ b) for x in x_crops) / (len(x_crops) * float(b @ b))))


def support_facility_location(z: np.ndarray, w: np.ndarray, counts, k: int, sim: str = "half"):
    """Greedy max of sum_g sum_{i in g} w_i max_{j in S cap g} s_ij, |S| <= k. Returns (order, gains)."""
    offs = np.cumsum([0] + list(counts))
    sims, cov, gains = [], [], []
    for g in range(len(counts)):
        zg = z[offs[g]:offs[g + 1]]
        c = zg @ zg.T
        s = (1.0 + c) / 2.0 if sim == "half" else np.clip(c, 0.0, None)
        sims.append(s)
        cov.append(np.zeros(len(zg)))
        gains.append((w[offs[g]:offs[g + 1]][:, None] * s).sum(axis=0))
    chosen = np.zeros(int(offs[-1]), dtype=bool)
    order, got = [], []
    for _ in range(min(k, int(offs[-1]))):
        flat = np.concatenate(gains)
        flat[chosen] = -np.inf
        j = int(np.argmax(flat))
        g = int(np.searchsorted(offs, j, side="right") - 1)
        order.append(j)
        got.append(float(flat[j]))
        chosen[j] = True
        cov[g] = np.maximum(cov[g], sims[g][:, j - offs[g]])
        wg = w[offs[g]:offs[g + 1]]
        gains[g] = (wg[:, None] * np.clip(sims[g] - cov[g][:, None], 0.0, None)).sum(axis=0)
    return np.array(order), got


def per_crop_counts(S, counts):
    offs = np.cumsum([0] + list(counts))
    S = np.asarray(S)
    return [int(((S >= offs[g]) & (S < offs[g + 1])).sum()) for g in range(len(counts))]


def draw(case, images, mags, grids, sels, hot_cells, out_png, dpi=80):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    counts = [gh * gw for gh, gw in grids]
    offs = np.cumsum([0] + counts)
    names = list(sels)
    hot_idx = [r * 16 + c for r, c in hot_cells]
    fig, axes = plt.subplots(1 + len(names), len(images), figsize=(2.3 * len(images), 2.45 * (1 + len(names))))

    def outline(ax, im, gh, gw):
        cw, ch = im.size[0] / gw, im.size[1] / gh
        for r, c in hot_cells:
            ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, fill=False, edgecolor="#00d5ff", linewidth=1.3))

    for ii, im in enumerate(images):
        gh, gw = grids[ii]
        ax = axes[0, ii]
        ax.imshow(im)
        outline(ax, im, gh, gw)
        ax.set_title(f"P{ii + 1} x{mags[ii]} ({gh}×{gw})", fontsize=8)
        ax.axis("off")
        for r, name in enumerate(names, start=1):
            ax = axes[r, ii]
            mask = np.zeros(counts[ii])
            S = np.asarray(sels[name])
            mask[S[(S >= offs[ii]) & (S < offs[ii + 1])] - offs[ii]] = 1.0
            ax.imshow(im, alpha=0.55)
            ax.imshow(mask.reshape(gh, gw), extent=(0, im.size[0], im.size[1], 0), cmap="Reds", alpha=0.6,
                      vmin=0, vmax=1, interpolation="nearest")
            outline(ax, im, gh, gw)
            ax.axis("off")
            hk = int(mask[hot_idx].sum())
            if ii == 0:
                ax.set_title(f"{name}   [{int(mask.sum())}/{counts[ii]} · hot {hk}/8]", fontsize=7.5, loc="left")
            else:
                ax.set_title(f"{int(mask.sum())}/{counts[ii]} · hot {hk}/8", fontsize=7)
    fig.suptitle(f"{case}: kept tokens per crop (red), k=25% · cyan = hot 8 cells (block-23 null template)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ds", default="gtex")
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--arrays-only", action="store_true", help="save z/selections per case for the 2-D (UMAP) figure; skip plots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from memory.qasc_select import cross_scale_factor, logdet_greedy
    from memory.stratified import random_keep_mask, stratified_keep_mask
    from vision_text_mas.cli import _open_slide
    from vision_text_mas.dataset import load_cases
    from vision_text_mas.geometry import Box
    from vision_text_mas.navigation_render import render_box

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    null = np.load(NULL)["maps"][:, BLOCK]
    b = centred_log(null).mean(axis=0)
    hot8 = np.argsort(-b)[:8]
    hot_cells = [divmod(int(i), 16) for i in sorted(hot8.tolist())]

    manifest = [r for r in json.loads((AUDIT / "manifest.json").read_text())
                if r["ds"] == args.ds and r.get("status") in ("ok", "cached")]
    summary, why_rows = {}, []
    for row in manifest:
        t0 = time.time()
        z_npz = np.load(AUDIT / "cases" / f"{row['key']}.npz")
        pb, thw, tissue, mags = z_npz["per_block"][BLOCK], z_npz["thw"], z_npz["tissue"], [int(m) for m in z_npz["mags"]]
        result = json.loads(Path(row["result"]).read_text())
        case = load_cases(DATA / f"{row['ds']}.json", slide_root=DATA / "slides", indices=(int(row["dataset_index"]),))[0]
        slide = _open_slide(case)
        images = [render_box(slide, Box(**p["box"]), max_side=512) for p in result["patches"]]
        boxes = [[int(p["box"]["x"]), int(p["box"]["y"]), int(p["box"]["width"]), int(p["box"]["height"])] for p in result["patches"]]
        enc = proc.image_processor(images=images, return_tensors="pt")
        assert torch.equal(enc["image_grid_thw"], torch.from_numpy(thw)), (enc["image_grid_thw"], thw)
        with torch.no_grad():
            feats = model.model.get_image_features(enc["pixel_values"].float(), enc["image_grid_thw"], return_dict=True).pooler_output
        X = (torch.cat(list(feats), dim=0) if isinstance(feats, (list, tuple)) else feats).float().numpy()
        spans = crop_spans(thw)
        grids = [(sp[2], sp[3]) for sp in spans]
        counts = [sp[1] - sp[0] for sp in spans]
        offs = np.cumsum([0] + counts)
        N = int(offs[-1])
        assert X.shape[0] == N == pb.shape[0], (X.shape, N, pb.shape)
        k = int(round(args.keep * N))
        z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        a_u = [pb[offs[g]:offs[g + 1]] * counts[g] for g in range(len(counts))]
        x_crops = [centred_log(a) for a in a_u]
        gamma = fit_gamma(x_crops, b)
        resid = [np.exp(x - gamma * b) for x in x_crops]
        w_page = np.concatenate([r / r.sum() for r in resid])
        w_slide = np.concatenate(resid) / sum(r.sum() for r in resid)
        w_uni = np.concatenate([np.full(c, 1.0 / c) for c in counts])
        w_raw = np.concatenate([a / a.sum() for a in a_u])

        n_cross, related = cross_scale_factor(torch.from_numpy(z).float(), counts, grids, mags, boxes)
        a_q = torch.from_numpy(pb / 4.0).float() * n_cross
        q_order, _ = logdet_greedy(a_q.clamp_min(0).sqrt().unsqueeze(1).double() * torch.from_numpy(z).double(), k)
        fl_order, fl_gains = support_facility_location(z, w_page, counts, k)

        sels = {
            "Stratified (C1)": np.flatnonzero(stratified_keep_mask(tuple(counts), tuple(grids), k).numpy()),
            "Random": np.flatnonzero(random_keep_mask(N, k, 0).numpy()),
            "Top-k ViT block-23 raw a": np.sort(np.argsort(-np.concatenate(a_u))[:k]),
            "#14 QASC (raw a · log-det)": np.sort(q_order.numpy()),
            f"#14.6 residual top-k (γ̂={gamma:.2f})": np.sort(np.argsort(-w_page)[:k]),
            "#14.6 residual + support coverage": np.sort(fl_order),
        }
        if args.arrays_only:
            np.savez_compressed(out / f"{row['key']}_arrays.npz", X=X.astype(np.float16), mags=np.array(mags),
                                counts=np.array(counts), a_raw=np.concatenate(a_u), w_page=w_page, gamma=gamma,
                                hot8=hot8, tissue=tissue, sel_names=np.array(list(sels)),
                                **{f"sel_{i}": S for i, S in enumerate(sels.values())})
            print(f"[{row['key']}] arrays saved {time.time() - t0:.0f}s", flush=True)
            continue
        draw(row["key"], images, mags, grids, sels, hot_cells, out / f"{row['key']}_spatial.png")

        # tissue per crop from the audit (same rule) and within-crop morphology spread
        crop_tissue = [float(tissue[offs[g]:offs[g + 1]].mean()) for g in range(len(counts))]
        cos_in = [z[offs[g]:offs[g + 1]] @ z[offs[g]:offs[g + 1]].T for g in range(len(counts))]
        hetero = [float(1.0 - c[np.triu_indices(len(c), 1)].mean()) for c in cos_in]
        cos_all = np.concatenate([c[np.triu_indices(len(c), 1)] for c in cos_in])

        rec = {"k": k, "gamma": round(gamma, 3), "cross_related": int(related), "crop_tissue": crop_tissue,
               "crop_hetero": hetero, "mags": mags,
               "within_crop_cos_pct": [round(float(v), 3) for v in np.percentile(cos_all, [5, 25, 50, 75, 95])],
               "methods": {}}
        local = np.concatenate([np.arange(c) for c in counts])
        for name, S in sels.items():
            pc = per_crop_counts(S, counts)
            rec["methods"][name] = {"per_crop": pc, "hot8_share": round(float(np.isin(local[S], hot8).mean()), 3),
                                    "tissue_sel": round(float(tissue[S].mean()), 3)}

        # why: allocation under variants + gain trajectory of the page variant
        fl_pc = per_crop_counts(fl_order, counts)
        first_gain, per_crop_gain_at = [], []
        seen = {}
        for t, (j, gval) in enumerate(zip(fl_order, fl_gains)):
            g = int(np.searchsorted(offs, j, side="right") - 1)
            seen[g] = seen.get(g, 0) + 1
            if seen[g] == 1:
                first_gain.append(gval)
        variants = {"V0 page": (w_page, "half"), "V1 s=max(cos,0)": (w_page, "clip"), "V2 coverage only": (w_uni, "half"),
                    "V3 slide-level w": (w_slide, "half"), "V4 raw a (no residual)": (w_raw, "half")}
        vres = {}
        for vname, (wv, simk) in variants.items():
            o, gv = (fl_order, fl_gains) if vname == "V0 page" else support_facility_location(z, wv, counts, k, sim=simk)
            pc = per_crop_counts(o, counts)
            vres[vname] = {"per_crop": pc, "sd": float(np.std(pc)), "range": [min(pc), max(pc)],
                           "x5_share": float(sum(c for c, m in zip(pc, mags) if m == 5) / k),
                           "rho_count_hetero": spearman(pc, hetero), "rho_count_tissue": spearman(pc, crop_tissue),
                           "last_gain": float(gv[-1]), "first_gain_mean": None}
        vres["V0 page"]["first_gain_mean"] = float(np.mean(first_gain))
        vres["V0 page"]["total_value_after_first_picks"] = float(np.sum(first_gain))
        vres["V0 page"]["objective_total"] = float(np.sum(fl_gains))
        rec["why"] = vres
        summary[row["key"]] = rec
        print(f"[{row['key']}] {time.time() - t0:.0f}s gamma={gamma:.3f} related={related} cos50={rec['within_crop_cos_pct'][2]} "
              + " | ".join(f"{v}: sd {r['sd']:.1f} {r['range']} ρhet {r['rho_count_hetero'] if r['rho_count_hetero'] is None else round(r['rho_count_hetero'], 2)}"
                           for v, r in vres.items()), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    if args.arrays_only:
        return
    # aggregate
    print("\n== aggregate over cases")
    for v in next(iter(summary.values()))["why"]:
        rs = [s["why"][v] for s in summary.values()]
        rh = [r["rho_count_hetero"] for r in rs if r["rho_count_hetero"] is not None]
        rt = [r["rho_count_tissue"] for r in rs if r["rho_count_tissue"] is not None]
        print(f"{v:24s} crop-count sd {np.mean([r['sd'] for r in rs]):5.1f} min {min(r['range'][0] for r in rs):3d} max {max(r['range'][1] for r in rs):3d} "
              f"x5 share {np.mean([r['x5_share'] for r in rs]):.3f} ρ(count,hetero) {np.median(rh) if rh else float('nan'):+.2f} "
              f"ρ(count,tissue) {np.median(rt) if rt else float('nan'):+.2f} last gain {np.median([r['last_gain'] for r in rs]):.2e}")
    v0 = [s["why"]["V0 page"] for s in summary.values()]
    print(f"V0: first-pick gain per crop mean {np.mean([r['first_gain_mean'] for r in v0]):.3f} ; value from the 12 first picks "
          f"{np.mean([r['total_value_after_first_picks'] for r in v0]):.3f} of total {np.mean([r['objective_total'] for r in v0]):.3f} (max possible 12)")
    print("within-crop cos percentiles (5/25/50/75/95), case median:",
          np.median(np.array([s["within_crop_cos_pct"] for s in summary.values()]), axis=0).round(3).tolist())
    for name in next(iter(summary.values()))["methods"]:
        ms = [s["methods"][name] for s in summary.values()]
        pcs = np.array([m["per_crop"] for m in ms])
        mg = np.array([s["mags"] for s in summary.values()])
        print(f"{name[:40]:40s} per-crop sd {pcs.std(axis=1).mean():5.1f} range {pcs.min()}-{pcs.max()} x5 share {pcs[mg == 5].sum() / pcs.sum():.3f} "
              f"hot8 {np.mean([m['hot8_share'] for m in ms]):.3f} tissue_sel {np.mean([m['tissue_sel'] for m in ms]):.3f}")


if __name__ == "__main__":
    main()
