#!/usr/bin/env python3
"""C1 #14.8 SSDA offline on nav3 GTEx crops: kept-token maps (same rows as vit_ntrs_gtex + #14.8), allocation stats,
and per-case selection files for the 2-D (UMAP) view. Uses memory/ssda_select.select — the runtime code path.

Inputs: runs/vit_ntrs_gtex/arrays/<key>_arrays.npz (X merger output fp16, a_raw block-23 uniform-scaled, counts, mags,
tissue, saved selections), memory/ssda_null_block23.npz, audit manifest (crop boxes for thumbnails).
Outputs: runs/vit_ssda_gtex/<key>_spatial.png, <key>_ssda.npz, summary.json

usage: vit_ssda_gtex.py <out_dir> [--keep 0.25]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sender_relay_exp"))

import numpy as np  # noqa: E402

from memory import ssda_select as ss  # noqa: E402

ARR = REPO / "sender_relay_exp" / "runs" / "vit_ntrs_gtex" / "arrays"
AUDIT = REPO / "sender_relay_exp" / "runs" / "vit_sal_audit" / "full"
DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--keep", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from vit_ntrs_gtex import draw
    from vision_text_mas.cli import _open_slide
    from vision_text_mas.dataset import load_cases
    from vision_text_mas.geometry import Box
    from vision_text_mas.navigation_render import render_box

    manifest = {r["key"]: r for r in json.loads((AUDIT / "manifest.json").read_text())}
    b16 = ss.load_null_b16()
    hot8 = np.argsort(-b16)[:8]
    hot_cells = [divmod(int(i), 16) for i in sorted(hot8.tolist())]
    summary = {}
    for f in sorted(ARR.glob("*_arrays.npz")):
        key = f.name.replace("_arrays.npz", "")
        a = np.load(f)
        X = a["X"].astype(np.float64)
        z = X / np.linalg.norm(X, axis=1, keepdims=True)
        counts = a["counts"].tolist()
        mags = a["mags"].tolist()
        grids = [(16, 16)] * len(counts)
        tissue = a["tissue"]
        offs = np.cumsum([0] + counts)
        N = int(offs[-1])
        B = int(round(args.keep * N))
        keep, info = ss.select(z, a["a_raw"], counts, grids, b16, B)
        # the user's unweighted variant of the allocation (uniform p) for comparison
        spec_u = [ss.weighted_spectrum(z[offs[g]:offs[g + 1]], np.full(counts[g], 1.0 / counts[g])) for g in range(len(counts))]
        k_uniform, _ = ss.water_fill(spec_u, B, counts)

        names = [str(n) for n in a["sel_names"]]
        old = {n: np.asarray(a[f"sel_{i}"]) for i, n in enumerate(names)}
        rows = {}
        for n in ("Stratified", "Random", "#14 QASC", "#14.6 residual + support"):
            m = [nn for nn in old if nn.startswith(n)]
            if m:
                rows[m[0]] = old[m[0]]
        rows["#14.8 SSDA (spectral k_g + FL)"] = keep

        crop_tissue = [float(tissue[offs[g]:offs[g + 1]].mean()) for g in range(len(counts))]
        local = np.concatenate([np.arange(c) for c in counts])
        stats = {}
        for n, S in rows.items():
            S = np.asarray(S)
            pc = [int(((S >= offs[g]) & (S < offs[g + 1])).sum()) for g in range(len(counts))]
            stats[n] = {"per_crop": pc, "sd": float(np.std(pc)), "x5_share": float(sum(c for c, m in zip(pc, mags) if m == 5) / len(S)),
                        "hot8_share": float(np.isin(local[S], hot8).mean()), "tissue_sel": float(tissue[S].mean()),
                        "mean_dmin": float((1 - (z @ z[S].T).max(axis=1)).mean()),
                        "rho_count_tissue": spearman(pc, crop_tissue), "rho_count_trace": spearman(pc, info["trace"])}
        fl = next(v for n, v in rows.items() if n.startswith("#14.6"))
        stats["#14.8 SSDA (spectral k_g + FL)"]["overlap_14.6"] = len(set(keep.tolist()) & set(np.asarray(fl).tolist())) / B
        summary[key] = {"B": B, "gamma": info["gamma"], "tau": info["tau"], "k": info["k"], "k_uniform_p": k_uniform,
                        "trace": info["trace"], "energy_kept": info["energy_kept"], "mags": mags, "crop_tissue": crop_tissue,
                        "stats": stats}
        np.savez(out / f"{key}_ssda.npz", sel=keep, k=np.array(info["k"]), k_uniform_p=np.array(k_uniform))

        row = manifest[key]
        result = json.loads(Path(row["result"]).read_text())
        case = load_cases(DATA / f"{row['ds']}.json", slide_root=DATA / "slides", indices=(int(row["dataset_index"]),))[0]
        slide = _open_slide(case)
        images = [render_box(slide, Box(**p["box"]), max_side=256) for p in result["patches"]]
        labelled = {}
        for n, S in rows.items():
            labelled[f"{n} · sd {stats[n]['sd']:.1f}"] = S
        draw(key, images, mags, grids, labelled, hot_cells, out / f"{key}_spatial.png", dpi=80)
        s8 = stats["#14.8 SSDA (spectral k_g + FL)"]
        print(f"[{key}] gamma={info['gamma']:.3f} k={info['k']} sd={s8['sd']:.1f} k_uniform_p_sd={np.std(k_uniform):.1f} "
              f"dmin={s8['mean_dmin']:.4f} hot={s8['hot8_share']:.3f} ov14.6={s8['overlap_14.6']:.2f} "
              f"rho_tissue={s8['rho_count_tissue']} rho_trace={s8['rho_count_trace']}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print("\n== mean over cases")
    names = list(next(iter(summary.values()))["stats"])
    for n in names:
        v = [s["stats"][n] for s in summary.values()]
        pcs = [x for s in v for x in s["per_crop"]]
        rt = [s["rho_count_tissue"] for s in v if s["rho_count_tissue"] is not None]
        rh = [s["rho_count_trace"] for s in v if s["rho_count_trace"] is not None]
        print(f"{n[:34]:34s} crop sd {np.mean([s['sd'] for s in v]):5.1f} range {min(pcs)}-{max(pcs)} x5 {np.mean([s['x5_share'] for s in v]):.3f} "
              f"hot8 {np.mean([s['hot8_share'] for s in v]):.3f} tissue {np.mean([s['tissue_sel'] for s in v]):.3f} "
              f"d_min {np.mean([s['mean_dmin'] for s in v]):.4f} rho(count,tissue) {np.median(rt) if rt else float('nan'):+.2f} "
              f"rho(count,trace) {np.median(rh) if rh else float('nan'):+.2f}")
    ku = [x for s in summary.values() for x in s["k_uniform_p"]]
    print(f"unweighted spectral k (user spec): per-case sd {np.mean([np.std(s['k_uniform_p']) for s in summary.values()]):.1f} "
          f"range {min(ku)}-{max(ku)}")
    print(f"#14.8 overlap with #14.6: {np.mean([s['stats']['#14.8 SSDA (spectral k_g + FL)']['overlap_14.6'] for s in summary.values()]):.3f}; "
          f"empty crops {sum(int(k == 0) for s in summary.values() for k in s['k'])}")


if __name__ == "__main__":
    main()
