#!/usr/bin/env python3
"""Per-crop morphology spectrum + global water-filling allocation k_g (nav3 GTEx crops), CPU.

User spec (09-15):
    z_{g,i} = h_{g,i} / ||h_{g,i}||                      h = merger output (LLM-input visual token)
    X_g     = rows z_{g,i} - mean_i z_{g,i}              (crop-common component removed)
    C_g     = X_g^T X_g / N_g,  eigenvalues lambda_{g,1} >= lambda_{g,2} >= ...
    global water-filling: one water level tau over ALL crops' spectra,
    k_g = #{ j : lambda_{g,j} > tau }  with  sum_g k_g = B  (B = 25% of tokens, same budget as QASC)
    (equivalently: the B largest eigenvalues of the whole slide's crop spectra, counted per crop)

Inputs: runs/vit_ntrs_gtex/arrays/<key>_arrays.npz (X float16 merger output, counts, mags, tissue, saved selections of
Stratified / Random / ViT raw top-k / QASC / #14.6) and the audit manifest (to re-render crop thumbnails).
Outputs: runs/vit_waterfill_gtex/<key>_waterfill.png, aggregate.png, summary.json

usage: vit_waterfill_gtex.py <out_dir> [--keep 0.25]
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

ARR = REPO / "sender_relay_exp" / "runs" / "vit_ntrs_gtex" / "arrays"
AUDIT = REPO / "sender_relay_exp" / "runs" / "vit_sal_audit" / "full"
DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")


# --------------------------------------------------------------------------- pure


def crop_spectrum(h: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of C = X^T X / N for X = unit rows minus their mean. Length min(N, d)."""
    z = h.astype(np.float64)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    x = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    return s ** 2 / x.shape[0]


def water_fill(spectra: list[np.ndarray], budget: int, caps: list[int]) -> tuple[list[int], float]:
    """B largest eigenvalues over all crops, per-crop cap -> (k_g, water level tau = smallest admitted lambda)."""
    vals = np.concatenate(spectra)
    owner = np.concatenate([np.full(len(s), g) for g, s in enumerate(spectra)])
    order = np.argsort(-vals, kind="mergesort")
    k = [0] * len(spectra)
    tau = float("nan")
    taken = 0
    for idx in order:
        g = int(owner[idx])
        if k[g] >= caps[g]:
            continue
        k[g] += 1
        taken += 1
        tau = float(vals[idx])
        if taken == budget:
            break
    return k, tau


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--only", default="", help="comma list of case keys (default all)")
    args = ap.parse_args()
    only = set(k for k in args.only.split(",") if k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vision_text_mas.cli import _open_slide
    from vision_text_mas.dataset import load_cases
    from vision_text_mas.geometry import Box
    from vision_text_mas.navigation_render import render_box

    manifest = {r["key"]: r for r in json.loads((AUDIT / "manifest.json").read_text())}
    summary = {}
    agg = {"k": [], "tissue": [], "hetero": [], "mag": [], "energy_frac": [], "top1_frac": []}
    for f in sorted(ARR.glob("*_arrays.npz")):
        key = f.name.replace("_arrays.npz", "")
        if only and key not in only:
            continue
        a = np.load(f)
        X = a["X"]
        counts = a["counts"].tolist()
        mags = a["mags"].tolist()
        tissue = a["tissue"]
        offs = np.cumsum([0] + counts)
        N = int(offs[-1])
        B = int(round(args.keep * N))
        spectra = [crop_spectrum(X[offs[g]:offs[g + 1]]) for g in range(len(counts))]
        k, tau = water_fill(spectra, B, [c for c in counts])
        crop_tissue = [float(tissue[offs[g]:offs[g + 1]].mean()) for g in range(len(counts))]
        hetero = [float(s.sum()) for s in spectra]                      # trace C_g = total centred variance
        energy = [float(s[:kk].sum() / s.sum()) if s.sum() > 0 else 0.0 for s, kk in zip(spectra, k)]
        top1 = [float(s[0] / s.sum()) for s in spectra]
        names = [str(n) for n in a["sel_names"]]
        sels = {n: np.asarray(a[f"sel_{i}"]) for i, n in enumerate(names)}
        alloc = {n: [int(((S >= offs[g]) & (S < offs[g + 1])).sum()) for g in range(len(counts))] for n, S in sels.items()}
        summary[key] = {"B": B, "tau": tau, "k": k, "mags": mags, "crop_tissue": crop_tissue, "trace": hetero,
                        "energy_captured": energy, "top1_frac": top1, "alloc_other": alloc,
                        "rho_k_tissue": spearman(k, crop_tissue), "rho_k_trace": spearman(k, hetero)}
        agg["k"] += k
        agg["tissue"] += crop_tissue
        agg["hetero"] += hetero
        agg["mag"] += mags
        agg["energy_frac"] += energy
        agg["top1_frac"] += top1

        # ---------------- per-case figure
        row = manifest[key]
        result = json.loads(Path(row["result"]).read_text())
        case = load_cases(DATA / f"{row['ds']}.json", slide_root=DATA / "slides", indices=(int(row["dataset_index"]),))[0]
        slide = _open_slide(case)
        imgs = [render_box(slide, Box(**p["box"]), max_side=256) for p in result["patches"]]
        G = len(counts)
        fig = plt.figure(figsize=(max(16, 1.55 * G), 9.2))
        gs = fig.add_gridspec(3, G, height_ratios=[1.0, 1.35, 1.1], hspace=0.45, wspace=0.08)
        for g in range(G):
            ax = fig.add_subplot(gs[0, g])
            ax.imshow(imgs[g])
            ax.set_title(f"P{g + 1} x{mags[g]}\nk={k[g]}  tissue {crop_tissue[g]:.2f}", fontsize=8,
                         color="#1f4fa0" if mags[g] == 5 else "#b3261e")
            ax.axis("off")
        ax = fig.add_subplot(gs[1, :G // 2])
        for g, s in enumerate(spectra):
            col = plt.get_cmap("Blues" if mags[g] == 5 else "Reds")(0.45 + 0.5 * g / max(1, G - 1))
            idx = np.arange(1, len(s) + 1)
            ax.plot(idx, s, color=col, lw=1.2, label=f"P{g + 1} x{mags[g]}")
            if k[g] > 0:
                ax.plot([k[g]], [s[k[g] - 1]], "o", color=col, ms=4)
        ax.axhline(tau, color="k", ls="--", lw=1)
        ax.text(2, tau * 1.25, f"water level τ = {tau:.2e}", fontsize=8)
        ax.set_yscale("log")
        ax.set_xscale("log")
        top = max(float(s[0]) for s in spectra)
        ax.set_ylim(tau / 30.0, top * 1.5)             # hide the numerically-zero tail (rank <= N-1)
        ax.set_xlim(1, min(len(s) for s in spectra) - 1)
        ax.set_xlabel("eigen-direction j (log)")
        ax.set_ylabel("λ_{g,j} (log)")
        ax.set_title(f"crop spectra of C_g · global water-filling picks the B={B} largest λ over all crops", fontsize=9)
        ax.legend(fontsize=6, ncol=2, loc="lower left")
        ax = fig.add_subplot(gs[1, G // 2:])
        for g, s in enumerate(spectra):
            col = plt.get_cmap("Blues" if mags[g] == 5 else "Reds")(0.45 + 0.5 * g / max(1, G - 1))
            ax.plot(np.arange(1, len(s) + 1), np.cumsum(s) / s.sum(), color=col, lw=1.2)
            ax.plot([k[g]], [energy[g]], "o", color=col, ms=4)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("eigen-direction j (log)")
        ax.set_ylabel("cumulative energy fraction")
        ax.set_title("cumulative spectrum · dot = energy kept at k_g", fontsize=9)
        ax = fig.add_subplot(gs[2, :])
        xs = np.arange(G)
        show = [("water-filling k_g", k, "#222222")]
        for n, col in (("Random", "#9e9e9e"), ("#14 QASC", "#e08a00"), ("#14.6 residual + support", "#2e7d32")):
            m = [nn for nn in alloc if nn.startswith(n)]
            if m:
                show.append((m[0], alloc[m[0]], col))
        w = 0.8 / len(show)
        for i, (lab, vals, col) in enumerate(show):
            ax.bar(xs + (i - (len(show) - 1) / 2) * w, vals, width=w, color=col, label=f"{lab} (sd {np.std(vals):.1f})")
        ax.axhline(B / G, color="#1565c0", ls=":", lw=1, label=f"stratified {B // G}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"P{g + 1} x{mags[g]}" for g in range(G)], fontsize=8)
        ax.set_ylabel("tokens per crop")
        ax.legend(fontsize=7, ncol=5, loc="upper right")
        ax.set_title(f"allocation · ρ(k_g, crop tissue) = {summary[key]['rho_k_tissue']} · ρ(k_g, trace C_g) = {summary[key]['rho_k_trace']}",
                     fontsize=9)
        fig.suptitle(f"{key}: per-crop morphology spectrum → global water-filling budget (B = {B} of {N})", fontsize=11)
        fig.savefig(out / f"{key}_waterfill.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"[{key}] B={B} tau={tau:.3e} k={k} sd={np.std(k):.1f} rho_tissue={summary[key]['rho_k_tissue']} "
              f"rho_trace={summary[key]['rho_k_trace']}", flush=True)

    # ---------------- aggregate
    k = np.array(agg["k"])
    tis = np.array(agg["tissue"])
    het = np.array(agg["hetero"])
    mg = np.array(agg["mag"])
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    ax = axes[0]
    ax.hist([k[mg == 5], k[mg == 20]], bins=30, color=["#1f4fa0", "#b3261e"], label=["x5", "x20"], stacked=True)
    ax.axvline(64, color="#1565c0", ls=":", label="stratified 64")
    ax.set_xlabel("k_g")
    ax.set_title(f"k_g over {len(k)} crops · median {np.median(k):.0f} · range {k.min()}–{k.max()}", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(tis, k, s=10, c=np.where(mg == 5, "#1f4fa0", "#b3261e"))
    ax.set_xlabel("crop tissue fraction")
    ax.set_ylabel("k_g")
    ax.set_title(f"k_g vs tissue · pooled ρ = {spearman(k, tis):.2f}", fontsize=9)
    ax = axes[2]
    ax.scatter(het, k, s=10, c=np.where(mg == 5, "#1f4fa0", "#b3261e"))
    ax.set_xlabel("trace C_g (total centred variance)")
    ax.set_ylabel("k_g")
    ax.set_title(f"k_g vs trace · pooled ρ = {spearman(k, het):.2f}", fontsize=9)
    ax = axes[3]
    rows = []
    for name in ("water-filling", "Stratified", "Random", "Top-k ViT", "#14 QASC", "#14.6 residual top-k", "#14.6 residual + support"):
        sds, x5 = [], []
        for s in summary.values():
            if name == "water-filling":
                v = s["k"]
            else:
                m = [nn for nn in s["alloc_other"] if nn.startswith(name)]
                if not m:
                    continue
                v = s["alloc_other"][m[0]]
            sds.append(np.std(v))
            x5.append(sum(vv for vv, mm in zip(v, s["mags"]) if mm == 5) / max(1, sum(v)))
        if sds:
            rows.append((name, np.mean(sds), np.mean(x5)))
    ax.axis("off")
    ax.table(cellText=[[n, f"{sd:.1f}", f"{x:.3f}"] for n, sd, x in rows],
             colLabels=["allocation", "mean per-crop sd", "x5 share"], loc="center", cellLoc="left").scale(1, 1.5)
    ax.set_title("allocation spread vs other selectors (GTEx nav3, 20 cases)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "aggregate.png", dpi=100)
    plt.close(fig)
    summary["_aggregate"] = {"k_median": float(np.median(k)), "k_min": int(k.min()), "k_max": int(k.max()),
                             "k_sd_per_case_mean": float(np.mean([np.std(s["k"]) for kk, s in summary.items() if not kk.startswith("_")])),
                             "rho_pooled_tissue": spearman(k, tis), "rho_pooled_trace": spearman(k, het),
                             "x5_share": float(k[mg == 5].sum() / k.sum()),
                             "zero_crops": int((k == 0).sum()), "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["_aggregate"], indent=1))


if __name__ == "__main__":
    main()
