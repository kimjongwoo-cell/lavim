#!/usr/bin/env python3
"""Mean 5x/20x heatmaps of the vision-encoder salience a_i and its position correlations.

Per crop (a_i sums to 1 inside a crop, rescaled so uniform = 1), per vision block and for the
block mean:
  r_idx          Spearman(a_i, raster token index)
  r_dist         Spearman(a_i, distance to crop centre)       negative = centre-favoured
  r_tissue       Spearman(a_i, stained-pixel fraction of the token's 32x32 pixels)
  r_dist|tissue  r_dist with tissue rank regressed out (centre effect beyond tissue layout)
  r_idx|tissue   same for token index
  r_template     Spearman(a_i, mean map of the same dataset x magnification from OTHER cases)
                 high = a fixed spatial template regardless of the tissue shown
  cb_ratio       centre half-square mean / outer ring mean
  max            largest token value (uniform = 1)
  argmax_mode    share of crops whose top token sits on the most common top position
Only square 16x16 crops enter heatmaps and templates (non-square crops at slide edges are
counted and reported).

usage: vit_sal_analyze.py <out_dir>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_sal_stats import (centre_border_ratio, crop_map, crop_spans, grid_coords,  # noqa: E402
                           loo_template_r, partial_spearman, spearman, summarize)

DSN = {"tcga_expert_vqa": "ExpertVQA", "gtex": "GTEx", "tcga": "TCGA",
       "tcga_slidebench": "SlideBench", "panda": "PANDA"}
ORDER = tuple(DSN)
MAGS = (5, 20)
SHOW_BLOCKS = (0, 3, 7, 11, 15, 19, 22, 23)

out = Path(sys.argv[1])
crops = []
for f in sorted((out / "cases").glob("*.npz")):
    z = np.load(f)
    ds = f.stem.rsplit("_", 1)[0]
    pb = z["per_block"]
    nb = pb.shape[0]
    for sp, mag in zip(crop_spans(z["thw"]), z["mags"]):
        blocks = np.stack([crop_map(pb[b], sp) for b in range(nb)])
        crops.append({"ds": ds, "case": f.stem, "mag": int(mag), "H": sp[2], "W": sp[3],
                      "maps": np.vstack([blocks, blocks.mean(0, keepdims=True)]),
                      "tissue": z["tissue"][sp[0]:sp[1]]})
if not crops:
    sys.exit("no cases")
NB = crops[0]["maps"].shape[0] - 1
LAYERS = list(range(NB)) + ["mean"]
LI = {l: i for i, l in enumerate(LAYERS)}

# per-crop metrics
for c in crops:
    g = grid_coords(c["H"], c["W"])
    c["m"] = defaultdict(dict)
    for li, l in enumerate(LAYERS):
        v = c["maps"][li]
        c["m"][l] = {
            "r_idx": spearman(v, g["idx"]), "r_dist": spearman(v, g["dist"]),
            "r_tissue": spearman(v, c["tissue"]),
            "r_dist|tissue": partial_spearman(v, g["dist"], c["tissue"]),
            "r_idx|tissue": partial_spearman(v, g["idx"], c["tissue"]),
            "cb_ratio": centre_border_ratio(v.reshape(c["H"], c["W"])),
            "max": float(v.max()), "argmax": int(v.argmax()),
        }

sq = [c for c in crops if (c["H"], c["W"]) == (16, 16)]
for key_fn in (lambda c: (c["ds"], c["mag"]), lambda c: ("ALL", c["mag"])):
    groups = defaultdict(list)
    for c in sq:
        groups[key_fn(c)].append(c)
    for gk, members in groups.items():
        tag = "r_template" if gk[0] != "ALL" else "r_template_pooled"
        for l in LAYERS:
            rs = loo_template_r([m["maps"][LI[l]] for m in members], [m["case"] for m in members])
            for m, r in zip(members, rs):
                m["m"][l][tag] = r

METRICS = ("r_idx", "r_dist", "r_tissue", "r_dist|tissue", "r_idx|tissue", "r_template",
           "cb_ratio", "max")


def agg(members, l):
    row = {k: summarize([m["m"][l].get(k) for m in members]) for k in METRICS}
    row["r_template_pooled"] = summarize([m["m"][l].get("r_template_pooled") for m in members])
    arg = Counter(m["m"][l]["argmax"] for m in members if (m["H"], m["W"]) == (16, 16))
    if arg:
        pos, cnt = arg.most_common(1)[0]
        row["argmax_mode"] = {"pos_rc": list(divmod(pos, 16)), "share": cnt / sum(arg.values()),
                              "n": sum(arg.values())}
    row["n_crops"] = len(members)
    row["n_square"] = sum((m["H"], m["W"]) == (16, 16) for m in members)
    return row


summary = {"n_cases": len({c["case"] for c in crops}), "n_crops": len(crops),
           "non_square": dict(Counter(f"{c['H']}x{c['W']}" for c in crops if (c["H"], c["W"]) != (16, 16))),
           "cases_per_ds": dict(Counter(c["case"].rsplit("_", 1)[0] for c in crops[::1] if True)),
           "by": {}}
summary["cases_per_ds"] = {ds: len({c["case"] for c in crops if c["ds"] == ds}) for ds in ORDER}
for l in LAYERS:
    for ds in ORDER + ("ALL",):
        for mag in MAGS:
            members = [c for c in crops if (ds == "ALL" or c["ds"] == ds) and c["mag"] == mag]
            if members:
                summary["by"][f"{l}|{ds}|{mag}"] = agg(members, l)
(out / "summary.json").write_text(json.dumps(summary, indent=1))


def med(l, ds, mag, k):
    r = summary["by"].get(f"{l}|{ds}|{mag}", {}).get(k, {})
    return r.get("median")


def fmt(x, p=2):
    return "  –  " if x is None else f"{x:+.{p}f}"


print(f"cases={summary['n_cases']} crops={summary['n_crops']} per ds={summary['cases_per_ds']} "
      f"non-square={summary['non_square']}")
for l in ("mean", 3, 22, 23):
    print(f"\n== layer {l}  (median over crops; r_dist<0 = centre-favoured)")
    print(f"{'ds':<11}{'mag':>4}{'n':>5}{'r_idx':>8}{'r_dist':>8}{'r_tis':>8}{'dist|t':>8}{'idx|t':>8}"
          f"{'r_tmpl':>8}{'tmplALL':>8}{'c/b':>6}{'max':>6}  argmax-mode")
    for ds in ORDER + ("ALL",):
        for mag in MAGS:
            row = summary["by"].get(f"{l}|{ds}|{mag}")
            if not row:
                continue
            am = row.get("argmax_mode", {})
            print(f"{DSN.get(ds, ds):<11}{mag:>4}{row['n_crops']:>5}"
                  + "".join(f"{fmt(row[k].get('median')):>8}" for k in
                            ("r_idx", "r_dist", "r_tissue", "r_dist|tissue", "r_idx|tissue", "r_template",
                             "r_template_pooled"))
                  + f"{row['cb_ratio'].get('median', float('nan')):>6.2f}{row['max'].get('median', float('nan')):>6.1f}"
                  + f"  rc={am.get('pos_rc')} {am.get('share', 0):.2f}")
print("\n== per block, ALL datasets (median r_idx / r_dist / r_dist|tissue / r_tissue / r_template_pooled)")
for mag in MAGS:
    print(f"-- {mag}x")
    for l in LAYERS:
        print(f"  {str(l):>4}: " + " ".join(fmt(med(l, 'ALL', mag, k)) for k in
                                          ("r_idx", "r_dist", "r_dist|tissue", "r_tissue", "r_template_pooled")))


# ---------------------------------------------------------------- figures
def mean_map(members, l):
    ms = [m["maps"][LI[l]].reshape(16, 16) for m in members if (m["H"], m["W"]) == (16, 16)]
    return (np.mean(ms, axis=0), len(ms)) if ms else (None, 0)


def draw(ax, mm, n, title, lim):
    if mm is None:
        ax.axis("off")
        return None
    im = ax.imshow(np.log2(mm), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(f"{title}\nn={n} c/b={centre_border_ratio(mm):.2f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return im


panels = [(ds, mag) for ds in ORDER + ("ALL",) for mag in MAGS]
maps = {p: mean_map([c for c in crops if (p[0] == "ALL" or c["ds"] == p[0]) and c["mag"] == p[1]], "mean")
        for p in panels}
vals = np.concatenate([np.abs(np.log2(m)).ravel() for m, _ in maps.values() if m is not None])
lim = float(np.percentile(vals, 99)) or 1.0
fig, axes = plt.subplots(len(MAGS), len(ORDER) + 1, figsize=(2.3 * (len(ORDER) + 1), 5.2))
for j, ds in enumerate(ORDER + ("ALL",)):
    for i, mag in enumerate(MAGS):
        im = draw(axes[i, j], *maps[(ds, mag)], f"{DSN.get(ds, 'All sets')} {mag}x", lim)
fig.colorbar(im, ax=axes, shrink=0.7, label="log2(a_i / uniform)")
fig.suptitle("Vision-encoder salience a_i, block mean — mean over crops (16x16 merged grid)", fontsize=10)
fig.savefig(out / "fig_heatmap_mean.png", dpi=130, bbox_inches="tight")
plt.close(fig)

cols = list(SHOW_BLOCKS) + ["mean"]
lm = {(mag, l): mean_map([c for c in crops if c["mag"] == mag], l) for mag in MAGS for l in cols}
vals = np.concatenate([np.abs(np.log2(m)).ravel() for m, _ in lm.values() if m is not None])
lim = float(np.percentile(vals, 99)) or 1.0
fig, axes = plt.subplots(len(MAGS), len(cols), figsize=(2.1 * len(cols), 5.0))
for j, l in enumerate(cols):
    for i, mag in enumerate(MAGS):
        im = draw(axes[i, j], *lm[(mag, l)], f"{mag}x block {l}", lim)
fig.colorbar(im, ax=axes, shrink=0.7, label="log2(a_i / uniform)")
fig.suptitle("a_i mean map by vision block, all datasets", fontsize=10)
fig.savefig(out / "fig_heatmap_blocks.png", dpi=130, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, len(MAGS), figsize=(11, 3.6), sharey=True)
for ax, mag in zip(axes, MAGS):
    x = np.arange(NB)
    for k, style in (("r_idx", "-"), ("r_dist", "-"), ("r_dist|tissue", "--"), ("r_tissue", "-"),
                     ("r_template_pooled", "-")):
        y = [med(l, "ALL", mag, k) for l in range(NB)]
        ax.plot(x, [np.nan if v is None else v for v in y], style, marker=".", label=k)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"{mag}x — median over crops, all datasets")
    ax.set_xlabel("vision block")
axes[0].set_ylabel("Spearman r")
axes[-1].legend(fontsize=7, loc="best")
fig.savefig(out / "fig_corr_blocks.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nwritten {out}/summary.json fig_heatmap_mean.png fig_heatmap_blocks.png fig_corr_blocks.png")
