#!/usr/bin/env python3
"""Does null calibration remove the positional prior from the vision-encoder salience a_i?

Notion C1 #14.5 "Null-Calibrated Visual Salience + Physical-Support Constrained Morphology Coverage" §4:
    pi_p   = a_p(x0)                      x0 = blank white input, block l_v = 23, stored once offline
    r_i(x) = (a_i(x) + eps) / (pi_p(i) + eps)
§6 falsifier: if r still follows grid coordinates and does not respond to tissue/content, drop the attention term.

Offline, CPU. Reuses the 1182 audit crops (runs/vit_sal_audit/full/cases/*.npz, per_block a_i, tissue per token)
and recomputes the content-free maps once (5 synthetic 512 px images, same hooks as vit_sink_probe).

Prior variants (applied per grid position, uniform-scaled maps so uniform attention = 1):
  none       raw a_i
  white      Notion spec (white image only)
  null5      geometric mean of white / black / gray / H&E pink / noise
  real_loco  mean map of the OTHER cases' real crops (reference for "perfect" positional removal, not the page method)

usage: vit_nullcal.py <out_dir> [--threads T] [--eps E]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sender_relay_exp"))

import numpy as np  # noqa: E402

from vit_sal_stats import crop_map, crop_spans, grid_coords, spearman  # noqa: E402

AUDIT = REPO / "sender_relay_exp" / "runs" / "vit_sal_audit" / "full"
NULL_NAMES = ("white", "black", "gray", "he_pink", "noise")
VARIANTS = ("none", "white", "null5", "real_loco")
GRID = 16
N = GRID * GRID
TOPK = N // 4


# --------------------------------------------------------------------------- pure helpers


def calibrate(a: np.ndarray, prior: np.ndarray, eps: float) -> np.ndarray:
    """(a + eps) / (prior + eps), element-wise; broadcasting over leading axes."""
    return (np.asarray(a, dtype=float) + eps) / (np.asarray(prior, dtype=float) + eps)


def loco_means(maps: np.ndarray, groups) -> np.ndarray:
    """Row i -> mean of rows whose group differs from group i. maps [n, d]."""
    maps = np.asarray(maps, dtype=float)
    groups = np.asarray(groups)
    total, n = maps.sum(axis=0), len(maps)
    out = np.empty_like(maps)
    for g in np.unique(groups):
        m = groups == g
        k = int(m.sum())
        if k == n:
            out[m] = np.nan
            continue
        out[m] = (total - maps[m].sum(axis=0)) / (n - k)
    return out


def position_eta2(logmaps: np.ndarray) -> float:
    """Share of the variance of log maps explained by grid position (column means). logmaps [n, d]."""
    x = np.asarray(logmaps, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)          # remove per-crop level (maps are per-crop normalised)
    tot = float(((x - x.mean()) ** 2).sum())
    col = x.mean(axis=0)
    between = float(len(x) * ((col - col.mean()) ** 2).sum())
    return between / tot if tot > 0 else float("nan")


def mean_percentile(v: np.ndarray, cells: np.ndarray) -> float:
    """Mean percentile rank (0 = lowest, 1 = highest) of the given cells in v."""
    order = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort")
    return float(order[cells].mean() / (len(v) - 1))


def border_mask(grid: int) -> np.ndarray:
    m = np.ones((grid, grid), dtype=bool)
    m[1:-1, 1:-1] = False
    return m.ravel()


def crop_metrics(v: np.ndarray, raw: np.ndarray, tissue: np.ndarray, tmpl_loco: np.ndarray,
                 hot: np.ndarray, coords: dict, border: np.ndarray) -> dict:
    top = np.argsort(-v, kind="mergesort")[:TOPK]
    bot = np.argsort(v, kind="mergesort")[:TOPK]
    raw_top = set(np.argsort(-raw, kind="mergesort")[:TOPK].tolist())
    return {
        "rho_template_loco": spearman(v, tmpl_loco),
        "r_dist": spearman(v, coords["dist"]),
        "r_idx": spearman(v, coords["idx"]),
        "r_tissue": spearman(v, tissue),
        "hot8_pct": mean_percentile(v, hot),
        "hot8_in_top25": float(np.isin(hot, top).mean()),
        "top25_border": float(border[top].mean()),
        "top25_tissue_minus_crop": float(tissue[top].mean() - tissue.mean()),
        "top25_tissue_minus_bottom": float(tissue[top].mean() - tissue[bot].mean()),
        "top25_overlap_raw": len(raw_top & set(top.tolist())) / TOPK,
    }


def med(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return None if not v else float(np.median(v))


# --------------------------------------------------------------------------- model part (once)


def null_maps(out: Path, threads: int) -> np.ndarray:
    """[5, n_blocks, 256] content-free maps (uniform = 1); cached in out/null_maps.npz."""
    f = out / "null_maps.npz"
    if f.exists():
        return np.load(f)["maps"]
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from vit_sink_probe import MODEL, collect_maps_and_norms, encode, maps_per_crop

    torch.set_num_threads(threads)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    rng = np.random.default_rng(0)                  # identical to vit_sink_probe check 1
    synth = {
        "white": Image.new("RGB", (512, 512), (255, 255, 255)),
        "black": Image.new("RGB", (512, 512), (0, 0, 0)),
        "gray": Image.new("RGB", (512, 512), (128, 128, 128)),
        "he_pink": Image.new("RGB", (512, 512), (230, 180, 210)),
        "noise": Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)),
    }
    pv, thw = encode(proc, [synth[k] for k in NULL_NAMES])
    attn, _ = collect_maps_and_norms(model, pv, thw)
    maps = np.stack([m for _sp, m in maps_per_crop(attn, thw)])
    assert maps.shape[-1] == N, maps.shape
    np.savez(f, maps=maps, names=np.array(NULL_NAMES))
    return maps


# --------------------------------------------------------------------------- main


def load_audit():
    crops = []
    for f in sorted((AUDIT / "cases").glob("*.npz")):
        z = np.load(f)
        ds = f.stem.rsplit("_", 1)[0]
        for sp, mag in zip(crop_spans(z["thw"]), z["mags"]):
            if sp[2] != GRID or sp[3] != GRID:
                continue
            crops.append({"case": f.stem, "ds": ds, "mag": int(mag),
                          "pb": np.stack([crop_map(z["per_block"][b], sp) for b in range(z["per_block"].shape[0])]),
                          "tissue": z["tissue"][sp[0]:sp[1]].astype(float)})
    return crops


def analyse_block(crops, block, nulls, eps, raw_template_hot) -> dict:
    """block: int or 'mean'."""
    def pick(pb):
        return pb.mean(axis=0) if block == "mean" else pb[block]

    raw = np.stack([pick(c["pb"]) for c in crops])                  # [n, 256]
    groups = [c["case"] for c in crops]
    nb = np.stack([pick(m) for m in nulls])                          # [5, 256]
    priors = {"none": np.ones(N), "white": nb[0],
              "null5": np.exp(np.log(nb + eps).mean(axis=0)) - eps}
    raw_loco = loco_means(raw, groups)
    coords = grid_coords(GRID, GRID)
    border = border_mask(GRID)
    hot = raw_template_hot

    res = {"null_images_vs_white": {}}
    wlog = np.log(nb[0] + eps)
    for i, name in enumerate(NULL_NAMES[1:], start=1):              # how well does white explain other null images?
        r = np.log(nb[i] + eps) - wlog
        res["null_images_vs_white"][name] = {
            "rho_raw_null_vs_white": spearman(nb[i], nb[0]),
            "rho_calibrated_vs_raw_template": spearman(r, raw.mean(axis=0)),
            "sd_log_calibrated": float(r.std()), "sd_log_raw": float(np.log(nb[i] + eps).std())}

    for var in VARIANTS:
        if var == "real_loco":
            cal = calibrate(raw, raw_loco, eps)
        else:
            cal = calibrate(raw, priors[var][None, :], eps)
        tmpl = loco_means(cal, groups)
        rows = []
        for c, v, r0, t in zip(crops, cal, raw, tmpl):
            rows.append({"ds": c["ds"], "mag": c["mag"],
                         **crop_metrics(v, r0, c["tissue"], t, hot, coords, border)})
        by = defaultdict(list)
        for row in rows:
            by["ALL"].append(row)
            by[f"{row['ds']}"].append(row)
            by[f"mag{row['mag']}"].append(row)
        summ = {}
        for key, members in by.items():
            summ[key] = {"n": len(members), **{m: med([r[m] for r in members]) for m in members[0] if m not in ("ds", "mag")}}
        summ["ALL"]["position_eta2"] = position_eta2(np.log(cal))
        if var in ("white", "null5"):
            # expected calibrated value of the hot cells vs the rest (over-correction check)
            p = priors[var]
            summ["ALL"]["hot8_mean_calibrated"] = float(cal[:, hot].mean())
            summ["ALL"]["rest_mean_calibrated"] = float(np.delete(cal, hot, axis=1).mean())
            summ["ALL"]["prior_hot8_mean"] = float(p[hot].mean())
        else:
            summ["ALL"]["hot8_mean_calibrated"] = float(cal[:, hot].mean())
            summ["ALL"]["rest_mean_calibrated"] = float(np.delete(cal, hot, axis=1).mean())
        res[var] = summ
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-3, help="in uniform-scaled units (uniform attention = 1)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nulls = null_maps(out, args.threads)
    print(f"[null] maps {nulls.shape}", flush=True)
    crops = load_audit()
    print(f"[audit] crops {len(crops)} cases {len({c['case'] for c in crops})}", flush=True)

    summary = {"eps": args.eps, "n_crops": len(crops), "blocks": {}}
    for block in (23, "mean"):
        tm = np.stack([(c["pb"].mean(axis=0) if block == "mean" else c["pb"][block]) for c in crops]).mean(axis=0)
        hot = np.argsort(-tm)[:8]
        res = analyse_block(crops, block, nulls, args.eps, hot)
        res["hot8_rc"] = [list(divmod(int(i), GRID)) for i in hot]
        summary["blocks"][str(block)] = res
        print(f"\n== block {block}  hot8 {res['hot8_rc']}")
        cols = ("rho_template_loco", "r_dist", "r_idx", "r_tissue", "hot8_pct", "hot8_in_top25", "top25_border",
                "top25_tissue_minus_crop", "top25_tissue_minus_bottom", "top25_overlap_raw")
        print("variant    " + " ".join(f"{c[:14]:>14s}" for c in cols) + "   eta2  hot8/rest")
        for var in VARIANTS:
            s = res[var]["ALL"]
            print(f"{var:10s} " + " ".join(f"{(s[c] if s[c] is not None else float('nan')):14.3f}" for c in cols)
                  + f"  {s['position_eta2']:.3f}  {s['hot8_mean_calibrated']:.2f}/{s['rest_mean_calibrated']:.2f}")
        print("per dataset r_tissue / top25_tissue_minus_bottom / rho_template_loco:")
        for ds in sorted(k for k in res["none"] if k not in ("ALL",) and not k.startswith("mag")):
            print(f"  {ds:18s} " + "  ".join(
                f"{var}:{res[var][ds]['r_tissue'] or float('nan'):+.2f}/{res[var][ds]['top25_tissue_minus_bottom']:+.3f}/{res[var][ds]['rho_template_loco']:+.2f}"
                for var in VARIANTS))
        print("null images calibrated by white:", json.dumps(res["null_images_vs_white"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[done] -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
