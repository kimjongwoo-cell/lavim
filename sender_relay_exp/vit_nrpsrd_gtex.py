#!/usr/bin/env python3
"""C1 #16 NR-PSRD (Null-Residual Physical-Support Rate-Distortion Selection) pre-registered
falsifiers A and B. Notion 3ddd771b-c9e2-81ab.

Page §12 asks two things BEFORE any downstream run, both on existing offline dumps:
  A  does the null residual actually separate white/background from tissue?
     the only quantity is ||phi_i||^2 ; tissue annotation is used for EVALUATION only.
  B  does the selection actually differ from #14.6 / #15 at the same budget?
     subset overlap, selected white fraction, support allocation dispersion.

Method as written on the page (§2-§6):
  z_i   = h_i / ||h_i||                        C1-boundary merged visual state (2560-d)
  U0    = orth([nu_1 ... nu_P])                span of content-free post-merge states
  phi_i = (I - U0 U0^T) z_i                    the ONLY token representation; NOT renormalised
  D_g(S_g) = sum_i min( ||phi_i||^2 , min_{j in S_g} ||phi_i - phi_j||^2 )
  S* = argmin sum_g D_g  s.t. |S| <= B  and  |S cap V_g| >= 1 for every support g
  first token of each support = argmax [D_g(empty) - D_g({j})]  (same objective, no anchor rule)
  then global greedy on marginal distortion reduction.

The page does not fix the null-subspace DIMENSION (P = 5 templates x 256 merged states here), so
--ranks sweeps it: r = 0 is the no-residualisation control and r = "rank" the numerical rank.
Everything else is fixed by the page.

Inputs reused, no pipeline re-run and no LLM:
  runs/vit_ntrs_gtex/arrays/<key>_arrays.npz   X [N, 2560] (= pooler_output), counts, mags,
                                               tissue, a_raw, w_page, sel_names + sel_<i>
  content-free templates are re-encoded here (5 images, vision tower only, CPU fp32).

usage: vit_nrpsrd_gtex.py <out_dir> [--arrays runs/vit_ntrs_gtex/arrays] [--keep 0.25]
                          [--ranks 0,1,4,16,64,256,rank] [--select-ranks rank,64]
                          [--threads 16] [--cases N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
# inputs are read-only from the 09-15 NTRS audit in the original tree
ARRAYS = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vit_ntrs_gtex/arrays")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sender_relay_exp"))

import numpy as np  # noqa: E402

MODEL = Path("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
BG_TISSUE = 0.10      # evaluation label only: background-like token
TIS_TISSUE = 0.90     # evaluation label only: tissue token


# --------------------------------------------------------------------------- pure


def null_basis(nu: np.ndarray, rank=None, rtol: float = 1e-6):
    """Orthonormal basis of span(nu) [P, d] -> (U [d, r], singular values).

    rank=None uses the numerical rank at rtol; an int takes the top-r left singular vectors.
    """
    A = np.asarray(nu, dtype=np.float64)
    U, s, _ = np.linalg.svd(A.T, full_matrices=False)     # columns of U span the row space of nu
    num = int((s > rtol * s[0]).sum()) if s.size else 0
    r = num if rank is None else max(0, min(int(rank), U.shape[1]))
    return U[:, :r], s


def residualize(z: np.ndarray, U: np.ndarray) -> np.ndarray:
    """phi = (I - U U^T) z, rows of z. U may have zero columns (control: phi = z)."""
    z = np.asarray(z, dtype=np.float64)
    if U.size == 0 or U.shape[1] == 0:
        return z.copy()
    return z - (z @ U) @ U.T


def support_distortion(phi_g: np.ndarray, S_local) -> float:
    """D_g of page §4 for one support, by the definition (reference implementation)."""
    null = (phi_g ** 2).sum(axis=1)
    S = list(S_local)
    if not S:
        return float(null.sum())
    d = ((phi_g[:, None, :] - phi_g[None, S, :]) ** 2).sum(axis=-1).min(axis=1)
    return float(np.minimum(null, d).sum())


def rd_greedy(phi: np.ndarray, counts, k: int):
    """§5-§6: min-1 per support, then global greedy on marginal distortion reduction.

    Returns (order, gains, D_after, D0). order is the pick sequence (global token indices).
    """
    counts = [int(c) for c in counts]
    offs = np.cumsum([0] + counts)
    N = int(offs[-1])
    k = min(int(k), N)
    G = len(counts)
    sq, cur, cost = [], [], []
    for g in range(G):
        p = phi[offs[g]:offs[g + 1]]
        n2 = (p ** 2).sum(axis=1)
        # squared euclidean distances inside the support
        d = n2[:, None] + n2[None, :] - 2.0 * (p @ p.T)
        np.fill_diagonal(d, 0.0)
        sq.append(np.maximum(d, 0.0))
        cur.append(n2.copy())          # current explanation cost per token (null at the start)
        cost.append(n2.copy())         # null cost, kept for reference
    D0 = float(sum(c.sum() for c in cur))

    def gains_of(g):
        # gain of adding candidate j in support g = sum_i (cur_i - d_ij)+
        return np.clip(cur[g][:, None] - sq[g], 0.0, None).sum(axis=0)

    gains = [gains_of(g) for g in range(G)]
    chosen = np.zeros(N, dtype=bool)
    order, got = [], []

    def take(j_global, g):
        chosen[j_global] = True
        order.append(int(j_global))
        jl = j_global - offs[g]
        got.append(float(gains[g][jl]))
        cur[g] = np.minimum(cur[g], sq[g][:, jl])
        gains[g] = gains_of(g)
        gains[g][chosen[offs[g]:offs[g + 1]]] = -np.inf

    for g in range(min(G, k)):                     # feasibility: one real token per support
        jl = int(np.argmax(gains[g]))
        take(offs[g] + jl, g)
    while len(order) < k:
        flat = np.concatenate([gains[g] for g in range(G)])
        flat[chosen] = -np.inf
        j = int(np.argmax(flat))
        if not np.isfinite(flat[j]) or flat[j] <= 0:
            # no candidate reduces distortion; fill by the largest remaining null cost
            rest = np.flatnonzero(~chosen)
            if rest.size == 0:
                break
            j = int(rest[np.argmax(np.concatenate(cost)[rest])])
        g = int(np.searchsorted(offs, j, side="right") - 1)
        take(j, g)
    D_after = float(sum(c.sum() for c in cur))
    return np.array(order, dtype=int), got, D_after, D0


def distortion_of(phi: np.ndarray, counts, S) -> float:
    """Total sum_g D_g for an arbitrary selection (used to score the other selectors)."""
    counts = [int(c) for c in counts]
    offs = np.cumsum([0] + counts)
    S = np.asarray(S)
    tot = 0.0
    for g in range(len(counts)):
        loc = S[(S >= offs[g]) & (S < offs[g + 1])] - offs[g]
        tot += support_distortion(phi[offs[g]:offs[g + 1]], loc)
    return float(tot)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)) with ties at 0.5 (rank statistic, no test)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.empty_like(allv)
    order = np.argsort(allv, kind="mergesort")
    sa = allv[order]
    i = 0
    while i < len(sa):                              # average ranks within ties
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def per_crop_counts(S, counts):
    offs = np.cumsum([0] + [int(c) for c in counts])
    S = np.asarray(S)
    return [int(((S >= offs[g]) & (S < offs[g + 1])).sum()) for g in range(len(counts))]


# --------------------------------------------------------------------------- null states


def null_states(threads: int):
    """Post-merge visual states of the 5 content-free templates (same images as vit_sink_probe)."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(threads)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    rng = np.random.default_rng(0)
    synth = {
        "white": Image.new("RGB", (512, 512), (255, 255, 255)),
        "black": Image.new("RGB", (512, 512), (0, 0, 0)),
        "gray": Image.new("RGB", (512, 512), (128, 128, 128)),
        "he_pink": Image.new("RGB", (512, 512), (230, 180, 210)),
        "noise": Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)),
    }
    enc = proc.image_processor(images=list(synth.values()), return_tensors="pt")
    with torch.no_grad():
        feats = model.model.get_image_features(enc["pixel_values"].float(), enc["image_grid_thw"],
                                               return_dict=True).pooler_output
    X = (torch.cat(list(feats), dim=0) if isinstance(feats, (list, tuple)) else feats).float().numpy()
    names = np.repeat(np.array(list(synth)), X.shape[0] // len(synth))
    return X.astype(np.float64), names, enc["image_grid_thw"].numpy()


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arrays", default=str(ARRAYS))
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--ranks", default="0,1,4,16,64,256,rank")
    ap.add_argument("--select-ranks", default="rank,64")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cases", type=int, default=0, help="0 = all")
    ap.add_argument("--null-cache", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache = Path(args.null_cache) if args.null_cache else out / "null_states.npz"
    if cache.exists():
        z = np.load(cache)
        nu, nu_names = z["nu"], z["names"]
        print(f"[null] cached {nu.shape}", flush=True)
    else:
        nu, nu_names, thw = null_states(args.threads)
        np.savez_compressed(cache, nu=nu.astype(np.float32), names=nu_names, thw=thw)
        print(f"[null] encoded {nu.shape} in {time.time() - t0:.0f}s", flush=True)
    nu = nu.astype(np.float64)
    Ufull, sv = null_basis(nu)
    rank_num = Ufull.shape[1]
    energy = np.cumsum(sv ** 2) / (sv ** 2).sum()
    print(f"[null] P={nu.shape[0]} d={nu.shape[1]} numerical rank={rank_num} "
          f"sv[0]={sv[0]:.3f} sv[-1]={sv[-1]:.2e} "
          f"dims for 90/99/99.9% energy = "
          f"{int(np.searchsorted(energy, 0.90) + 1)}/{int(np.searchsorted(energy, 0.99) + 1)}/"
          f"{int(np.searchsorted(energy, 0.999) + 1)}", flush=True)

    def parse_rank(tok):
        return rank_num if tok.strip() == "rank" else int(tok)

    ranks = [parse_rank(t) for t in args.ranks.split(",")]
    sel_ranks = [parse_rank(t) for t in args.select_ranks.split(",")]
    bases = {r: null_basis(nu, rank=r)[0] for r in sorted(set(ranks) | set(sel_ranks))}

    files = sorted(Path(args.arrays).glob("*_arrays.npz"))
    if args.cases:
        files = files[:args.cases]
    per_case, agg_A, agg_B = {}, {r: [] for r in ranks}, {r: [] for r in sel_ranks}
    for f in files:
        tc = time.time()
        d = np.load(f, allow_pickle=True)
        key = f.name.replace("_arrays.npz", "")
        X = d["X"].astype(np.float64)
        counts = [int(c) for c in d["counts"]]
        tissue = d["tissue"].astype(float)
        N = X.shape[0]
        k = int(round(args.keep * N))
        zc = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        # the saved #14.6 top-k name carries that case's gamma-hat; strip it so keys match across cases
        refs = {re.sub(r" \(.*\)$", "", str(n)): np.asarray(d[f"sel_{i}"])
                for i, n in enumerate(d["sel_names"])}
        bg = tissue <= BG_TISSUE
        tis = tissue >= TIS_TISSUE
        rec = {"N": N, "k": k, "n_bg": int(bg.sum()), "n_tis": int(tis.sum()),
               "tissue_mean": float(tissue.mean()), "A": {}, "B": {}}
        # baseline separations from the signals the previous methods used
        rec["A_baseline"] = {"a_raw": auc(d["a_raw"][tis], d["a_raw"][bg]),
                             "w_page": auc(d["w_page"][tis], d["w_page"][bg])}
        for r in ranks:
            phi = residualize(zc, bases[r])
            n2 = (phi ** 2).sum(axis=1)
            rec["A"][str(r)] = {
                "auc_tis_vs_bg": auc(n2[tis], n2[bg]),
                "med_bg": float(np.median(n2[bg])) if bg.any() else None,
                "med_tis": float(np.median(n2[tis])) if tis.any() else None,
                "med_all": float(np.median(n2)),
                "frac_energy_kept": float(n2.mean()),      # z is unit norm, so this is the mean kept share
                "rho_tissue": float(np.corrcoef(n2, tissue)[0, 1]),
            }
            if r in agg_A:
                agg_A[r].append(rec["A"][str(r)])
        for r in sel_ranks:
            phi = residualize(zc, bases[r])
            order, gains, D_after, D0 = rd_greedy(phi, counts, k)
            S = np.sort(order)
            pc = per_crop_counts(S, counts)
            b = {"D0": D0, "D_after": D_after, "reduction": 1.0 - D_after / D0,
                 "per_crop": pc, "sd": float(np.std(pc)), "min": int(min(pc)), "max": int(max(pc)),
                 "sel_tissue_mean": float(tissue[S].mean()),
                 "sel_bg_frac": float(bg[S].mean()), "all_bg_frac": float(bg.mean()),
                 "first_g_gain_share": float(sum(gains[:len(counts)]) / (D0 - D_after)) if D0 > D_after else None,
                 "last_gain": float(gains[-1]),
                 "overlap": {n: float(len(np.intersect1d(S, R)) / max(len(R), 1)) for n, R in refs.items()},
                 "ref_bg_frac": {n: float(bg[R].mean()) for n, R in refs.items()},
                 "ref_sd": {n: float(np.std(per_crop_counts(R, counts))) for n, R in refs.items()},
                 "D_of_ref": {n: distortion_of(phi, counts, R) for n, R in refs.items()}}
            rec["B"][str(r)] = b
            agg_B[r].append(b)
        per_case[key] = rec
        a0 = rec["A"][str(sel_ranks[0])]
        b0 = rec["B"][str(sel_ranks[0])]
        print(f"[{key}] {time.time() - tc:.0f}s N={N} k={k} bg={int(bg.sum())} tis={int(tis.sum())} | "
              f"r={sel_ranks[0]} AUC {a0['auc_tis_vs_bg']:.3f} (a_raw {rec['A_baseline']['a_raw']:.3f}, "
              f"w_page {rec['A_baseline']['w_page']:.3f}) | RD sd {b0['sd']:.1f} "
              f"bg {b0['sel_bg_frac']:.3f} vs all {b0['all_bg_frac']:.3f} | "
              f"overlap 14.6full {b0['overlap'].get('#14.6 residual + support coverage', float('nan')):.3f}",
              flush=True)

    summary = {"null": {"P": int(nu.shape[0]), "d": int(nu.shape[1]), "rank": rank_num,
                        "sv_first": float(sv[0]), "sv_last": float(sv[-1]),
                        "dim_90": int(np.searchsorted(energy, 0.90) + 1),
                        "dim_99": int(np.searchsorted(energy, 0.99) + 1),
                        "dim_999": int(np.searchsorted(energy, 0.999) + 1)},
               "keep": args.keep, "cases": per_case}
    mean = lambda xs: (float(np.mean([x for x in xs if x is not None and np.isfinite(x)]))
                       if any(x is not None and np.isfinite(x) for x in xs) else None)
    summary["A_mean"] = {str(r): {k2: mean([c[k2] for c in agg_A[r]])
                                  for k2 in agg_A[r][0]} for r in ranks if agg_A[r]}
    summary["A_baseline_mean"] = {k2: mean([per_case[c]["A_baseline"][k2] for c in per_case])
                                  for k2 in ("a_raw", "w_page")}
    summary["B_mean"] = {}
    for r in sel_ranks:
        if not agg_B[r]:
            continue
        rows = agg_B[r]
        names = list(rows[0]["overlap"])
        summary["B_mean"][str(r)] = {
            "reduction": mean([x["reduction"] for x in rows]),
            "sd": mean([x["sd"] for x in rows]), "min": mean([x["min"] for x in rows]),
            "max": mean([x["max"] for x in rows]),
            "sel_tissue_mean": mean([x["sel_tissue_mean"] for x in rows]),
            "sel_bg_frac": mean([x["sel_bg_frac"] for x in rows]),
            "all_bg_frac": mean([x["all_bg_frac"] for x in rows]),
            "first_g_gain_share": mean([x["first_g_gain_share"] for x in rows]),
            "overlap": {n: mean([x["overlap"][n] for x in rows]) for n in names},
            "ref_bg_frac": {n: mean([x["ref_bg_frac"][n] for x in rows]) for n in names},
            "ref_sd": {n: mean([x["ref_sd"][n] for x in rows]) for n in names},
            "D_ratio_ref": {n: mean([x["D_of_ref"][n] / x["D_after"] for x in rows]) for n in names},
        }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    print("\n== Falsifier A: ||phi||^2 separates tissue from background (label = eval only)")
    print(f"{'null rank r':>12} {'AUC tis>bg':>11} {'med bg':>10} {'med tis':>10} "
          f"{'mean ||phi||^2':>15} {'rho(n2,tissue)':>15}")
    for r in ranks:
        a = summary["A_mean"].get(str(r))
        if a:
            print(f"{r:>12} {a['auc_tis_vs_bg']:>11.3f} {a['med_bg']:>10.4f} {a['med_tis']:>10.4f} "
                  f"{a['frac_energy_kept']:>15.4f} {a['rho_tissue']:>15.3f}")
    bl = summary["A_baseline_mean"]
    print(f"{'a_raw (NTRS)':>12} {bl['a_raw']:>11.3f}   <- salience already in use")
    print(f"{'w_page (14.6)':>12} {bl['w_page']:>11.3f}")

    print("\n== Falsifier B: does the selection differ at the same budget")
    for r in sel_ranks:
        s = summary["B_mean"].get(str(r))
        if not s:
            continue
        print(f"-- null rank r={r}: distortion reduction {s['reduction']:.3f} · "
              f"crop-count sd {s['sd']:.1f} ({s['min']:.0f}-{s['max']:.0f}) · "
              f"selected bg {s['sel_bg_frac']:.3f} vs all-token bg {s['all_bg_frac']:.3f} · "
              f"first-per-support share of the gain {s['first_g_gain_share']:.3f}")
        for n in s["overlap"]:
            print(f"   overlap {s['overlap'][n]:.3f}  ref bg {s['ref_bg_frac'][n]:.3f}  "
                  f"ref sd {s['ref_sd'][n]:>4.1f}  D(ref)/D(NR-PSRD) {s['D_ratio_ref'][n]:.3f}   {n}")
    print(f"\n[done] {len(per_case)} cases {time.time() - t0:.0f}s -> {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
