#!/usr/bin/env python3
"""#14.6 with ONLY the coverage term swapped for the clamped (rate-distortion) form.

Everything else is #14.6 exactly as the Notion page defines it (3dbd771b-c9e2-81ec):
  demand   w_{g,i} = softmax_g(x_{g,i} - gamma_hat * b_i)    (null-template residual salience)
  support  g = crop, single global budget, no quota
  greedy   marginal gain, same order rule
  z_i      = h_i / ||h_i||   (NO null-subspace residualisation: that part was measured inert)

What changes is only how much credit a representative gives a token.

  #14.6      F  = sum_g sum_i w_gi * max_{j in S_g} (1 + cos_ij) / 2          -> maximise
  V1 (audit) F  = sum_g sum_i w_gi * max_{j in S_g} max(cos_ij, 0)            -> identical allocation
  clamp      D  = sum_g sum_i w_gi * min[ tau , min_{j in S_g} ||z_i-z_j||^2 ] -> minimise
             with ||z_i-z_j||^2 = 2(1-cos_ij) and a per-token no-store cost tau.

The clamp is the one piece of NR-PSRD that worked (runs/vit_nrpsrd_gtex): a representative only
earns credit where it beats the token's own no-store cost. Writing the clamp as a similarity,

             s'_ij = 1 - min[1, 2(1-cos_ij)/tau] = max(0, 1 - 2(1-cos_ij)/tau)

makes the difference explicit: #14.6 pays 0.5 at cos=0, V1 pays 0+ from cos=0, and the clamp pays
nothing until cos > 1 - tau/2. tau = 2 reproduces V1 exactly, so the sweep interpolates from the
known-inert case to strict forms. tau = 1 is the value the rate-distortion form implies for
unit-norm z (no-store cost ||z||^2 = 1).

usage: vit_146clamp_gtex.py <out_dir> [--arrays ...] [--keep 0.25] [--taus 2,1,0.5,0.25]
                            [--threads 12] [--cases N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
ARRAYS = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vit_ntrs_gtex/arrays")
sys.path.insert(0, str(REPO / "sender_relay_exp"))

import numpy as np  # noqa: E402

NULL = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vit_nullcal/null_maps.npz")
BLOCK = 23
EPS = 1e-3
BG_TISSUE = 0.10
TIS_TISSUE = 0.90


# --------------------------------------------------------------------------- pure


def centred_log(maps, eps: float = EPS) -> np.ndarray:
    lg = np.log(np.asarray(maps, dtype=float) + eps)
    return lg - lg.mean(axis=-1, keepdims=True)


def demands(a_raw: np.ndarray, gamma: float, b: np.ndarray, counts) -> dict:
    """The four demand weights of the #14.6 audit, rebuilt from the saved a_raw and gamma.

    page   w_gi = softmax_g(x_gi - gamma b_i)        support-local salience   (= V0, stored w_page)
    slide  w_i  = exp(x_i - gamma b) / sum over ALL  global salience          (= V3)
    unif_g w_gi = 1 / N_g                            support-local uniform    (= V2)
    unif_n w_i  = 1 / N                              global uniform
    """
    counts = [int(c) for c in counts]
    offs = np.cumsum([0] + counts)
    N = int(offs[-1])
    resid = []
    for g in range(len(counts)):
        x = centred_log(a_raw[offs[g]:offs[g + 1]])
        resid.append(np.exp(x - gamma * b))
    tot = sum(float(r.sum()) for r in resid)
    return {
        "page": np.concatenate([r / r.sum() for r in resid]),
        "slide": np.concatenate(resid) / tot,
        "unif_g": np.concatenate([np.full(c, 1.0 / c) for c in counts]),
        "unif_n": np.full(N, 1.0 / N),
    }


def credit(cos: np.ndarray, tau: float) -> np.ndarray:
    """s'_ij = max(0, 1 - 2(1-cos)/tau). tau=2 -> max(cos,0); tau=inf -> 1 (any rep covers all)."""
    return np.clip(1.0 - 2.0 * (1.0 - cos) / float(tau), 0.0, None)


def weighted_greedy(z: np.ndarray, w: np.ndarray, counts, k: int, tau: float):
    """Same greedy as #14.6 (support-local max coverage, global budget) with credit() as s.

    Returns (order, gains, F_final, F_max) where F_max = sum_i w_i (the value if every token had a
    perfect representative), so F_final / F_max is comparable across tau.
    """
    counts = [int(c) for c in counts]
    offs = np.cumsum([0] + counts)
    N = int(offs[-1])
    k = min(int(k), N)
    sims, cov, gains = [], [], []
    for g in range(len(counts)):
        zg = z[offs[g]:offs[g + 1]]
        s = credit(zg @ zg.T, tau)
        np.fill_diagonal(s, 1.0)                      # a token represents itself perfectly
        sims.append(s)
        cov.append(np.zeros(len(zg)))
        gains.append((w[offs[g]:offs[g + 1]][:, None] * s).sum(axis=0))
    chosen = np.zeros(N, dtype=bool)
    order, got = [], []
    for _ in range(k):
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
    F_final = float(sum(float((w[offs[g]:offs[g + 1]] * cov[g]).sum()) for g in range(len(counts))))
    return np.array(order, dtype=int), got, F_final, float(w.sum())


def per_crop_counts(S, counts):
    offs = np.cumsum([0] + [int(c) for c in counts])
    S = np.asarray(S)
    return [int(((S >= offs[g]) & (S < offs[g + 1])).sum()) for g in range(len(counts))]


def first_pick_share(z, w, counts, k, tau):
    """What fraction of the achieved coverage the FIRST pick of each support takes (page §4 of the
    #14.6 audit reported 0.799 for the original form)."""
    counts = [int(c) for c in counts]
    offs = np.cumsum([0] + counts)
    tot, best = 0.0, 0.0
    for g in range(len(counts)):
        zg = z[offs[g]:offs[g + 1]]
        wg = w[offs[g]:offs[g + 1]]
        s = credit(zg @ zg.T, tau)
        np.fill_diagonal(s, 1.0)
        single = (wg[:, None] * s).sum(axis=0)
        best += float(single.max())
        tot += float(wg.sum())               # the support's max possible coverage = sum of demand
    return best / tot if tot else float("nan")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arrays", default=str(ARRAYS))
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--taus", default="2,1,0.5,0.25")
    ap.add_argument("--demands", default="page,slide,unif_g,unif_n")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cases", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taus = [float(t) for t in args.taus.split(",")]
    dems = [x.strip() for x in args.demands.split(",")]
    b = centred_log(np.load(NULL)["maps"][:, BLOCK]).mean(axis=0)
    t0 = time.time()

    files = sorted(Path(args.arrays).glob("*_arrays.npz"))
    if args.cases:
        files = files[:args.cases]
    per_case, agg = {}, {f"{dm}|{t}": [] for dm in dems for t in taus}
    for f in files:
        tc = time.time()
        d = np.load(f, allow_pickle=True)
        key = f.name.replace("_arrays.npz", "")
        X = d["X"].astype(np.float64)
        counts = [int(c) for c in d["counts"]]
        tissue = d["tissue"].astype(float)
        w_page = d["w_page"].astype(np.float64)
        hot8 = set(int(h) for h in d["hot8"])
        N = X.shape[0]
        k = int(round(args.keep * N))
        z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        refs = {re.sub(r" \(.*\)$", "", str(n)): np.asarray(d[f"sel_{i}"])
                for i, n in enumerate(d["sel_names"])}
        ref146 = refs["#14.6 residual + support coverage"]
        bg = tissue <= BG_TISSUE
        offs = np.cumsum([0] + counts)
        grid = np.concatenate([np.arange(c) % 256 for c in counts])   # local cell index per token
        W = demands(d["a_raw"].astype(np.float64), float(d["gamma"]), b, counts)
        chk = float(np.abs(W["page"] - w_page).max())
        rec = {"N": N, "k": k, "gamma": float(d["gamma"]), "w_page_maxdiff": chk, "taus": {}}
        assert chk < 1e-9, f"w_page rebuild mismatch {chk}"
        for dem in dems:
          for tau in taus:
            w = W[dem]
            order, gains, F, Fmax = weighted_greedy(z, w, counts, k, tau)
            S = np.sort(order)
            pc = per_crop_counts(S, counts)
            hot_mass = float(np.mean([1.0 if int(g) in hot8 else 0.0 for g in grid[S]]))
            rec["taus"][f"{dem}|{tau}"] = {
                "sd": float(np.std(pc)), "min": int(min(pc)), "max": int(max(pc)), "per_crop": pc,
                "first_share": first_pick_share(z, w, counts, k, tau),
                "coverage_frac": F / Fmax,
                "sel_tissue": float(tissue[S].mean()), "sel_bg": float(bg[S].mean()),
                "all_tissue": float(tissue.mean()), "all_bg": float(bg.mean()),
                "hot8_share": hot_mass,
                "last_gain": float(gains[-1]), "first_gain": float(gains[0]),
                "overlap": {n: float(len(np.intersect1d(S, R)) / max(len(R), 1))
                            for n, R in refs.items()},
            }
            agg[f"{dem}|{tau}"].append(rec["taus"][f"{dem}|{tau}"])
        rec["ref146"] = {"sd": float(np.std(per_crop_counts(ref146, counts))),
                         "sel_tissue": float(tissue[ref146].mean()),
                         "sel_bg": float(bg[ref146].mean()),
                         "hot8_share": float(np.mean([1.0 if int(g) in hot8 else 0.0
                                                      for g in grid[ref146]]))}
        per_case[key] = rec
        r0 = rec["taus"][f"{dems[0]}|{taus[0]}"]
        r1 = rec["taus"][f"{dems[min(1, len(dems) - 1)]}|{taus[min(1, len(taus) - 1)]}"]
        print(f"[{key}] {time.time() - tc:.0f}s N={N} k={k} gamma={rec['gamma']:.2f} | "
              f"{dems[0]}|{taus[0]} sd {r0['sd']:.1f} first {r0['first_share']:.3f} | "
              f"{dems[min(1, len(dems) - 1)]}|{taus[min(1, len(taus) - 1)]} sd {r1['sd']:.1f} "
              f"first {r1['first_share']:.3f} ov146 {r1['overlap']['#14.6 residual + support coverage']:.3f} "
              f"tis {r1['sel_tissue']:.3f} | #14.6 sd {rec['ref146']['sd']:.1f} "
              f"tis {rec['ref146']['sel_tissue']:.3f}", flush=True)

    mean = lambda xs: float(np.mean([x for x in xs if x is not None and np.isfinite(x)]))
    summary = {"keep": args.keep, "taus": taus, "cases": per_case, "mean": {}}
    keys = [f"{dm}|{t}" for dm in dems for t in taus]
    names = list(next(iter(per_case.values()))["taus"][keys[0]]["overlap"])
    for kk in keys:
        rows = agg[kk]
        summary["mean"][kk] = {
            k2: mean([r[k2] for r in rows]) for k2 in
            ("sd", "min", "max", "first_share", "coverage_frac", "sel_tissue", "sel_bg",
             "all_tissue", "all_bg", "hot8_share", "first_gain", "last_gain")}
        summary["mean"][kk]["overlap"] = {n: mean([r["overlap"][n] for r in rows]) for n in names}
    summary["ref146_mean"] = {k2: mean([per_case[c]["ref146"][k2] for c in per_case])
                              for k2 in ("sd", "sel_tissue", "sel_bg", "hot8_share")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    r = summary["ref146_mean"]
    print("\n== #14.6 coverage term -> clamped form, same demand / support / budget / greedy")
    print(f"   s'_ij = max(0, 1 - 2(1-cos)/tau).  tau=2 is exactly the audit's V1 (max(cos,0)).")
    print(f"{'demand':>7} {'tau':>5} {'cutoff':>7} {'crop sd':>8} {'min':>5} {'max':>5} "
          f"{'first':>6} {'cover':>6} {'tissue':>7} {'bg':>6} {'hot8':>6} "
          f"{'ov146':>6} {'ov topk':>8}")
    for dm in dems:
        for tau in taus:
            m = summary["mean"][f"{dm}|{tau}"]
            print(f"{dm:>7} {tau:>5} {1 - tau / 2:>7.2f} {m['sd']:>8.1f} {m['min']:>5.0f} "
                  f"{m['max']:>5.0f} {m['first_share']:>6.3f} {m['coverage_frac']:>6.3f} "
                  f"{m['sel_tissue']:>7.3f} {m['sel_bg']:>6.3f} {m['hot8_share']:>6.3f} "
                  f"{m['overlap']['#14.6 residual + support coverage']:>6.3f} "
                  f"{m['overlap']['#14.6 residual top-k']:>8.3f}")
    m0 = summary["mean"][keys[0]]
    print(f"\n   #14.6 as published: crop sd {r['sd']:.1f} · sel tissue {r['sel_tissue']:.3f} · "
          f"sel bg {r['sel_bg']:.3f} · hot8 {r['hot8_share']:.3f} · first-pick share 0.799 (prior audit)")
    print(f"   all-token reference: tissue {m0['all_tissue']:.3f} · bg {m0['all_bg']:.3f} · "
          f"hot8 uniform 8/256 = 0.031")
    print("\n   overlap with every stored selector:")
    for n in names:
        row = " ".join(f"{summary['mean'][kk]['overlap'][n]:.3f}" for kk in keys)
        print(f"     {row}   {n}")
    print(f"     (columns: {' '.join(keys)})")
    print(f"\n[done] {len(per_case)} cases {time.time() - t0:.0f}s -> {out / 'summary.json'}",
          flush=True)


if __name__ == "__main__":
    main()
