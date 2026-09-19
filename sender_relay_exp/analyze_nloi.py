"""Offline analysis for the NLOI Stage A runs (memory/nloi_diag.py jsonl).

Fixed BEFORE looking at any Stage A result (2026-09-13 20:0x KST):
  row          decision row r=0 (row predicting the first answer-content token) is primary;
               the mean over the probed answer rows is reported as secondary.
  per case     median over the 28 true pairs of psi_rel = ||Psi|| / (||D_i|| + ||D_j||) and
               C_rel = C / max(||D_i||, ||D_j||); the same for the 28 shuffled-group pairs.
  true vs shuf case-level win counts (true median > shuffled median) per dataset.
  distance     within-case rank tertiles of d_ij over the 28 pairs: near = ranks 0-8,
               middle = 9-18, far = 19-27 (ties broken by pair index).
  far/near     per case far-bin median minus near-bin median; dataset mean of that
               difference vs the distance-label permutation null (distances permuted among
               the 28 pairs inside each case, 2000 draws, same aggregation) -> reported as
               the observed value next to the null's 5th-95th percentile band. No p-values.
  composition  counts of parent-child / same-magnification pairs per bin, and a far-vs-near
               split restricted to same-magnification x20-x20 pairs (median distance split).
  Stage B      (only if the user accepts Stage A) per case, in each distance tertile, the
               true pair with max psi_rel and the pair with min psi_rel at r=0 (<= 6 pairs)
               plus their singles, at from_layer in {18, 24, 27, 30, 33}. `stage_b_pairs`
               below writes that list; it never looks at outputs or correctness.

usage: analyze_nloi.py <runs/nloi dir> [arm prefix nat|can] [--stageb out.json] [--max-case N]
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics as st
import sys

DS = ["gtex", "tcga_expert_vqa", "tcga_slidebench", "panda", "tcga"]
LABEL = {"gtex": "GTEx", "tcga_expert_vqa": "EVQA", "tcga_slidebench": "SB", "panda": "PANDA", "tcga": "TCGA"}


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def load(path):
    rows, seen = [], set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["case"] in seen:          # first row wins (score_dup2 convention)
            continue
        seen.add(r["case"])
        rows.append(r)
    return rows


def pair_val(p, key, row):
    if row == "mean":
        vs = [x[key] for x in p["rows"] if x.get(key) is not None]
        return sum(vs) / len(vs) if vs else None
    return p["rows"][row][key]


def tertiles(pairs):
    order = sorted(range(len(pairs)), key=lambda k: (pairs[k]["geom"]["dist"], k))
    n = len(order)
    a, b = round(n / 3), round(2 * n / 3)
    lab = [None] * n
    for rank, k in enumerate(order):
        lab[k] = "near" if rank < a else ("mid" if rank < b else "far")
    return lab


def bin_diff(vals, labs):
    far = med([v for v, l in zip(vals, labs) if l == "far"])
    near = med([v for v, l in zip(vals, labs) if l == "near"])
    return None if far is None or near is None else far - near


def analyze(rows, row=0, n_perm=2000, seed=0):
    rng = random.Random(seed)
    out = {"n": len(rows)}
    per = []
    for r in rows:
        tp = r["true"]["pairs"]
        sp = (r.get("shuffled") or {}).get("pairs") or []
        d_all = r["zero_all"]["d"][0 if row == "mean" else row]
        c = {
            "psi_t": med([pair_val(p, "psi_rel", row) for p in tp]),
            "C_t": med([pair_val(p, "C_rel", row) for p in tp]),
            "cos_t": med([pair_val(p, "cos_ij", row) for p in tp]),
            "psi_s": med([pair_val(p, "psi_rel", row) for p in sp]) if sp else None,
            "C_s": med([pair_val(p, "C_rel", row) for p in sp]) if sp else None,
            "cos_s": med([pair_val(p, "cos_ij", row) for p in sp]) if sp else None,
            "d1_over_all": med([s["d"][0 if row == "mean" else row] / d_all for s in r["true"]["singles"]]) if d_all else None,
            "d1s_over_all": med([s["d"][0 if row == "mean" else row] / d_all for s in r["shuffled"]["singles"]]) if (d_all and sp) else None,
            "psi_all_rel": (r["true"]["all"][0 if row == "mean" else row]["psi_all"] / d_all) if d_all else None,
            "drift_over_all": (r["path_drift"][0 if row == "mean" else row] / d_all) if d_all else None,
            "pair_flip": sum(p["out"]["flip"][0] for p in tp), "single_flip": sum(s["out"]["flip"][0] for s in r["true"]["singles"]),
            "shuf_pair_flip": sum(p["out"]["flip"][0] for p in sp), "zero_flip": int(r["zero_all"]["out"]["flip"][0]),
        }
        if all(p.get("geom") and p["geom"]["dist"] is not None for p in tp):
            labs = tertiles(tp)
            psi = [pair_val(p, "psi_rel", row) for p in tp]
            cr = [pair_val(p, "C_rel", row) for p in tp]
            c["bins"] = {b: {"psi": med([v for v, l in zip(psi, labs) if l == b]),
                             "C": med([v for v, l in zip(cr, labs) if l == b]),
                             "pc": sum(1 for p, l in zip(tp, labs) if l == b and p["geom"]["parent_child"]),
                             "same": sum(1 for p, l in zip(tp, labs) if l == b and p["geom"]["same_mag"])}
                         for b in ("near", "mid", "far")}
            c["fn_psi"] = bin_diff(psi, labs)
            c["fn_C"] = bin_diff(cr, labs)
            c["_psi"], c["_C"], c["_labs"] = psi, cr, labs
            x20 = [k for k, p in enumerate(tp) if p["geom"]["mag"] == [20, 20]]
            if len(x20) >= 4:
                dd = sorted(x20, key=lambda k: (tp[k]["geom"]["dist"], k))
                h = len(dd) // 2
                c["x20_fn_psi"] = med([psi[k] for k in dd[h:]]) - med([psi[k] for k in dd[:h]])
                c["x20_fn_C"] = med([cr[k] for k in dd[h:]]) - med([cr[k] for k in dd[:h]])
        per.append(c)
    out["per"] = per
    g = lambda k: [c[k] for c in per if c.get(k) is not None]
    out["psi_t"], out["psi_s"] = med(g("psi_t")), med(g("psi_s"))
    out["C_t"], out["C_s"] = med(g("C_t")), med(g("C_s"))
    out["cos_t"], out["cos_s"] = med(g("cos_t")), med(g("cos_s"))
    both = [c for c in per if c.get("psi_s") is not None]
    out["win_psi"] = (sum(c["psi_t"] > c["psi_s"] for c in both), sum(c["psi_t"] < c["psi_s"] for c in both), len(both))
    out["win_C"] = (sum(c["C_t"] > c["C_s"] for c in both), sum(c["C_t"] < c["C_s"] for c in both), len(both))
    out["d1_over_all"], out["d1s_over_all"] = med(g("d1_over_all")), med(g("d1s_over_all"))
    out["psi_all_rel"], out["drift_over_all"] = med(g("psi_all_rel")), med(g("drift_over_all"))
    for k in ("pair_flip", "single_flip", "shuf_pair_flip", "zero_flip"):
        out[k] = sum(c[k] for c in per)
    geo = [c for c in per if "bins" in c]
    out["n_geo"] = len(geo)
    if geo:
        for b in ("near", "mid", "far"):
            out[f"bin_{b}"] = {"psi": med([c["bins"][b]["psi"] for c in geo]), "C": med([c["bins"][b]["C"] for c in geo]),
                               "pc": sum(c["bins"][b]["pc"] for c in geo), "same": sum(c["bins"][b]["same"] for c in geo)}
        for key, src in (("fn_psi", "_psi"), ("fn_C", "_C")):
            obs = st.mean([c[key] for c in geo if c[key] is not None])
            null = []
            for _ in range(n_perm):
                vals = []
                for c in geo:
                    labs = list(c["_labs"]); rng.shuffle(labs)
                    d = bin_diff(c[src], labs)
                    if d is not None:
                        vals.append(d)
                null.append(st.mean(vals))
            null.sort()
            out[key] = {"obs_mean": obs, "null_p05": null[int(0.05 * n_perm)], "null_p95": null[int(0.95 * n_perm) - 1],
                        "cases_far_ge_near": sum(1 for c in geo if c[key] is not None and c[key] >= 0)}
        out["x20_fn_psi"] = med(g("x20_fn_psi")); out["x20_fn_C"] = med(g("x20_fn_C"))
        out["x20_cases_far_ge_near_psi"] = sum(1 for c in geo if c.get("x20_fn_psi") is not None and c["x20_fn_psi"] >= 0)
        out["x20_n"] = sum(1 for c in geo if c.get("x20_fn_psi") is not None)
    for c in per:
        for k in ("_psi", "_C", "_labs"):
            c.pop(k, None)
    return out


def stage_b_pairs(rows):
    sel = {}
    for r in rows:
        tp = r["true"]["pairs"]
        if not all(p.get("geom") and p["geom"]["dist"] is not None for p in tp):
            continue
        labs = tertiles(tp)
        pick = []
        for b in ("near", "mid", "far"):
            cand = [k for k, l in enumerate(labs) if l == b and tp[k]["rows"][0]["psi_rel"] is not None]
            if not cand:
                continue
            hi = max(cand, key=lambda k: (tp[k]["rows"][0]["psi_rel"], -k))
            lo = min(cand, key=lambda k: (tp[k]["rows"][0]["psi_rel"], k))
            for tag, k in (("max", hi), ("min", lo)):
                pick.append({"bin": b, "tag": tag, "i": tp[k]["i"], "j": tp[k]["j"]})
        sel[str(r["case"])] = pick
    return sel


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    base = sys.argv[1]
    arm = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "nat"
    stageb = sys.argv[sys.argv.index("--stageb") + 1] if "--stageb" in sys.argv else None
    max_case = int(sys.argv[sys.argv.index("--max-case") + 1]) if "--max-case" in sys.argv else None
    sel_all = {}
    for row in (0, "mean"):
        print(f"\n=== arm={arm} row={row} ===")
        print("set | n | psi_rel true/shuf | win psi (T>S / T<S) | C_rel true/shuf | win C | cos true/shuf | "
              "|D_i|/|D_all| true/shuf | psi_all/|D_all| | drift/|D_all| | flips pair/single/shufpair/zeroA")
        for ds in DS:
            p = os.path.join(base, f"nloi_{arm}_{ds}.jsonl")
            if not os.path.exists(p):
                continue
            rows = [r for r in load(p) if max_case is None or r["case"] < max_case]
            if not rows:
                continue
            a = analyze(rows, row=row)
            print(f"{LABEL[ds]} | {a['n']} | {f(a['psi_t'])}/{f(a['psi_s'])} | {a['win_psi'][0]}/{a['win_psi'][1]} of {a['win_psi'][2]} | "
                  f"{f(a['C_t'])}/{f(a['C_s'])} | {a['win_C'][0]}/{a['win_C'][1]} | {f(a['cos_t'])}/{f(a['cos_s'])} | "
                  f"{f(a['d1_over_all'])}/{f(a['d1s_over_all'])} | {f(a['psi_all_rel'])} | {f(a['drift_over_all'])} | "
                  f"{a['pair_flip']}/{a['single_flip']}/{a['shuf_pair_flip']}/{a['zero_flip']}")
            if row == 0 and a.get("n_geo"):
                print(f"   bins psi near/mid/far {f(a['bin_near']['psi'])}/{f(a['bin_mid']['psi'])}/{f(a['bin_far']['psi'])} "
                      f"C {f(a['bin_near']['C'])}/{f(a['bin_mid']['C'])}/{f(a['bin_far']['C'])} "
                      f"parent-child {a['bin_near']['pc']}/{a['bin_mid']['pc']}/{a['bin_far']['pc']} "
                      f"same-mag {a['bin_near']['same']}/{a['bin_mid']['same']}/{a['bin_far']['same']}")
                for key in ("fn_psi", "fn_C"):
                    d = a[key]
                    print(f"   {key}: obs mean {f(d['obs_mean'], 4)} null5-95 [{f(d['null_p05'], 4)}, {f(d['null_p95'], 4)}] "
                          f"cases far>=near {d['cases_far_ge_near']}/{a['n_geo']}")
                print(f"   x20-x20 only far-near psi med {f(a['x20_fn_psi'], 4)} C med {f(a['x20_fn_C'], 4)} "
                      f"cases far>=near {a['x20_cases_far_ge_near_psi']}/{a['x20_n']}")
            if row == 0 and stageb:
                sel_all[ds] = stage_b_pairs(rows)
    if stageb:
        json.dump(sel_all, open(stageb, "w"), indent=1)
        print(f"\nStage B pair list -> {stageb}")


if __name__ == "__main__":
    main()
