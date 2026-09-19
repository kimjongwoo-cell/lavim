"""Analysis for memory/rss_diag.py outputs.

usage:
  rss_analyze.py --gate <stageA_dir>          first line PASS/FAIL (Stage A implementation gate)
  rss_analyze.py <dir> [<dir> ...]            N=25 estimator validation report (+ analysis_summary.json)
"""
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

DATASETS = ["tcga_expert_vqa", "gtex", "tcga_slidebench", "tcga", "panda"]


def load(dirs):
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "rss_*.jsonl"))):
            ds = os.path.basename(f)[4:-6]
            for line in open(f):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["ds"] = ds
                    r["_file"] = f
                    recs.append(r)
    return recs


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def argmax_set(v):
    top = max(v)
    return {i for i, x in enumerate(v) if x == top}


def med(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def gate(stage_dir):
    recs = load([stage_dir])
    problems = []
    by_ds = defaultdict(list)
    for r in recs:
        by_ds[r["ds"]].append(r)
    if not recs:
        problems.append("no records")
    for r in recs:
        tag = f"{r['ds']}#{r.get('case')}"
        if r.get("skip") or r.get("error"):
            problems.append(f"{tag}: skip={r.get('skip')} error={r.get('error')}")
            continue
        px = r.get("proxy") or {}
        if px.get("identity_max_rel_err") is None or px["identity_max_rel_err"] > 1e-3:
            problems.append(f"{tag}: identity {px.get('identity_max_rel_err')}")
        if px.get("recon_rel_err") is None or px["recon_rel_err"] > 0.05:
            problems.append(f"{tag}: recon {px.get('recon_rel_err')}")
        fl = r.get("floor") or {}
        if fl.get("S_A") is None or fl["S_A"] > 1e-3 or not fl.get("top_same", False):
            problems.append(f"{tag}: floor {fl}")
        loo = r.get("loo")
        if loo is not None and (len(loo) != len(px.get("s_hat", [])) or [x["support"] for x in loo] != px.get("supports")):
            problems.append(f"{tag}: LOO/proxy support mismatch")
    head = "PASS" if not problems else "FAIL " + "; ".join(problems[:6])
    print(head)
    for r in recs:
        px = r.get("proxy") or {}
        fl = r.get("floor") or {}
        print(f"  {r['ds']:16} case={r.get('case')} supports={len(px.get('s_hat', []))} ident={px.get('identity_max_rel_err')} "
              f"recon={px.get('recon_rel_err')} rho_v={px.get('rho_v')} floor_SR={fl.get('S_R')} floor_SA={fl.get('S_A')} "
              f"top_same={fl.get('top_same')} proxy_sec={px.get('proxy_sec')} sec={r.get('sec')} "
              f"maxSA={max((x['S_A'] for x in r.get('loo') or []), default=None)}")
    return 0 if not problems else 1


def bootstrap_mean_ci(v, n=2000, seed=0):
    if len(v) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def report(dirs):
    recs = [r for r in load(dirs) if r.get("loo") and not r.get("error") and not r.get("skip")]
    bad = [r for r in load(dirs) if not (r.get("loo") and not r.get("error") and not r.get("skip"))]
    print(f"records with LOO: {len(recs)}  (skipped/errored: {len(bad)})")
    for r in bad:
        print(f"  skipped {r['ds']}#{r.get('case')}: skip={r.get('skip')} error={r.get('error')}")
    cases = []
    for r in recs:
        px = r["proxy"]
        loo = r["loo"]
        if [x["support"] for x in loo] != px["supports"]:
            print(f"  support mismatch {r['ds']}#{r['case']} — excluded")
            continue
        fl = r.get("floor") or {}
        c = {
            "ds": r["ds"], "case": r["case"], "G": len(loo),
            "s_hat": px["s_hat"], "mass": px["mass_sum_heads"], "size": [float(x) for x in px["size"]],
            "qk": px.get("qk_lme") or [0.0] * len(px["s_hat"]),
            "S_A": [x["S_A"] for x in loo], "S_R": [x["S_R"] for x in loo],
            "S_A_rep": [x["S_A_vs_replay"] for x in loo], "S_R_rep": [x["S_R_vs_replay"] for x in loo],
            "flip": [x["top_flip"] for x in loo], "floor_SA": fl.get("S_A"), "floor_SR": fl.get("S_R"),
            "rho_v": px.get("rho_v"), "ident": px.get("identity_max_rel_err"), "recon": px.get("recon_rel_err"),
            "proxy_sec": px.get("proxy_sec"), "sec": r.get("sec"), "floor_sec": fl.get("sec"),
            "mag": px.get("mag"),
        }
        for tgt in ("S_A", "S_R"):
            for base in ("s_hat", "mass", "size", "qk"):
                c[f"rho_{base}_{tgt}"] = spearman(c[base], c[tgt])
            t = argmax_set(c[tgt])
            for base in ("s_hat", "mass", "size", "qk"):
                b = argmax_set(c[base])
                c[f"top1_{base}_{tgt}"] = (len(b & t) > 0) if len(b) == 1 else None  # ties in the estimator = undefined
        c["rho_SR_SA"] = spearman(c["S_R"], c["S_A"])
        cases.append(c)
    if not cases:
        print("no usable cases")
        return 1

    def pooled(base, tgt, within=False):
        xs, ys = [], []
        for c in cases:
            if within:
                rb, rt = ranks(c[base]), ranks(c[tgt])
                n = max(1, c["G"] - 1)
                xs += [x / n for x in rb]
                ys += [y / n for y in rt]
            else:
                xs += c[base]
                ys += c[tgt]
        return spearman(xs, ys)

    summary = {"n_cases": len(cases), "by_dataset": {}, "overall": {}}
    for tgt in ("S_A", "S_R"):
        print(f"\n=== target {tgt} (per-case Spearman over supports; G≈{med([c['G'] for c in cases])}) ===")
        print(f"{'dataset':16} {'n':>3} {'ρ(Ŝ)':>7} {'ρ(mass)':>8} {'ρ(size)':>8} {'ρ(qk)':>7} {'Ŝ>mass':>7} {'top1 Ŝ':>7} {'top1 mass':>9} {'top1 size':>9} {'top1 qk':>8}")
        for ds in DATASETS + ["ALL"]:
            cs = [c for c in cases if ds == "ALL" or c["ds"] == ds]
            if not cs:
                continue
            rh = [c[f"rho_s_hat_{tgt}"] for c in cs if c[f"rho_s_hat_{tgt}"] is not None]
            rm = [c[f"rho_mass_{tgt}"] for c in cs if c[f"rho_mass_{tgt}"] is not None]
            rs = [c[f"rho_size_{tgt}"] for c in cs if c[f"rho_size_{tgt}"] is not None]
            rq = [c[f"rho_qk_{tgt}"] for c in cs if c[f"rho_qk_{tgt}"] is not None]
            paired = [(c[f"rho_s_hat_{tgt}"], c[f"rho_mass_{tgt}"]) for c in cs
                      if c[f"rho_s_hat_{tgt}"] is not None and c[f"rho_mass_{tgt}"] is not None]
            win = sum(a > b for a, b in paired)
            t1 = lambda k: (sum(1 for c in cs if c[k] is True), sum(1 for c in cs if c[k] is not None))
            a1, a2 = t1(f"top1_s_hat_{tgt}")
            b1, b2 = t1(f"top1_mass_{tgt}")
            s1, s2 = t1(f"top1_size_{tgt}")
            q1, q2 = t1(f"top1_qk_{tgt}")
            print(f"{ds:16} {len(cs):>3} {fmt(med(rh)):>7} {fmt(med(rm)):>8} {fmt(med(rs)):>8} {fmt(med(rq)):>7} {win:>3}/{len(paired):<3} "
                  f"{a1:>3}/{a2:<3} {b1:>5}/{b2:<3} {s1:>5}/{s2:<3} {q1:>4}/{q2:<3}")
            row = {"n": len(cs), "rho_hat_median": med(rh), "rho_mass_median": med(rm), "rho_size_median": med(rs),
                   "rho_qk_median": med(rq), "hat_beats_mass": [win, len(paired)], "top1_hat": [a1, a2],
                   "top1_mass": [b1, b2], "top1_size": [s1, s2], "top1_qk": [q1, q2]}
            if ds == "ALL":
                d = [a - b for a, b in paired]
                row["delta_rho_mean"] = sum(d) / len(d) if d else None
                row["delta_rho_ci95"] = bootstrap_mean_ci(d)
                row["pooled_raw"] = {b: pooled(b, tgt) for b in ("s_hat", "mass", "size", "qk")}
                row["pooled_within_rank"] = {b: pooled(b, tgt, within=True) for b in ("s_hat", "mass", "size", "qk")}
                row["chance_top1"] = sum(1.0 / c["G"] for c in cs)
                summary["overall"][tgt] = row
                print(f"  Δρ(Ŝ−mass) mean {fmt(row['delta_rho_mean'])} · bootstrap 95% range {row['delta_rho_ci95']}")
                print(f"  pooled raw ρ: Ŝ {fmt(row['pooled_raw']['s_hat'])} · mass {fmt(row['pooled_raw']['mass'])} · size {fmt(row['pooled_raw']['size'])}")
                print(f"  pooled within-case rank ρ: Ŝ {fmt(row['pooled_within_rank']['s_hat'])} · mass {fmt(row['pooled_within_rank']['mass'])} · size {fmt(row['pooled_within_rank']['size'])}")
                print(f"  top-1 chance (sum 1/G) {row['chance_top1']:.1f} of {len(cs)}")
            else:
                summary["by_dataset"].setdefault(ds, {})[tgt] = row

    print("\n=== magnitudes and checks ===")
    all_SA = [x for c in cases for x in c["S_A"]]
    all_SR = [x for c in cases for x in c["S_R"]]
    floors_A = [c["floor_SA"] for c in cases if c["floor_SA"] is not None]
    floors_R = [c["floor_SR"] for c in cases if c["floor_SR"] is not None]
    thr = max([1e-9] + floors_A) * 10
    print(f"S_A LOO median {med(all_SA):.3g} · max {max(all_SA):.3g} · floor max {max(floors_A) if floors_A else None} · "
          f"supports with S_A > 10×max floor: {sum(x > thr for x in all_SA)}/{len(all_SA)}")
    print(f"S_R LOO median {med(all_SR):.3g} · max {max(all_SR):.3g} · floor max {max(floors_R) if floors_R else None}")
    flips = sum(sum(1 for f in c["flip"] if f) for c in cases)
    print(f"decision-row top token changed by deleting one support: {flips}/{len(all_SA)} supports · cases with ≥1 change "
          f"{sum(1 for c in cases if any(c['flip']))}/{len(cases)}")
    rsa = [c["rho_SR_SA"] for c in cases if c["rho_SR_SA"] is not None]
    print(f"per-case ρ(S_R, S_A) median {fmt(med(rsa))} (role-state LOO ↔ Answerer LOO link)")
    print(f"rho_v at proxy row median {fmt(med([c['rho_v'] for c in cases]), 4)} · identity max {max(c['ident'] for c in cases):.2e} · "
          f"recon max {max(c['recon'] for c in cases):.2e}")
    print(f"proxy sec median {fmt(med([c['proxy_sec'] for c in cases]), 4)} · one no-delete replay sec median {fmt(med([c['floor_sec'] for c in cases]), 2)} · "
          f"case LOO total sec median {fmt(med([c['sec'] for c in cases]), 1)}")
    summary["magnitudes"] = {"S_A_median": med(all_SA), "S_A_max": max(all_SA), "floor_SA_max": max(floors_A) if floors_A else None,
                             "S_A_above_10x_floor": [sum(x > thr for x in all_SA), len(all_SA)],
                             "S_R_median": med(all_SR), "top_flip_supports": [flips, len(all_SA)],
                             "rho_SR_SA_median": med(rsa), "rho_v_median": med([c["rho_v"] for c in cases]),
                             "proxy_sec_median": med([c["proxy_sec"] for c in cases]),
                             "replay_sec_median": med([c["floor_sec"] for c in cases])}
    summary["cases"] = [{k: v for k, v in c.items()} for c in cases]
    out = os.path.join(os.path.dirname(os.path.abspath(dirs[0])), "analysis_summary.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"\nsummary -> {out}")
    return 0


def profile(dirs):
    """Stage B: cheap Reasoner profile only (no LOO) — flatness and agreement with baselines."""
    recs = [r for r in load(dirs) if r.get("proxy") and not r.get("error") and not r.get("skip")]
    print(f"records with proxy: {len(recs)}")
    rows = []
    for r in recs:
        px = r["proxy"]
        s = px["s_hat"]
        srt = sorted(s)
        rows.append({"case": r["case"], "G": len(s), "s_max": max(s), "s_med": med(s), "max_over_med": max(s) / max(med(s), 1e-12),
                     "s_over_o": max(s) / max(px.get("o_norm", 1e-12), 1e-12), "rho_v": px.get("rho_v"),
                     "rho_mass": spearman(s, px["mass_sum_heads"]), "rho_size": spearman(s, [float(x) for x in px["size"]]),
                     "rho_qk": spearman(s, px.get("qk_lme") or [0.0] * len(s)),
                     "top_is_x5": (px["mag"][s.index(max(s))] == min(m for m in px["mag"] if m)) if any(px["mag"]) else None,
                     "second_gap": (srt[-1] - srt[-2]) / max(srt[-1], 1e-12) if len(srt) > 1 else None})
        print(f"  case {r['case']:>3} G={len(s)} Ŝmax={max(s):.4g} Ŝmed={med(s):.4g} max/med={rows[-1]['max_over_med']:.2f} "
              f"Ŝmax/|o|={rows[-1]['s_over_o']:.3f} rho_v={px.get('rho_v'):.4f} ρ(Ŝ,mass)={fmt(rows[-1]['rho_mass'])} "
              f"ρ(Ŝ,size)={fmt(rows[-1]['rho_size'])} ρ(Ŝ,qk)={fmt(rows[-1]['rho_qk'])} top=x5:{rows[-1]['top_is_x5']}")
    if rows:
        print(f"median max/med {fmt(med([x['max_over_med'] for x in rows]), 2)} · median Ŝmax/|o| {fmt(med([x['s_over_o'] for x in rows]), 3)} · "
              f"median ρ(Ŝ,mass) {fmt(med([x['rho_mass'] for x in rows]))} · ρ(Ŝ,size) {fmt(med([x['rho_size'] for x in rows]))} · "
              f"ρ(Ŝ,qk) {fmt(med([x['rho_qk'] for x in rows]))} · top support is x5 in {sum(1 for x in rows if x['top_is_x5'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--gate":
        sys.exit(gate(sys.argv[2]))
    if len(sys.argv) >= 3 and sys.argv[1] == "--profile":
        sys.exit(profile(sys.argv[2:]))
    sys.exit(report(sys.argv[1:]))
