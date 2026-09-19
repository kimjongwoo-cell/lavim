#!/usr/bin/env python3
"""Exact scoring of an arm vs NTRS50 (+decode fix) and vs base (+decode fix) on the SAME dataset_index set.
usage: c2_pair_score.py <runs subdir> [ds ...]   (default gtex)   -> prints a row per dataset, writes runs/<subdir>/pair_score.json
Also aggregates the arm's per-case jsonl diagnostics if present (lu_shard*.jsonl / velh_shard*.jsonl / textcut_shard*.jsonl)."""
import glob, importlib.util, json, sys
from pathlib import Path
from statistics import mean
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
SUB = sys.argv[1]; DSS = sys.argv[2:] or ["gtex"]
nrm = rs.am.normalize
def g(roots, ds):
    c, t = rs.collect(rs.run_dirs(roots, None, []), {ds}); return c.get(ds, {}), t.get(ds, [])
def eff(root, ids):
    out = {}
    for f in glob.glob(str(root / "**/efficiency.json"), recursive=True):
        try:
            d = json.load(open(f)); i = d.get("dataset_index", json.load(open(Path(f).parent / "result.json")).get("dataset_index"))
        except Exception:
            continue
        if i in ids: out[i] = d
    keys = ("flops_total", "prefill_tokens_total", "decode_steps_total", "peak_gpu_mem_bytes")
    return {k: round(mean(float(v[k]) for v in out.values() if k in v), 2) for k in keys if any(k in v for v in out.values())} if out else {}
res = {}
for ds in DSS:
    arm, sec = g([R / SUB / ds], ds)
    if not arm:
        print(f"{ds}: no results in {SUB}"); continue
    recs = rs.load_records(ds)
    n50, _ = g([R / "rtask_ntrs50_nav3" / ds], ds)
    base, _ = g([R / "greedy_base_rtask_nav3" / ds], ds)
    ids = sorted(set(arm) & set(n50))
    ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]
    sc = lambda c, I: rs.score_dataset(ds, {i: c[i] for i in I}, recs)["exact"]
    a, n = sc(arm, ids), sc(n50, ids)
    gain = [i for i in ids if ok(arm, i) and not ok(n50, i)]; loss = [i for i in ids if not ok(arm, i) and ok(n50, i)]
    same = sum(nrm(arm[i][1]) == nrm(n50[i][1]) for i in ids)
    bad = sum(1 for i in ids if not any(nrm(ch) and nrm(ch) == nrm(arm[i][1]) for ch in (recs[i].get("Choice") or [])))
    lab = lambda L: {k: sum(1 for i in L if nrm(arm[i][0]) == k) for k in sorted({nrm(arm[i][0]) for i in L})}
    pred = {}
    for i in ids: pred[nrm(arm[i][1])] = pred.get(nrm(arm[i][1]), 0) + 1
    row = {"n": len(ids), "arm": a, "ntrs50": n, "same_as_ntrs50": same, "gain": len(gain), "loss": len(loss),
           "gain_by_gold": lab(gain), "loss_by_gold": lab(loss), "gain_ids": gain, "loss_ids": loss, "non_choice": bad,
           "pred_top": sorted(pred.items(), key=lambda kv: -kv[1])[:5], "eff": eff(R / SUB / ds, set(ids)),
           "sec_case": round(mean(sec), 1) if sec else None}
    ib = sorted(set(arm) & set(base))
    if ib:
        row["vs_base"] = {"n": len(ib), "arm": sc(arm, ib), "base": sc(base, ib),
                          "gain": sum(ok(arm, i) and not ok(base, i) for i in ib), "loss": sum((not ok(arm, i)) and ok(base, i) for i in ib)}
    diag = {}
    for pat in ("lu_shard[0-9]*.jsonl", "velh_shard[0-9]*.jsonl", "textcut_shard[0-9]*.jsonl"):
        rows = [json.loads(l) for f in glob.glob(str(R / SUB / pat)) for l in open(f) if l.strip()]
        rows = [r for r in rows if "skip" not in r]
        if not rows: continue
        num = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        diag[pat.split("_")[0]] = {"n": len(rows), **{k: round(mean(float(r[k]) for r in rows if k in r), 5) for k in sorted(num) if k not in ("case", "m")}}
        if pat.startswith("lu"):
            diag["lu"]["skip"] = sum(1 for f in glob.glob(str(R / SUB / pat)) for l in open(f) if '"skip"' in l)
            s2 = [r["s2"]["best_iter"] for r in rows if "s2" in r]
            if s2: diag["lu"]["s2_best_iter_hist"] = {str(k): s2.count(k) for k in sorted(set(s2))}
    row["diag"] = diag
    res[ds] = row
    print(f"{ds:16s} {SUB}: n={len(ids)} ntrs50 {n['score']} ({n['correct']}) -> arm {a['score']} ({a['correct']}) | same {same} gain {len(gain)} {row['gain_by_gold']} loss {len(loss)} {row['loss_by_gold']} | non-choice {bad} | pred_top {row['pred_top'][:3]} | sec/case {row['sec_case']} eff {row['eff']}" + (f" | vs base(+decode) n={row['vs_base']['n']} base {row['vs_base']['base']['score']} arm {row['vs_base']['arm']['score']} +{row['vs_base']['gain']}/-{row['vs_base']['loss']}" if "vs_base" in row else ""))
    for k, v in diag.items(): print(f"   diag[{k}] " + " ".join(f"{kk}={vv}" for kk, vv in v.items() if kk in ("n","skip","R_native","R_final","R_s1","dz_rel_s1","dz_rel_s2","dz_rel_total","cos_pos_native","cos_pos_s1","cos_neg_native","cos_neg_s1","sec","act_rate","gap_mean","mass_ep","mass_anchor","mass_vis","n_cut","s2_best_iter_hist")))
(R / SUB / "pair_score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
