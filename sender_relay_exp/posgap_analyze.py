#!/usr/bin/env python3
"""Analyse runs/posgap: post-hoc key re-rotation vs consistent recomputation of the same position gap.

Teacher-forced (choice log-prob argmax, memory/canon_diag.py):
  P0      posthoc run, native arm                 (reference)
  Pg512   posthoc run, gap512 arm                 (keys of cols 0..last-visual re-rotated by -512)
  Pg2048  posthoc run, gap2048 arm
  Pshift  posthoc run, shift arm                  (canon_diag replication)
  C2048   consistent run (VLMAS_POS_GAP=2048), native arm
Per arm vs P0: answers changed, gained, lost; mean Jensen-Shannon divergence of the choice distribution;
decision-row attention masses L0-17 / L27-35. Also generated answers (result.json, exact) consistent vs posthoc.

usage: posgap_analyze.py [root]
"""
import glob
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

T = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else T + "sender_relay_exp/runs/posgap")
s = importlib.util.spec_from_file_location("rescore", T + "scripts/rescore.py")
rs = importlib.util.module_from_spec(s); sys.modules["rescore"] = rs; s.loader.exec_module(rs)
am = rs.am
DSN = {"tcga_expert_vqa": "ExpertVQA", "tcga_slidebench": "SlideBench", "gtex": "GTEx", "tcga": "TCGA", "panda": "PANDA"}


def load_diag(sub, ds):
    rows = {}
    for f in sorted(glob.glob(str(ROOT / sub / ds / "diag_gpu*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            rows[(r["case"], r.get("slide"))] = r
    return rows


def gold_of(sub, ds):
    """dataset_index -> (gold, generated answer) from result.json (latest mtime)."""
    c, _ = rs.collect(rs.run_dirs([ROOT / sub / ds], None, []), {ds})
    return c.get(ds, {})


def dist(lp):
    v = [x if x is not None else -1e9 for x in lp]
    m = max(v); e = [math.exp(x - m) for x in v]; z = sum(e)
    return [x / z for x in e]


def js(p, q):
    def kl(a, b):
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)
    mm = [(x + y) / 2 for x, y in zip(p, q)]
    return 0.5 * kl(p, mm) + 0.5 * kl(q, mm)


def band(arm_rec, key, lo, hi):
    v = [x for x in arm_rec["layers"][key][lo:hi + 1] if x is not None]
    return st.median(v) if v else None


def norm(x):
    return am.normalize(str(x)) if x is not None else None


summary = {}
pooled = {}
for ds in DSN:
    P = load_diag("posthoc", ds)
    C = load_diag("consistent2048", ds)
    if not P:
        continue
    gP = gold_of("posthoc", ds); gC = gold_of("consistent2048", ds)
    keys = sorted(k for k in P if k in C) if C else sorted(P)
    arms = [("P0", "posthoc", "native"), ("Pg512", "posthoc", "gap512"), ("Pg2048", "posthoc", "gap2048"),
            ("Pshift", "posthoc", "shift"), ("C2048", "consistent", "native")]
    out = {"n_posthoc": len(P), "n_consistent": len(C), "n_common": len(keys)}
    for name, src, arm in arms:
        rows = C if src == "consistent" else P
        if not rows:
            continue
        chg = gain = loss = corr = 0; jss = []; rv_lo = []; rp_lo = []; rv_hi = []; n = 0
        for k in keys:
            if k not in rows or arm not in rows[k]["arms"]:
                continue
            idx = k[0]
            gold = gP.get(idx, (None,))[0] if idx in gP else None
            ref = P[k]["arms"]["native"]; cur = rows[k]["arms"][arm]
            ok_ref = gold is not None and norm(ref["answer"]) == norm(gold)
            ok_cur = gold is not None and norm(cur["answer"]) == norm(gold)
            n += 1; corr += ok_cur
            chg += norm(ref["answer"]) != norm(cur["answer"])
            gain += ok_cur and not ok_ref; loss += ok_ref and not ok_cur
            jss.append(js(dist(ref["logp"]), dist(cur["logp"])))
            if "layers" in cur:
                rv_lo.append(band(cur, "rho_v", 0, 17)); rp_lo.append(band(cur, "rho_prompt", 0, 17))
                rv_hi.append(band(cur, "rho_v", 27, 35))
        med = lambda v: None if not [x for x in v if x is not None] else round(st.median([x for x in v if x is not None]), 4)
        out[name] = {"n": n, "correct": corr, "changed": chg, "gain": gain, "loss": loss,
                     "js_mean": round(sum(jss) / len(jss), 4) if jss else None, "js_median": med(jss),
                     "rho_v_L0_17": med(rv_lo), "rho_prompt_L0_17": med(rp_lo), "rho_v_L27_35": med(rv_hi)}
        pl = pooled.setdefault(name, {"n": 0, "correct": 0, "changed": 0, "gain": 0, "loss": 0, "js": []})
        pl["n"] += n; pl["correct"] += corr; pl["changed"] += chg; pl["gain"] += gain; pl["loss"] += loss; pl["js"] += jss
    # generated answers: consistent vs posthoc (native pipeline)
    if gC:
        com = sorted(set(gP) & set(gC))
        okp = sum(norm(gP[i][1]) == norm(gP[i][0]) for i in com)
        okc = sum(norm(gC[i][1]) == norm(gC[i][0]) for i in com)
        chg = sum(norm(gP[i][1]) != norm(gC[i][1]) for i in com)
        out["generated"] = {"n": len(com), "posthoc_native_correct": okp, "consistent_correct": okc, "changed": chg}
    summary[ds] = out

for name, pl in pooled.items():
    pl["js_mean"] = round(sum(pl["js"]) / len(pl["js"]), 4) if pl["js"] else None
    pl.pop("js")
summary["ALL"] = pooled
(ROOT / "posgap_summary.json").write_text(json.dumps(summary, indent=1))

print(f"{'ds':<11}{'arm':<8}{'n':>4}{'정답':>6}{'바뀜':>6}{'얻음':>6}{'잃음':>6}{'JS평균':>9}{'ρv L0-17':>10}{'ρp L0-17':>10}{'ρv L27-35':>10}")
for ds, out in summary.items():
    if ds == "ALL":
        continue
    for name in ("P0", "Pg512", "Pg2048", "Pshift", "C2048"):
        r = out.get(name)
        if not r:
            continue
        f = lambda x: "–" if x is None else f"{x:.4f}"
        print(f"{DSN[ds]:<11}{name:<8}{r['n']:>4}{r['correct']:>6}{r['changed']:>6}{r['gain']:>6}{r['loss']:>6}"
              f"{f(r['js_mean']):>9}{f(r['rho_v_L0_17']):>10}{f(r['rho_prompt_L0_17']):>10}{f(r['rho_v_L27_35']):>10}")
    if "generated" in out:
        g = out["generated"]
        print(f"{'':<11}생성 답: n={g['n']} posthoc(native) 정답 {g['posthoc_native_correct']} · consistent 정답 {g['consistent_correct']} · 바뀜 {g['changed']}")
print("\nALL", json.dumps(pooled))
