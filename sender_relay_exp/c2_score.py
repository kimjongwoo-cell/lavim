#!/usr/bin/env python3
"""Exact scoring of a C2 arm on top of nav3 + NTRS against NTRS alone (same dataset_index) and nav3 base.
usage: c2_score.py <runs subdir, e.g. lpvh_ntrs_nav3> [label]   -> runs/<subdir>/score.json
PANDA base = nav3clamp_base (x5 clamp, 196)."""
import importlib.util, json, sys
from pathlib import Path
from statistics import mean
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
SUB = sys.argv[1]; LABEL = sys.argv[2] if len(sys.argv) > 2 else SUB
DS = {"tcga_expert_vqa": 128, "tcga_slidebench": 197, "gtex": 190, "tcga": 221, "panda": 196}
nrm = rs.am.normalize
def g(roots, ds):
    c, t = rs.collect(rs.run_dirs(roots, None, []), {ds}); return c.get(ds, {}), t.get(ds, [])
out = {}
for ds, n_exp in DS.items():
    arm, sec = g([R / SUB / ds], ds)
    if not arm:
        continue
    recs = rs.load_records(ds)
    ntrs, _ = g([R / "ntrs_nav3" / ds], ds)
    base, _ = g([R / "nav3clamp_base" / "panda"] if ds == "panda" else sorted(R.glob(f"nav3_prompt1_base_{ds}_gpu*")), ds)
    ids = sorted(set(arm) & set(ntrs) & set(base))
    ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]
    sc = lambda c: rs.score_dataset(ds, {i: c[i] for i in ids}, recs)["exact"]
    a, nt, b = sc(arm), sc(ntrs), sc(base)
    gain = [i for i in ids if ok(arm, i) and not ok(ntrs, i)]; loss = [i for i in ids if not ok(arm, i) and ok(ntrs, i)]
    same = sum(nrm(arm[i][1]) == nrm(ntrs[i][1]) for i in ids)
    bad = sum(1 for i in ids if not any(nrm(ch) and nrm(ch) == nrm(arm[i][1]) for ch in (recs[i].get("Choice") or [])))
    lab = lambda L: {k: sum(1 for i in L if nrm(arm[i][0]) == k) for k in sorted({nrm(arm[i][0]) for i in L})}
    row = {"n": len(ids), "expected": n_exp, "arm": a, "ntrs": nt, "base": b, "same_as_ntrs": same,
           "gain_vs_ntrs": len(gain), "loss_vs_ntrs": len(loss), "gain_by_gold": lab(gain), "loss_by_gold": lab(loss),
           "non_choice": bad, "sec": round(mean(sec), 1) if sec else None}
    out[ds] = row
    print(f"{ds:16s} {LABEL} {len(arm)}/{n_exp} common={len(ids)} | base {b['score']} ({b['correct']}) ntrs {nt['score']} ({nt['correct']}) "
          f"{LABEL} {a['score']} ({a['correct']}) | same_as_ntrs {same} gain {len(gain)} {row['gain_by_gold']} loss {len(loss)} {row['loss_by_gold']} "
          f"| non-choice {bad} | sec {row['sec']}")
(R / SUB / "score.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
