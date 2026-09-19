#!/usr/bin/env python3
"""Exact scoring of nav3 + C1 #14.6 NTRS (runs/ntrs_nav3) against nav3 base and #14 QASC on the same dataset_index.
Works on partial runs: only indices present in NTRS (and base) are compared. Output runs/ntrs_score.json.
usage: ntrs_score.py
"""
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
R = T / "sender_relay_exp/runs"


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


rs = load("rescore", T / "scripts/rescore.py")
DS = ["tcga_expert_vqa", "tcga_slidebench", "gtex", "tcga", "panda"]
EXPECTED = {"tcga_expert_vqa": 128, "tcga_slidebench": 197, "gtex": 190, "tcga": 221, "panda": 185}


def gather(roots, ds):
    dirs = rs.run_dirs(roots, None, [])
    c, t = rs.collect(dirs, {ds})
    return c.get(ds, {}), t.get(ds, [])


out = {}
for ds in DS:
    nt, nt_sec = gather([R / "ntrs_nav3" / ds], ds)
    if not nt:
        print(f"{ds:16s} NTRS 0건")
        continue
    base, _ = gather(sorted(R.glob(f"nav3_prompt1_base_{ds}_gpu*")), ds)
    qa, _ = gather([R / "qasc_nav3" / ds], ds)
    recs = rs.load_records(ds)
    common = sorted(set(nt) & set(base))
    ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], (recs[i].get("Choice") if i < len(recs) else None), "exact")[0]
    sb = rs.score_dataset(ds, {i: base[i] for i in common}, recs)
    sn = rs.score_dataset(ds, {i: nt[i] for i in common}, recs)
    cq = [i for i in common if i in qa]
    sq = rs.score_dataset(ds, {i: qa[i] for i in cq}, recs) if cq else None
    gain = [i for i in common if ok(nt, i) and not ok(base, i)]
    loss = [i for i in common if not ok(nt, i) and ok(base, i)]
    same = sum(1 for i in common if rs.am.normalize(nt[i][1]) == rs.am.normalize(base[i][1]))
    lab = lambda ids: {g: sum(1 for i in ids if rs.am.normalize(nt[i][0]) == g) for g in sorted({rs.am.normalize(nt[i][0]) for i in ids})}
    row = {"n_ntrs": len(nt), "expected": EXPECTED[ds], "n_common": len(common),
           "base": sn and {"score": sb["exact"]["score"], "correct": sb["exact"]["correct"]},
           "ntrs": {"score": sn["exact"]["score"], "correct": sn["exact"]["correct"]},
           "qasc_same_idx": sq and {"n": sq["n"], "score": sq["exact"]["score"], "correct": sq["exact"]["correct"]},
           "same_answer": same, "gain": len(gain), "loss": len(loss), "gain_by_gold": lab(gain), "loss_by_gold": lab(loss),
           "sec": round(mean(nt_sec), 1) if nt_sec else None, "score_name": sn["score_name"]}
    out[ds] = row
    print(f"{ds:16s} NTRS {len(nt)}/{EXPECTED[ds]} common={len(common)} | {sn['score_name']} base {row['base']['score']} ({row['base']['correct']}) "
          f"ntrs {row['ntrs']['score']} ({row['ntrs']['correct']}) qasc {sq and sq['exact']['score']} ({sq and sq['exact']['correct']}) "
          f"| same {same} gain {len(gain)} {row['gain_by_gold']} loss {len(loss)} {row['loss_by_gold']} | sec {row['sec']}")
(R / "ntrs_score.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
