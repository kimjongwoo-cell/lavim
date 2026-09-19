#!/usr/bin/env python3
"""Exact scoring + efficiency for runs/greedy_base_rtask_nav3 (copy tree) vs nav3 greedy base (same dataset_index).

Also compares with the other session's sampling base without the prompt (runs/sample_t06_nav3/base) where present.
Writes <copy runs>/sample_base_rtask_nav3/score.json and prints one line per finished dataset.
usage: score_sample_rtask.py [dataset ...]
"""
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

T = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj")
R = T / "sender_relay_exp/runs"
import os
C = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs") / os.environ["ARM"]
DB = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/greedy_base_rtask_nav3")
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py")
rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs
s.loader.exec_module(rs)
# eff_report.py has no __main__ guard (importing it rewrites runs/eff_report_0915.json in the original tree),
# so load only its imports, constants and function definitions.
import ast as _ast
import types as _types
_src = (T / "sender_relay_exp/eff_report.py").read_text()
_tree = _ast.parse(_src)
_tree.body = [n for n in _tree.body if isinstance(n, (_ast.Import, _ast.ImportFrom, _ast.FunctionDef, _ast.Assign, _ast.Expr))
              and not (isinstance(n, _ast.Expr) and not isinstance(getattr(n, "value", None), _ast.Constant))]
ef = _types.ModuleType("eff_report_defs")
exec(compile(_tree, "eff_report_defs", "exec"), ef.__dict__)

EXPECT = {"tcga_expert_vqa": 128, "tcga_slidebench": 197, "gtex": 190, "tcga": 221, "panda": 196}
nrm = rs.am.normalize


def g(roots, ds):
    c, t = rs.collect(rs.run_dirs(roots, None, []), {ds})
    return c.get(ds, {}), t.get(ds, [])


def attempts_wall(ds):
    walls = []
    for f in (C / ds).glob("gpu*_attempt_*/attempt_metrics.jsonl"):
        walls += [json.loads(l)["wall_seconds"] for l in open(f)]
    return round(st.mean(walls), 1) if walls else None


def eff(pattern_dirs):
    rows = []
    for d in pattern_dirs:
        for case in sorted(Path(d).glob("*/result.json")):
            try:
                m = ef.case_metrics(case.parent)
            except Exception:  # noqa: BLE001
                m = None
            if m:
                rows.append(m)
    if not rows:
        return None
    keys = rows[0].keys()
    return {k: round(st.mean(r[k] for r in rows if isinstance(r.get(k), (int, float))), 1)
            for k in keys if isinstance(rows[0].get(k), (int, float))}


out = {}
if (C / "score.json").exists():
    out = json.loads((C / "score.json").read_text())
targets = sys.argv[1:] or list(EXPECT)
for ds in targets:
    arm, sec = g([C / ds], ds)
    if not arm:
        continue
    recs = rs.load_records(ds)
    base, bsec = g([DB / ds], ds)  # Latent base + decode fix (same tree/settings, no C1)
    old_roots = sorted(R.glob(f"nav3_prompt1_base_{ds}_gpu*")) if ds != "panda" else [R / "nav3clamp_base" / "panda"]
    samp, _ = g(old_roots, ds)  # old nav3 base (reference only)
    ids = sorted(set(arm) & set(base))

    def ok(c, i):
        return rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]

    def sc(c, idx):
        return rs.score_dataset(ds, {i: c[i] for i in idx}, recs)["exact"]

    def nonchoice(c, idx):
        return sum(1 for i in idx if not any(nrm(ch) and nrm(ch) == nrm(c[i][1]) for ch in (recs[i].get("Choice") or [])))

    a_full = sc(arm, sorted(arm))
    a, b = sc(arm, ids), sc(base, ids)
    gain = [i for i in ids if ok(arm, i) and not ok(base, i)]
    loss = [i for i in ids if not ok(arm, i) and ok(base, i)]
    same = sum(nrm(arm[i][1]) == nrm(base[i][1]) for i in ids)
    row = {
        "n": len(arm), "expected": EXPECT[ds], "score": a_full["score"], "correct": a_full["correct"],
        "paired_n": len(ids), "paired_arm": a, "paired_base": b, "same_as_base": same,
        "gain": len(gain), "loss": len(loss),
        "non_choice_arm": nonchoice(arm, sorted(arm)), "non_choice_base_paired": nonchoice(base, ids),
        "non_choice_arm_paired": nonchoice(arm, ids),
        "sec_model": round(st.mean(sec), 1) if sec else None, "sec_wall": attempts_wall(ds),
    }
    sids = sorted(set(arm) & set(samp))
    if sids:
        row["old_nav3_base"] = {"n": len(sids), "arm": sc(arm, sids), "samp": sc(samp, sids),
                                "same": sum(nrm(arm[i][1]) == nrm(samp[i][1]) for i in sids),
                                "non_choice_arm": nonchoice(arm, sids), "non_choice_samp": nonchoice(samp, sids)}
    row["eff"] = eff(sorted((C / ds).glob("gpu*_attempt_*")))
    out[ds] = row
    print(json.dumps({ds: row}, ensure_ascii=False))
(C / "score.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
