#!/usr/bin/env python3
"""Score matched-budget accuracy arms with the canonical dashboard rule.

Per item: expand_letter(pred/gt, choices) -> acc_of_seq (code/eval/metrics.py).
Aggregation mirrors dashboard/server.py multipath_quality_metrics: class key = expanded gold;
gtex / tcga / panda -> balanced accuracy (macro over classes), others -> micro.
Dedup: newest result.json per dataset_index within a run root.

usage: score_accq.py RUNS_ROOT ARM[,ARM...] DS[,DS...]
       reads <RUNS_ROOT>/<arm>_<ds>/**/result.json and the dataset json of <ds>.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

TREE = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
DS_ROOT = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
BALANCED = {"gtex", "tcga", "panda"}
CASE_RE = re.compile(r"^(\d+)_")

spec = importlib.util.spec_from_file_location("strict_metrics", TREE / "code/eval/metrics.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dataset_records(ds):
    """The runs used the fixed 20-case subsets, and result.json's dataset_index is the
    position INSIDE that subset, so scoring must read the subset file (it carries the same
    Answer/Choice fields as the full dataset)."""
    for cand in (TREE / f"sender_relay_exp/smoke/dcs20_{ds}.json", DS_ROOT / f"{ds}.json"):
        if cand.is_file():
            return json.loads(cand.read_text()), str(cand)
    raise SystemExit(f"case list for {ds} not found")


def collect(root):
    latest = {}
    root = Path(root)
    if not root.is_dir():
        return {}
    for rj in root.glob("**/result.json"):
        mo = CASE_RE.match(rj.parent.name)
        try:
            d = json.loads(rj.read_text())
            mtime = rj.stat().st_mtime_ns
        except Exception:
            continue
        idx = d.get("dataset_index", int(mo.group(1)) if mo else None)
        if not isinstance(idx, int):
            continue
        prev = latest.get(idx)
        if prev is None or mtime > prev[0]:
            latest[idx] = (mtime, d)
    return {i: d for i, (_, d) in latest.items()}


def score(records, cases, balanced):
    by_class, missing = {}, 0
    for idx, result in cases.items():
        if not 0 <= idx < len(records):
            missing += 1
            continue
        rec = records[idx]
        choices, gt = rec.get("Choice"), rec.get("Answer")
        ans = result.get("answer")
        pred = ans.get("answer") if isinstance(ans, dict) else ans
        if not (isinstance(choices, list) and isinstance(gt, str) and isinstance(pred, str)):
            missing += 1
            continue
        ge = m.expand_letter(gt.strip(), choices)
        pe = m.expand_letter(pred.strip(), choices)
        by_class.setdefault(ge, []).append(int(bool(m.acc_of_seq(choices, ge, pe))))
    n = sum(len(v) for v in by_class.values())
    if n == 0:
        return {"n": 0, "micro": None, "bacc": None, "primary": None, "classes": 0, "unscored": missing}
    micro = sum(sum(v) for v in by_class.values()) / n
    bacc = sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)
    return {"n": n, "micro": round(micro, 4), "bacc": round(bacc, 4),
            "primary": round(bacc if balanced else micro, 4),
            "classes": len(by_class), "unscored": missing}


root = Path(sys.argv[1])
arms = sys.argv[2].split(",")
dsets = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gtex", "tcga_expert_vqa", "tcga_slidebench", "tcga", "panda"]
rows = []
for ds in dsets:
    records, path = dataset_records(ds)
    print(f"# {ds}: {len(records)} records from {path}")
    for arm in arms:
        # arm "-" scores a bare <root>/<ds> layout (older run folders)
        cases = collect(root / (ds if arm == "-" else f"{arm}_{ds}"))
        s = score(records, cases, ds in BALANCED)
        rows.append({"dataset": ds, "arm": arm, **s})
        print(f"{ds:16s} {arm:10s} n={s['n']:3d} primary={s['primary']} "
              f"(bacc {s['bacc']} micro {s['micro']}, classes {s['classes']}, unscored {s['unscored']})")
json.dump(rows, open(root / "accq_scores.json", "w"), ensure_ascii=False, indent=1)
print(f"\nwritten {root}/accq_scores.json  (metric: BACC for {sorted(BALANCED)}, micro otherwise)")
