#!/usr/bin/env python3
"""Arm-to-arm answer agreement for the matched-budget accuracy comparison.

Reuses score_accq.py's parsing (expand_letter on the subset case list) so the
comparison is on the same expanded predictions the scorer grades.

usage: accq_agree.py RUNS_ROOT ARM[,ARM...] DS[,DS...]
"""
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

TREE = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
BALANCED = {"gtex", "tcga", "panda"}
CASE_RE = re.compile(r"^(\d+)_")

spec = importlib.util.spec_from_file_location("strict_metrics", TREE / "code/eval/metrics.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dataset_records(ds):
    p = TREE / f"sender_relay_exp/smoke/dcs20_{ds}.json"
    return json.loads(p.read_text())


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


def expanded(result, rec):
    """Expanded prediction string, or None when unusable."""
    choices = rec.get("Choice")
    ans = result.get("answer")
    pred = ans.get("answer") if isinstance(ans, dict) else ans
    if not (isinstance(choices, list) and isinstance(pred, str)):
        return None
    return m.expand_letter(pred.strip(), choices)


root = Path(sys.argv[1])
arms = sys.argv[2].split(",")
dsets = sys.argv[3].split(",")

out = {"per_dataset": {}, "pairwise_overall": {}, "flips_vs_base": {}}
pair_tot = {p: [0, 0] for p in itertools.combinations(arms, 2)}
allsame_tot = [0, 0]
flip_tot = {a: {"win": 0, "loss": 0, "tie_right": 0, "tie_wrong": 0} for a in arms if a != "base"}

for ds in dsets:
    records = dataset_records(ds)
    preds = {a: {} for a in arms}
    correct = {a: {} for a in arms}
    for a in arms:
        for idx, res in collect(root / f"{a}_{ds}").items():
            if not 0 <= idx < len(records):
                continue
            rec = records[idx]
            pe = expanded(res, rec)
            if pe is None:
                continue
            preds[a][idx] = pe
            gt = rec.get("Answer")
            if isinstance(gt, str):
                ge = m.expand_letter(gt.strip(), records[idx]["Choice"])
                correct[a][idx] = int(bool(m.acc_of_seq(records[idx]["Choice"], ge, pe)))

    common = set.intersection(*[set(preds[a]) for a in arms]) if arms else set()
    row = {"n_common": len(common), "n_per_arm": {a: len(preds[a]) for a in arms}, "pairwise": {}}
    for p in itertools.combinations(arms, 2):
        same = sum(1 for i in common if preds[p[0]][i] == preds[p[1]][i])
        row["pairwise"]["|".join(p)] = f"{same}/{len(common)}"
        pair_tot[p][0] += same
        pair_tot[p][1] += len(common)
    allsame = sum(1 for i in common if len({preds[a][i] for a in arms}) == 1)
    row["all_arms_identical"] = f"{allsame}/{len(common)}"
    allsame_tot[0] += allsame
    allsame_tot[1] += len(common)

    # distinct answer strings actually produced, per arm (mode collapse check)
    row["distinct_answers"] = {a: len({preds[a][i] for i in common}) for a in arms}

    if "base" in arms:
        row["flips_vs_base"] = {}
        for a in arms:
            if a == "base":
                continue
            w = sum(1 for i in common if correct[a].get(i) == 1 and correct["base"].get(i) == 0)
            l = sum(1 for i in common if correct[a].get(i) == 0 and correct["base"].get(i) == 1)
            tr = sum(1 for i in common if correct[a].get(i) == 1 and correct["base"].get(i) == 1)
            tw = sum(1 for i in common if correct[a].get(i) == 0 and correct["base"].get(i) == 0)
            row["flips_vs_base"][a] = {"win": w, "loss": l, "tie_right": tr, "tie_wrong": tw}
            flip_tot[a]["win"] += w
            flip_tot[a]["loss"] += l
            flip_tot[a]["tie_right"] += tr
            flip_tot[a]["tie_wrong"] += tw
    out["per_dataset"][ds] = row

    print(f"# {ds}  common={len(common)}  all-4-identical={row['all_arms_identical']}")
    for k, v in row["pairwise"].items():
        print(f"    {k:22s} {v}")
    print(f"    distinct answers: {row['distinct_answers']}")
    if "flips_vs_base" in row:
        for a, f in row["flips_vs_base"].items():
            print(f"    vs base  {a:9s} win {f['win']}  loss {f['loss']}  "
                  f"tie_right {f['tie_right']}  tie_wrong {f['tie_wrong']}")
    print()

print("== 전체 합계 ==")
for p, (s, t) in pair_tot.items():
    out["pairwise_overall"]["|".join(p)] = f"{s}/{t}"
    print(f"  {'|'.join(p):22s} {s}/{t} = {s / t:.3f}" if t else f"  {'|'.join(p)} n/a")
print(f"  all-4-identical        {allsame_tot[0]}/{allsame_tot[1]}")
out["all_arms_identical_overall"] = f"{allsame_tot[0]}/{allsame_tot[1]}"
out["flips_vs_base"] = flip_tot
for a, f in flip_tot.items():
    print(f"  vs base {a:9s} win {f['win']}  loss {f['loss']}  tie_right {f['tie_right']}  tie_wrong {f['tie_wrong']}")

json.dump(out, open(root / "accq_agreement.json", "w"), ensure_ascii=False, indent=1)
print(f"\nwritten {root}/accq_agreement.json")
