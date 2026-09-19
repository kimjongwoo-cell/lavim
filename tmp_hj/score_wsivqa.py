#!/usr/bin/env python3
"""WSI-VQA (735) scoring for runs/nav4_wsivqa/<arm>/wsivqa.

Scored items: 390 with Choice + 269 open items that carry canonical labels (histological_type / PR / vital / HER2).
Survival-time items (76) ARE scored (09-19 user: nothing excluded): exact string equality of the day count; no C-index.

Two scores per arm, on the same case set:
  (a) code/eval/metrics.py (the original WSI-VQA evaluator), fed a jsonl of question/prediction/ground_truth/Choice
      (open items without Choice, as in WsiVQA_test.json) -> mcq_accuracy, open_substring_accuracy, total_accuracy
  (b) exact (eval/answer_match.py normalize equality) -> MCQ Acc, open Acc, open BAcc per field + macro, overall Acc
With 2+ arms, only dataset_index present in every arm is scored, and arm[1:] get gained/lost vs arm[0].

usage: score_wsivqa.py ARM [ARM ...]      (ARM = dir under runs/nav4_wsivqa, e.g. base latplan)
writes runs/nav4_wsivqa/score_<arms>.json and runs/nav4_wsivqa/<arm>/metrics_py/
"""
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
RUNS = REPO / "sender_relay_exp/runs/nav4_wsivqa"
import os
DS = Path(os.environ.get("DSJSON", REPO / "sender_relay_exp/data_wsivqa/wsivqa.json"))  # smoke check: DSJSON=smoke/wsivqa2.json SUB=smoke_<arm>
METRICS = REPO / "code/eval/metrics.py"
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"

sys.path.insert(0, str(REPO))
from vision_text_mas.prompts import canonical_open_answer_options  # noqa: E402

_s = importlib.util.spec_from_file_location(
    "answer_match", "/home/users/whddn12316/wsi_latent_0902_2155_hj/eval/answer_match.py")
am = importlib.util.module_from_spec(_s)
_s.loader.exec_module(am)

records = json.loads(DS.read_text())


def category(rec):
    if rec.get("Choice"):
        return "mcq", None
    q = rec["Question"].lower()
    if canonical_open_answer_options(rec["Question"]):
        field = next(f for f in ("progesterone", "vital", "histological", "her2") if f in q)
        return "open", field
    return "survival", None


def collect(arm):
    latest = {}
    for f in (RUNS / arm / os.environ.get("SUB", "wsivqa").replace("<arm>", arm)).glob("gpu*_attempt_*/*/result.json"):
        try:
            p = json.loads(f.read_text())
            i = int(p["dataset_index"])
            ans = p.get("answer", {})
            pred = str(ans.get("answer", "") if isinstance(ans, dict) else ans)
            gold = str(p.get("gold_answer", ""))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        t = f.stat().st_mtime
        if i not in latest or t > latest[i][0]:
            latest[i] = (t, gold, pred)
    return {i: (g, a) for i, (_, g, a) in latest.items()}


arms = sys.argv[1:] or ["base"]
cases = {a: collect(a) for a in arms}
ids = sorted(set.intersection(*(set(c) for c in cases.values())))
# 0919: all 735 are scored (survival included; exact = number string equality, no C-index)

out = {"arms": arms, "n_scored": len(ids), "arms_detail": {}}
verdict = {}
for arm in arms:
    rows, jl = defaultdict(list), []
    by_field = defaultdict(lambda: defaultdict(list))
    gold_mismatch = 0
    for i in ids:
        rec = records[i]
        gold, pred = cases[arm][i]
        gold_mismatch += am.normalize(gold) != am.normalize(rec["Answer"])
        cat, field = category(rec)
        ok = am.normalize(pred) == am.normalize(rec["Answer"])
        verdict[(arm, i)] = ok
        rows[cat].append(ok)
        rows["all"].append(ok)
        if cat == "open":
            by_field[field][am.normalize(rec["Answer"])].append(ok)
        row = {"question_id": str(i), "Id": rec["Id"], "question": rec["Question"],
               "prediction": pred, "ground_truth": rec["Answer"]}
        if rec.get("Choice"):
            row["Choice"] = rec["Choice"]
        jl.append(row)

    def acc(v):
        return round(100.0 * sum(v) / len(v), 2) if v else None

    def bacc(labels):
        return round(100.0 * sum(sum(v) / len(v) for v in labels.values()) / len(labels), 2) if labels else None

    fields = {f: {"n": sum(len(v) for v in lab.values()), "acc": acc([x for v in lab.values() for x in v]),
                  "bacc": bacc(lab)} for f, lab in sorted(by_field.items())}
    exact = {"surv_acc": acc(rows["survival"]), "n_surv": len(rows["survival"]),
             "mcq_acc": acc(rows["mcq"]), "n_mcq": len(rows["mcq"]),
             "open_acc": acc(rows["open"]), "n_open": len(rows["open"]),
             "open_bacc_macro_fields": round(sum(v["bacc"] for v in fields.values()) / len(fields), 2) if fields else None,
             "open_fields": fields, "overall_acc": acc(rows["all"]), "correct": sum(rows["all"]),
             "gold_mismatch": gold_mismatch}

    mdir = RUNS / arm / ("metrics_py" if "SUB" not in os.environ else "metrics_py_smoke")
    mdir.mkdir(parents=True, exist_ok=True)
    inp = mdir / "wsivqa_scored.jsonl"
    inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in jl))
    mp = None
    if jl:
        subprocess.run([PY, str(METRICS), "--input", str(inp), "--output-dir", str(mdir), "--name", arm],
                       check=True, capture_output=True, text=True)
        s = json.loads((mdir / f"{arm}_summary.json").read_text())
        mp = {k: s.get(k) for k in ("num_valid_for_eval", "total_mcq", "correct_mcq", "mcq_accuracy", "total_open",
                                     "correct_open_substring", "open_substring_accuracy", "total_accuracy")}
    out["arms_detail"][arm] = {"exact": exact, "metrics_py": mp}

for arm in arms[1:]:
    g = sum(verdict[(arm, i)] and not verdict[(arms[0], i)] for i in ids)
    l_ = sum(verdict[(arms[0], i)] and not verdict[(arm, i)] for i in ids)
    same = sum(am.normalize(cases[arm][i][1]) == am.normalize(cases[arms[0]][i][1]) for i in ids)
    out["arms_detail"][arm]["vs_" + arms[0]] = {"gained": g, "lost": l_, "same_answer": same}

(RUNS / f"score{'_smoke' if 'SUB' in os.environ else ''}_{'_'.join(arms)}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
