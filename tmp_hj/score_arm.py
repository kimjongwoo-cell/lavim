#!/usr/bin/env python3
"""Score one expq arm: merge shard roots, dedupe by dataset_index, acc + sec.

Usage: score_arm.py <dataset_key> <root> [<root> ...]
Per-item scorer = eval/metrics.py expand_letter -> acc_of_seq (0904 canonical).
gtex/tcga/panda -> balanced accuracy (macro over gt classes); others -> micro.
sec = mean over cases of sum(role_calls[].elapsed_seconds).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

CODE = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/code")
DS_ROOT = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
BALANCED = {"gtex", "tcga", "panda"}

spec = importlib.util.spec_from_file_location("strict_metrics", CODE / "eval/metrics.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

CASE_RE = re.compile(r"^(\d+)_")


def collect(roots):
    latest = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for rj in root.glob("**/result.json"):
            mo = CASE_RE.match(rj.parent.name)
            if not mo:
                continue
            try:
                mtime = rj.stat().st_mtime_ns
            except OSError:
                continue
            try:
                d = json.loads(rj.read_text())
            except Exception:
                continue
            idx = d.get("dataset_index", int(mo.group(1)))
            if not isinstance(idx, int):
                idx = int(mo.group(1))
            prev = latest.get(idx)
            if prev is None or mtime > prev[0]:
                latest[idx] = (mtime, d)
    return {i: d for i, (_, d) in latest.items()}


def case_seconds(result):
    calls = result.get("role_calls")
    if not isinstance(calls, list):
        return None
    total = 0.0
    seen = False
    for call in calls:
        if isinstance(call, dict) and isinstance(
            call.get("elapsed_seconds"), (int, float)
        ):
            total += call["elapsed_seconds"]
            seen = True
    return total if seen else None


def main():
    ds_key = sys.argv[1]
    roots = sys.argv[2:]
    records = json.loads((DS_ROOT / f"{ds_key}.json").read_text())
    cases = collect(roots)
    by_class = {}
    secs = []
    for idx, result in cases.items():
        if not 0 <= idx < len(records):
            continue
        rec = records[idx]
        choices = rec.get("Choice")
        gt = rec.get("Answer")
        ans = result.get("answer")
        pred = ans.get("answer") if isinstance(ans, dict) else ans
        sec = case_seconds(result)
        if sec is not None:
            secs.append(sec)
        if not (
            isinstance(choices, list)
            and isinstance(gt, str)
            and isinstance(pred, str)
        ):
            continue
        pe = m.expand_letter(pred.strip(), choices)
        ge = m.expand_letter(gt.strip(), choices)
        by_class.setdefault(ge, []).append(int(bool(m.acc_of_seq(choices, ge, pe))))
    n = sum(len(v) for v in by_class.values())
    micro = sum(sum(v) for v in by_class.values()) / n if n else 0.0
    bacc = (
        sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)
        if by_class
        else 0.0
    )
    metric = bacc if ds_key in BALANCED else micro
    mean_sec = sum(secs) / len(secs) if secs else None
    print(
        json.dumps(
            {
                "dataset": ds_key,
                "n_scored": n,
                "n_cases": len(cases),
                "micro": round(micro, 4),
                "bacc": round(bacc, 4),
                "metric": round(metric, 4),
                "sec": round(mean_sec, 1) if mean_sec is not None else None,
                "classes": len(by_class),
            }
        )
    )


if __name__ == "__main__":
    main()
