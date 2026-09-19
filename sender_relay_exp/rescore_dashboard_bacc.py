#!/usr/bin/env python3
"""Rescore all artifact cells with the 0806 dashboard's exact per-item scorer.

Per-item: expand_letter(pred/gt, choices) -> acc_of_seq (eval/metrics.py, byte-identical
in both trees). Aggregation mirrors dashboard/server.py multipath_quality_metrics:
  - class key = expand_letter(Answer, Choice)
  - gtex/tcga/panda -> balanced accuracy (macro over classes); others -> micro.
Dedup mirrors latest_cases: newest result.json mtime per dataset_index across roots.
Output: JSON rows {dataset, arm, param, step, n, micro, bacc}.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

CODE = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/code")
DS_ROOT = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
R0806 = Path("/home/users/whddn12316/wsi_latentmas_0806/results/runs")
RHJ = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs")
BALANCED = {"gtex", "tcga", "panda"}

spec = importlib.util.spec_from_file_location("strict_metrics", CODE / "eval/metrics.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

DATASETS = {
    "gtex": json.loads((DS_ROOT / "gtex.json").read_text()),
    "tcga_expert_vqa": json.loads((DS_ROOT / "tcga_expert_vqa.json").read_text()),
}

CASE_RE = re.compile(r"^(\d+)_")


def collect(roots):
    """Newest result.json per dataset_index across all roots (recursive gpu* ok)."""
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
            idx_dir = int(mo.group(1))
            try:
                d = json.loads(rj.read_text())
            except Exception:
                continue
            idx = d.get("dataset_index", idx_dir)
            if not isinstance(idx, int):
                idx = idx_dir
            prev = latest.get(idx)
            if prev is None or mtime > prev[0]:
                latest[idx] = (mtime, d)
    return {i: d for i, (_, d) in latest.items()}


def score(ds_key, cases):
    records = DATASETS[ds_key]
    by_class = {}
    for idx, result in cases.items():
        if not 0 <= idx < len(records):
            continue
        rec = records[idx]
        choices = rec.get("Choice")
        gt = rec.get("Answer")
        ans = result.get("answer")
        pred = ans.get("answer") if isinstance(ans, dict) else ans
        if not (isinstance(choices, list) and isinstance(gt, str) and isinstance(pred, str)):
            continue
        pe = m.expand_letter(pred.strip(), choices)
        ge = m.expand_letter(gt.strip(), choices)
        by_class.setdefault(ge, []).append(int(bool(m.acc_of_seq(choices, ge, pe))))
    n = sum(len(v) for v in by_class.values())
    if n == 0:
        return {"n": 0, "micro": None, "bacc": None}
    micro = sum(sum(v) for v in by_class.values()) / n
    bacc = sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)
    return {"n": n, "micro": round(micro, 4), "bacc": round(bacc, 4)}


def base_roots(ds, param, step):
    if ds == "gtex":
        roots = [R0806 / f"multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/{param}/step{step}"]
        if step == 5:
            roots.append(R0806 / f"multipathqa_gtex_qwen_gpu678_v2_20260901/{param}/latent_base_step5")
        return roots
    return [R0806 / f"multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/{ds}/{param}/step{step}"]


CELLS = []
for ds in ("gtex", "tcga_expert_vqa"):
    for param in ("2b", "4b", "8b"):
        for step in (5, 10, 20, 30):
            CELLS.append((ds, "Latent base(기존)", param, step, base_roots(ds, param, step)))
            for arm, name in (("pruning_v3", "Pruning V3"), ("reallocation_v2", "Realloc V2")):
                CELLS.append((ds, name, param, step,
                              [RHJ / f"allsteps_matrix_20260903/{ds}/{param}/{arm}_step{step}"]))
    mp = RHJ / f"multipath_qwen4b_step5_20260903/{ds}"
    for arm, name in (("base", "Latent base(재현)"), ("pruning_v3", "Pruning V3(재현)"),
                      ("pr_key_norm", "P+R key_norm"), ("pr_uniform", "P+R uniform"),
                      ("pr_sender", "P+R sender"), ("pr_shuffled", "P+R shuffled")):
        CELLS.append((ds, name, "4b", 5, [mp / arm]))
    CELLS.append((ds, "Base+Refeed8", "4b", 5, [RHJ / f"refeed_qwen4b_step5_20260903/{ds}/base_refeed8"]))
    for arm, name in (("C_rescue", "C1 Rescue"), ("D_cross", "C2 Cross-scale"), ("E_both", "C1+C2")):
        CELLS.append((ds, name, "4b", 5, [RHJ / f"consolidation_ablation/{ds}/{arm}"]))
    for step in (10, 20):
        CELLS.append((ds, "C2 Cross-scale", "4b", step, [RHJ / f"c2_replication/{ds}/4b/D_cross_step{step}"]))

rows = []
for ds, name, param, step, roots in CELLS:
    s = score(ds, collect(roots))
    if s["n"] == 0:
        continue
    rows.append({"dataset": ds, "task": name, "param": param, "step": step, **s})

json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)
print()
