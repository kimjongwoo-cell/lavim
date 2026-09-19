"""Summarize sender-relay smoke arms (§11/§15E). PYTHONPATH-free, stdlib only."""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

SMOKE = Path(__file__).resolve().parent / "runs" / "smoke"
BASE_RUN = Path("/home/super/hj/0902_2155/runs/canonical/wsi-vqa/qwen-4b/latent-base-step5/gpu0")
INDICES = list(range(10))


def load_arm(root: Path) -> dict[int, dict]:
    cases: dict[int, dict] = {}
    for result_path in root.glob("*/result.json"):
        result = json.loads(result_path.read_text())
        index = int(result["dataset_index"])
        answer_object = result.get("answer") or {}
        entry = {
            "answer": str(answer_object.get("answer", "")),
            "gold": str(result.get("gold_answer", "")),
        }
        relay_path = result_path.parent / "relay_audit.json"
        if relay_path.exists():
            relay = json.loads(relay_path.read_text())
            receiver = relay.get("receiver") or {}
            entry["relay_source"] = receiver.get("relay_source")
            entry["mass_before"] = receiver.get("vision_mass_before")
            entry["mass_after"] = receiver.get("vision_mass_after")
            entry["realloc_mass"] = receiver.get("total_reallocated_mass")
            entry["fallback"] = receiver.get("fallback_used")
            entry["n_visual"] = len(receiver.get("visual_columns") or [])
            diag = relay.get("diagnostics") or {}
            entry["sp_kn_sender"] = diag.get("spearman_key_norm_vs_sender")
            sender = relay.get("sender") or {}
            entry["shuffle_seed"] = (receiver.get("build") or {}).get("shuffle_seed")
            entry["has_sender_audit"] = bool(sender)
        cases[index] = entry
    metrics_path = root / "attempt_metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            metric = json.loads(line)
            index = int(metric["dataset_index"])
            if index in cases:
                cases[index]["wall_s"] = metric["wall_seconds"]
    log_path = root.with_suffix(".log")
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        kept = re.findall(r"hierarchy-constrained visual tokens: (\d+)/(\d+)", text)
        if kept:
            cases.setdefault(-1, {})["retained_visual"] = kept
    return cases


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    arms = ["pruning_v3", "pr_key_norm", "pr_uniform", "pr_sender_priority", "pr_shuffled"]
    base_cases: dict[int, dict] = {}
    for index in INDICES:
        matches = list(BASE_RUN.glob(f"{index:03d}_*/result.json"))
        if matches:
            result = json.loads(matches[0].read_text())
            base_cases[index] = {
                "answer": str((result.get("answer") or {}).get("answer", "")),
                "gold": str(result.get("gold_answer", "")),
            }
    metrics_path = BASE_RUN / "attempt_metrics.jsonl"
    for line in metrics_path.read_text().splitlines():
        metric = json.loads(line)
        if int(metric["dataset_index"]) in base_cases:
            base_cases[int(metric["dataset_index"])]["wall_s"] = metric["wall_seconds"]

    data = {"latent_base(reused)": base_cases}
    for arm in arms:
        root = SMOKE / arm
        if root.exists():
            data[arm] = load_arm(root)

    print("=== per-arm summary (10 fixed cases, seed 42) ===")
    header = f"{'arm':26s} {'done':>4s} {'gold==ans':>9s} {'wall med':>8s} {'massB':>7s} {'massA':>7s} {'Δmass':>7s} {'fallback':>8s}"
    print(header)
    for arm, cases in data.items():
        real = {k: v for k, v in cases.items() if k >= 0}
        n = len(real)
        correct = sum(
            1 for v in real.values()
            if v.get("answer", "").strip().lower() == v.get("gold", "").strip().lower()
        )
        walls = [v["wall_s"] for v in real.values() if "wall_s" in v]
        mb = [v["mass_before"] for v in real.values() if v.get("mass_before") is not None]
        ma = [v["mass_after"] for v in real.values() if v.get("mass_after") is not None]
        dm = [v["realloc_mass"] for v in real.values() if v.get("realloc_mass") is not None]
        fb = sum(1 for v in real.values() if v.get("fallback"))
        print(
            f"{arm:26s} {n:4d} {correct:>7d}/{n:<2d}"
            f" {fmt(statistics.median(walls), 1) if walls else '-':>7s}s"
            f" {fmt(statistics.mean(mb)) if mb else '-':>7s}"
            f" {fmt(statistics.mean(ma)) if ma else '-':>7s}"
            f" {fmt(statistics.mean(dm)) if dm else '-':>7s}"
            f" {fb:>8d}"
        )

    print("\n=== per-case answers ===")
    arm_names = list(data)
    print("idx  " + "  ".join(f"{a[:14]:14s}" for a in arm_names) + "  gold")
    for index in INDICES:
        row = []
        gold = ""
        for arm in arm_names:
            v = data[arm].get(index, {})
            gold = v.get("gold", gold)
            answer = v.get("answer", "?")[:14]
            row.append(f"{answer:14s}")
        print(f"{index:3d}  " + "  ".join(row) + f"  {gold[:30]}")

    print("\n=== relay diagnostics (sender arm) ===")
    for arm in ("pr_sender_priority", "pr_shuffled", "pr_key_norm", "pr_uniform"):
        cases = data.get(arm, {})
        for index in INDICES:
            v = cases.get(index, {})
            if v.get("relay_source"):
                print(
                    f"{arm:20s} idx={index} n_vis={v.get('n_visual')} "
                    f"spearman(kn,sender)={fmt(v.get('sp_kn_sender'))} "
                    f"seed={v.get('shuffle_seed')} fallback={v.get('fallback')} "
                    f"sender_audit={v.get('has_sender_audit')}"
                )

    print("\n=== retained visual tokens (from logs) ===")
    for arm in arms:
        cases = data.get(arm, {})
        kept = cases.get(-1, {}).get("retained_visual")
        if kept:
            ratios = [f"{k}/{t}" for k, t in kept[:12]]
            print(f"{arm:20s} {' '.join(ratios)}")


if __name__ == "__main__":
    main()
    sys.exit(0)
