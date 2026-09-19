#!/usr/bin/env python3
"""HJ 실험 대시보드(hj_dashboard.html)의 ROWS 행을 런 디렉토리에서 생성한다.

채점은 0806 대시보드 / sender_relay_exp/rescore_dashboard_bacc.py와 같은 규칙:
  - per-item: expand_letter(pred/gt, choices) -> acc_of_seq   (eval/metrics.py)
  - class key: expand_letter(Answer, Choice)
  - gtex / tcga / panda -> balanced accuracy, 그 외 -> micro
  - 같은 dataset_index가 여러 번 나오면 산출물 mtime이 최신인 것만 채택
sec(케이스당 초)는 latent/vlmas는 role_calls의 elapsed_seconds 합, single은
inference_time_sec 의 케이스 평균이다.

usage:
  # 0913 트리의 runs/baselines_* 전부
  dashboard_rows.py
  # 임의 런 루트 (0902 런으로 대조할 때)
  dashboard_rows.py --scan /home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/mp_slidebench_20260905
  # arm 하나만, 경로 추론 없이
  dashboard_rows.py --arm <dir> --dataset gtex --param 4b --method latent-base --step 5
options:
  --task-template "{label}(0913)"  task 이름 서식. {label}은 Single/VLMAS/Latent base
  --full                           micro/bacc를 함께 출력 (대시보드 붙여넣기용이 아닌 점검용)

주의 — 대시보드 쪽 손질이 하나 필요하다: hj_dashboard.html 의 fam() 기준선 정규식이
/기존|Latent base|InternVL3|Lingshu/ 라서 "Single(0913)"·"VLMAS(0913)" 는 기준선으로
분류되지 않고 기본값(재배치·C2)으로 떨어진다. 행을 붙일 때 그 정규식에 Single|VLMAS를
같이 넣어야 한다. baseMap 도 "Latent base(기존)" 문자열에 고정이라 Δ 기준선은 그대로다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
DS_ROOT = TREE.parent / "datasets" / "MultiPathQA" / "ready_wsivqa" / "full_no_panda"
BALANCED = {"gtex", "tcga", "panda"}

# 드라이버/디렉토리에서 쓰는 이름 -> 대시보드 ROWS의 dataset 키
DATASET_KEYS = {
    "gtex": "gtex",
    "tcga": "tcga",
    "panda": "panda",
    "expertvqa": "tcga_expert_vqa",
    "tcga_expert_vqa": "tcga_expert_vqa",
    "slidebench": "tcga_slidebench",
    "tcga_slidebench": "tcga_slidebench",
}
# method 디렉토리 이름 -> 대시보드 task 라벨
METHOD_LABELS = {
    "single": "Single",
    "single-no-thinking": "Single",
    "vlmas": "VLMAS",
    "latent-base": "Latent base",
    "base": "Latent base",
    "latent_base": "Latent base",
}
CASE_RE = re.compile(r"^(\d+)_")
STEP_RE = re.compile(r"--latent-steps (\d+)")

_metrics = None


def metrics():
    """eval/metrics.py 를 파일에서 직접 로드한다 (패키지가 아니다)."""
    global _metrics
    if _metrics is None:
        spec = importlib.util.spec_from_file_location(
            "strict_metrics", TREE / "eval" / "metrics.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _metrics = module
    return _metrics


_ANAGRAM_CHECKED: set[str] = set()


def _warn_order_blind_choices(key: str, records: list) -> None:
    """Warn when a dataset has choices that differ only in the order of characters.

    `acc_of_seq` ranks candidates with `difflib.SequenceMatcher.quick_ratio()`, which
    compares character multisets and ignores order: "T1N0M0" and "T0N1M0" score 1.000
    against each other. Two such choices cannot be told apart, so a wrong pick is
    counted as correct. The five MultiPathQA sets have zero such items (932/932
    checked 2026-09-13); a new dataset may not, and it would fail silently.
    """
    if key in _ANAGRAM_CHECKED:
        return
    _ANAGRAM_CHECKED.add(key)
    hits: list[tuple[int, list[str]]] = []
    for index, record in enumerate(records):
        choices = record.get("Choice") or []
        if len(choices) < 2:
            continue
        signatures = [
            tuple(sorted(str(choice).lower().replace(" ", ""))) for choice in choices
        ]
        if len(set(signatures)) == len(signatures):
            continue
        clashing = [
            str(choice)
            for choice, signature in zip(choices, signatures)
            if signatures.count(signature) > 1
        ]
        hits.append((index, clashing[:3]))
    if hits:
        index, clashing = hits[0]
        print(
            f"[dashboard_rows] WARNING {key}: {len(hits)}/{len(records)} items have "
            "choices that differ only in character order; acc_of_seq uses "
            "quick_ratio() and scores them identically, so a wrong pick counts as "
            f"correct. first at dataset_index {index}: {clashing}",
            file=sys.stderr,
        )


def load_dataset(key: str) -> list:
    path = DS_ROOT / f"{key}.json"
    if not path.is_file():
        raise SystemExit(f"dataset not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    _warn_order_blind_choices(key, records)
    return records


def _as_index(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def collect_latent(arm: Path) -> dict[int, tuple[str, float | None]]:
    """result.json 산출물에서 인덱스별 (예측, 소요초)를 모은다."""
    latest: dict[int, tuple[int, str, float | None]] = {}
    for result_path in arm.glob("**/result.json"):
        matched = CASE_RE.match(result_path.parent.name)
        if not matched:
            continue
        try:
            mtime = result_path.stat().st_mtime_ns
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        index = _as_index(data.get("dataset_index"), int(matched.group(1)))
        answer = data.get("answer")
        prediction = answer.get("answer") if isinstance(answer, dict) else answer
        if not isinstance(prediction, str):
            continue
        seconds = None
        role_calls = data.get("role_calls")
        if isinstance(role_calls, list):
            values = [
                float(call["elapsed_seconds"])
                for call in role_calls
                if isinstance(call, dict)
                and isinstance(call.get("elapsed_seconds"), (int, float))
            ]
            if values:
                seconds = sum(values)
        previous = latest.get(index)
        if previous is None or mtime > previous[0]:
            latest[index] = (mtime, prediction, seconds)
    return {index: (prediction, seconds) for index, (_, prediction, seconds) in latest.items()}


def collect_single(arm: Path) -> dict[int, tuple[str, float | None]]:
    """single 워커/머지 predictions jsonl 에서 인덱스별 (예측, 소요초)를 모은다."""
    latest: dict[int, tuple[int, str, float | None]] = {}
    for predictions in arm.glob("**/predictions/qwen3vl_predictions.jsonl"):
        try:
            mtime = predictions.stat().st_mtime_ns
            lines = predictions.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            index = _as_index(row.get("sample_index"), _as_index(row.get("question_id"), -1))
            if index < 0:
                continue
            prediction = row.get("prediction")
            if not isinstance(prediction, str):
                continue
            seconds = None
            elapsed = row.get("inference_time_sec")
            try:
                if elapsed is not None:
                    seconds = float(elapsed)
            except (TypeError, ValueError):
                seconds = None
            previous = latest.get(index)
            if previous is None or mtime > previous[0]:
                latest[index] = (mtime, prediction, seconds)
    return {index: (prediction, seconds) for index, (_, prediction, seconds) in latest.items()}


def score(dataset_key: str, cases: dict[int, tuple[str, float | None]]) -> dict:
    """대시보드와 동일한 집계로 n / micro / bacc / sec 를 낸다."""
    module = metrics()
    records = load_dataset(dataset_key)
    by_class: dict[str, list[int]] = {}
    seconds: list[float] = []
    for index, (prediction, elapsed) in cases.items():
        if not 0 <= index < len(records):
            continue
        record = records[index]
        choices = record.get("Choice")
        gold = record.get("Answer")
        if not (isinstance(choices, list) and isinstance(gold, str)):
            continue
        predicted = module.expand_letter(prediction.strip(), choices)
        expected = module.expand_letter(gold.strip(), choices)
        by_class.setdefault(expected, []).append(
            int(bool(module.acc_of_seq(choices, expected, predicted)))
        )
        if elapsed is not None and elapsed >= 0:
            seconds.append(elapsed)
    total = sum(len(values) for values in by_class.values())
    if total == 0:
        return {"n": 0, "micro": None, "bacc": None, "sec": None}
    micro = sum(sum(values) for values in by_class.values()) / total
    balanced = sum(sum(values) / len(values) for values in by_class.values()) / len(by_class)
    return {
        "n": total,
        "micro": round(micro, 4),
        "bacc": round(balanced, 4),
        "sec": round(sum(seconds) / len(seconds), 1) if seconds else None,
    }


def normalize_dataset(name: str) -> str | None:
    candidate = re.sub(r"^mp_", "", name)
    candidate = re.sub(r"_\d{8}$", "", candidate)
    return DATASET_KEYS.get(candidate)


def normalize_param(name: str) -> str | None:
    matched = re.search(r"(\d+b)", name.lower())
    return matched.group(1) if matched else None


def arm_step(arm: Path, method: str) -> int:
    """run_meta.json > 드라이버 로그의 --latent-steps > 0."""
    meta_path = arm / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if METHOD_LABELS.get(method) == "Latent base":
                return int(meta.get("steps", 0))
            return 0
        except (OSError, ValueError, TypeError):
            pass
    if METHOD_LABELS.get(method) != "Latent base":
        return 0
    log_path = arm.parent / f"{arm.name}.log"
    if log_path.is_file():
        try:
            matched = STEP_RE.search(log_path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            matched = None
        if matched:
            return int(matched.group(1))
    return 0


def discover_arms(root: Path) -> list[Path]:
    arms = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_dir() and path.name in METHOD_LABELS
    ]
    return arms


def row_for_arm(
    arm: Path,
    dataset_key: str,
    param: str,
    method: str,
    step: int,
    template: str,
    full: bool,
) -> dict | None:
    cases = collect_single(arm) if METHOD_LABELS[method] == "Single" else collect_latent(arm)
    if not cases:
        return None
    result = score(dataset_key, cases)
    if result["n"] == 0:
        return None
    accuracy = result["bacc"] if dataset_key in BALANCED else result["micro"]
    row = {
        "dataset": dataset_key,
        "task": template.format(label=METHOD_LABELS[method]),
        "param": param,
        "step": step,
        "n": result["n"],
        "acc": accuracy,
        "sec": result["sec"],
    }
    if full:
        row["micro"] = result["micro"]
        row["bacc"] = result["bacc"]
        row["root"] = str(arm)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=TREE / "runs")
    parser.add_argument("--glob", default="baselines_*", help="runs-root 아래에서 스캔할 런 그룹 패턴")
    parser.add_argument("--scan", type=Path, action="append", default=[], help="임의 런 루트 (반복 가능)")
    parser.add_argument("--arm", type=Path, help="arm 디렉토리 하나만 채점")
    parser.add_argument("--dataset", help="--arm 과 함께: 데이터셋 이름")
    parser.add_argument("--param", help="--arm 과 함께: 2b|4b|8b")
    parser.add_argument("--method", help="--arm 과 함께: single|vlmas|latent-base")
    parser.add_argument("--step", type=int, help="--arm 과 함께: latent step")
    parser.add_argument("--task-template", default="{label}(0913)")
    parser.add_argument("--full", action="store_true", help="micro/bacc/root 를 함께 출력")
    arguments = parser.parse_args()

    rows: list[dict] = []
    if arguments.arm is not None:
        if not (arguments.dataset and arguments.param and arguments.method):
            raise SystemExit("--arm 은 --dataset/--param/--method 가 함께 필요하다")
        dataset_key = DATASET_KEYS.get(arguments.dataset)
        if dataset_key is None:
            raise SystemExit(f"알 수 없는 dataset: {arguments.dataset}")
        method = arguments.method
        if method not in METHOD_LABELS:
            raise SystemExit(f"알 수 없는 method: {method}")
        step = arguments.step if arguments.step is not None else arm_step(arguments.arm, method)
        row = row_for_arm(
            arguments.arm, dataset_key, arguments.param, method, step,
            arguments.task_template, arguments.full,
        )
        if row is not None:
            rows.append(row)
    else:
        roots = list(arguments.scan)
        if not roots:
            roots = sorted(arguments.runs_root.glob(arguments.glob))
        for root in roots:
            if not root.is_dir():
                print(f"skip (not a directory): {root}", file=sys.stderr)
                continue
            for arm in discover_arms(root):
                dataset_key = None
                param = None
                for parent in arm.parents:
                    if param is None:
                        param = normalize_param(parent.name)
                    if dataset_key is None:
                        dataset_key = normalize_dataset(parent.name)
                    if parent == root.parent:
                        break
                if dataset_key is None or param is None:
                    print(
                        f"skip (dataset/param 추론 실패, --arm 으로 지정하라): {arm}",
                        file=sys.stderr,
                    )
                    continue
                row = row_for_arm(
                    arm, dataset_key, param, arm.name, arm_step(arm, arm.name),
                    arguments.task_template, arguments.full,
                )
                if row is not None:
                    rows.append(row)

    rows.sort(key=lambda row: (row["dataset"], row["param"], row["step"], row["task"]))
    json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)
    print()


if __name__ == "__main__":
    main()
