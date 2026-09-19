#!/usr/bin/env python3
"""MultiPathQA 런을 exact / norm / nospace 세 level 로 채점한다 (판정 규칙: eval/answer_match.py).

집계는 0912/live_dashboard.py collect_uncached() 와 같다:
  - run_dir/<case>/result.json 을 읽는다. --variant 를 주면 run_dir/run_manifest.json 의 variant 로 거른다
  - dataset 은 slide_id 접두사("gtex__…")로 정한다
  - 같은 dataset_index 는 result.json mtime 이 최신인 것 하나만 쓴다
  - gtex / tcga / panda 는 BAcc(gold 라벨별 recall 평균), 그 외는 Acc
  - BAcc 클래스 키는 모든 level 에서 exact 로 정리한 gold 다 (level 이 클래스를 합치지 않게)
  - sec = run_dir/attempt_metrics.jsonl 의 성공 role_call_seconds 평균 (index 별 최신 파일)

보기(Choice)는 result.json 에 없어서 데이터셋 JSON 에서 dataset_index 로 찾는다. 그 레코드의
Answer 가 result.json 의 gold_answer 와 exact 로 같을 때만 쓰고, 다르면 choices_missing 으로 세고
그 문항은 exact 로 판정한다. 20케이스 부분집합 런처럼 인덱스가 다른 파일 기준이면
--dataset-json 으로 그 파일을 준다.

usage:
  rescore.py --root '0912/runs/nav2v2_base_expert_*' --dataset tcga_expert_vqa --variant base
  rescore.py --root '0912/runs/*nav2*' --exclude prompt2 --dataset gtex --variant base --changed
  rescore.py --root sender_relay_exp/runs/sb_sdpa_base --dataset tcga_slidebench
  rescore.py --collisions          # 데이터셋 5개 보기 충돌만
상대 경로·glob 은 이 트리 루트 기준이다.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

TREE = Path(__file__).resolve().parents[1]
DS_ROOT = TREE.parent / "datasets" / "MultiPathQA" / "ready_wsivqa" / "full_no_panda"
DATASETS = ("tcga_expert_vqa", "gtex", "tcga_slidebench", "tcga", "panda")
BALANCED = frozenset({"gtex", "tcga", "panda"})


def _load_answer_match():
    spec = importlib.util.spec_from_file_location("answer_match", TREE / "eval" / "answer_match.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("answer_match", module)
    spec.loader.exec_module(module)
    return module


am = _load_answer_match()


def resolve_roots(patterns: list[str]) -> list[Path]:
    roots: list[Path] = []
    for pattern in patterns:
        full = pattern if Path(pattern).is_absolute() else str(TREE / pattern)
        matches = sorted(glob.glob(full))
        if not matches:
            print(f"[rescore] no match: {pattern}", file=sys.stderr)
        roots.extend(Path(match) for match in matches if Path(match).is_dir())
    return roots


def dataset_from_slide(slide_id: str) -> str | None:
    head = slide_id.split("__", 1)[0]
    return head if head in DATASETS and slide_id.startswith(f"{head}__") else None


def run_dirs(roots: list[Path], variant: str | None, exclude: list[str]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        found.update(path.parent for path in root.rglob("run_manifest.json"))
        found.update(path.parent.parent for path in root.rglob("result.json"))
    kept: list[Path] = []
    for run_dir in sorted(found):
        if any(token in str(run_dir) for token in exclude):
            continue
        if variant is not None:
            try:
                manifest = json.loads((run_dir / "run_manifest.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if str(manifest.get("variant")) != variant:
                continue
        kept.append(run_dir)
    return kept


def collect(dirs: list[Path], datasets: set[str]):
    """dataset -> {index: (gold, prediction)}, dataset -> [seconds]."""
    cases: dict[str, dict[int, tuple[float, str, str]]] = {}
    timings: dict[str, dict[int, tuple[float, float]]] = {}
    for run_dir in dirs:
        for result_path in run_dir.glob("*/result.json"):
            try:
                payload = json.loads(result_path.read_text())
                index = int(payload["dataset_index"])
                dataset = dataset_from_slide(str(payload["slide_id"]))
                answer = str(payload.get("answer", {}).get("answer", ""))
                gold = str(payload.get("gold_answer", ""))
                stamp = result_path.stat().st_mtime
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
                continue
            if dataset is None or dataset not in datasets:
                continue
            current = cases.setdefault(dataset, {}).get(index)
            if current is None or stamp > current[0]:
                cases[dataset][index] = (stamp, gold, answer)
        metrics_path = run_dir / "attempt_metrics.jsonl"
        if not metrics_path.exists():
            continue
        stamp = metrics_path.stat().st_mtime
        for line in metrics_path.read_text().splitlines():
            try:
                metric = json.loads(line)
                if not metric.get("success"):
                    continue
                index = int(metric["dataset_index"])
                dataset = dataset_from_slide(str(metric["slide_id"]))
                seconds = float(metric["role_call_seconds"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
                continue
            if dataset is None or dataset not in datasets:
                continue
            current = timings.setdefault(dataset, {}).get(index)
            if current is None or stamp > current[0]:
                timings[dataset][index] = (stamp, seconds)
    return (
        {ds: {i: (g, a) for i, (_, g, a) in rows.items()} for ds, rows in cases.items()},
        {ds: [s for _, s in rows.values()] for ds, rows in timings.items()},
    )


def load_records(dataset: str, path: Path | None = None) -> list[dict]:
    return json.loads((path or DS_ROOT / f"{dataset}.json").read_text(encoding="utf-8"))


def score_dataset(dataset: str, cases: dict[int, tuple[str, str]], records: list[dict],
                  seconds: list[float] | None = None, changed: bool = False) -> dict:
    items = []
    missing = 0
    for index in sorted(cases):
        gold, prediction = cases[index]
        choices = None
        if 0 <= index < len(records):
            record = records[index]
            if isinstance(record.get("Choice"), list) and am.normalize(record.get("Answer", "")) == am.normalize(gold):
                choices = record["Choice"]
        if choices is None:
            missing += 1
        items.append((index, gold, prediction, choices))

    row: dict = {
        "dataset": dataset,
        "score_name": "BAcc" if dataset in BALANCED else "Acc",
        "n": len(items),
        "sec": round(mean(seconds), 2) if seconds else None,
        "choices_missing": missing,
    }
    verdicts: dict[str, list[bool]] = {}
    for level in am.LEVELS:
        by_label: dict[str, list[bool]] = {}
        oks: list[bool] = []
        fallback = 0
        for _, gold, prediction, choices in items:
            ok, fell = am.judge(prediction, gold, choices, level)
            fallback += fell and level != "exact" and choices is not None
            by_label.setdefault(am.normalize(gold), []).append(ok)
            oks.append(ok)
        verdicts[level] = oks
        if not oks:
            value = None
        elif dataset in BALANCED:
            value = 100.0 * mean(sum(v) / len(v) for v in by_label.values())
        else:
            value = 100.0 * sum(oks) / len(oks)
        row[level] = {"score": round(value, 2) if value is not None else None, "correct": sum(oks)}
        if level != "exact":
            row[level]["collision_fallback"] = fallback
    if changed:
        row["changed"] = [
            {"index": index, "gold": gold, "prediction": prediction[:200],
             "correct_at": [lv for lv in am.LEVELS if verdicts[lv][pos]]}
            for pos, (index, gold, prediction, _) in enumerate(items)
            if len({verdicts[lv][pos] for lv in am.LEVELS}) > 1
        ]
    return row


def collision_report() -> list[dict]:
    report = []
    for dataset in DATASETS:
        records = load_records(dataset)
        row: dict = {"dataset": dataset, "items": len(records)}
        for level in am.LEVELS:
            hit_items, gold_hit, examples = 0, 0, []
            for index, record in enumerate(records):
                choices = record.get("Choice") or []
                groups = am.collisions(choices, level)
                if not groups:
                    continue
                hit_items += 1
                target = am.normalize(record.get("Answer", ""), level)
                gold_hit += any(am.normalize(choices[g[0]], level) == target for g in groups)
                if len(examples) < 3:
                    examples.append({"index": index, "choices": [[choices[i] for i in g] for g in groups]})
            labels = {}
            for record in records:
                labels.setdefault(am.normalize(record.get("Answer", ""), level), set()).add(
                    am.normalize(record.get("Answer", "")))
            row[level] = {"items_with_collision": hit_items, "gold_in_collision": gold_hit,
                          "merged_gold_labels": sum(len(v) > 1 for v in labels.values()),
                          "examples": examples}
        report.append(row)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", default=[], help="런 루트 (glob 가능, 반복 가능)")
    parser.add_argument("--dataset", action="append", choices=DATASETS, help="생략하면 5개 전부")
    parser.add_argument("--variant", help="run_manifest.json variant 로 거른다 (예: base)")
    parser.add_argument("--exclude", action="append", default=[], help="경로에 이 문자열이 들어간 run_dir 제외")
    parser.add_argument("--dataset-json", type=Path, help="--dataset 하나와 함께: 보기를 찾을 JSON")
    parser.add_argument("--changed", action="store_true", help="level 간 판정이 갈린 문항을 함께 출력")
    parser.add_argument("--collisions", action="store_true", help="데이터셋 보기 충돌만 출력")
    arguments = parser.parse_args()

    if arguments.collisions:
        json.dump(collision_report(), sys.stdout, ensure_ascii=False, indent=1)
        print()
        return
    if not arguments.root:
        parser.error("--root 가 필요하다 (또는 --collisions)")
    datasets = set(arguments.dataset or DATASETS)
    if arguments.dataset_json is not None and len(datasets) != 1:
        parser.error("--dataset-json 은 --dataset 하나와 함께 쓴다")

    dirs = run_dirs(resolve_roots(arguments.root), arguments.variant, arguments.exclude)
    cases, timings = collect(dirs, datasets)
    rows = [
        score_dataset(ds, cases[ds], load_records(ds, arguments.dataset_json),
                      timings.get(ds), arguments.changed)
        for ds in DATASETS if ds in cases
    ]
    json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)
    print()


if __name__ == "__main__":
    main()
