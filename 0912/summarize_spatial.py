#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow", "duckdb", "pydantic"]
# ///
# How to run: uv run 0912/summarize_spatial.py
"""Audit the fixed pilot and write a reproducible paired-comparison artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb
import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

ROOT: Final = Path(__file__).resolve().parent
PILOT_IDS: Final = tuple(range(0, 128, 8))


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class Box(Record):
    x: int
    y: int
    width: int
    height: int


class Patch(Record):
    patch_id: str
    magnification: int
    box: Box
    source_x5_anchor: str


class Answer(Record):
    answer: str


class RoleCall(Record):
    role: str
    prompt: str


class Result(Record):
    dataset_index: int
    slide_id: str
    gold_answer: str
    answer: Answer
    patches: tuple[Patch, ...]
    role_calls: tuple[RoleCall, ...]


class Metric(Record):
    dataset_index: int
    success: bool
    wall_seconds: float


class Step(Record):
    step: int
    patch_indices: tuple[int, ...]
    visual_tokens: int
    biased_layer_calls: int
    cache_tokens: int


class Trace(Record):
    dataset_index: int
    groups: tuple[tuple[int, ...], ...]
    original_parents: tuple[int, ...]
    patch_token_counts: tuple[int, ...]
    steps: tuple[Step, ...]


@dataclass(frozen=True, slots=True)
class Run:
    results: dict[int, Result]
    metrics: dict[int, Metric]


def read_run(directory: Path) -> Run:
    """Parse completed artifacts; terminal failures are not treated as successes."""
    results = [
        Result.model_validate_json(p.read_text())
        for p in directory.glob("*/result.json")
    ]
    metrics_path = directory / "attempt_metrics.jsonl"
    metrics = (
        [
            Metric.model_validate_json(line)
            for line in metrics_path.read_text().splitlines()
        ]
        if metrics_path.exists()
        else []
    )
    return Run(
        {result.dataset_index: result for result in results},
        {metric.dataset_index: metric for metric in metrics if metric.success},
    )


def main() -> None:
    """Compare all arms on the exact pilot IDs, and fail on protocol drift."""
    directories = {
        "Base (existing SDPA)": ROOT / "runs/base_sdpa_gpu6",
        "Pruning-B (existing SDPA)": ROOT / "runs/pruning_b_sdpa_gpu8",
        "Spatial latent": ROOT / "runs/spatial_spatial_pilot_gpu6",
        "Shuffled latent": ROOT / "runs/spatial_shuffled_pilot_gpu8",
    }
    runs = {name: read_run(path) for name, path in directories.items()}
    common = sorted(
        set(PILOT_IDS).intersection(
            *(set(run.results) & set(run.metrics) for run in runs.values())
        )
    )
    base = runs["Base (existing SDPA)"]
    rows = []
    for name, run in runs.items():
        for index in common:
            result = run.results[index]
            original = base.results[index]
            assert result.slide_id == original.slide_id
            assert result.gold_answer == original.gold_answer
            assert result.patches == original.patches
            assert result.role_calls == original.role_calls
            if name in {"Spatial latent", "Shuffled latent"}:
                trace_path = directories[name] / "spatial_traces" / f"{index:03d}.json"
                trace = Trace.model_validate_json(trace_path.read_text())
                assert trace.dataset_index == index
                assert [step.step for step in trace.steps] == list(range(1, 11))
                assert [step.biased_layer_calls for step in trace.steps] == [13] * 8 + [
                    0,
                    0,
                ]
                assert sorted(
                    patch for group in trace.groups for patch in group
                ) == list(range(12))
                assert all(step.visual_tokens == 768 for step in trace.steps[:8])
                assert all(0 < step.cache_tokens <= 12288 for step in trace.steps[:8])
                assert all(not step.patch_indices for step in trace.steps[8:])
                spatial = name == "Spatial latent"
                assert all(
                    (trace.original_parents[child] == group[0]) == spatial
                    for group in trace.groups
                    for child in group[1:]
                )
            pred = result.answer.answer.strip().casefold()
            gold = result.gold_answer.strip().casefold()
            rows.append(
                {
                    "method": name,
                    "id": index,
                    "correct": pred == gold,
                    "base_correct": original.answer.answer.strip().casefold() == gold,
                    "prediction": result.answer.answer,
                    "gold": result.gold_answer,
                    "wall_seconds": run.metrics[index].wall_seconds,
                }
            )
    if not rows:
        raise SystemExit("No shared completed pilot IDs yet")
    with duckdb.connect() as connection:
        connection.register("cases", pa.Table.from_pylist(rows))
        summary = (
            connection.sql("""
            SELECT method, count(*) n, sum(correct::INT)::BIGINT correct,
              100.0 * avg(correct::INT) accuracy, avg(wall_seconds) mean_seconds,
              sum((correct AND NOT base_correct)::INT)::BIGINT fixed,
              sum((NOT correct AND base_correct)::INT)::BIGINT regressed
            FROM cases GROUP BY method ORDER BY method
        """)
            .arrow()
            .read_all()
            .to_pylist()
        )
    for row in summary:
        numpy_mean = np.mean(
            [r["wall_seconds"] for r in rows if r["method"] == row["method"]]
        )
        assert np.isclose(row["mean_seconds"], numpy_mean)
    report = {
        "planned_ids": PILOT_IDS,
        "common_completed_ids": common,
        "complete": common == list(PILOT_IDS),
        "summary": summary,
        "cases": rows,
        "audits": (
            "Exact patches, prompts, labels, schedule, hook calls, "
            "group size and parent identity checked"
        ),
        "timing_note": (
            "Base/Pruning timings are historical, "
            "not simultaneous controlled latency measurements."
        ),
        "sources": {name: str(path) for name, path in directories.items()},
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "spatial_latent").glob("*.py"))
        },
    }
    (ROOT / "spatial_pilot_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"n": len(common), "complete": report["complete"], "summary": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
