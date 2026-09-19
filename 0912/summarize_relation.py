#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow", "duckdb", "pydantic"]
# ///
# How to run: uv run 0912/summarize_relation.py
"""Audit and aggregate the fixed cross-scale relation-latent pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import duckdb
import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict
from summarize_spatial import PILOT_IDS, Trace, read_run

ROOT: Final = Path(__file__).resolve().parent


class Readout(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_index: int
    relation_columns: tuple[int, ...]
    alpha: float
    applied_layer_calls: int
    maximum_cache_tokens: int


def main() -> None:
    """Use only fully paired IDs and reject every known form of protocol drift."""
    directories = {
        "Base": ROOT / "runs/base_sdpa_gpu6",
        "Pruning-B": ROOT / "runs/pruning_b_sdpa_gpu8",
        "Relation real + readout": ROOT / "runs/relation_spatial_alpha0.1_gpu6",
        "Relation shuffled + readout": ROOT / "runs/relation_shuffled_alpha0.1_gpu8",
        "Relation real, no readout": ROOT / "runs/relation_spatial_alpha0.0_gpu6",
    }
    runs = {name: read_run(path) for name, path in directories.items()}
    common = sorted(
        set(PILOT_IDS).intersection(
            *(set(run.results) & set(run.metrics) for run in runs.values())
        )
    )
    base = runs["Base"]
    rows = []
    for name, run in runs.items():
        for index in common:
            result = run.results[index]
            original = base.results[index]
            assert result.slide_id == original.slide_id
            assert result.gold_answer == original.gold_answer
            assert result.patches == original.patches
            assert result.role_calls == original.role_calls
            if name.startswith("Relation"):
                trace = Trace.model_validate_json(
                    (
                        directories[name] / "spatial_traces" / f"{index:03d}.json"
                    ).read_text()
                )
                readout = Readout.model_validate_json(
                    (
                        directories[name] / "relation_readout" / f"{index:03d}.json"
                    ).read_text()
                )
                assert trace.dataset_index == readout.dataset_index == index
                assert [step.step for step in trace.steps] == list(range(1, 11))
                assert [step.biased_layer_calls for step in trace.steps] == [13] * 10
                assert [step.visual_tokens for step in trace.steps] == [
                    256,
                    512,
                ] * 4 + [0, 0]
                assert all(0 < step.cache_tokens <= 12288 for step in trace.steps)
                assert len(readout.relation_columns) == 4
                first = readout.relation_columns[0]
                assert readout.relation_columns == tuple(
                    first + 2 * i for i in range(4)
                )
                has_readout = "no readout" not in name
                assert (readout.alpha == 0.1) == has_readout
                assert (readout.applied_layer_calls > 0) == has_readout
                if has_readout:
                    assert 0 < readout.maximum_cache_tokens <= 12288
                spatial = "shuffled" not in name
                assert all(
                    (trace.original_parents[child] == group[0]) == spatial
                    for group in trace.groups
                    for child in group[1:]
                )
            prediction = result.answer.answer.strip().casefold()
            gold = result.gold_answer.strip().casefold()
            base_prediction = original.answer.answer.strip().casefold()
            rows.append(
                {
                    "method": name,
                    "id": index,
                    "correct": prediction == gold,
                    "base_correct": base_prediction == gold,
                    "changed_from_base": prediction != base_prediction,
                    "prediction": result.answer.answer,
                    "gold": result.gold_answer,
                    "wall_seconds": run.metrics[index].wall_seconds,
                }
            )
    if not rows:
        raise SystemExit("No fully paired relation-latent IDs yet")
    with duckdb.connect() as connection:
        connection.register("cases", pa.Table.from_pylist(rows))
        summary = (
            connection.sql("""
                SELECT method, count(*) n, sum(correct::INT)::BIGINT correct,
                  100.0 * avg(correct::INT) accuracy,
                  avg(wall_seconds) mean_seconds,
                  sum((correct AND NOT base_correct)::INT)::BIGINT fixed,
                  sum((NOT correct AND base_correct)::INT)::BIGINT regressed,
                  sum(changed_from_base::INT)::BIGINT changed
                FROM cases GROUP BY method ORDER BY method
            """)
            .arrow()
            .read_all()
            .to_pylist()
        )
    for aggregate in summary:
        observed = [
            row["wall_seconds"] for row in rows if row["method"] == aggregate["method"]
        ]
        assert np.isclose(aggregate["mean_seconds"], np.mean(observed))
    report = {
        "planned_ids": PILOT_IDS,
        "common_completed_ids": common,
        "complete": common == list(PILOT_IDS),
        "summary": summary,
        "cases": rows,
        "audits": (
            "Exact patches/prompts/labels, relation schedule, real/shuffled parents, "
            "latent addresses, readout alpha and hook calls checked"
        ),
        "timing_note": "Base and Pruning-B timings are historical.",
        "sources": {name: str(path) for name, path in directories.items()},
    }
    (ROOT / "relation_pilot_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"n": len(common), "complete": report["complete"], "summary": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
