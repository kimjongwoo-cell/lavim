"""Dataset validation shared by every experiment method."""

from __future__ import annotations

import json
from pathlib import Path

from ..config.experiment import ExperimentSpec


class DatasetUnavailableError(RuntimeError):
    """Raised when a requested canonical dataset is not locally available."""


def count_cases(spec: ExperimentSpec) -> int:
    """Validate all immutable inputs and return the number of question cases."""
    for path, label in (
        (spec.dataset_path, "dataset"),
        (spec.slide_root, "slide root"),
        (spec.model_path, "model"),
        (spec.contract_path, "contract"),
    ):
        if not path.exists():
            raise DatasetUnavailableError(f"{label} is unavailable: {path}")
    payload = json.loads(spec.dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise DatasetUnavailableError(f"dataset must be a non-empty JSON array: {spec.dataset_path}")
    return len(payload)


def partition_case_indices(total: int, shard: int, shard_count: int) -> tuple[int, ...]:
    """Assign every question to exactly one GPU in round-robin order."""
    return tuple(range(shard, total, shard_count))
