"""Per-attempt reliability and efficiency accounting for direct-HF runs."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import torch

from vision_text_mas.contracts import CaseInput, RunResult
from vision_text_mas.errors import PipelineFailure


@dataclass(frozen=True, slots=True)
class AttemptMetric:
    """One physical case attempt; failures remain observable but are not efficient work."""

    dataset_index: int
    slide_id: str
    attempt: int
    success: bool
    wall_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    role_call_seconds: float | None
    physical_calls: int | None
    format_repairs: int | None


def _memory_peaks() -> tuple[int, int]:
    """Return this process's CUDA peak allocator measurements when available."""
    if not torch.cuda.is_available():
        return 0, 0
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _reset_memory_peaks() -> None:
    """Start a new per-attempt allocator peak window without changing model residency."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _append_metric(path: Path, metric: AttemptMetric) -> None:
    """Append one completed attempt atomically enough for a sequential worker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metric), sort_keys=True))
        handle.write("\n")


def load_attempts(path: Path) -> tuple[AttemptMetric, ...]:
    """Read persisted attempt records after a run has finished or been interrupted."""
    if not path.exists():
        return ()
    return tuple(
        AttemptMetric(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def run_case_with_retries(
    *,
    case: CaseInput,
    max_attempts: int,
    operation: Callable[[], RunResult],
    metrics_path: Path,
) -> RunResult:
    """Run a case immediately up to ``max_attempts`` and persist every attempt."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        _reset_memory_peaks()
        started = time.perf_counter()
        try:
            result = operation()
        except (PipelineFailure, torch.OutOfMemoryError):
            allocated, reserved = _memory_peaks()
            _append_metric(
                metrics_path,
                AttemptMetric(
                    case.dataset_index,
                    case.item.slide_id,
                    attempt,
                    False,
                    time.perf_counter() - started,
                    allocated,
                    reserved,
                    None,
                    None,
                    None,
                ),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == max_attempts:
                raise
        else:
            allocated, reserved = _memory_peaks()
            _append_metric(
                metrics_path,
                AttemptMetric(
                    case.dataset_index,
                    case.item.slide_id,
                    attempt,
                    True,
                    time.perf_counter() - started,
                    allocated,
                    reserved,
                    sum(call.elapsed_seconds for call in result.role_calls),
                    sum(call.physical_calls for call in result.role_calls),
                    sum(call.format_repairs for call in result.role_calls),
                ),
            )
            return result
    raise AssertionError("retry loop must either return or raise")


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    """Summarize observed values without inventing a denominator for missing data."""
    if not values:
        return None
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_successes(attempts: Sequence[AttemptMetric]) -> dict[str, object]:
    """Report success-only efficiency and separately retain excluded failure counts."""
    successful = tuple(attempt for attempt in attempts if attempt.success)
    failed_attempts = len(attempts) - len(successful)
    return {
        "successful_cases": len(successful),
        "failed_attempts_excluded": failed_attempts,
        "case_wall_seconds": _distribution([attempt.wall_seconds for attempt in successful]),
        "peak_allocated_bytes": _distribution(
            [float(attempt.peak_allocated_bytes) for attempt in successful]
        ),
        "peak_reserved_bytes": _distribution(
            [float(attempt.peak_reserved_bytes) for attempt in successful]
        ),
        "role_call_seconds": _distribution(
            [attempt.role_call_seconds for attempt in successful if attempt.role_call_seconds is not None]
        ),
        "physical_calls": _distribution(
            [float(attempt.physical_calls) for attempt in successful if attempt.physical_calls is not None]
        ),
        "format_repairs": _distribution(
            [float(attempt.format_repairs) for attempt in successful if attempt.format_repairs is not None]
        ),
    }
