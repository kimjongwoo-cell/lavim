"""Strict-v2 metric primitives for the live matched-prefix plot."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CompletedCase:
    """One completed latent answer, ordered by its result-file timestamp."""

    dataset_index: int
    result_path: Path
    modified_ns: int


@dataclass(frozen=True, slots=True)
class Indicator:
    """The strict-v2 accuracy numerators and denominators for one answer."""

    total_correct: int
    total_count: int
    mcq_correct: int
    mcq_count: int
    open_correct: int
    open_count: int


@dataclass(frozen=True, slots=True)
class RunningPoint:
    """A method score over one common completed-case prefix."""

    completed_cases: int
    total_accuracy: float
    mcq_accuracy: float | None
    open_substring_accuracy: float | None
    mcq_count: int = 0
    open_count: int = 0


@dataclass(frozen=True, slots=True)
class StrictFunctions:
    """The project evaluator primitives needed for the displayed metrics."""

    expand_letter: Callable[[str, list[str]], str]
    acc_of_seq: Callable[[list[str], str, str], bool | None]
    substring_correct: Callable[[str, str], bool]


def read_json(path: Path) -> dict[str, object]:
    """Read one JSON object and reject non-object payloads at the file boundary."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        msg = f"Expected a JSON object: {path}"
        raise TypeError(msg)
    return data


def dataset_index_from_result(path: Path) -> int:
    """Extract a validated dataset index from one latent result."""
    value = read_json(path).get("dataset_index")
    if isinstance(value, int):
        return value
    msg = f"Missing integer dataset_index: {path}"
    raise TypeError(msg)


def select_latest_cases(output_root: Path) -> tuple[CompletedCase, ...]:
    """Select one newest result per question so retries cannot duplicate a prefix."""
    latest: dict[int, CompletedCase] = {}
    for result_path in output_root.rglob("result.json"):
        case = CompletedCase(dataset_index_from_result(result_path), result_path, result_path.stat().st_mtime_ns)
        previous = latest.get(case.dataset_index)
        if previous is None or case.modified_ns > previous.modified_ns:
            latest[case.dataset_index] = case
    return tuple(sorted(latest.values(), key=lambda item: (item.modified_ns, item.dataset_index)))


def case_signature(cases: tuple[CompletedCase, ...]) -> tuple[tuple[int, int], ...]:
    """Identify the exact prefix without rereading unchanged source results."""
    return tuple((case.dataset_index, case.modified_ns) for case in cases)


def load_details(path: Path) -> dict[int, Indicator]:
    """Load precomputed strict indicators for a completed baseline evaluation."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        msg = f"Expected a JSON list: {path}"
        raise TypeError(msg)
    indicators: dict[int, Indicator] = {}
    for row in raw:
        if not isinstance(row, dict):
            msg = f"Expected a JSON object in {path}"
            raise TypeError(msg)
        question_id = row.get("question_id")
        answer_type = row.get("answer_type")
        if not isinstance(question_id, str) or not isinstance(answer_type, str):
            msg = f"Malformed evaluator detail in {path}"
            raise TypeError(msg)
        match answer_type:
            case "CLOSED":
                correct = int(bool(row["mcq_seq_correct"]))
                indicators[int(question_id)] = Indicator(correct, 1, correct, 1, 0, 0)
            case "OPEN":
                correct = int(bool(row["open_substring_correct"]))
                indicators[int(question_id)] = Indicator(correct, 1, 0, 0, correct, 1)
            case unexpected:
                msg = f"Unknown answer_type {unexpected!r} in {path}"
                raise ValueError(msg)
    return indicators


def load_eval_rows(path: Path) -> dict[int, dict[str, object]]:
    """Load shared strict-v2 question and choice metadata by question index."""
    rows: dict[int, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            msg = f"Expected a JSON object in {path}"
            raise TypeError(msg)
        question_id = row.get("question_id")
        if not isinstance(question_id, str):
            msg = f"Missing question_id in {path}"
            raise TypeError(msg)
        rows[int(question_id)] = row
    return rows


def load_strict_functions(evaluator_path: Path) -> StrictFunctions:
    """Load the exact matching helpers from the project strict evaluator."""
    spec = importlib.util.spec_from_file_location("strict_metrics_live_plot", evaluator_path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load strict evaluator: {evaluator_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return StrictFunctions(module.expand_letter, module.acc_of_seq, module.substring_correct)


def latent_indicator(case: CompletedCase, baseline_row: dict[str, object], strict: StrictFunctions) -> Indicator:
    """Score one latent answer with the same strict primitives as evaluation."""
    result = read_json(case.result_path)
    answer = result.get("answer")
    prediction = answer.get("answer") if isinstance(answer, dict) else None
    ground_truth = result.get("gold_answer")
    choices = baseline_row.get("choices")
    is_mcq = baseline_row.get("is_mcq")
    if not isinstance(prediction, str) or not isinstance(ground_truth, str):
        return Indicator(0, 0, 0, 0, 0, 0)
    if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
        msg = f"Malformed choices for question {case.dataset_index}"
        raise TypeError(msg)
    if not isinstance(is_mcq, bool):
        msg = f"Missing is_mcq for question {case.dataset_index}"
        raise TypeError(msg)
    if not prediction.strip() or not ground_truth.strip():
        return Indicator(0, 0, 0, 0, 0, 0)
    gt_eval = strict.expand_letter(ground_truth.strip(), choices)
    prediction_eval = strict.expand_letter(prediction.strip(), choices)
    if is_mcq or choices:
        correct = int(bool(strict.acc_of_seq(choices, gt_eval, prediction_eval)))
        return Indicator(correct, 1, correct, 1, 0, 0)
    correct = int(strict.substring_correct(gt_eval, prediction_eval))
    return Indicator(correct, 1, 0, 0, correct, 1)


def cumulative_points(indicators: list[Indicator]) -> list[RunningPoint]:
    """Convert per-item strict indicators into fair, running-prefix points."""
    total_correct = total_count = mcq_correct = mcq_count = open_correct = open_count = 0
    points: list[RunningPoint] = []
    for completed_cases, indicator in enumerate(indicators, start=1):
        total_correct += indicator.total_correct
        total_count += indicator.total_count
        mcq_correct += indicator.mcq_correct
        mcq_count += indicator.mcq_count
        open_correct += indicator.open_correct
        open_count += indicator.open_count
        points.append(
            RunningPoint(
                completed_cases,
                total_correct / total_count if total_count else 0.0,
                mcq_correct / mcq_count if mcq_count else None,
                open_correct / open_count if open_count else None,
                mcq_count,
                open_count,
            )
        )
    return points
