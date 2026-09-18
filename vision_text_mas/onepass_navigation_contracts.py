"""Validated one-call spatial plan for the OnePass Navigator."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, override

from pydantic import Field, field_validator

from vision_text_mas.contracts import FrozenModel


DETAIL_FALLBACK_IDS: Final = (1, 6, 11, 16, 8, 9)
INTEGER_PATTERN: Final = re.compile(r"\d+")
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def recover_one_shot_navigation_slots(raw: str) -> dict[str, JsonValue]:
    """Recover bounded ID fields, rejecting candidate-list echoes/repetition."""
    root_marker = '"root_ids"'
    detail_marker = '"detail_cell_ids"'
    root_start = raw.find(root_marker)
    detail_start = raw.find(detail_marker)
    if root_start < 0 or detail_start <= root_start:
        return {}
    root_segment = raw[root_start + len(root_marker) : detail_start]
    detail_segment = raw[detail_start + len(detail_marker) :]
    root_ids = [int(value) for value in INTEGER_PATTERN.findall(root_segment)]
    detail_ids = [int(value) for value in INTEGER_PATTERN.findall(detail_segment)]
    if len(root_ids) > 5 or len(detail_ids) > 20:
        return {}
    root_values: list[JsonValue] = list(root_ids)
    detail_values: list[JsonValue] = list(detail_ids)
    return {
        "root_ids": root_values,
        "detail_cell_ids": detail_values,
        "selection_reason": "bounded slot recovery",
    }


@dataclass(frozen=True, slots=True)
class InvalidOneShotNavigationError(ValueError):
    """One-shot selection violates its bounded discrete coordinate system."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


class OneShotNavigation(FrozenModel):
    """Whole-slide regions and within-anchor x20 targets from one model call."""

    root_ids: tuple[int, ...] = Field(min_length=1, max_length=5)
    detail_cell_ids: tuple[int, ...] = Field(min_length=1, max_length=20)
    selection_reason: str = Field(min_length=1)

    @field_validator("root_ids")
    @classmethod
    def require_positive_root_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(candidate_id < 1 for candidate_id in value):
            raise InvalidOneShotNavigationError("root IDs must be positive")
        return value

    @field_validator("detail_cell_ids")
    @classmethod
    def require_x20_grid_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(candidate_id not in range(1, 17) for candidate_id in value):
            raise InvalidOneShotNavigationError(
                "x20 grid IDs must be between 1 and 16"
            )
        return value


def _integer_ids(value: JsonValue | None, *, allowed: frozenset[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    return [
        candidate_id
        for candidate_id in value
        if isinstance(candidate_id, int)
        and not isinstance(candidate_id, bool)
        and candidate_id in allowed
    ]


def _fill(values: list[int], *, fallback: tuple[int, ...], count: int) -> tuple[int, ...]:
    repaired = list(values[:count])
    while len(repaired) < count:
        repaired.append(fallback[len(repaired) % len(fallback)])
    return tuple(repaired)


def _unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def _fill_unique(
    values: list[int], *, fallback: tuple[int, ...], count: int
) -> tuple[int, ...]:
    repaired = list(values[:count])
    for candidate_id in fallback:
        if len(repaired) == count:
            break
        if candidate_id not in repaired:
            repaired.append(candidate_id)
    return _fill(repaired, fallback=fallback, count=count)


def repair_one_shot_navigation_output(
    value: JsonValue,
    *,
    root_candidate_ids: frozenset[int],
    overview_count: int,
    detail_count: int,
) -> JsonValue:
    """Repair malformed discrete IDs locally instead of issuing a new query."""
    raw: dict[str, JsonValue] = value if isinstance(value, dict) else {}
    root_fallback = tuple(sorted(root_candidate_ids))
    if not root_fallback:
        return raw
    root_ids = _fill_unique(
        _unique(_integer_ids(raw.get("root_ids"), allowed=root_candidate_ids)),
        fallback=root_fallback,
        count=overview_count,
    )
    detail_ids = _fill(
        _integer_ids(raw.get("detail_cell_ids"), allowed=frozenset(range(1, 17))),
        fallback=DETAIL_FALLBACK_IDS,
        count=detail_count,
    )
    reason = raw.get("selection_reason")
    return {
        "root_ids": list(root_ids),
        "detail_cell_ids": list(detail_ids),
        "selection_reason": (
            reason
            if isinstance(reason, str) and reason.strip()
            else "model-selected tissue regions"
        ),
    }
