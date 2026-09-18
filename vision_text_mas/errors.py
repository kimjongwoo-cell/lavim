"""Typed terminal failures for Vision-TextMAS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCode(str, Enum):
    """Stable failure codes written to case artifacts."""

    MODEL_OUTPUT = "FAILED_MODEL_OUTPUT"
    SELECTION_CONTRACT = "FAILED_SELECTION_CONTRACT"
    INSUFFICIENT_CANDIDATES = "FAILED_INSUFFICIENT_CANDIDATES"
    IMAGE_IO = "FAILED_IMAGE_IO"
    MODEL_EXECUTION = "FAILED_MODEL_EXECUTION"
    ANSWER_CONTRACT = "FAILED_ANSWER_CONTRACT"


@dataclass(frozen=True, slots=True)
class PipelineFailure(Exception):
    """One explicit technical or contract failure."""

    code: FailureCode
    stage: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code.value} at {self.stage}: {self.detail}"
