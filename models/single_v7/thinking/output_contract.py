"""Parse Single-v7 generations and detect unfinished thinking outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SingleResponse:
    """Parsed response plus whether the model emitted an explicit final answer."""

    answer: str
    explanation: str
    explicit: bool


def extract_last_boxed_answer(text: str) -> str | None:
    """Return the last box, matching official LatentMAS answer ordering."""
    boxes: list[str] = re.findall(r"\\boxed\{([^}]*)\}", text)
    return boxes[-1].strip() if boxes else None


def parse_single_response(raw_output: str) -> SingleResponse:
    """Parse labels or a box, retaining the legacy last-line fallback."""
    visible = raw_output.rpartition("</think>")[2].strip()
    json_match = re.search(r"\{.*?\}", visible, flags=re.DOTALL)
    if json_match is not None:
        try:
            json_value = json.loads(json_match.group())
        except json.JSONDecodeError:
            json_value = None
        if isinstance(json_value, dict):
            answer_value = json_value.get("answer")
            rationale_value = json_value.get("rationale")
            if isinstance(answer_value, str) and answer_value.strip():
                explanation = (
                    rationale_value.strip()
                    if isinstance(rationale_value, str) and rationale_value.strip()
                    else ""
                )
                return SingleResponse(answer_value.strip(), explanation, explicit=True)
    answer = ""
    explanation = ""
    for line in visible.splitlines():
        label, separator, value = line.strip().partition(":")
        if not separator:
            continue
        normalized_label = label.strip().lower()
        if normalized_label == "answer":
            answer = value.strip()
        elif normalized_label == "explanation":
            explanation = value.strip()
    if answer:
        return SingleResponse(answer, explanation, explicit=True)

    boxed = extract_last_boxed_answer(visible)
    if boxed:
        return SingleResponse(boxed, "", explicit=True)

    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    fallback = lines[-1] if lines else "unknown"
    fallback = re.sub(
        r"^(?:final\s+answer|answer)\s*[:：]\s*",
        "",
        fallback,
        flags=re.IGNORECASE,
    ).strip()
    return SingleResponse(fallback or "unknown", "", explicit=False)


def extract_single_response(raw_output: str) -> tuple[str, str]:
    """Return the answer and explanation expected by the stock adapter."""
    parsed = parse_single_response(raw_output)
    return parsed.answer, parsed.explanation


def requires_finalization(
    raw_output: str,
    output_tokens: int,
    token_limit: int,
) -> bool:
    """Detect a generation that exhausted its budget before a final answer."""
    return output_tokens >= token_limit and not parse_single_response(raw_output).explicit


def to_stock_labeled_output(answer: str, explanation: str = "") -> str:
    """Bridge the shared labeled response into the stock WSI evaluator schema."""
    return f"answer: {answer}\nexplanation: {explanation}".rstrip()
