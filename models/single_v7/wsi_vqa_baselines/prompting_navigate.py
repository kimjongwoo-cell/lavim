"""PathNavigate-style prompting for the single-image VQA baseline.

This mirrors the prompting *format* of SurpriseNav's final adjudicator
(specifically the vision adjudicator, ``surprisenav/vl_adjudicator.py``), but
keeps the single-image premise of the baseline: the model is shown ONE
pathology image and answers in one pass. There is no multi-region evidence,
no navigation/controller loop, and no debiasing/verification here.

The point of this module is to unify the *answer output format* with
PathNavigate: the model is asked to respond with a JSON object
``{"answer": ..., "explanation": ...}`` instead of the ``answer:/explanation:``
labelled text used by ``prompting.py``.

Rendered layout (system block shown only when include_system_prompt=True)::

    You are an expert pathologist examining a pathology image. ...
    <q_type-specific guidance>
    <guardrail>
    <answer-format instruction + JSON schema>

    Question: <question>
    Choices:
    A. <option 1>
    B. <option 2>
    ...

    Examine the image carefully, then answer the question. Respond in JSON.

MCQ answer representation is selectable via ``mcq_style`` ("letter" | "text"),
exactly like ``prompting.format_vqa_prompt``. Choice mapping for scoring reuses
``prompting.select_choice_prediction`` so both styles round-trip identically.
"""

from __future__ import annotations

import json
import re
from typing import Final

from . import prompting

_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- Prompt fragments, ported from SurpriseNav and adapted to a single image ---

_HEADER: Final[str] = (
    "You are an expert pathologist examining a pathology image. "
)

_GUIDANCE_MORPHOLOGY: Final[str] = (
    "You are performing DIFFERENTIAL DIAGNOSIS based on microscopic morphology.\n"
    "Key discriminators: single-file invasion -> lobular; tubule formation -> ductal; "
    "medullary pattern -> syncytial growth + lymphocytes; mucinous -> extracellular mucin pools.\n"
    "For grading: assess tubule formation, nuclear pleomorphism, and mitotic count.\n"
    "IMPORTANT: Always commit to a specific diagnosis based on the morphological evidence you observe.\n"
)

_GUIDANCE_OTHER: Final[str] = (
    "Synthesize your observations across the image to answer the clinical question.\n"
    "Identify the most consistent pattern, resolve contradictions by favoring the predominant finding.\n"
    "IMPORTANT: Always commit to a specific answer. Do NOT answer 'cannot be determined' "
    "unless truly no relevant evidence is visible.\n"
)

# Hallucination guardrail (ported from the text adjudicator; the vision
# adjudicator omits it, but we include it here per design choice).
_GUARDRAIL: Final[str] = (
    "A single pathology image cannot establish exact tumor size, TNM stage, lymph node "
    "status, surgical margin status, or clinical history. "
    "Do not trust image-level claims about receptor staining or HER2 score unless "
    "the image explicitly shows an immunostain. "
    "Prefer the best morphology-supported answer rather than inventing a gross measurement.\n"
)

_HER2_KEYWORDS: Final[tuple[str, ...]] = ("her2", "erbb2", "her-2")


def build_system_prompt(
    question: str,
    choices: list[str],
    *,
    q_type: str = "other",
    mcq_style: str = "letter",
    use_guardrail: bool = True,
) -> str:
    """Assemble the navigate-style system prompt (guidance + guardrail + format)."""
    guidance = _GUIDANCE_MORPHOLOGY if q_type == "morphology" else _GUIDANCE_OTHER
    parts = [_HEADER + guidance]
    if use_guardrail:
        parts.append(_GUARDRAIL)

    if choices:
        if mcq_style == "text":
            answer_rule = (
                "For multiple choice: assess each option against what you observe. "
                "Eliminate contradicted options, select the one with strongest visual support.\n"
                "Your answer MUST be the exact option text from the given choices.\n"
                'Respond in JSON: {"answer": "<exact choice text>", "explanation": "brief reasoning"}'
            )
        else:
            answer_rule = (
                "For multiple choice: assess each option against what you observe. "
                "Eliminate contradicted options, select the one with strongest visual support.\n"
                "Your answer MUST be ONLY the letter (A, B, C, or D).\n"
                'Respond in JSON: {"answer": "A", "explanation": "brief reasoning"}'
            )
    elif any(kw in str(question).lower() for kw in _HER2_KEYWORDS):
        answer_rule = (
            "For HER2: answer MUST be one of: '0', '1+', '2+', or '3+'.\n"
            'Respond in JSON: {"answer": "your answer", "explanation": "brief reasoning"}'
        )
    else:
        answer_rule = 'Respond in JSON: {"answer": "your answer", "explanation": "brief reasoning"}'

    parts.append(answer_rule)
    return "".join(parts[:-1]) + parts[-1]


def format_navigate_prompt(
    question: str,
    choices: list[str],
    *,
    q_type: str = "other",
    force_choice_answer: bool = True,
    mcq_style: str = "letter",
    include_system_prompt: bool = True,
    use_guardrail: bool = True,
) -> str:
    """Build the single-image, navigate-style prompt as one string.

    Signature mirrors ``prompting.format_vqa_prompt`` (plus ``q_type`` and
    ``use_guardrail``) so a runner can swap modules with minimal changes.
    ``force_choice_answer`` is accepted for parity; the choice constraint is
    always expressed through the system prompt's answer rule.
    """
    prompt_parts: list[str] = []
    if include_system_prompt:
        prompt_parts.append(
            build_system_prompt(
                question,
                choices,
                q_type=q_type,
                mcq_style=mcq_style,
                use_guardrail=use_guardrail,
            )
        )

    prompt_parts.append(f"Question: {str(question).strip()}")

    if choices:
        choice_lines = "\n".join(
            f"{_LETTERS[index] if index < len(_LETTERS) else index + 1}. {choice}"
            for index, choice in enumerate(choices)
        )
        prompt_parts.append(f"Choices:\n{choice_lines}")
        prompt_parts.append(
            "Examine the image carefully, then answer the question based on your "
            "observations. Respond in JSON."
        )
        return "\n\n".join(part for part in prompt_parts if part).strip()

    prompt_parts.append(
        "Examine the image carefully, then answer the question based on your "
        "observations. Respond in JSON."
    )
    return "\n\n".join(part for part in prompt_parts if part).strip()


def _extract_json(text: str) -> dict | None:
    """First ``{`` to last ``}`` JSON extraction (ported from json_utils)."""
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx + 1])
        except Exception:
            return None
    return None


def parse_navigate_output(text: str) -> tuple[str, str]:
    """Unified parser: JSON first, then fall back to the labelled-text parser.

    Returns ``(answer, explanation)``. Robust to models that ignore the JSON
    instruction and emit ``answer:/explanation:`` text or a bare answer instead,
    so both output styles can be received without changing the caller.
    """
    cleaned = prompting.strip_thinking(text)

    parsed = _extract_json(cleaned)
    if isinstance(parsed, dict):
        answer = str(parsed.get("answer", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
        if answer:
            return answer, explanation

    # Fallback: labelled "answer:/explanation:" text or first non-empty line.
    return prompting.parse_labeled_output(cleaned)


# Re-export shared choice-mapping helpers so callers can use a single module.
select_choice_prediction = prompting.select_choice_prediction
normalize_text = prompting.normalize_text
as_choice_list = prompting.as_choice_list
strip_thinking = prompting.strip_thinking
count_text_tokens = prompting.count_text_tokens
