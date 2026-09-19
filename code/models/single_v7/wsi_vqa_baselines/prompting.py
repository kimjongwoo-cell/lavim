from __future__ import annotations

import difflib
import re
from typing import Final


def _letter_seq(count):
    """Option labels A..Z then AA, AB, ... — 26-choice behavior unchanged."""
    base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labels = list(base)
    i = 0
    while len(labels) < count:
        labels.append(base[i // 26] + base[i % 26])
        i += 1
    return labels[:count]



# SYSTEM_PROMPT restored (baselines/inference/*-inference.py imports it). The 0710
# refactor commented it out in favor of ANSWER_RULES (used by text_mas diagnosis)
# but did not update the baseline inference imports → ImportError. Both now coexist.
# SYSTEM_PROMPT: Final[str] = (
#     "You are an expert pathology visual question answering assistant. "
#     "Answer based only on the provided pathology image.\n\n"
#     "Rules:\n"
#     "1. For biomarker questions (HER2, PR, ER, Ki-67 etc.): output the observed status or score only "
#     "(e.g. 'Positive', 'Negative', '2+', 'High'). Do not define the term.\n"
#     "2. For survival time questions: output the number only (e.g. '1881'). Do not explain.\n"
#     "3. For vital status questions: output 'Alive' or 'Dead' only.\n"
#     "4. For histological type questions: output the type name only.\n"
#     # "5. If choices are provided: your answer MUST be exactly one of the given option letters.\n"
#     "5. **If choices are provided, your 'answer' must be exactly one of the given options. "
#     "Never generate an answer outside the provided choices.**\n"
#     "6. For open-ended questions: answer using a single word or short phrase.\n\n"
#     "Always respond in this exact format:\n"
#     "answer: <exact choice text or short phrase>\n"
#     "explanation: <one short sentence>"
# )

SYSTEM_PROMPT: Final[str] = (
    # "You are an expert pathology visual question answering assistant. "
    "You are a helpful pathology assistant analyzing whole-slide images (WSI). "
    "Rules:\n"
    "1. For biomarker questions (HER2, PR, ER, Ki-67 etc.): output the observed status or score only "
    "(e.g. 'Positive', 'Negative', '2+', 'High'). Do not define the term.\n"
    # "2. For survival time questions: output the number only (e.g. '1881'). Do not explain.\n"
    "2. When the question asks about survival time or other quantitative results, "
    "output ONLY the numerical value for the answer key in the response JSON.\n"
    "3. For vital status questions: output 'Alive' or 'Dead' only.\n"
    # "4. For histological type questions: output the type name only.\n"
    "4. For histological type questions: commit to a specific diagnosis.\n"
    "5. **If choices are provided, your 'answer' must be exactly one of the given options. "
    "Never generate an answer outside the provided choices.**\n"
    "6. For open-ended questions: answer using a single word or short phrase."
    "Always respond in this exact format:\n"
    "answer: <exact choice text or short phrase>\n"
    "explanation: <one short sentence>"
)

def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def as_choice_list(raw_choices) -> list[str]:
    if raw_choices is None:
        return []
    if isinstance(raw_choices, list):
        return [str(choice) for choice in raw_choices if str(choice).strip()]
    return [str(raw_choices)] if str(raw_choices).strip() else []


def format_vqa_prompt(
    question: str,
    choices: list[str],
    *,
    force_choice_answer: bool = True,
    mcq_style: str = "letter",
    include_system_prompt: bool = True,
) -> str:
    prompt_parts = []
    if include_system_prompt:
        prompt_parts.append(SYSTEM_PROMPT)

    prompt_parts.append(f"Question: {question.strip()}")
    if choices:
        letters = _letter_seq(len(choices))
        choice_lines = "\n".join(f"{letters[index]}. {choice}" for index, choice in enumerate(choices))
        prompt_parts.append(f"Choices:\n{choice_lines}")
        if force_choice_answer:
            if mcq_style == "letter":
                prompt_parts.append(
                    "For this multiple-choice question, put only the option letter in the answer field."
                )
            else:
                prompt_parts.append(
                    "For this multiple-choice question, put only the exact choice text in the answer field."
                )
        return "\n\n".join(part for part in prompt_parts if part).strip()

    prompt_parts.append("Put a single word or short phrase in the answer field.")
    return "\n\n".join(part for part in prompt_parts if part).strip()


def parse_labeled_output(text: str) -> tuple[str, str]:
    answer = ""
    explanation = ""
    for line in str(text).strip().splitlines():
        stripped = line.strip()
        label, sep, value = stripped.partition(":")
        if not sep:
            continue
        normalized_label = label.strip().lower()
        if normalized_label == "answer":
            answer = value.strip()
        elif normalized_label == "explanation":
            explanation = value.strip()

    if not answer:
        for line in str(text).strip().splitlines():
            stripped = line.strip()
            if stripped:
                answer = stripped
                break
    return answer, explanation


def count_text_tokens(tokenizer, text: str) -> int:
    value = str(text).strip()
    if not value:
        return 0
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(value, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(value))
    return len(value.split())


def strip_thinking(text: str) -> str:
    stripped = str(text)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.I | re.S)
    stripped = re.sub(r"<thinking>.*?</thinking>", "", stripped, flags=re.I | re.S)
    stripped = re.sub(r"^\s*(?:reasoning|thinking)\s*[:：].*?(?:\n\s*\n|$)", "", stripped, flags=re.I | re.S)
    return stripped.strip()


def strip_choice_prefix(text: str) -> str:
    return re.sub(
        r"^\s*(?:answer\s*[:\-]\s*)?(?:option\s*)?([A-Z]|\d+)[\.\):\-]\s*",
        "",
        str(text).strip(),
        flags=re.I,
    ).strip()


def _choice_letter_from_prediction(raw_prediction: str, choices: list[str]) -> tuple[str, str] | None:
    letters = _letter_seq(len(choices))
    text = str(raw_prediction).strip()
    pred_norm = normalize_text(text)

    first_token = re.match(r"^\s*(?:answer\s*[:\-]\s*)?(?:option\s*)?([A-Z]|\d+)\b", text, flags=re.I)
    if first_token:
        token = first_token.group(1).upper()
        if token in letters:
            return token, "letter"
        if token.isdigit():
            # Only treat a leading number as an option INDEX when the whole answer is
            # that bare number (e.g. "3", "option 2", "3."). A number that begins a
            # measured value like "3.0 cm" / "3 mm" is an answer VALUE, not an index —
            # mapping it by index silently sends the correct value to the wrong choice.
            bare_number = re.fullmatch(
                r"\s*(?:answer\s*[:\-]\s*)?(?:option\s*)?\d+\s*[.\):]?\s*", text, flags=re.I
            )
            if bare_number:
                index = int(token) - 1
                if 0 <= index < len(choices):
                    return letters[index], "number"

    prefix_norm = normalize_text(strip_choice_prefix(text))
    for index, choice in enumerate(choices):
        choice_norm = normalize_text(choice)
        if pred_norm == choice_norm or prefix_norm == choice_norm:
            return letters[index], "choice_to_letter"

    contained = [
        (len(normalize_text(choice)), letters[index])
        for index, choice in enumerate(choices)
        if normalize_text(choice) and normalize_text(choice) in pred_norm
    ]
    if contained:
        return max(contained, key=lambda item: item[0])[1], "contains_choice_to_letter"
    return None


def select_choice_prediction(raw_prediction: str, choices: list[str], *, mcq_style: str = "letter") -> tuple[str, str]:
    if not choices:
        return raw_prediction, "no_choices"

    letter_result = _choice_letter_from_prediction(raw_prediction, choices)
    if mcq_style == "letter":
        if letter_result is not None:
            return letter_result
        return raw_prediction, "unparsed_letter"

    if letter_result is not None:
        letter, method = letter_result
        index = _letter_seq(len(choices)).index(letter)
        if 0 <= index < len(choices):
            return str(choices[index]), method.replace("letter", "choice")

    # MCQ always has a correct option and the model reliably emits a listed choice
    # verbatim (choice_to_choice); the fuzzy fallback is a rare last resort. Always
    # commit to the nearest option rather than abstain — a guess can be right, an
    # abstention is an automatic miss.
    pred_norm = normalize_text(raw_prediction)
    scored = [
        (_sequence_ratio(pred_norm, normalize_text(choice)), str(choice))
        for choice in choices
    ]
    return max(scored, key=lambda item: item[0])[1], "nearest_choice"


def _sequence_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()
