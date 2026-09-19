"""BCNB-style prompting: a terse multiple-choice prompt that asks the model to
answer with the option letter directly (no rules list, no answer/explanation
labels).

Rendered layout (system line shown only when include_system_prompt=True):

    You are an expert pathology visual question answering assistant. Answer based only on the provided pathology image.
    <question>
    A. <option 1>
    B. <option 2>
    C. <option 3>
    D. <option 4>
    Answer with the option's letter from the given choices directly.

The adapters keep ground truth as the choice *text*, so a predicted letter is
mapped back to its choice text for scoring via
wsi_vqa_baselines.prompting.select_choice_prediction(..., mcq_style="text").
"""

from __future__ import annotations

from typing import Final

SYSTEM_PROMPT_BCNB: Final[str] = (
    "You are an expert pathology visual question answering assistant. "
    "Answer based only on the provided pathology image."
)

_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_ANSWER_INSTRUCTION_LETTER: Final[str] = (
    "Answer with the option's letter from the given choices directly."
)
_ANSWER_INSTRUCTION_TEXT: Final[str] = (
    "Answer with the exact option text from the given choices directly, and nothing else."
)


def format_bcnb_prompt(
    question: str,
    choices: list[str],
    *,
    include_system_prompt: bool = False,
    answer_mode: str = "letter",
) -> str:
    """Build the BCNB multiple-choice prompt.

    answer_mode="letter": options shown as "A. <opt>" and the model is asked for
    the option letter. answer_mode="text": options shown as a plain bulleted list
    (no A/B/C/D labels) and the model is asked for the exact option text, which
    removes the positional-letter shortcut.

    include_system_prompt=True prepends SYSTEM_PROMPT_BCNB as the first line
    (for single-string interfaces such as HuatuoGPT-Vision). When False, the
    caller is expected to pass SYSTEM_PROMPT_BCNB as a separate system message.
    """
    lines: list[str] = [str(question).strip()]
    if answer_mode == "text":
        for choice in choices:
            lines.append(f"- {choice}")
        lines.append(_ANSWER_INSTRUCTION_TEXT)
    else:
        for index, choice in enumerate(choices):
            letter = _LETTERS[index] if index < len(_LETTERS) else str(index + 1)
            lines.append(f"{letter}. {choice}")
        lines.append(_ANSWER_INSTRUCTION_LETTER)
    body = "\n".join(lines)
    if include_system_prompt:
        # System instruction first, then a blank line, then the MCQ body.
        return f"{SYSTEM_PROMPT_BCNB}\n\n{body}"
    return body
