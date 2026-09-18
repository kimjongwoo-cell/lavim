"""Opt-in prompt2 builders for the latent Reasoner and Answerer."""

from __future__ import annotations

from enum import StrEnum
import os
from typing import Final

from typing_extensions import assert_never

from vision_text_mas.contracts import PatchMetadata

LATENT_REASONER_SYSTEM: Final = (
    "You are Reasoner. Analyze the supplied pathology patches and summarize the "
    "visual evidence relevant to the target question; never answer the question."
)
LATENT_JUDGER_SYSTEM: Final = (
    "You are Judger. Use the pathology evidence available in this turn and return "
    "one grounded final answer."
)
PROMPT_SET_ENV: Final = "VLMAS_PROMPT_SET"
PROMPT2_HANDOFF: Final = (
    "You are provided with latent information for reference and a target question "
    "to solve. The latent information might contain irrelevant content. Ignore it "
    "if it is not helpful for answering the target question."
)


def prompt2_enabled() -> bool:
    """Return whether the compact latent prompt set was explicitly requested."""
    return os.environ.get(PROMPT_SET_ENV, "").strip().casefold() == "prompt2"


class AnswererProtocol(StrEnum):
    """Terminal surface used to read the final answer from latent state."""

    TAG = "tag"
    BOXED = "boxed"
    JSON = "json"
    STRUCTURED_JSON = "structured_json"


def latent_reasoner_prompt(
    *, question: str, patches: tuple[PatchMetadata, ...]
) -> str:
    """Build the opt-in compact Reasoner prompt and response schema."""
    labeled_patches = "\n".join(
        f"- {patch.patch_id}: x{patch.magnification.value}, box={patch.box}"
        for patch in patches
    )
    return (
        f"Question stem: {question}\n"
        "The evidence plan is provided in latent KV representation format. It may "
        "contain irrelevant or uncertain information, so use only what helps.\n"
        f"Labeled patches:\n{labeled_patches}\n"
        "Assess every labeled patch for question-relevant visible morphology, "
        "including architecture, cellularity, cytology, stroma, necrosis, and tissue "
        "boundary. Note image quality or artifact only when it affects interpretation. "
        'Use concise phrases. Do not invent or force unsupported findings; use "uncertain" '
        "when a feature cannot be reliably assessed. Then integrate the evidence "
        "across patches, identifying consistent findings, contradictions, and unresolved "
        "evidence. Do not diagnose, choose an answer, or navigate. Return exactly this "
        "JSON shape, replacing placeholder text: "
        '{"patches":[{"patch_id":"R1-P1","morphology":{"architecture":"text",'
        '"cellularity":"text","cytology":"text","stroma":"text","necrosis":"text",'
        '"boundary":"text"},"quality_or_artifact":"text","confidence":0.0}],'
        '"consistent_findings":[{"finding":"text","patch_ids":["R1-P1"]}],'
        '"contradictions":[],"unresolved_evidence":["text"]}'
    )


def latent_answerer_prompt(
    *,
    question: str,
    choices: tuple[str, ...],
    decision_basis: str = "",
    cited_patch_ids: tuple[str, ...] = (),
    include_rationale: bool = False,
    protocol: AnswererProtocol = AnswererProtocol.TAG,
) -> str:
    """Request the configured terminal envelope from accumulated latent evidence."""
    choice_lines = "\n".join(
        f"CHOICE {index}: {choice}"
        for index, choice in enumerate(choices, start=1)
    )
    answer_placeholder = "<exact choice text>" if choices else "short answer phrase"
    choice_rule = (
        "If choices are provided, the answer must exactly match one choice. "
        if choices
        else ""
    )
    match protocol:
        case AnswererProtocol.BOXED:
            envelope = f"Output only \\boxed{{{answer_placeholder}}}."
        case AnswererProtocol.JSON:
            rationale = (
                ',"rationale":"brief evidence-grounded explanation"'
                if include_rationale
                else ""
            )
            envelope = f'Output only {{"answer":"{answer_placeholder}"{rationale}}}.'
        case AnswererProtocol.STRUCTURED_JSON:
            envelope = (
                "Output only one complete JSON object: "
                f'{{"answer":"{answer_placeholder}","rationale":"brief '
                'evidence-grounded explanation","confidence":0.0}}'
            )
        case AnswererProtocol.TAG:
            envelope = f"Output only <answer>{answer_placeholder}</answer>"
            if include_rationale:
                envelope += (
                    "<rationale>brief evidence-grounded explanation</rationale>"
                )
            envelope += "."
        case unreachable:
            assert_never(unreachable)
    _ = (decision_basis, cited_patch_ids)
    return (
        f"Question: {question}\n{choice_lines}\n"
        f"{PROMPT2_HANDOFF} Return the best-supported answer from the current latent "
        f"evidence. {choice_rule}"
        f"Reason step by step internally. {envelope}"
    )
