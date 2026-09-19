"""Choice-isolated prompt builders for the four runtime agents."""

from __future__ import annotations

from collections.abc import Iterable
import json

from vision_text_mas.contracts import PatchId, PatchMetadata
from vision_text_mas.navigation.navigation_contracts import EvidencePlan


NAVIGATOR_SYSTEM = "You are Navigator. Retrieve visual evidence; never diagnose."
REASONER_SYSTEM = "You are Reasoner. Report visible morphology; never answer."
VERIFIER_SYSTEM = "You are Verifier. Audit grounding and evidence sufficiency."
ANSWERER_SYSTEM = "You are Answerer. Select the answer from grounded evidence."


def canonical_open_answer_options(question: str) -> tuple[str, ...]:
    """Return the dataset's declared labels for recurring open-answer fields."""
    normalized = question.casefold()
    if "progesterone_receptor" in normalized or "progesterone receptor" in normalized:
        return ("Positive", "Negative")
    if "vital_status" in normalized or "vital status" in normalized:
        return ("Alive", "Dead")
    if "histological_type" in normalized or "histological type" in normalized:
        return (
            "Infiltrating Ductal Carcinoma",
            "Infiltrating Lobular Carcinoma",
            "Medullary Carcinoma",
            "Other, specify",
        )
    if "her2" in normalized:
        return ("0", "1+", "2+", "3+")
    return ()


def _ranked_ids(values: Iterable[int]) -> str:
    return ", ".join(str(value) for value in sorted(values))


def navigator_candidate_prompt(
    *,
    question: str,
    candidate_ids: frozenset[int],
    selection_count: int,
    retrieval_target: str,
) -> str:
    """Request one overcomplete ranking from a rendered discrete grid."""
    return (
        f"Question stem: {question}\n"
        f"Visual retrieval target: {retrieval_target}\n"
        f"Available candidate IDs: {_ranked_ids(candidate_ids)}\n"
        f"Rank exactly {selection_count} unique available candidate ID(s) in "
        "order. Do not diagnose. Return JSON with only the cells list."
    )


def reasoner_prompt(
    *,
    question: str,
    patches: tuple[PatchMetadata, ...],
    evidence_plan: EvidencePlan,
) -> str:
    """Request one scale-aware observation for every labeled patch."""
    patch_lines = "\n".join(
        f"- {patch.patch_id}: x{patch.magnification.value}, box={patch.box}"
        for patch in patches
    )
    observation_template = [
        {
            "patch_id": str(patch.patch_id),
            "quality": "text",
            "architecture": "text",
            "cellularity": "text",
            "cytology": "text",
            "stroma": "text",
            "necrosis": "text",
            "boundary": "text",
            "artifact": "text",
            "question_relevance": "text",
            "confidence": 0.0,
        }
        for patch in patches
    ]
    first_patch_id = str(patches[0].patch_id)
    output_template = json.dumps(
        {
            "patches": observation_template,
            "consistent_findings": [
                {"finding": "text", "patch_ids": [first_patch_id]}
            ],
            "contradictions": [],
            "unresolved_evidence": ["text"],
            "diagnostically_usable": [first_patch_id],
        },
        ensure_ascii=False,
    )
    return (
        f"Question stem: {question}\nEvidence plan: {evidence_plan.model_dump_json()}\n"
        f"Labeled patches:\n{patch_lines}\n"
        "Describe every patch using patch_id, quality, architecture, cellularity, "
        "cytology, stroma, necrosis, boundary, artifact, question_relevance, and "
        "confidence. Use terse 2-4 word phrases per text field. "
        "State your best morphological read for every field even when "
        "uncertain; do not report a feature as unresolvable. Then provide "
        "consistent_findings, contradictions, unresolved_evidence, and "
        "diagnostically_usable. Use one short synthesis finding. "
        "Confidence must be a JSON number from 0.0 to 1.0, "
        "never a label such as low, medium, or high. The four synthesis fields "
        "must be JSON arrays; finding objects must cite patch_ids, and "
        "diagnostically_usable may contain only patch IDs. Use this exact JSON "
        "shape. Reasoner finding citations and diagnostically_usable may include "
        f"up to all {len(patches)} input patch IDs; the final Answerer evidence is bounded "
        "separately. "
        f"Replace placeholder text: {output_template}\n"
        "Do not diagnose, choose an answer, or navigate."
    )


def verifier_prompt(
    *,
    question: str,
    evidence_plan: EvidencePlan,
    evidence_json: str,
    patch_ids: tuple[PatchId, ...],
    round_number: int,
    max_rounds: int,
) -> str:
    """Request a grounded controller action without answer choices."""
    if not patch_ids:
        msg = "Verifier requires at least one known patch ID"
        raise ValueError(msg)
    legal_actions = (
        "PASS or INSUFFICIENT"
        if round_number == max_rounds
        else "PASS, NEW_REGION, or ZOOM_DETAIL"
    )
    illegal_action_rule = (
        "NEW_REGION and ZOOM_DETAIL are invalid on this final round."
        if round_number == max_rounds
        else "INSUFFICIENT is invalid before the final round."
    )
    first_patch_id = str(patch_ids[0])
    finding = {"finding": "text", "patch_ids": [first_patch_id]}
    examples = [
        json.dumps(
            {
                "action": "PASS",
                "verified_findings": [finding],
                "supporting_patches": [first_patch_id],
                "reason": "text",
            }
        )
    ]
    if round_number == max_rounds:
        examples.append(
            json.dumps(
                {
                    "action": "INSUFFICIENT",
                    "best_available_patches": [first_patch_id],
                    "verified_findings": [finding],
                    "uncertainty": "text",
                }
            )
        )
    else:
        examples.extend(
            (
                json.dumps(
                    {
                        "action": "NEW_REGION",
                        "missing_evidence": "text",
                        "retrieval_query": "text",
                        "reason": "text",
                    }
                ),
                json.dumps(
                    {
                        "action": "ZOOM_DETAIL",
                        "source_x5_anchor": first_patch_id,
                        "missing_evidence": "text",
                        "target_magnification": 20,
                        "reason": "text",
                    }
                ),
            )
        )
    action_shapes = "\n".join(examples)
    scale = evidence_plan.scale_plan
    return (
        f"Question stem: {question}\nEvidence plan: {evidence_plan.model_dump_json()}\n"
        f"Round: {round_number}/{max_rounds}\n"
        f"Validated structured evidence: {evidence_json}\n"
        f"Legal actions now: {legal_actions}. {illegal_action_rule} Audit claims "
        "against the labeled "
        "evidence-board image. Cite only visible patch IDs and use at most six patch IDs "
        "in every citation list. PASS requires diagnostic "
        "support and every evidence-plan success criterion must be visibly met. "
        f"PASS is forbidden unless the current evidence contains exactly "
        f"{scale.overview_patch_count} x5 and {scale.detail_patch_count} x20 "
        "patches and satisfies every scale_success_criteria item; "
        "NEW_REGION is for coverage; ZOOM_DETAIL is for scale. On the final "
        "round return INSUFFICIENT when evidence remains answer-changingly incomplete. "
        "Return exactly one JSON object using one of these complete legal shapes; "
        "replace placeholder text and never omit a shown field:\n"
        f"{action_shapes}"
    )


def answerer_prompt(
    *,
    question: str,
    choices: tuple[str, ...],
    evidence_json: str,
    decision_basis: str,
    cited_patch_ids: tuple[PatchId, ...],
) -> str:
    """Expose choices only at the final grounded answer boundary."""
    if not cited_patch_ids:
        msg = "Answerer requires at least one cited patch ID"
        raise ValueError(msg)
    choice_lines = "\n".join(
        f"CHOICE {index}: {choice}" for index, choice in enumerate(choices, start=1)
    )
    open_options = canonical_open_answer_options(question)
    if choices:
        answer_rule = "The answer must exactly equal one CHOICE text."
    elif open_options:
        answer_rule = f"The answer must exactly equal one of: {open_options}."
    else:
        answer_rule = (
            "Return only the shortest literal answer value or category, without "
            "explanation, qualifiers, or a repeated question phrase."
        )
    output_template = json.dumps(
        {
            "decision_basis": decision_basis,
            "answer": "<exact choice text>" if choices else "short answer phrase",
            "evidence_patches": [str(cited_patch_ids[0])],
            "rationale": "text grounded in the cited patches",
            "confidence": 0.0,
        },
        ensure_ascii=False,
    )
    return (
        f"Question: {question}\n{choice_lines}\nDecision basis: {decision_basis}\n"
        f"Validated evidence: {evidence_json}\n{answer_rule} Re-inspect the supplied "
        "whole-slide thumbnail and raw patches, cite only unique supplied patch IDs, "
        "and do not request more "
        "evidence. Confidence must be a JSON number from 0.0 to 1.0. Return one "
        "complete JSON object with this exact shape and replace placeholder text: "
        f"{output_template}"
    )
