"""Choice-free prompts for evidence planning and spatial ranking."""

from __future__ import annotations

import json
from typing import Literal

from typing_extensions import assert_never

from vision_text_mas.navigation.navigation_contracts import EvidencePlan


EVIDENCE_PLANNER_SYSTEM = (
    "You are Evidence Planner. Decide what visual evidence the question requires "
    "from the supplied whole-slide thumbnail; never select locations or answer."
)
PATCH_NAVIGATOR_SYSTEM = (
    "You are Patch Navigator. Rank visible candidate locations against a supplied "
    "evidence plan; never reinterpret the question or diagnose."
)
ONE_SHOT_NAVIGATOR_SYSTEM = (
    "You are Navigator. Select all planned multi-scale tissue regions from the "
    "supplied whole-slide views in one response; never answer or diagnose."
)


def evidence_planner_prompt(
    *,
    question: str,
    missing_evidence: str | None,
    patch_budget: Literal[5, 8, 25] = 5,
) -> str:
    """Request a spatially neutral visual-evidence plan."""
    match patch_budget:
        case 5:
            overview_patch_count = 2
            detail_patch_count = 3
            detail_anchor_count = 2
        case 8:
            overview_patch_count = 3
            detail_patch_count = 5
            detail_anchor_count = 2
        case 25:
            overview_patch_count = 5
            detail_patch_count = 20
            detail_anchor_count = 5
        case unreachable:
            assert_never(unreachable)
    template = json.dumps(
        {
            "question_focus": "text",
            "needed_visual_information": ["observable visual requirement"],
            "thumbnail_observations": ["non-diagnostic overview observation"],
            "search_instruction": "text for ranking visible regions",
            "success_criteria": ["observable condition for useful evidence"],
            "detail_trigger": "observable condition requiring finer scale",
            "scale_plan": {
                "overview_target": "architecture/context required at x5",
                "overview_patch_count": overview_patch_count,
                "detail_target": "cellular evidence required at x20",
                "detail_patch_count": detail_patch_count,
                "detail_anchor_count": detail_anchor_count,
                "scale_success_criteria": [
                    "observable condition requiring both x5 and x20"
                ],
                "patch_budget": patch_budget,
            },
        }
    )
    follow_up = (
        f"\nVerifier missing evidence: {missing_evidence}"
        if missing_evidence is not None
        else ""
    )
    return (
        f"Question stem: {question}{follow_up}\n"
        "Use only the question and supplied thumbnail. State what visual evidence "
        "is needed and how another agent should recognize useful regions. "
        "Do not output grid IDs, coordinates, candidate "
        "ranks, or answer choices. Do not "
        "propose a diagnosis or answer. Return exactly this JSON shape, replacing "
        f"placeholder text: {template}"
    )


def patch_navigator_prompt(
    *,
    evidence_plan: EvidencePlan,
    candidate_ids: frozenset[int],
    selection_count: int,
) -> str:
    """Request a plan-conditioned ranking without the raw question."""
    candidate_text = ", ".join(str(value) for value in sorted(candidate_ids))
    template = json.dumps(
        {
            "ranked": [
                {"id": 1, "match_reason": "visible match to the evidence plan"}
            ]
        }
    )
    return (
        f"Validated evidence plan: {evidence_plan.model_dump_json()}\n"
        f"Available candidate IDs: {candidate_text}\n"
        f"Rank exactly {selection_count} unique available IDs by visible match to "
        "the plan. Tissue amount alone is not relevance. Do not reinterpret the "
        "question or diagnose. Return the ranking in this JSON shape, with a "
        f"specific visible match reason per ID: {template}"
    )


def one_shot_navigator_prompt(
    *,
    evidence_plan: EvidencePlan,
    root_candidate_ids: frozenset[int],
) -> str:
    """Request all thumbnail-level x5 and x20 targets in one call."""
    scale = evidence_plan.scale_plan
    root_ids = ", ".join(str(value) for value in sorted(root_candidate_ids))
    return (
        f"Validated evidence plan: {evidence_plan.model_dump_json()}\n"
        "Image 1 is the whole-slide thumbnail. Image 2 is the same slide with "
        f"root-grid IDs. Available root IDs: {root_ids}.\n"
        f"Choose exactly {scale.overview_patch_count} root IDs for x5 overview "
        f"evidence. Choose exactly {scale.detail_patch_count} x20 cell IDs. "
        "An x20 cell ID is row-major in a shared 4x4 grid: 1 is top-left and "
        "16 is bottom-right; the deterministic cropper distributes these targets "
        "across the planned x5 detail anchors. Select every requested region now. "
        "Do not request another view, output coordinates, answer the question, or "
        "diagnose. Output ONLY JSON with these exact keys, replacing the "
        'placeholders with actual integer IDs: {"root_ids": [<id>, ...], '
        '"detail_cell_ids": [<cell 1-16>, ...]}'
    )
