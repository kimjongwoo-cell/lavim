"""Question-and-thumbnail Evidence Planner role."""

from __future__ import annotations

import os
from typing import Literal

from PIL import Image
from pydantic import TypeAdapter

from vision_text_mas.contracts import RoleCall
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.navigation.navigation_contracts import ScalePlan
from vision_text_mas.navigation.navigation_contracts import EvidencePlan
from vision_text_mas.navigation.navigation_prompts import (
    EVIDENCE_PLANNER_SYSTEM,
    evidence_planner_prompt,
)
from vision_text_mas.qwen_client import ParsedRoleCall, QwenJsonClient


class EvidencePlannerAgent:
    """Convert a question and WSI overview into a spatially neutral plan."""

    def __init__(self, client: QwenJsonClient) -> None:
        self._client = client

    def plan(
        self,
        *,
        question: str,
        thumbnail: Image.Image,
        missing_evidence: str | None,
        patch_budget: Literal[5, 8, 25] = 5,
    ) -> ParsedRoleCall[EvidencePlan]:
        """Return a validated plan without choices, IDs, or coordinates."""
        prompt = evidence_planner_prompt(
            question=question,
            missing_evidence=missing_evidence,
            patch_budget=patch_budget,
        )
        if os.environ.get("MATCHED_VLMAS_DETERMINISTIC_PLANNER", "0") == "1":
            value = _fallback_evidence_plan(
                question=question,
                patch_budget=patch_budget,
            )
            return ParsedRoleCall(
                value=value,
                record=RoleCall(
                    role="evidence_planner",
                    prompt=prompt,
                    image_labels=("whole-slide-thumbnail",),
                    reasoning="deterministic evidence plan requested by runtime config",
                    final_outputs=("deterministic evidence plan",),
                    format_repairs=0,
                    physical_calls=1,
                    elapsed_seconds=0.0,
                ),
            )

        def fallback(_outputs: tuple[str, ...], last_error: str) -> EvidencePlan:
            """Use a conservative visual-search plan when JSON decoding fails."""
            return _fallback_evidence_plan(question=question, patch_budget=patch_budget)

        try:
            return self._client.generate_json(
                role="evidence_planner",
                images=(thumbnail,),
                image_labels=("whole-slide-thumbnail",),
                system_prompt=EVIDENCE_PLANNER_SYSTEM,
                user_prompt=prompt,
                output_adapter=TypeAdapter(EvidencePlan),
                final_tokens=768,
                semantic_validator=(
                    lambda plan: (
                        None
                        if plan.scale_plan.patch_budget == patch_budget
                        else f"scale plan must use patch_budget={patch_budget}"
                    )
                ),
                fallback_factory=fallback,
            )
        except PipelineFailure as failure:
            if failure.code not in {
                FailureCode.MODEL_EXECUTION,
                FailureCode.MODEL_OUTPUT,
            }:
                raise
            value = _fallback_evidence_plan(
                question=question,
                patch_budget=patch_budget,
            )
            return ParsedRoleCall(
                value=value,
                record=RoleCall(
                    role="evidence_planner",
                    prompt=prompt,
                    image_labels=("whole-slide-thumbnail",),
                    reasoning=f"deterministic fallback after {failure}",
                    final_outputs=(f"deterministic fallback after {failure}",),
                    format_repairs=0,
                    physical_calls=1,
                    elapsed_seconds=0.0,
                ),
            )


def _fallback_evidence_plan(
    *,
    question: str,
    patch_budget: Literal[5, 8, 25],
) -> EvidencePlan:
    """Construct an executable, spatially neutral evidence plan."""
    match patch_budget:
        case 5:
            overview_count = 2
            detail_count = 3
            detail_anchor_count = 2
        case 8:
            overview_count = 3
            detail_count = 5
            detail_anchor_count = 2
        case 25:
            overview_count = 5
            detail_count = 20
            detail_anchor_count = 5
        case _:
            msg = "patch_budget must be 5, 8, or 25"
            raise ValueError(msg)
    return EvidencePlan(
        question_focus=question[:240],
        needed_visual_information=(
            "representative tumor architecture",
            "cellular morphology relevant to the question",
            "context needed to support the final answer",
        ),
        thumbnail_observations=(
            "whole-slide overview available for non-diagnostic region search",
        ),
        search_instruction=(
            "rank tissue-rich regions likely to contain diagnostic morphology"
        ),
        success_criteria=(
            "selected patches contain usable tissue",
            "overview and detail patches cover representative morphology",
            "evidence can support a concise answer and rationale",
        ),
        detail_trigger="zoom into representative tissue for cellular detail",
        scale_plan=ScalePlan(
            overview_target="representative tissue architecture at low power",
            overview_patch_count=overview_count,
            detail_target="cellular and lesion morphology at high power",
            detail_patch_count=detail_count,
            detail_anchor_count=detail_anchor_count,
            scale_success_criteria=(
                "combine low-power context with high-power cellular evidence",
            ),
            patch_budget=patch_budget,
        ),
    )
