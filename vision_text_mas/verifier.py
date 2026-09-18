"""Grounding and sufficiency role over accumulated evidence."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import re
from typing import Annotated
from typing_extensions import assert_never

from PIL import Image
from pydantic import BeforeValidator, TypeAdapter

from vision_text_mas.contracts import (
    InsufficientDecision,
    NavigationAction,
    NewRegionDecision,
    PatchId,
    PassDecision,
    VerifierDecision,
    ZoomDetailDecision,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import Magnification
from vision_text_mas.prompts import VERIFIER_SYSTEM, verifier_prompt
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.output_repair import repair_verifier_output
from vision_text_mas.qwen_client import ParsedRoleCall, QwenJsonClient
from vision_text_mas.verifier_schema import verifier_json_schema


def scale_allocation_error(
    evidence_plan: EvidencePlan,
    memory: EvidenceMemory,
) -> str | None:
    """Return why the current round cannot satisfy the planned scale bundle."""
    current_patches = memory.patches
    actual_x5 = sum(
        patch.magnification is Magnification.X5 for patch in current_patches
    )
    actual_x20 = sum(
        patch.magnification is Magnification.X20 for patch in current_patches
    )
    scale = evidence_plan.scale_plan
    if (
        actual_x5 >= scale.overview_patch_count
        and actual_x20 >= scale.detail_patch_count
    ):
        return None
    return (
        f"planned scale allocation requires {scale.overview_patch_count} x5 and "
        f"{scale.detail_patch_count} x20 patches; actual evidence has "
        f"{actual_x5} x5 and {actual_x20} x20"
    )


def zoomable_original_x5_ids(memory: EvidenceMemory) -> tuple[PatchId, ...]:
    """Offer one full zoom round at most; later refinement must change region."""
    if any(
        evidence_round.action is NavigationAction.ZOOM_DETAIL
        for evidence_round in memory.rounds
    ):
        return ()
    return tuple(
        patch.patch_id
        for patch in memory.patches
        if patch.magnification is Magnification.X5
        and patch.source_x5_anchor == patch.patch_id
    )


class VerifierAgent:
    """Audit structured claims against the labeled evidence board."""

    def __init__(self, client: QwenJsonClient) -> None:
        self._client = client

    def verify(
        self,
        *,
        question: str,
        evidence_plan: EvidencePlan,
        evidence_json: str,
        evidence_board: Path,
        memory: EvidenceMemory,
        round_number: int,
        max_rounds: int,
    ) -> ParsedRoleCall[VerifierDecision]:
        """Return one legal, grounded controller decision."""
        try:
            with Image.open(evidence_board) as source:
                board = source.convert("RGB")
        except OSError as error:
            raise PipelineFailure(
                code=FailureCode.IMAGE_IO,
                stage="verifier",
                detail=f"failed to read {evidence_board}: {error}",
            ) from error
        known_ids = tuple(patch.patch_id for patch in memory.patches)
        known_id_set = set(known_ids)
        original_x5_ids = zoomable_original_x5_ids(memory)
        legal_actions = (
            frozenset({"PASS", "INSUFFICIENT"})
            if round_number == max_rounds
            else frozenset({"PASS", "NEW_REGION", "ZOOM_DETAIL"})
        )

        def validate_raw(json_text: str) -> str | None:
            action_match = re.search(r'"action"\s*:\s*"([A-Z_]+)"', json_text)
            if action_match is None:
                return None
            action = action_match.group(1)
            if action not in legal_actions:
                return (
                    f"action {action} is illegal in round {round_number}; choose "
                    f"exactly one of {sorted(legal_actions)}"
                )
            return None

        def validate(decision: VerifierDecision) -> str | None:
            match decision:
                case PassDecision():
                    allocation_error = scale_allocation_error(evidence_plan, memory)
                    if allocation_error is not None:
                        return allocation_error
                    cited = set(decision.supporting_patches)
                    cited.update(
                        patch_id
                        for finding in decision.verified_findings
                        for patch_id in finding.patch_ids
                    )
                case InsufficientDecision():
                    if round_number != max_rounds:
                        return "INSUFFICIENT is legal only on the final round"
                    cited = set(decision.best_available_patches)
                    cited.update(
                        patch_id
                        for finding in decision.verified_findings
                        for patch_id in finding.patch_ids
                    )
                case NewRegionDecision():
                    if round_number == max_rounds:
                        return "NEW_REGION is forbidden on the final round"
                    return None
                case ZoomDetailDecision():
                    if round_number == max_rounds:
                        return "ZOOM_DETAIL is forbidden on the final round"
                    try:
                        anchor = memory.find_patch(decision.source_x5_anchor)
                    except PipelineFailure:
                        return f"unknown x5 anchor: {decision.source_x5_anchor}"
                    if anchor.source_x5_anchor != anchor.patch_id:
                        return "source_x5_anchor must name an original x5 patch"
                    return None
                case _ as unreachable:
                    assert_never(unreachable)
            unknown = cited.difference(known_id_set)
            return f"unknown cited patch IDs: {sorted(unknown)}" if unknown else None

        return self._client.generate_json(
            role="verifier",
            images=(board,),
            image_labels=("evidence-board",),
            system_prompt=VERIFIER_SYSTEM,
            user_prompt=verifier_prompt(
                question=question,
                evidence_plan=evidence_plan,
                evidence_json=evidence_json,
                patch_ids=known_ids,
                round_number=round_number,
                max_rounds=max_rounds,
            ),
            output_adapter=TypeAdapter(
                Annotated[
                    VerifierDecision,
                    BeforeValidator(partial(repair_verifier_output, memory=memory)),
                ]
            ),
            final_tokens=768,
            json_schema=verifier_json_schema(
                known_ids=known_ids,
                original_x5_ids=original_x5_ids,
                allow_pass=scale_allocation_error(evidence_plan, memory) is None,
                final_round=round_number == max_rounds,
            ),
            raw_semantic_validator=validate_raw,
            semantic_validator=validate,
            semantic_failure_code=FailureCode.SELECTION_CONTRACT,
        )
