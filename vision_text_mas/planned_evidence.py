"""Typed Answerer evidence derived from one planned multi-scale pass."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from vision_text_mas.contracts import (
    MAX_FINAL_EVIDENCE_PATCHES,
    FrozenModel,
    PatchId,
    ReasonerReport,
    VerifierDecision,
)
from vision_text_mas.evidence import EvidenceMemory


class PlannedEvidence(FrozenModel):
    """Reasoner-supported evidence without a Verifier decision."""

    action: Literal["PLANNED_EVIDENCE"] = "PLANNED_EVIDENCE"
    supporting_patches: tuple[PatchId, ...] = Field(
        min_length=1,
        max_length=MAX_FINAL_EVIDENCE_PATCHES,
    )
    reason: str = Field(min_length=1)


type AnswerEvidence = VerifierDecision | PlannedEvidence


def planned_evidence_from_reports(
    memory: EvidenceMemory,
    reports: tuple[ReasonerReport, ...],
) -> PlannedEvidence:
    """Select up to six known patches in stable Reasoner-first order."""
    known = frozenset(patch.patch_id for patch in memory.patches)
    selected: list[PatchId] = []
    for report in reports:
        candidates = (
            *report.diagnostically_usable,
            *(patch_id for finding in report.consistent_findings for patch_id in finding.patch_ids),
            *(patch_id for finding in report.contradictions for patch_id in finding.patch_ids),
        )
        for patch_id in candidates:
            if patch_id in known and patch_id not in selected:
                selected.append(patch_id)
            if len(selected) == MAX_FINAL_EVIDENCE_PATCHES:
                break
        if len(selected) == MAX_FINAL_EVIDENCE_PATCHES:
            break
    if not selected:
        selected.extend(
            patch.patch_id for patch in memory.patches[:MAX_FINAL_EVIDENCE_PATCHES]
        )
    return PlannedEvidence(
        supporting_patches=tuple(selected),
        reason="one-pass planner-selected multi-scale evidence",
    )
