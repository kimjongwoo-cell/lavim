"""Evidence-plan acquisition and artifact persistence."""

from __future__ import annotations

from PIL import Image

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.navigation.navigation_contracts import EvidencePlan
from vision_text_mas.qwen_client import ParsedRoleCall
from vision_text_mas.role_protocols import EvidencePlannerRole


def acquire_plan(
    planner: EvidencePlannerRole,
    artifacts: ArtifactStore,
    *,
    question: str,
    thumbnail: Image.Image,
    round_number: int,
    missing_evidence: str | None,
) -> ParsedRoleCall[EvidencePlan]:
    """Obtain one validated plan and persist its complete model call."""
    call = planner.plan(
        question=question,
        thumbnail=thumbnail,
        missing_evidence=missing_evidence,
    )
    artifacts.write_plan(round_number, call.value, call.record)
    return call
