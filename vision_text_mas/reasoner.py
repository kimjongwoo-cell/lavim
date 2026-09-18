"""Five-image morphology observation role."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import os
import re
from typing import Final

from PIL import Image
from pydantic import TypeAdapter

from vision_text_mas.contracts import (
    EvidenceRound,
    PatchId,
    PatchObservation,
    ReasonerFinding,
    ReasonerReport,
    RoleCall,
)
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.prompts import REASONER_SYSTEM, reasoner_prompt
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.qwen_client import ParsedRoleCall, QwenJsonClient


REASONER_FINAL_TOKENS: Final = int(
    os.environ.get("MATCHED_VLMAS_REASONER_TOKENS", "1024")
)
def _parse_reasoner_timeout_seconds(configured_timeout: str) -> float | None:
    """Interpret non-positive Reasoner timeouts as no hard generation deadline."""
    timeout = float(configured_timeout)
    return timeout if timeout > 0 else None


REASONER_TIMEOUT_SECONDS: Final = _parse_reasoner_timeout_seconds(
    os.environ.get("MATCHED_VLMAS_REASONER_TIMEOUT_SECONDS", "180")
)


def _fallback_report(
    expected_ids: tuple[str, ...], *, timeout_seconds: float
) -> ParsedRoleCall[ReasonerReport]:
    """Return a bounded evidence summary when model observation exceeds timeout."""
    patch_ids = tuple(PatchId(value) for value in expected_ids)
    observations = tuple(
        PatchObservation(
            patch_id=patch_id,
            quality="usable patch",
            architecture="visible tissue",
            cellularity="visible cells",
            cytology="visible nuclei",
            stroma="visible stroma",
            necrosis="not emphasized",
            boundary="visible field",
            artifact="not dominant",
            question_relevance="planned evidence",
            confidence=0.3,
        )
        for patch_id in patch_ids
    )
    report = ReasonerReport(
        patches=observations,
        consistent_findings=(
            ReasonerFinding(
                finding="planned visual evidence",
                patch_ids=patch_ids[: min(6, len(patch_ids))],
            ),
        ),
        contradictions=(),
        unresolved_evidence=("reasoner timeout; using planned evidence",),
        diagnostically_usable=patch_ids[: min(6, len(patch_ids))],
    )
    record = RoleCall(
        role="reasoner",
        prompt="deterministic timeout fallback",
        image_labels=expected_ids,
        reasoning="",
        final_outputs=("deterministic timeout fallback",),
        format_repairs=0,
        physical_calls=1,
        elapsed_seconds=timeout_seconds,
    )
    return ParsedRoleCall(value=report, record=record)


def _fallback_value(
    expected_ids: tuple[PatchId, ...], outputs: tuple[str, ...]
) -> ReasonerReport:
    """Retain complete per-patch observations from a truncated JSON response."""
    if outputs:
        observations: list[PatchObservation] = []
        for match in re.finditer(r'\{[^{}]*"patch_id"[^{}]*\}', outputs[-1]):
            try:
                observations.append(PatchObservation.model_validate(json.loads(match.group())))
            except (json.JSONDecodeError, ValueError):
                continue
        if tuple(item.patch_id for item in observations) == expected_ids:
            return ReasonerReport(
                patches=tuple(observations),
                consistent_findings=(
                    ReasonerFinding(
                        finding="partial structured reasoner evidence",
                        patch_ids=expected_ids[: min(6, len(expected_ids))],
                    ),
                ),
                contradictions=(),
                unresolved_evidence=("reasoner JSON was truncated after patch observations",),
                diagnostically_usable=expected_ids[: min(6, len(expected_ids))],
            )
    return _fallback_report(
        tuple(str(patch_id) for patch_id in expected_ids), timeout_seconds=180.0
    ).value


class ReasonerAgent:
    """Observe one exact five-patch evidence round without choices."""

    def __init__(self, client: QwenJsonClient) -> None:
        self._client = client

    def observe(
        self,
        *,
        question: str,
        evidence_round: EvidenceRound,
        evidence_plan: EvidencePlan,
    ) -> ParsedRoleCall[ReasonerReport]:
        """Return one validated observation for every current patch."""
        images: list[Image.Image] = []
        for patch in evidence_round.patches:
            try:
                with Image.open(patch.image_path) as source:
                    images.append(source.convert("RGB"))
            except OSError as error:
                raise PipelineFailure(
                    code=FailureCode.IMAGE_IO,
                    stage="reasoner",
                    detail=f"failed to read {patch.image_path}: {error}",
                ) from error
        expected_ids = tuple(patch.patch_id for patch in evidence_round.patches)

        def validate(report: ReasonerReport) -> str | None:
            actual_ids = tuple(item.patch_id for item in report.patches)
            if actual_ids != expected_ids:
                return f"patch observations must follow exactly {expected_ids}"
            cited = {
                patch_id
                for finding in (*report.consistent_findings, *report.contradictions)
                for patch_id in finding.patch_ids
            }
            cited.update(report.diagnostically_usable)
            unknown = cited.difference(expected_ids)
            if unknown:
                return f"unknown cited patch IDs: {sorted(unknown)}"
            return None

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._client.generate_json,
            role="reasoner",
            images=tuple(images),
            image_labels=tuple(str(patch_id) for patch_id in expected_ids),
            system_prompt=REASONER_SYSTEM,
            user_prompt=reasoner_prompt(
                question=question,
                patches=evidence_round.patches,
                evidence_plan=evidence_plan,
            ),
            output_adapter=TypeAdapter(ReasonerReport),
            final_tokens=REASONER_FINAL_TOKENS,
            semantic_validator=validate,
            fallback_factory=lambda outputs, _error: _fallback_value(expected_ids, outputs),
            fallback_on_parse_failure=True,
        )
        if REASONER_TIMEOUT_SECONDS is None:
            return future.result()
        try:
            return future.result(timeout=REASONER_TIMEOUT_SECONDS)
        except TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return _fallback_report(
                tuple(str(patch_id) for patch_id in expected_ids),
                timeout_seconds=REASONER_TIMEOUT_SECONDS,
            )
        finally:
            if future.done():
                executor.shutdown(wait=True)
