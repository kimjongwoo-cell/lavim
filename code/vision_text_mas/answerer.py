"""Choice-visible grounded final-answer role."""

from __future__ import annotations

from functools import partial
from typing import Annotated
from typing_extensions import assert_never

from pydantic import BeforeValidator, TypeAdapter

from vision_text_mas.answerer_schema import answerer_json_schema
from vision_text_mas.contracts import (
    AnswerOutput,
    DecisionBasis,
    InsufficientDecision,
    NewRegionDecision,
    PassDecision,
    ReasonerReport,
    ZoomDetailDecision,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.prompts import (
    ANSWERER_SYSTEM,
    answerer_prompt,
    canonical_open_answer_options,
)
from vision_text_mas.qwen_client import ParsedRoleCall, QwenJsonClient
from vision_text_mas.output_repair import repair_answer_output
from vision_text_mas.planned_evidence import AnswerEvidence, PlannedEvidence


class AnswererAgent:
    """Map verified visual evidence to one exact final answer."""

    def __init__(self, client: QwenJsonClient) -> None:
        self._client = client

    def answer(
        self,
        *,
        question: str,
        choices: tuple[str, ...],
        thumbnail: Image.Image,
        memory: EvidenceMemory,
        evidence_board: Path,
        reasoner_reports: tuple[ReasonerReport, ...],
        verifier_decision: AnswerEvidence,
        decision_basis: DecisionBasis,
    ) -> ParsedRoleCall[AnswerOutput]:
        """Answer solely from the Reasoner's textual report and evidence decision."""
        match verifier_decision:
            case PassDecision():
                cited_ids = verifier_decision.supporting_patches
            case InsufficientDecision():
                cited_ids = verifier_decision.best_available_patches
            case PlannedEvidence():
                cited_ids = verifier_decision.supporting_patches
            case NewRegionDecision() | ZoomDetailDecision():
                raise PipelineFailure(
                    code=FailureCode.ANSWER_CONTRACT,
                    stage="answerer",
                    detail="navigation decisions cannot be answered",
                )
            case _ as unreachable:
                assert_never(unreachable)
        _ = (thumbnail, memory, evidence_board)
        evidence_json = TypeAdapter(
            tuple[tuple[ReasonerReport, ...], AnswerEvidence]
        ).dump_json((reasoner_reports, verifier_decision)).decode()
        open_options = canonical_open_answer_options(question) if not choices else ()

        def validate(output: AnswerOutput) -> str | None:
            if output.decision_basis is not decision_basis:
                return f"decision_basis must be {decision_basis.value}"
            if choices and output.answer not in choices:
                return (
                    "answer must exactly match one supplied choice; "
                    f"received {output.answer!r}"
                )
            if open_options and output.answer not in open_options:
                return f"answer must exactly match one canonical label: {open_options}"
            unknown = set(output.evidence_patches).difference(cited_ids)
            if unknown:
                return f"answer cites unsupported patch IDs: {sorted(unknown)}"
            return None

        return self._client.generate_json(
            role="answerer",
            images=(),
            image_labels=(),
            system_prompt=ANSWERER_SYSTEM,
            user_prompt=answerer_prompt(
                question=question,
                choices=choices,
                evidence_json=evidence_json,
                decision_basis=decision_basis.value,
                cited_patch_ids=cited_ids,
            ),
            output_adapter=TypeAdapter(
                Annotated[
                    AnswerOutput,
                    BeforeValidator(
                        partial(
                            repair_answer_output,
                            choices=choices,
                            cited_ids=cited_ids,
                        )
                    ),
                ]
            ),
            final_tokens=1_024,
            json_schema=answerer_json_schema(
                decision_basis=decision_basis,
                cited_ids=cited_ids,
                answer_options=choices or open_options,
            ),
            semantic_validator=validate,
            semantic_failure_code=FailureCode.ANSWER_CONTRACT,
        )
