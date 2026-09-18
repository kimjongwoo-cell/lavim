"""One-round four-agent controller with planner-fixed multi-scale evidence."""

from __future__ import annotations

import os
from typing import Final, Literal, final

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.controller_answer import finalize_answer
from vision_text_mas.contracts import (
    CaseInput,
    DecisionBasis,
    ReasonerReport,
    RoleCall,
    RunResult,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import PipelineFailure
from vision_text_mas.navigation_render import SlideLike, slide_thumbnail
from vision_text_mas.planned_evidence import planned_evidence_from_reports
from vision_text_mas.role_protocols import (
    AnswererRole,
    EvidencePlannerRole,
    OnePassNavigatorRole,
    ReasonerRole,
)


def _onepass_patch_budget() -> Literal[5, 8, 12]:
    value = os.environ.get("MATCHED_VLMAS_PATCH_BUDGET", "8")
    budgets: dict[str, Literal[5, 8, 12]] = {"5": 5, "8": 8, "12": 12}
    budget = budgets.get(value)
    if budget is None:
        msg = "MATCHED_VLMAS_PATCH_BUDGET must be 5, 8, or 12"
        raise ValueError(msg)
    return budget


ONEPASS_PATCH_BUDGET: Final = _onepass_patch_budget()


@final
class OnePassController:
    """Run Planner, Navigator, Reasoner, and Answerer exactly once."""

    def __init__(
        self,
        *,
        planner: EvidencePlannerRole,
        navigator: OnePassNavigatorRole,
        reasoner: ReasonerRole,
        answerer: AnswererRole,
        artifacts: ArtifactStore,
    ) -> None:
        self._planner = planner
        self._navigator = navigator
        self._reasoner = reasoner
        self._answerer = answerer
        self._artifacts = artifacts

    def run_case(self, case: CaseInput, *, slide: SlideLike) -> RunResult:
        """Materialize one Planner-allocated multi-scale patch bundle and answer once."""
        memory = EvidenceMemory.empty()
        calls: tuple[RoleCall, ...] = ()
        try:
            thumbnail = slide_thumbnail(slide)
            plan_call = self._planner.plan(
                question=case.item.question,
                thumbnail=thumbnail,
                missing_evidence=None,
                patch_budget=ONEPASS_PATCH_BUDGET,
            )
            calls = (*calls, plan_call.record)
            _ = self._artifacts.write_plan(1, plan_call.value, plan_call.record)
            evidence_round = self._navigator.initial(
                slide,
                thumbnail=thumbnail,
                evidence_plan=plan_call.value,
                round_number=1,
            )
            navigator_calls = self._navigator.records
            calls = (*calls, *navigator_calls)
            _ = self._artifacts.write_navigator_calls(1, navigator_calls)
            _ = self._artifacts.write_selection_traces(1, self._navigator.traces)
            memory = memory.append(evidence_round)
            _ = self._artifacts.write_round(evidence_round)
            evidence_board = self._artifacts.render_board(memory)
            reasoner_call = self._reasoner.observe(
                question=case.item.question,
                evidence_round=evidence_round,
                evidence_plan=plan_call.value,
            )
            reports: tuple[ReasonerReport, ...] = (reasoner_call.value,)
            calls = (*calls, reasoner_call.record)
            _ = self._artifacts.write_reasoner(
                1,
                reasoner_call.value,
                reasoner_call.record,
            )
            evidence = planned_evidence_from_reports(memory, reports)
            return finalize_answer(
                self._answerer,
                self._artifacts,
                case=case,
                thumbnail=thumbnail,
                memory=memory,
                board=evidence_board,
                reports=reports,
                decision=evidence,
                basis=DecisionBasis.PLANNED_EVIDENCE,
                calls=calls,
                transitions=(),
            )
        except PipelineFailure as failure:
            _ = self._artifacts.write_failure(failure, case)
            raise
