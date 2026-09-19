"""Four-agent OnePass controller whose intermediate messages remain latent."""

from __future__ import annotations

from typing import Literal, final

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.controller_answer import finalize_answer
from vision_text_mas.contracts import CaseInput, DecisionBasis, RoleCall, RunResult
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import PipelineFailure
from vision_text_mas.latent_onepass_agents import (
    LatentEvidencePlannerAgent,
    LatentReasonerAgent,
)
from vision_text_mas.navigation.navigation_render import SlideLike, slide_thumbnail
from vision_text_mas.navigation.onepass_navigator import OnePassNavigator
from vision_text_mas.planned_evidence import planned_evidence_from_reports
from vision_text_mas.role_protocols import AnswererRole


@final
class LatentOnePassController:
    """Execute Planner -> Navigator -> Reasoner -> Answerer once per slide."""

    def __init__(
        self,
        *,
        planner: LatentEvidencePlannerAgent,
        navigator: OnePassNavigator,
        reasoner: LatentReasonerAgent,
        answerer: AnswererRole,
        artifacts: ArtifactStore,
        patch_budget: Literal[8, 25] = 8,
    ) -> None:
        self._planner = planner
        self._navigator = navigator
        self._reasoner = reasoner
        self._answerer = answerer
        self._artifacts = artifacts
        self._patch_budget: Literal[8, 25] = patch_budget

    def run_case(self, case: CaseInput, *, slide: SlideLike) -> RunResult:
        """Materialize one latent-planned patch bundle and decode one answer."""
        memory = EvidenceMemory.empty()
        calls: tuple[RoleCall, ...] = ()
        try:
            thumbnail = slide_thumbnail(slide)
            plan_call = self._planner.plan(
                question=case.item.question,
                thumbnail=thumbnail,
                missing_evidence=None,
                patch_budget=self._patch_budget,
            )
            calls = (*calls, plan_call.record)
            self._artifacts.write_plan(1, plan_call.value, plan_call.record)
            evidence_round = self._navigator.initial(
                slide,
                thumbnail=thumbnail,
                evidence_plan=plan_call.value,
                round_number=1,
            )
            navigator_calls = self._navigator.records
            calls = (*calls, *navigator_calls)
            self._artifacts.write_navigator_calls(1, navigator_calls)
            self._artifacts.write_selection_traces(1, self._navigator.traces)
            memory = memory.append(evidence_round)
            self._artifacts.write_round(evidence_round)
            board = self._artifacts.render_board(memory)
            reasoner_call = self._reasoner.observe(
                question=case.item.question,
                evidence_round=evidence_round,
                evidence_plan=plan_call.value,
            )
            calls = (*calls, reasoner_call.record)
            self._artifacts.write_reasoner(
                1,
                reasoner_call.value,
                reasoner_call.record,
            )
            decision = planned_evidence_from_reports(memory, (reasoner_call.value,))
            return finalize_answer(
                self._answerer,
                self._artifacts,
                case=case,
                thumbnail=thumbnail,
                memory=memory,
                board=board,
                reports=(reasoner_call.value,),
                decision=decision,
                basis=DecisionBasis.PLANNED_EVIDENCE,
                calls=calls,
                transitions=(),
            )
        except PipelineFailure as failure:
            self._artifacts.write_failure(failure, case)
            raise
