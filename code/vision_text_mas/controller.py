"""Exhaustive maximum-three-round Vision-TextMAS controller."""

from __future__ import annotations

from typing_extensions import assert_never

from pydantic import TypeAdapter

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.contracts import (
    CaseInput,
    DecisionBasis,
    EvidenceRound,
    InsufficientDecision,
    NewRegionDecision,
    PassDecision,
    ReasonerReport,
    RoleCall,
    RunResult,
    StateTransition,
    VerifierDecision,
    ZoomDetailDecision,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.controller_state import illegal, require_navigation_round, transition
from vision_text_mas.controller_planning import acquire_plan
from vision_text_mas.controller_answer import finalize_answer
from vision_text_mas.navigation.navigation_render import SlideLike, slide_thumbnail
from vision_text_mas.role_protocols import (
    AnswererRole,
    EvidencePlannerRole,
    NavigatorRole,
    ReasonerRole,
    VerifierRole,
)


class Controller:
    """Own navigation transitions while keeping technical failures terminal."""

    def __init__(
        self,
        *,
        planner: EvidencePlannerRole,
        navigator: NavigatorRole,
        reasoner: ReasonerRole,
        verifier: VerifierRole,
        answerer: AnswererRole,
        artifacts: ArtifactStore,
        max_rounds: int,
    ) -> None:
        if max_rounds < 1 or max_rounds > 3:
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="controller",
                detail="max_rounds must be between one and three",
            )
        self._planner = planner
        self._navigator = navigator
        self._reasoner = reasoner
        self._verifier = verifier
        self._answerer = answerer
        self._artifacts = artifacts
        self._max_rounds = max_rounds

    def run_case(self, case: CaseInput, *, slide: SlideLike) -> RunResult:
        """Run one case to a grounded answer or explicit technical failure."""
        memory = EvidenceMemory.empty()
        reports: tuple[ReasonerReport, ...] = ()
        calls: tuple[RoleCall, ...] = ()
        transitions: tuple[StateTransition, ...] = ()
        navigator_call_count = 0
        trace_count = 0
        try:
            thumbnail = slide_thumbnail(slide)
            plan_call = acquire_plan(
                self._planner,
                self._artifacts,
                question=case.item.question,
                thumbnail=thumbnail,
                round_number=1,
                missing_evidence=None,
            )
            active_plan = plan_call.value
            calls = (*calls, plan_call.record)
            current = self._navigator.initial(
                slide,
                thumbnail=thumbnail,
                evidence_plan=active_plan,
                round_number=1,
            )
            while True:
                navigator_calls = self._navigator.records[navigator_call_count:]
                navigator_call_count += len(navigator_calls)
                self._artifacts.write_navigator_calls(
                    current.round_number,
                    navigator_calls,
                )
                calls = (*calls, *navigator_calls)
                traces = self._navigator.traces[trace_count:]
                trace_count += len(traces)
                self._artifacts.write_selection_traces(current.round_number, traces)
                memory = memory.append(current)
                self._artifacts.write_round(current)
                board = self._artifacts.render_board(memory)
                reasoner_call = self._reasoner.observe(
                    question=case.item.question,
                    evidence_round=current,
                    evidence_plan=active_plan,
                )
                reports = (*reports, reasoner_call.value)
                calls = (*calls, reasoner_call.record)
                self._artifacts.write_reasoner(
                    current.round_number,
                    reasoner_call.value,
                    reasoner_call.record,
                )
                evidence_json = TypeAdapter(tuple[ReasonerReport, ...]).dump_json(
                    reports
                ).decode()
                verifier_call = self._verifier.verify(
                    question=case.item.question,
                    evidence_plan=active_plan,
                    evidence_json=evidence_json,
                    evidence_board=board,
                    memory=memory,
                    round_number=current.round_number,
                    max_rounds=self._max_rounds,
                )
                decision = verifier_call.value
                calls = (*calls, verifier_call.record)
                self._artifacts.write_verifier(
                    current.round_number,
                    decision,
                    verifier_call.record,
                )
                transitions = (
                    *transitions,
                    transition(current.round_number, decision),
                )
                match decision:
                    case PassDecision():
                        return finalize_answer(
                            self._answerer,
                            self._artifacts,
                            case=case,
                            thumbnail=thumbnail,
                            memory=memory,
                            board=board,
                            reports=reports,
                            decision=decision,
                            basis=DecisionBasis.VERIFIER_PASS,
                            calls=calls,
                            transitions=transitions,
                        )
                    case InsufficientDecision():
                        if current.round_number != self._max_rounds:
                            illegal("INSUFFICIENT is legal only on the final round")
                        return finalize_answer(
                            self._answerer,
                            self._artifacts,
                            case=case,
                            thumbnail=thumbnail,
                            memory=memory,
                            board=board,
                            reports=reports,
                            decision=decision,
                            basis=DecisionBasis.FORCED_ANSWER,
                            calls=calls,
                            transitions=transitions,
                        )
                    case NewRegionDecision():
                        require_navigation_round(current.round_number, self._max_rounds)
                        plan_call = acquire_plan(
                            self._planner,
                            self._artifacts,
                            question=case.item.question,
                            thumbnail=thumbnail,
                            round_number=current.round_number + 1,
                            missing_evidence=(
                                f"{decision.missing_evidence}; {decision.retrieval_query}"
                            ),
                        )
                        active_plan = plan_call.value
                        calls = (*calls, plan_call.record)
                        current = self._navigator.new_region(
                            slide,
                            evidence_plan=active_plan,
                            request=decision,
                            round_number=current.round_number + 1,
                            evidence=memory,
                        )
                    case ZoomDetailDecision():
                        require_navigation_round(current.round_number, self._max_rounds)
                        plan_call = acquire_plan(
                            self._planner,
                            self._artifacts,
                            question=case.item.question,
                            thumbnail=thumbnail,
                            round_number=current.round_number + 1,
                            missing_evidence=decision.missing_evidence,
                        )
                        active_plan = plan_call.value
                        calls = (*calls, plan_call.record)
                        current = self._navigator.zoom(
                            slide,
                            evidence_plan=active_plan,
                            request=decision,
                            round_number=current.round_number + 1,
                            evidence=memory,
                        )
                    case _ as unreachable:
                        assert_never(unreachable)
        except PipelineFailure as failure:
            self._artifacts.write_failure(failure, case)
            raise
