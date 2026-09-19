"""Typed role boundaries consumed by the deterministic controller."""

from __future__ import annotations

from pathlib import Path

from typing import Literal, Protocol

from PIL import Image

from vision_text_mas.contracts import (
    AnswerOutput,
    DecisionBasis,
    EvidenceRound,
    NewRegionDecision,
    ReasonerReport,
    RoleCall,
    VerifierDecision,
    ZoomDetailDecision,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.navigation.navigation_render import SlideLike
from vision_text_mas.navigation.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.navigation.onepass_navigation_contracts import OneShotNavigation
from vision_text_mas.planned_evidence import AnswerEvidence
from vision_text_mas.qwen_client import ParsedRoleCall


class EvidencePlannerRole(Protocol):
    def plan(
        self,
        *,
        question: str,
        thumbnail: Image.Image,
        missing_evidence: str | None,
        patch_budget: Literal[5, 8] = 5,
    ) -> ParsedRoleCall[EvidencePlan]: ...


class NavigatorRole(Protocol):
    @property
    def records(self) -> tuple[RoleCall, ...]: ...

    @property
    def traces(self) -> tuple[SelectionTrace, ...]: ...

    def initial(
        self,
        slide: SlideLike,
        *,
        thumbnail: Image.Image,
        evidence_plan: EvidencePlan,
        round_number: int,
    ) -> EvidenceRound: ...

    def new_region(
        self,
        slide: SlideLike,
        *,
        evidence_plan: EvidencePlan,
        request: NewRegionDecision,
        round_number: int,
        evidence: EvidenceMemory,
    ) -> EvidenceRound: ...

    def zoom(
        self,
        slide: SlideLike,
        *,
        evidence_plan: EvidencePlan,
        request: ZoomDetailDecision,
        round_number: int,
        evidence: EvidenceMemory,
    ) -> EvidenceRound: ...


class OnePassNavigatorRole(Protocol):
    """The Verifier-free controller can only materialize its initial bundle."""

    @property
    def records(self) -> tuple[RoleCall, ...]: ...

    @property
    def traces(self) -> tuple[SelectionTrace, ...]: ...

    def initial(
        self,
        slide: SlideLike,
        *,
        thumbnail: Image.Image,
        evidence_plan: EvidencePlan,
        round_number: int,
    ) -> EvidenceRound: ...


class OneShotNavigationRole(Protocol):
    """One-call spatial choice boundary used only by OnePass Navigator."""

    def select(
        self,
        *,
        thumbnail: Image.Image,
        root_grid: Image.Image,
        evidence_plan: EvidencePlan,
        root_candidate_ids: frozenset[int],
    ) -> ParsedRoleCall[OneShotNavigation]: ...


class ReasonerRole(Protocol):
    def observe(
        self,
        *,
        question: str,
        evidence_round: EvidenceRound,
        evidence_plan: EvidencePlan,
    ) -> ParsedRoleCall[ReasonerReport]: ...


class VerifierRole(Protocol):
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
    ) -> ParsedRoleCall[VerifierDecision]: ...


class AnswererRole(Protocol):
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
    ) -> ParsedRoleCall[AnswerOutput]: ...
