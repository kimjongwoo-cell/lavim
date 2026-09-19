"""Validated external contracts shared by the four agents."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Literal, NewType

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from vision_text_mas.geometry import Box, Magnification


PatchId = NewType("PatchId", str)
MAX_FINAL_EVIDENCE_PATCHES: Final = 6
MAX_REASONER_EVIDENCE_PATCHES: Final = 25
RootCellId = NewType("RootCellId", int)
CandidateId = NewType("CandidateId", int)


class NavigationAction(str, Enum):
    """Controller-owned evidence retrieval actions."""

    INITIAL = "INITIAL"
    NEW_REGION = "NEW_REGION"
    ZOOM_DETAIL = "ZOOM_DETAIL"


class DecisionBasis(str, Enum):
    """Why Answerer was invoked."""

    VERIFIER_PASS = "VERIFIER_PASS"
    FORCED_ANSWER = "FORCED_ANSWER"
    PLANNED_EVIDENCE = "PLANNED_EVIDENCE"


class FrozenModel(BaseModel):
    """Immutable Pydantic boundary model."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetItem(FrozenModel):
    """One parsed WSI-VQA row."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    slide_id: str = Field(alias="Id", min_length=1)
    question: str = Field(alias="Question", min_length=1)
    choices: tuple[str, ...] = Field(alias="Choice", default_factory=tuple)
    answer: str = Field(alias="Answer")
    task: str | None = Field(alias="Task", default=None)

    @field_validator("choices", mode="before")
    @classmethod
    def normalize_choices(
        cls,
        value: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """Represent open-ended questions with an empty tuple."""
        return () if value is None else tuple(value)


class VerifiedFinding(FrozenModel):
    """One image-grounded finding and its patch citations."""

    finding: str = Field(min_length=1)
    patch_ids: tuple[PatchId, ...] = Field(
        min_length=1,
        max_length=MAX_FINAL_EVIDENCE_PATCHES,
    )


class ReasonerFinding(FrozenModel):
    """One intermediate morphology finding over the full planned bundle."""

    finding: str = Field(
        min_length=1,
        validation_alias=AliasChoices("finding", "contradiction"),
    )
    patch_ids: tuple[PatchId, ...] = Field(
        min_length=1,
        max_length=MAX_REASONER_EVIDENCE_PATCHES,
    )


class PatchMetadata(FrozenModel):
    """Stable identity, geometry, lineage, and image artifact for one patch."""

    patch_id: PatchId
    round_number: int = Field(ge=1, le=3)
    rank: int = Field(ge=1, le=25)
    action: NavigationAction
    magnification: Magnification
    box: Box
    root_id: RootCellId
    source_x5_anchor: PatchId
    candidate_id: CandidateId
    tissue_fraction: float = Field(ge=0.10, le=1.0)
    image_path: Path


class EvidenceRound(FrozenModel):
    """Five through twenty-five patches retrieved by one navigation action."""

    round_number: int = Field(ge=1, le=3)
    action: NavigationAction
    patches: tuple[PatchMetadata, ...] = Field(min_length=5, max_length=25)

    @model_validator(mode="after")
    def require_consistent_patch_identity(self) -> Self:
        """Keep round labels, ranks, actions, and IDs internally coherent."""
        expected_ids = tuple(
            PatchId(f"R{self.round_number}-P{rank}")
            for rank in range(1, len(self.patches) + 1)
        )
        actual_ids = tuple(patch.patch_id for patch in self.patches)
        if actual_ids != expected_ids:
            msg = "patch IDs must be ordered R{round}-P1 through P5"
            raise ValueError(msg)
        if any(patch.round_number != self.round_number for patch in self.patches):
            msg = "patch round_number must match its evidence round"
            raise ValueError(msg)
        if any(patch.action is not self.action for patch in self.patches):
            msg = "patch action must match its evidence round"
            raise ValueError(msg)
        return self


class SelectionOutput(FrozenModel):
    """One ranked discrete Navigator response."""

    cells: tuple[int, ...] = Field(min_length=1, max_length=10)

    @field_validator("cells")
    @classmethod
    def require_unique_cells(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject duplicate candidate IDs."""
        if len(set(value)) != len(value):
            msg = "selected cells must be unique"
            raise ValueError(msg)
        return value


class PatchObservation(FrozenModel):
    """Scale-aware morphology observed in one labeled patch."""

    patch_id: PatchId
    quality: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    cellularity: str = Field(min_length=1)
    cytology: str = Field(min_length=1)
    stroma: str = Field(min_length=1)
    necrosis: str = Field(min_length=1)
    boundary: str = Field(min_length=1)
    artifact: str = Field(min_length=1)
    question_relevance: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class ReasonerReport(FrozenModel):
    """Five-through-twenty-five-patch observations and a non-diagnostic synthesis."""

    patches: tuple[PatchObservation, ...] = Field(min_length=5, max_length=25)
    consistent_findings: tuple[ReasonerFinding, ...]
    contradictions: tuple[ReasonerFinding, ...]
    unresolved_evidence: tuple[str, ...]
    diagnostically_usable: tuple[PatchId, ...] = Field(
        max_length=MAX_REASONER_EVIDENCE_PATCHES,
    )

    @field_validator("patches")
    @classmethod
    def require_unique_patch_observations(
        cls,
        value: tuple[PatchObservation, ...],
    ) -> tuple[PatchObservation, ...]:
        """Require one observation per distinct input patch."""
        if len({item.patch_id for item in value}) != len(value):
            msg = "reasoner patch observations must have unique patch IDs"
            raise ValueError(msg)
        return value


class AnswerOutput(FrozenModel):
    """Grounded final answer produced only by Answerer."""

    decision_basis: DecisionBasis
    answer: str = Field(min_length=1)
    evidence_patches: tuple[PatchId, ...] = Field(
        min_length=1,
        max_length=MAX_FINAL_EVIDENCE_PATCHES,
    )
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("evidence_patches", mode="before")
    @classmethod
    def deduplicate_evidence_patches(
        cls,
        value: tuple[PatchId, ...] | list[PatchId],
    ) -> tuple[PatchId, ...]:
        """Canonicalize repeated citations when the decoder cannot enforce uniqueness."""
        return tuple(dict.fromkeys(value))


class RoleCall(FrozenModel):
    """Serializable record of one logical agent invocation."""

    role: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    image_labels: tuple[str, ...]
    reasoning: str
    final_outputs: tuple[str, ...] = Field(min_length=1, max_length=3)
    format_repairs: int = Field(ge=0, le=2)
    physical_calls: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0.0)


class PassDecision(FrozenModel):
    """Evidence is sufficient for Answerer."""

    action: Literal["PASS"]
    verified_findings: tuple[VerifiedFinding, ...]
    supporting_patches: tuple[PatchId, ...] = Field(
        min_length=1,
        max_length=MAX_FINAL_EVIDENCE_PATCHES,
    )
    reason: str = Field(min_length=1)


class NewRegionDecision(FrozenModel):
    """Evidence coverage is wrong or incomplete."""

    action: Literal["NEW_REGION"]
    missing_evidence: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ZoomDetailDecision(FrozenModel):
    """A known x5 region requires finer evidence."""

    action: Literal["ZOOM_DETAIL"]
    source_x5_anchor: PatchId
    missing_evidence: str = Field(min_length=1)
    target_magnification: Magnification
    reason: str = Field(min_length=1)

    @field_validator("target_magnification")
    @classmethod
    def require_detail_magnification(
        cls,
        value: Magnification,
    ) -> Magnification:
        """Reject x5 as a zoom target."""
        if value is Magnification.X5:
            msg = "ZOOM_DETAIL target must be x20 or x40"
            raise ValueError(msg)
        return value


class InsufficientDecision(FrozenModel):
    """Final-round evidence remains answer-changingly incomplete."""

    action: Literal["INSUFFICIENT"]
    best_available_patches: tuple[PatchId, ...] = Field(
        min_length=1,
        max_length=MAX_FINAL_EVIDENCE_PATCHES,
    )
    verified_findings: tuple[VerifiedFinding, ...]
    uncertainty: str = Field(min_length=1)


VerifierDecision = Annotated[
    PassDecision
    | NewRegionDecision
    | ZoomDetailDecision
    | InsufficientDecision,
    Field(discriminator="action"),
]


class CaseInput(FrozenModel):
    """One resolved dataset row and its physical slide path."""

    dataset_index: int = Field(ge=0)
    item: DatasetItem
    slide_path: Path


class StateTransition(FrozenModel):
    """One verifier decision recorded by the deterministic controller."""

    round_number: int = Field(ge=1, le=3)
    action: Literal["PASS", "NEW_REGION", "ZOOM_DETAIL", "INSUFFICIENT"]
    reason: str = Field(min_length=1)


class RunResult(FrozenModel):
    """Successful per-case result with evidence and execution cost."""

    dataset_index: int = Field(ge=0)
    slide_id: str = Field(min_length=1)
    gold_answer: str
    answer: AnswerOutput
    decision_basis: DecisionBasis
    round_count: int = Field(ge=1, le=3)
    patches: tuple[PatchMetadata, ...] = Field(min_length=5, max_length=75)
    transitions: tuple[StateTransition, ...] = Field(max_length=3)
    role_calls: tuple[RoleCall, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def require_valid_patch_budget_per_round(self) -> Self:
        """Keep each evidence round within the supported patch budget."""
        expected_rounds = tuple(range(1, self.round_count + 1))
        actual_rounds = tuple(sorted({patch.round_number for patch in self.patches}))
        if actual_rounds != expected_rounds:
            msg = "run result must contain every sequential evidence round"
            raise ValueError(msg)
        for round_number in expected_rounds:
            patch_count = sum(
                patch.round_number == round_number for patch in self.patches
            )
            if not 5 <= patch_count <= 25:
                msg = "each evidence round must contain five through twenty-five patches"
                raise ValueError(msg)
        if self.answer.decision_basis is not self.decision_basis:
            msg = "answer and run decision basis must match"
            raise ValueError(msg)
        return self

    @property
    def correct(self) -> bool:
        """Use exact dataset-answer matching for the smoke comparison."""
        return self.answer.answer == self.gold_answer
