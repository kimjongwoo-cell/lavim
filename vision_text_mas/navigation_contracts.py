"""Typed contracts for question planning and spatial patch ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, model_validator
from typing_extensions import Self, assert_never

from vision_text_mas.contracts import FrozenModel


PlanText = Annotated[str, Field(min_length=1, max_length=240)]
PlanItems = Annotated[tuple[PlanText, ...], Field(min_length=1, max_length=3)]


@dataclass(frozen=True, slots=True)
class DuplicateCandidateError(ValueError):
    """A ranked reserve repeated one or more candidate IDs."""

    ids: tuple[int, ...]

    def __str__(self) -> str:
        return f"ranked candidate IDs must be unique: {self.ids}"


@dataclass(frozen=True, slots=True)
class InvalidScalePlanError(ValueError):
    """A scale allocation cannot produce one valid multi-scale bundle."""

    detail: str

    def __str__(self) -> str:
        return self.detail


class ScalePlan(FrozenModel):
    """Question-derived x5 context and x20 detail allocation."""

    overview_target: PlanText
    overview_patch_count: int = Field(ge=1, le=5)
    detail_target: PlanText
    detail_patch_count: int = Field(ge=1, le=20)
    detail_anchor_count: int = Field(ge=1, le=5)
    scale_success_criteria: PlanItems
    patch_budget: Literal[5, 8, 12, 20, 25] = 5

    @model_validator(mode="before")
    @classmethod
    def infer_omitted_patch_budget(cls, data: object) -> object:
        """Recover the only valid omitted budget from a complete allocation."""
        if not isinstance(data, dict) or "patch_budget" in data:
            return data
        overview_count = data.get("overview_patch_count")
        detail_count = data.get("detail_patch_count")
        if overview_count == 3 and detail_count == 5:
            return {**data, "patch_budget": 8}
        if overview_count == 5 and detail_count == 20:
            return {**data, "patch_budget": 25}
        if overview_count == 4 and detail_count == 16:
            return {**data, "patch_budget": 20}
        if overview_count == 4 and detail_count == 8:
            return {**data, "patch_budget": 12}
        return data

    @model_validator(mode="after")
    def require_executable_allocation(self) -> Self:
        """Require exactly five outputs and at least one detail per anchor."""
        allocated_patches = self.overview_patch_count + self.detail_patch_count
        if allocated_patches != self.patch_budget:
            raise InvalidScalePlanError(
                f"x5 and x20 patch counts must total {self.patch_budget}"
            )
        match self.patch_budget:
            case 5:
                pass
            case 8:
                if not 2 <= self.overview_patch_count <= 4:
                    raise InvalidScalePlanError(
                        "eight-patch plans require two through four x5 patches"
                    )
            case 25:
                if self.overview_patch_count != 5 or self.detail_patch_count != 20:
                    raise InvalidScalePlanError(
                        "twenty-five-patch plans require five x5 and twenty x20 patches"
                    )
            case 20:
                if self.overview_patch_count != 4 or self.detail_patch_count != 16:
                    raise InvalidScalePlanError(
                        "twenty-patch plans require four x5 and sixteen x20 patches"
                    )
            case 12:
                if self.overview_patch_count != 4 or self.detail_patch_count != 8:
                    raise InvalidScalePlanError(
                        "twelve-patch plans require four x5 and eight x20 patches"
                    )
            case unreachable:
                assert_never(unreachable)
        if self.detail_anchor_count > self.overview_patch_count:
            raise InvalidScalePlanError("x20 anchors must be retained x5 patches")
        if self.detail_anchor_count > self.detail_patch_count:
            raise InvalidScalePlanError("every x20 anchor must yield a detail patch")
        return self


class EvidencePlan(FrozenModel):
    """Question-derived visual evidence request without spatial decisions."""

    @model_validator(mode="before")
    @classmethod
    def normalize_planner_lists(cls, data: object) -> object:
        """Accept bounded planner lists while preserving deterministic priority."""
        if not isinstance(data, dict):
            return data
        normalized = {key: value for key, value in data.items() if key != "patch_budget"}
        for key in (
            "needed_visual_information",
            "thumbnail_observations",
            "success_criteria",
        ):
            value = normalized.get(key)
            if isinstance(value, list) and len(value) > 3:
                normalized[key] = value[:3]
        return normalized

    question_focus: PlanText = Field(
        validation_alias=AliasChoices("question_focus", "questionFocus"),
    )
    needed_visual_information: PlanItems = Field(
        validation_alias=AliasChoices(
            "needed_visual_information", "needed Visual Information"
        ),
    )
    thumbnail_observations: tuple[PlanText, ...] = Field(
        max_length=3,
        validation_alias=AliasChoices(
            "thumbnail_observations", "thumbnail Observations"
        ),
    )
    search_instruction: PlanText = Field(
        validation_alias=AliasChoices("search_instruction", "search Instruction"),
    )
    success_criteria: PlanItems = Field(
        validation_alias=AliasChoices("success_criteria", "success Criteria"),
    )
    detail_trigger: PlanText = Field(
        validation_alias=AliasChoices("detail_trigger", "detail Trigger"),
    )
    scale_plan: ScalePlan = Field(
        validation_alias=AliasChoices("scale_plan", "scalePlan"),
    )


class RankedCandidate(FrozenModel):
    """One Patch Navigator candidate and its visible plan match."""

    id: int = Field(ge=1)
    match_reason: str = Field(min_length=1)


class RankedSelection(FrozenModel):
    """One ordered overcomplete reserve from Patch Navigator."""

    ranked: tuple[RankedCandidate, ...] = Field(min_length=1, max_length=10)

    @property
    def ids(self) -> tuple[int, ...]:
        """Return candidate IDs without discarding ranking order."""
        return tuple(item.id for item in self.ranked)

    @model_validator(mode="after")
    def require_unique_ids(self) -> Self:
        """Reject duplicate candidate IDs at the model boundary."""
        if len(set(self.ids)) != len(self.ids):
            raise DuplicateCandidateError(ids=self.ids)
        return self


class SelectionTrace(FrozenModel):
    """Auditable controller filtering of one semantic ranking."""

    label: str = Field(min_length=1)
    ranked_ids: tuple[int, ...] = Field(min_length=1, max_length=10)
    skipped_low_tissue_ids: tuple[int, ...]
    selected_ids: tuple[int, ...] = Field(max_length=5)
