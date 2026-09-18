"""Shared discrete-selection session and auditable call records."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from vision_text_mas.contracts import RoleCall
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.navigation_contracts import (
    EvidencePlan,
    RankedSelection,
    SelectionTrace,
)
from vision_text_mas.qwen_client import ParsedRoleCall


class DiscreteSelector(Protocol):
    """One view-to-discrete-IDs selection boundary."""

    def select(
        self,
        *,
        image: Image.Image,
        image_label: str,
        evidence_plan: EvidencePlan,
        candidate_ids: frozenset[int],
        selection_count: int,
    ) -> ParsedRoleCall[RankedSelection]: ...


class SelectionSession:
    """Apply hard candidate capacity and preserve every Navigator call."""

    def __init__(self, selector: DiscreteSelector) -> None:
        self._selector = selector
        self._records: list[RoleCall] = []
        self._traces: list[SelectionTrace] = []

    @property
    def records(self) -> tuple[RoleCall, ...]:
        return tuple(self._records)

    @property
    def traces(self) -> tuple[SelectionTrace, ...]:
        return tuple(self._traces)

    def select(
        self,
        *,
        image: Image.Image,
        label: str,
        evidence_plan: EvidencePlan,
        candidate_ids: frozenset[int],
        eligible_ids: frozenset[int],
        count: int,
    ) -> tuple[int, ...]:
        """Filter one overcomplete ranking in place without re-querying."""
        if len(eligible_ids) < count:
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail=f"{label} has {len(eligible_ids)} eligible candidates; needs {count}",
            )
        selectable_ids = candidate_ids.intersection(eligible_ids)
        reserve_count = min(len(selectable_ids), count + 5)
        call = self._selector.select(
            image=image,
            image_label=label,
            evidence_plan=evidence_plan,
            candidate_ids=selectable_ids,
            selection_count=reserve_count,
        )
        self._records.append(call.record)
        selected = tuple(
            candidate_id
            for candidate_id in call.value.ids
            if candidate_id in eligible_ids
        )[:count]
        skipped = tuple(
            candidate_id
            for candidate_id in call.value.ids
            if candidate_id not in eligible_ids
        )
        self._traces.append(
            SelectionTrace(
                label=label,
                ranked_ids=call.value.ids,
                skipped_low_tissue_ids=skipped,
                selected_ids=selected,
            )
        )
        if len(selected) < count:
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail=(
                    f"{label} ranked reserve has {len(selected)} eligible candidates; "
                    f"needs {count}"
                ),
            )
        return selected
