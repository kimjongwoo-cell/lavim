"""Deterministic composition of ranked x5 scouts and x20 detail evidence."""

from __future__ import annotations

from vision_text_mas.contracts import EvidenceRound, NavigationAction, PatchMetadata
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.navigation_render import SlideLike
from vision_text_mas.navigator_zoom import ZoomNavigator


class MultiscaleComposer:
    """Build one exact five-patch bundle from an executable scale plan."""

    def __init__(self, zoom: ZoomNavigator) -> None:
        self._zoom = zoom

    def compose(
        self,
        slide: SlideLike,
        *,
        scout: EvidenceRound,
        evidence_plan: EvidencePlan,
    ) -> EvidenceRound:
        """Retain ranked x5 context and collect planned x20 detail."""
        scale = evidence_plan.scale_plan
        overview = scout.patches[: scale.overview_patch_count]
        detail_anchors = overview[: scale.detail_anchor_count]
        base_count, remainder = divmod(
            scale.detail_patch_count,
            scale.detail_anchor_count,
        )
        detail: list[PatchMetadata] = []
        next_rank = scale.overview_patch_count + 1
        for anchor_index, anchor in enumerate(detail_anchors):
            count = base_count + (1 if anchor_index < remainder else 0)
            patches = self._zoom.collect_x20(
                slide,
                evidence_plan=evidence_plan,
                anchor=anchor,
                count=count,
                start_rank=next_rank,
                round_number=scout.round_number,
                action=scout.action,
            )
            detail.extend(patches)
            next_rank += count
        return EvidenceRound(
            round_number=scout.round_number,
            action=scout.action,
            patches=(*overview, *detail),
        )
