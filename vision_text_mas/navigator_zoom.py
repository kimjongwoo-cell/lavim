"""Five-patch x20/x40 detail retrieval inside one original x5 anchor."""

from __future__ import annotations

from pathlib import Path

from vision_text_mas.contracts import (
    CandidateId,
    EvidenceRound,
    NavigationAction,
    PatchId,
    PatchMetadata,
    ZoomDetailDecision,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import Box, Magnification, eligible_tissue_ids, zoom_grid
from vision_text_mas.navigation_render import (
    SlideLike,
    numbered_grid,
    render_box,
    save_image,
    tissue_fractions,
)
from vision_text_mas.navigation_session import SelectionSession
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.navigator_regions import MINIMUM_TISSUE_FRACTION


class ZoomNavigator:
    """Enforce the x5-to-x20-to-x40 evidence ladder."""

    def __init__(self, *, session: SelectionSession, artifact_root: Path) -> None:
        self._session = session
        self._artifact_root = artifact_root

    def run(
        self,
        slide: SlideLike,
        *,
        evidence_plan: EvidencePlan,
        request: ZoomDetailDecision,
        round_number: int,
        evidence: EvidenceMemory,
    ) -> EvidenceRound:
        """Select five finer patches, falling through tissue-poor x5 anchors."""
        requested = evidence.find_patch(request.source_x5_anchor)
        if requested.magnification is not Magnification.X5:
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="navigator",
                detail="zoom source must be an original x5 patch",
            )
        original_x5 = tuple(
            patch
            for patch in evidence.patches
            if patch.magnification is Magnification.X5
            and patch.source_x5_anchor == patch.patch_id
        )
        anchors = (
            (requested,)
            if request.target_magnification is Magnification.X40
            else (requested, *(patch for patch in original_x5 if patch != requested))
        )
        last_capacity_failure: PipelineFailure | None = None
        patches: tuple[PatchMetadata, ...] | None = None
        for anchor in anchors:
            prior_x20 = any(
                patch.magnification is Magnification.X20
                and patch.source_x5_anchor == anchor.patch_id
                for patch in evidence.patches
            )
            if request.target_magnification is Magnification.X40 and not prior_x20:
                continue
            try:
                patches = self._collect(
                    slide,
                    evidence_plan=evidence_plan,
                    anchor=anchor,
                    target_magnification=request.target_magnification,
                    count=5,
                    start_rank=1,
                    round_number=round_number,
                    action=NavigationAction.ZOOM_DETAIL,
                    excluded_boxes=frozenset(
                        patch.box for patch in evidence.patches
                    ),
                )
                break
            except PipelineFailure as failure:
                if failure.code is not FailureCode.INSUFFICIENT_CANDIDATES:
                    raise
                last_capacity_failure = failure
        if patches is None:
            if last_capacity_failure is not None:
                raise last_capacity_failure
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="navigator",
                detail="no original x5 anchor supports the requested zoom",
            )
        return EvidenceRound(
            round_number=round_number,
            action=NavigationAction.ZOOM_DETAIL,
            patches=patches,
        )

    def collect_x20(
        self,
        slide: SlideLike,
        *,
        evidence_plan: EvidencePlan,
        anchor: PatchMetadata,
        count: int,
        start_rank: int,
        round_number: int,
        action: NavigationAction,
    ) -> tuple[PatchMetadata, ...]:
        """Collect planned x20 detail inside one retained x5 anchor."""
        if anchor.magnification is not Magnification.X5:
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="navigator",
                detail="planned x20 source must be a retained x5 patch",
            )
        return self._collect(
            slide,
            evidence_plan=evidence_plan,
            anchor=anchor,
            target_magnification=Magnification.X20,
            count=count,
            start_rank=start_rank,
            round_number=round_number,
            action=action,
        )

    def _collect(
        self,
        slide: SlideLike,
        *,
        evidence_plan: EvidencePlan,
        anchor: PatchMetadata,
        target_magnification: Magnification,
        count: int,
        start_rank: int,
        round_number: int,
        action: NavigationAction,
        excluded_boxes: frozenset[Box] = frozenset(),
    ) -> tuple[PatchMetadata, ...]:
        """Rank and materialize a requested number of finer patches."""
        boxes = zoom_grid(anchor.box, target_magnification)
        fractions = tissue_fractions(slide, boxes)
        eligible = eligible_tissue_ids(
            fractions,
            minimum_fraction=MINIMUM_TISSUE_FRACTION,
        )
        used_candidate_ids = frozenset(
            candidate_id
            for candidate_id, box in enumerate(boxes, start=1)
            if box in excluded_boxes
        )
        eligible = eligible.difference(used_candidate_ids)
        candidates = frozenset(range(1, len(boxes) + 1))
        view = numbered_grid(
            render_box(slide, anchor.box, max_side=768),
            parent=anchor.box,
            boxes=boxes,
            eligible_ids=eligible,
        )
        save_image(
            view,
            self._artifact_root
            / f"round_{round_number}_anchor_{anchor.patch_id}_x{target_magnification.value}_grid.png",
        )
        selected = self._session.select(
            image=view,
            label=(
                f"round-{round_number}-anchor-{anchor.patch_id}-"
                f"x{target_magnification.value}-grid"
            ),
            evidence_plan=evidence_plan,
            candidate_ids=candidates,
            eligible_ids=eligible,
            count=count,
        )
        patches = tuple(
            self._patch(
                slide,
                box=boxes[candidate_id - 1],
                fraction=fractions[candidate_id - 1],
                candidate_id=candidate_id,
                rank=start_rank + offset,
                round_number=round_number,
                target_magnification=target_magnification,
                action=action,
                anchor=anchor,
            )
            for offset, candidate_id in enumerate(selected)
        )
        return patches

    def _patch(
        self,
        slide: SlideLike,
        *,
        box: Box,
        fraction: float,
        candidate_id: int,
        rank: int,
        round_number: int,
        target_magnification: Magnification,
        action: NavigationAction,
        anchor: PatchMetadata,
    ) -> PatchMetadata:
        patch_id = PatchId(f"R{round_number}-P{rank}")
        path = save_image(
            render_box(slide, box, max_side=512),
            self._artifact_root / f"round_{round_number}" / f"{patch_id}.png",
        )
        return PatchMetadata(
            patch_id=patch_id,
            round_number=round_number,
            rank=rank,
            action=action,
            magnification=target_magnification,
            box=box,
            root_id=anchor.root_id,
            source_x5_anchor=anchor.patch_id,
            candidate_id=CandidateId(candidate_id),
            tissue_fraction=fraction,
            image_path=path,
        )
