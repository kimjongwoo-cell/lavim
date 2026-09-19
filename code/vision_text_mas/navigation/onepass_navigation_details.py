"""Deterministic tissue-safe x20 materialization for selected x5 anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_text_mas.contracts import CandidateId, NavigationAction, PatchId, PatchMetadata
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import Box, Magnification, eligible_tissue_ids, zoom_grid
from vision_text_mas.navigation.navigation_contracts import EvidencePlan
from vision_text_mas.navigation.navigation_render import (
    SlideLike,
    numbered_grid,
    render_box,
    save_image,
    tissue_fractions,
)
from vision_text_mas.navigation.onepass_navigation_contracts import OneShotNavigation
from vision_text_mas.navigation.onepass_navigation_roots import MINIMUM_TISSUE_FRACTION


@dataclass(frozen=True, slots=True)
class _DetailGrid:
    boxes: tuple[Box, ...]
    fractions: tuple[float, ...]
    eligible_ids: frozenset[int]


def _detail_grid(
    slide: SlideLike,
    *,
    anchor: PatchMetadata,
    artifact_root: Path,
    round_number: int,
) -> _DetailGrid:
    boxes = zoom_grid(anchor.box, Magnification.X20)
    fractions = tissue_fractions(slide, boxes)
    eligible = eligible_tissue_ids(
        fractions,
        minimum_fraction=MINIMUM_TISSUE_FRACTION,
    )
    _ = save_image(
        numbered_grid(
            render_box(slide, anchor.box, max_side=768),
            parent=anchor.box,
            boxes=boxes,
            eligible_ids=eligible,
        ),
        artifact_root / f"round_{round_number}_anchor_{anchor.patch_id}_x20_grid.png",
    )
    return _DetailGrid(boxes=boxes, fractions=fractions, eligible_ids=eligible)


def _x20_distance(left_id: int, right_id: int) -> int:
    left_row, left_col = divmod(left_id - 1, 4)
    right_row, right_col = divmod(right_id - 1, 4)
    return (left_row - right_row) ** 2 + (left_col - right_col) ** 2


def _choose_detail_id(
    grid: _DetailGrid,
    *,
    requested_id: int,
    used_ids: frozenset[int],
) -> int | None:
    available = grid.eligible_ids.difference(used_ids)
    if not available:
        return None
    return min(
        available,
        key=lambda candidate_id: (
            _x20_distance(candidate_id, requested_id),
            -grid.fractions[candidate_id - 1],
            candidate_id,
        ),
    )


def materialize_x20_patches(
    slide: SlideLike,
    *,
    x5_patches: tuple[PatchMetadata, ...],
    navigation: OneShotNavigation,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
) -> tuple[PatchMetadata, ...]:
    """Distribute x20 targets across planned anchors without another query."""
    anchor_count = evidence_plan.scale_plan.detail_anchor_count
    anchors = x5_patches[:anchor_count]
    grids = {
        anchor.patch_id: _detail_grid(
            slide,
            anchor=anchor,
            artifact_root=artifact_root,
            round_number=round_number,
        )
        for anchor in anchors
    }
    used_ids: dict[PatchId, frozenset[int]] = {
        anchor.patch_id: frozenset() for anchor in anchors
    }
    details: list[PatchMetadata] = []
    for offset, requested_id in enumerate(navigation.detail_cell_ids):
        preferred = anchors[offset % len(anchors)]
        search = (preferred, *(anchor for anchor in anchors if anchor != preferred))
        selected_anchor: PatchMetadata | None = None
        selected_id: int | None = None
        for anchor in search:
            candidate_id = _choose_detail_id(
                grids[anchor.patch_id],
                requested_id=requested_id,
                used_ids=used_ids[anchor.patch_id],
            )
            if candidate_id is not None:
                selected_anchor, selected_id = anchor, candidate_id
                break
        if selected_anchor is None or selected_id is None:
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail="no tissue-safe x20 candidate remains after local replacement",
            )
        used_ids[selected_anchor.patch_id] = frozenset(
            (*used_ids[selected_anchor.patch_id], selected_id)
        )
        grid = grids[selected_anchor.patch_id]
        patch_id = PatchId(f"R{round_number}-P{len(x5_patches) + offset + 1}")
        box = grid.boxes[selected_id - 1]
        path = save_image(
            render_box(slide, box, max_side=512),
            artifact_root / f"round_{round_number}" / f"{patch_id}.png",
        )
        details.append(
            PatchMetadata(
                patch_id=patch_id,
                round_number=round_number,
                rank=len(x5_patches) + offset + 1,
                action=NavigationAction.INITIAL,
                magnification=Magnification.X20,
                box=box,
                root_id=selected_anchor.root_id,
                source_x5_anchor=selected_anchor.patch_id,
                candidate_id=CandidateId(selected_id),
                tissue_fraction=grid.fractions[selected_id - 1],
                image_path=path,
            )
        )
    return tuple(details)
