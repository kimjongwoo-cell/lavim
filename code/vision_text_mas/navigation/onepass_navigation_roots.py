"""Root-grid construction and deterministic x5 materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vision_text_mas.contracts import (
    CandidateId,
    NavigationAction,
    PatchId,
    PatchMetadata,
    RootCellId,
)
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import Box, Magnification, coarse_grid, edge_anchored_grid, eligible_tissue_ids
from vision_text_mas.navigation.navigation_render import (
    SlideLike,
    numbered_grid,
    render_box,
    save_image,
    tissue_fractions,
)
from vision_text_mas.navigation.onepass_navigation_contracts import OneShotNavigation


EXACT_X5_SIDE = 4_096
MAX_ROOT_GRID_SIDE = 6
MINIMUM_TISSUE_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class RootGrid:
    """Root candidates visible to the one Navigator call."""

    dimensions: tuple[int, int]
    boxes: tuple[Box, ...]
    fractions: tuple[float, ...]
    image: Image.Image

    @property
    def candidate_ids(self) -> frozenset[int]:
        return frozenset(range(1, len(self.boxes) + 1))

    @property
    def eligible_ids(self) -> frozenset[int]:
        return eligible_tissue_ids(
            self.fractions,
            minimum_fraction=MINIMUM_TISSUE_FRACTION,
        )


def prepare_root_grid(slide: SlideLike, *, thumbnail: Image.Image) -> RootGrid:
    """Build the thumbnail-labelled root grid once, before model selection."""
    width, height = slide.dimensions
    rows = min(MAX_ROOT_GRID_SIDE, max(1, height // EXACT_X5_SIDE))
    cols = min(MAX_ROOT_GRID_SIDE, max(1, width // EXACT_X5_SIDE))
    boxes = coarse_grid(width, height, rows=rows, cols=cols)
    fractions = tissue_fractions(slide, boxes)
    eligible = eligible_tissue_ids(fractions, minimum_fraction=MINIMUM_TISSUE_FRACTION)
    if not eligible:
        raise PipelineFailure(
            code=FailureCode.INSUFFICIENT_CANDIDATES,
            stage="navigator",
            detail="whole-slide thumbnail has no tissue root at or above 10%",
        )
    return RootGrid(
        dimensions=slide.dimensions,
        boxes=boxes,
        fractions=fractions,
        image=numbered_grid(
            thumbnail.copy(),
            parent=Box(0, 0, width, height),
            boxes=boxes,
            eligible_ids=eligible,
        ),
    )


def _squared_distance(left: Box, right: Box) -> int:
    left_x = left.x * 2 + left.width
    left_y = left.y * 2 + left.height
    right_x = right.x * 2 + right.width
    right_y = right.y * 2 + right.height
    return (left_x - right_x) ** 2 + (left_y - right_y) ** 2


def _root_order(grid: RootGrid, requested_root_id: int) -> tuple[int, ...]:
    requested = grid.boxes[requested_root_id - 1]
    return tuple(
        sorted(
            grid.eligible_ids,
            key=lambda candidate_id: (
                _squared_distance(grid.boxes[candidate_id - 1], requested),
                -grid.fractions[candidate_id - 1],
                candidate_id,
            ),
        )
    )


def _best_x5_child(
    slide: SlideLike,
    *,
    root: Box,
    root_id: int,
    used_boxes: frozenset[Box],
    artifact_root: Path,
    round_number: int,
) -> tuple[Box, int, float] | None:
    try:
        boxes = edge_anchored_grid(root, side=EXACT_X5_SIDE, rows=4, cols=6)
    except ValueError:
        return None
    fractions = tissue_fractions(slide, boxes)
    eligible = eligible_tissue_ids(
        fractions,
        minimum_fraction=MINIMUM_TISSUE_FRACTION,
    )
    _ = save_image(
        numbered_grid(
            render_box(slide, root, max_side=768),
            parent=root,
            boxes=boxes,
            eligible_ids=eligible,
        ),
        artifact_root / f"round_{round_number}_root_{root_id}_children.png",
    )
    candidates = tuple(
        candidate_id
        for candidate_id in eligible
        if boxes[candidate_id - 1] not in used_boxes
    )
    if not candidates:
        return None
    chosen = min(
        candidates,
        key=lambda candidate_id: (
            -fractions[candidate_id - 1],
            _squared_distance(boxes[candidate_id - 1], root),
            candidate_id,
        ),
    )
    return boxes[chosen - 1], chosen, fractions[chosen - 1]


def _materialize_x5(
    slide: SlideLike,
    *,
    grid: RootGrid,
    requested_root_id: int,
    rank: int,
    used_boxes: frozenset[Box],
    artifact_root: Path,
    round_number: int,
) -> PatchMetadata:
    for root_id in _root_order(grid, requested_root_id):
        selected = _best_x5_child(
            slide,
            root=grid.boxes[root_id - 1],
            root_id=root_id,
            used_boxes=used_boxes,
            artifact_root=artifact_root,
            round_number=round_number,
        )
        if selected is None:
            continue
        box, candidate_id, fraction = selected
        patch_id = PatchId(f"R{round_number}-P{rank}")
        path = save_image(
            render_box(slide, box, max_side=512),
            artifact_root / f"round_{round_number}" / f"{patch_id}.png",
        )
        return PatchMetadata(
            patch_id=patch_id,
            round_number=round_number,
            rank=rank,
            action=NavigationAction.INITIAL,
            magnification=Magnification.X5,
            box=box,
            root_id=RootCellId(root_id),
            source_x5_anchor=patch_id,
            candidate_id=CandidateId(candidate_id),
            tissue_fraction=fraction,
            image_path=path,
        )
    raise PipelineFailure(
        code=FailureCode.INSUFFICIENT_CANDIDATES,
        stage="navigator",
        detail="no tissue-safe x5 candidate remains after local replacement",
    )


def materialize_x5_patches(
    slide: SlideLike,
    *,
    root_grid: RootGrid,
    navigation: OneShotNavigation,
    artifact_root: Path,
    round_number: int,
) -> tuple[PatchMetadata, ...]:
    """Render model-selected root regions with deterministic tissue replacement."""
    patches: list[PatchMetadata] = []
    used_boxes: frozenset[Box] = frozenset()
    for rank, requested_root_id in enumerate(navigation.root_ids, start=1):
        patch = _materialize_x5(
            slide,
            grid=root_grid,
            requested_root_id=requested_root_id,
            rank=rank,
            used_boxes=used_boxes,
            artifact_root=artifact_root,
            round_number=round_number,
        )
        patches.append(patch)
        used_boxes = frozenset((*used_boxes, patch.box))
    return tuple(patches)
