"""Whole-slide root selection and exact-x5 child retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from vision_text_mas.contracts import (
    CandidateId,
    EvidenceRound,
    NavigationAction,
    PatchId,
    PatchMetadata,
    RootCellId,
)
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import (
    Box,
    Magnification,
    coarse_grid,
    edge_anchored_grid,
    eligible_tissue_ids,
)
from vision_text_mas.navigation.navigation_render import (
    SlideLike,
    numbered_grid,
    render_box,
    save_image,
    tissue_fractions,
)
from vision_text_mas.navigation.navigation_contracts import EvidencePlan
from vision_text_mas.navigation.navigation_session import SelectionSession


MINIMUM_TISSUE_FRACTION = 0.10
EXACT_X5_SIDE: Final = 4_096
MAX_ROOT_GRID_SIDE: Final = 6


def _repeat_selected_roots(selected: tuple[int, ...], count: int) -> tuple[int, ...]:
    """Fill a five-patch bundle by cycling only roots already passing tissue filtering."""
    if not selected:
        raise PipelineFailure(
            code=FailureCode.INSUFFICIENT_CANDIDATES,
            stage="navigator",
            detail="cannot repeat an empty root selection",
        )
    return tuple(selected[index % len(selected)] for index in range(count))


def _selectable_root_ids(
    eligible: frozenset[int],
    visited: frozenset[int],
) -> frozenset[int]:
    """Prefer unseen tissue roots, then reuse tissue roots after exhaustion."""
    unvisited = eligible.difference(visited)
    return unvisited or eligible


@dataclass(frozen=True, slots=True)
class RootCache:
    """Whole-slide data reused across root navigation rounds."""

    dimensions: tuple[int, int]
    boxes: tuple[Box, ...]
    fractions: tuple[float, ...]
    thumbnail: Image.Image


class RegionNavigator:
    """Retrieve five x5 patches from five selected tissue roots."""

    def __init__(self, *, session: SelectionSession, artifact_root: Path) -> None:
        self._session = session
        self._artifact_root = artifact_root
        self._root_cache: RootCache | None = None

    def _roots(self, slide: SlideLike, thumbnail: Image.Image | None) -> RootCache:
        if self._root_cache is None:
            if thumbnail is None:
                raise PipelineFailure(
                    code=FailureCode.IMAGE_IO,
                    stage="navigator",
                    detail="initial navigation requires the planner thumbnail",
                )
            width, height = slide.dimensions
            rows = min(MAX_ROOT_GRID_SIDE, max(1, height // EXACT_X5_SIDE))
            cols = min(MAX_ROOT_GRID_SIDE, max(1, width // EXACT_X5_SIDE))
            boxes = coarse_grid(width, height, rows=rows, cols=cols)
            self._root_cache = RootCache(
                dimensions=slide.dimensions,
                boxes=boxes,
                fractions=tissue_fractions(slide, boxes),
                thumbnail=thumbnail.copy(),
            )
        if self._root_cache.dimensions != slide.dimensions:
            raise PipelineFailure(
                code=FailureCode.IMAGE_IO,
                stage="navigator",
                detail="Navigator cannot mix slides within one case",
            )
        return self._root_cache

    def run(
        self,
        slide: SlideLike,
        *,
        thumbnail: Image.Image | None,
        evidence_plan: EvidencePlan,
        round_number: int,
        action: NavigationAction,
        excluded_roots: frozenset[int],
    ) -> EvidenceRound:
        """Select five roots and exactly one tissue-eligible x5 child per root."""
        roots = self._roots(slide, thumbnail)
        tissue_eligible = eligible_tissue_ids(
            roots.fractions,
            minimum_fraction=MINIMUM_TISSUE_FRACTION,
        )
        eligible = _selectable_root_ids(tissue_eligible, excluded_roots)
        candidates = frozenset(range(1, len(roots.boxes) + 1))
        root_view = numbered_grid(
            roots.thumbnail.copy(),
            parent=Box(0, 0, *roots.dimensions),
            boxes=roots.boxes,
            eligible_ids=eligible,
        )
        save_image(root_view, self._artifact_root / f"round_{round_number}_roots.png")
        root_count = min(5, len(eligible))
        if root_count == 0:
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail=f"{root_view} has no eligible tissue roots",
            )
        selected_unique_roots = self._session.select(
            image=root_view,
            label=f"round-{round_number}-root-grid",
            evidence_plan=evidence_plan,
            candidate_ids=candidates,
            eligible_ids=eligible,
            count=root_count,
        )
        selected_roots = _repeat_selected_roots(selected_unique_roots, 5)
        patches = tuple(
            self._select_child(
                slide,
                root=roots.boxes[root_id - 1],
                root_id=root_id,
                rank=rank,
                round_number=round_number,
                action=action,
                evidence_plan=evidence_plan,
            )
            for rank, root_id in enumerate(selected_roots, start=1)
        )
        return EvidenceRound(round_number=round_number, action=action, patches=patches)

    def _select_child(
        self,
        slide: SlideLike,
        *,
        root: Box,
        root_id: int,
        rank: int,
        round_number: int,
        action: NavigationAction,
        evidence_plan: EvidencePlan,
    ) -> PatchMetadata:
        try:
            boxes = edge_anchored_grid(
                root,
                side=EXACT_X5_SIDE,
                rows=4,
                cols=6,
            )
        except ValueError as error:
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail=f"root {root_id} cannot contain exact x5 candidates",
            ) from error
        fractions = tissue_fractions(slide, boxes)
        eligible = eligible_tissue_ids(
            fractions,
            minimum_fraction=MINIMUM_TISSUE_FRACTION,
        )
        candidates = frozenset(range(1, len(boxes) + 1))
        view = numbered_grid(
            render_box(slide, root, max_side=768),
            parent=root,
            boxes=boxes,
            eligible_ids=eligible,
        )
        save_image(
            view,
            self._artifact_root / f"round_{round_number}_root_{root_id}_children.png",
        )
        selected = self._session.select(
            image=view,
            label=f"round-{round_number}-root-{root_id}-x5-grid",
            evidence_plan=evidence_plan,
            candidate_ids=candidates,
            eligible_ids=eligible,
            count=1,
        )[0]
        patch_id = PatchId(f"R{round_number}-P{rank}")
        box = boxes[selected - 1]
        rendered = render_box(slide, box, max_side=512)
        save_image(
            rendered,
            self._artifact_root
            / f"round_{round_number}_scout"
            / f"R{round_number}-P{rank}.png",
        )
        path = save_image(
            rendered,
            self._artifact_root / f"round_{round_number}" / f"{patch_id}.png",
        )
        return PatchMetadata(
            patch_id=patch_id,
            round_number=round_number,
            rank=rank,
            action=action,
            magnification=Magnification.X5,
            box=box,
            root_id=RootCellId(root_id),
            source_x5_anchor=patch_id,
            candidate_id=CandidateId(selected),
            tissue_fraction=fractions[selected - 1],
            image_path=path,
        )
