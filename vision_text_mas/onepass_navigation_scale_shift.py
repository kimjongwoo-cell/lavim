"""Opt-in Nav3 evidence shift from x5/x20 to x20/x40."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from vision_text_mas.contracts import CandidateId, PatchMetadata
from vision_text_mas.geometry import Box, Magnification, coarse_grid, zoom_grid
from vision_text_mas.navigation_render import SlideLike, render_box, save_image, tissue_fractions

X40_GRID_SIDE: Final = 2


def detail_boxes(anchor: Box, target: Magnification) -> tuple[Box, ...]:
    """Return native level-0 fields for one scale-shift target."""
    match target:
        case Magnification.X20:
            return zoom_grid(anchor, target)
        case Magnification.X40:
            cells = coarse_grid(
                anchor.width,
                anchor.height,
                rows=X40_GRID_SIDE,
                cols=X40_GRID_SIDE,
            )
            return tuple(
                Box(
                    x=anchor.x + cell.x,
                    y=anchor.y + cell.y,
                    width=cell.width,
                    height=cell.height,
                )
                for cell in cells
            )
        case Magnification.X5:
            msg = "x5 cannot be a scale-shift target"
            raise ValueError(msg)
        case _ as unreachable:
            from typing_extensions import assert_never

            assert_never(unreachable)


def shift_nav3_scales(
    slide: SlideLike,
    patches: tuple[PatchMetadata, ...],
    *,
    artifact_root: Path,
    round_number: int,
) -> tuple[PatchMetadata, ...]:
    """Replace Nav3 x5/x20 fields with tissue-rich x20/x40 children."""
    shifted: list[PatchMetadata] = []
    for patch in patches:
        target = (
            Magnification.X20
            if patch.magnification is Magnification.X5
            else Magnification.X40
        )
        boxes = detail_boxes(patch.box, target)
        fractions = tissue_fractions(slide, boxes)
        candidate_index = max(range(len(boxes)), key=fractions.__getitem__)
        box = boxes[candidate_index]
        image_path = save_image(
            render_box(slide, box, max_side=512),
            artifact_root / f"round_{round_number}" / f"{patch.patch_id}.png",
        )
        shifted.append(
            patch.model_copy(
                update={
                    "magnification": target,
                    "box": box,
                    "candidate_id": CandidateId(candidate_index + 1),
                    "tissue_fraction": fractions[candidate_index],
                    "image_path": image_path,
                }
            )
        )
    return tuple(shifted)
