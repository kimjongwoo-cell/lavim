"""Resolve actual image-token neighborhoods without changing Base acquisition."""

from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict
from spatial_latent.grouping import LayoutError as LegacyLayoutError
from spatial_latent.grouping import patch_columns

from pvcr.errors import LayoutError

type Variant = Literal[
    "spatial", "shuffled", "visual", "paired_context", "original", "off"
]


class ImageMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    patch_id: str = ""
    parent_id: str = ""
    magnification: int = 0
    box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class Neighborhoods:
    columns: torch.Tensor
    core: torch.Tensor
    context: torch.Tensor
    kind: str


def build_neighborhoods(
    columns: torch.Tensor,
    grids: torch.Tensor,
    metadata: tuple[ImageMetadata, ...],
    *,
    variant: Variant,
) -> Neighborhoods:
    """Use thumbnail grid adjacency or geometrically validated x5/x20 families."""
    try:
        runs = patch_columns(columns, len(metadata))
    except LegacyLayoutError as error:
        raise LayoutError(str(error)) from error
    if grids.shape != (len(metadata), 3):
        raise LayoutError("PVCR requires actual per-image grid_thw metadata")
    shapes = [(int(t), int(h) // 2, int(w) // 2) for t, h, w in grids.tolist()]
    if any(
        t != 1 or h * w != run.numel()
        for (t, h, w), run in zip(shapes, runs, strict=True)
    ):
        raise LayoutError("PVCR grid/visual column mismatch (pruning unsupported)")
    parents = [i for i, item in enumerate(metadata) if item.magnification == 5]
    if parents:
        if len(parents) != 4 or len(metadata) != 12:
            raise LayoutError("PVCR requires four x5 parents and eight x20 children")
        family_count = 8 if variant == "paired_context" else 4
        core = torch.zeros(family_count, columns.numel())
        context = torch.zeros_like(core)
        offsets = [0]
        for run in runs:
            offsets.append(offsets[-1] + run.numel())
        children_seen: set[int] = set()
        for row, parent in enumerate(parents):
            parent_meta = metadata[parent]
            if parent_meta.box is None:
                raise LayoutError("Missing parent spatial box")
            x, y, width, height = parent_meta.box
            children = [
                i
                for i, item in enumerate(metadata)
                if item.magnification == 20 and item.parent_id == parent_meta.patch_id
            ]
            if len(children) != 2:
                raise LayoutError("Each PVCR parent must have two children")
            for child in children:
                box = metadata[child].box
                if box is None:
                    raise LayoutError("Missing child spatial box")
                cx, cy, cw, ch = box
                if not (
                    x <= cx
                    and y <= cy
                    and cx + cw <= x + width
                    and cy + ch <= y + height
                ):
                    raise LayoutError("PVCR child is outside its parent")
                if variant == "paired_context":
                    sibling = next(candidate for candidate in children if candidate != child)
                    core[row * 2 + children.index(child), offsets[child] : offsets[child + 1]] = 1
                    family = row * 2 + children.index(child)
                    context[family, offsets[sibling] : offsets[sibling + 1]] = 1
                    context[family, offsets[parent] : offsets[parent + 1]] = 1
                else:
                    core[row, offsets[child] : offsets[child + 1]] = 1
                children_seen.add(child)
            if variant != "paired_context":
                context[row, offsets[parent] : offsets[parent + 1]] = 1
        if len(children_seen) != 8:
            raise LayoutError("PVCR children are not uniquely assigned")
        if variant == "shuffled":
            context = context.roll(1, dims=0)
        return Neighborhoods(columns, core, context, "x5_x20")
    if len(metadata) != 2:
        raise LayoutError("Navigator PVCR expects thumbnail and root-grid images")
    # The first image is the unannotated thumbnail; root-grid digits are excluded.
    _, height, width = shapes[0]
    count = height * width
    indices = torch.arange(count)
    rows, cols = indices // width, indices % width
    distance = torch.maximum((rows[:, None] - rows).abs(), (cols[:, None] - cols).abs())
    core = torch.eye(count)
    context = (distance == 1).float()
    if variant == "shuffled":
        if height < 4:
            raise LayoutError("Shuffled thumbnail requires at least four token rows")
        context = context.roll((height // 2) * width, dims=1)
    return Neighborhoods(runs[0], core, context, "thumbnail_8_connected")
