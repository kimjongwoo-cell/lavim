"""Pure Level-0 WSI geometry and stained-tissue rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing_extensions import assert_never

from PIL import Image


class Magnification(IntEnum):
    """Supported effective patch magnifications."""

    X5 = 5
    X20 = 20
    X40 = 40


@dataclass(frozen=True, slots=True)
class Box:
    """A rectangle in Level-0 pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, other: Box) -> bool:
        """Return whether another box lies fully inside this box."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )


def coarse_grid(
    slide_width: int,
    slide_height: int,
    *,
    rows: int,
    cols: int,
) -> tuple[Box, ...]:
    """Partition an entire slide into a row-major grid."""
    x_edges = tuple(int(index * slide_width / cols) for index in range(cols + 1))
    y_edges = tuple(int(index * slide_height / rows) for index in range(rows + 1))
    return tuple(
        Box(
            x=x_edges[col],
            y=y_edges[row],
            width=x_edges[col + 1] - x_edges[col],
            height=y_edges[row + 1] - y_edges[row],
        )
        for row in range(rows)
        for col in range(cols)
    )


def edge_anchored_grid(
    parent: Box,
    *,
    side: int,
    rows: int,
    cols: int,
) -> tuple[Box, ...]:
    """Spread exact square fields across a parent, including both edges."""
    x_extent = parent.width - side
    y_extent = parent.height - side
    if x_extent < 0 or y_extent < 0:
        msg = "candidate field exceeds its parent"
        raise ValueError(msg)
    x_positions = tuple(
        parent.x + round(index * x_extent / (cols - 1)) for index in range(cols)
    )
    y_positions = tuple(
        parent.y + round(index * y_extent / (rows - 1)) for index in range(rows)
    )
    return tuple(
        Box(x=x, y=y, width=side, height=side)
        for y in y_positions
        for x in x_positions
    )


def non_overlapping_grid(
    parent: Box,
    *,
    side: int,
    rows: int,
    cols: int,
) -> tuple[Box, ...]:
    """Tile a parent exactly with non-overlapping square fields."""
    if side * cols != parent.width or side * rows != parent.height:
        msg = "non-overlapping grid must exactly tile its parent"
        raise ValueError(msg)
    return tuple(
        Box(
            x=parent.x + col * side,
            y=parent.y + row * side,
            width=side,
            height=side,
        )
        for row in range(rows)
        for col in range(cols)
    )


def zoom_grid(anchor: Box, magnification: Magnification) -> tuple[Box, ...]:
    """Enumerate x20 or x40 fields inside one exact x5 anchor."""
    match magnification:
        case Magnification.X20:
            return non_overlapping_grid(anchor, side=1_024, rows=4, cols=4)
        case Magnification.X40:
            return non_overlapping_grid(anchor, side=512, rows=8, cols=8)
        case Magnification.X5:
            msg = "x5 is an anchor, not a zoom target"
            raise ValueError(msg)
        case _ as unreachable:
            assert_never(unreachable)


def tissue_fraction(image: Image.Image) -> float:
    """Return the fraction of pixels with HSV saturation at least 26."""
    histogram = image.convert("HSV").getchannel("S").histogram()
    stained_pixels = sum(histogram[26:])
    return stained_pixels / sum(histogram)


def eligible_tissue_ids(
    fractions: tuple[float, ...],
    *,
    minimum_fraction: float,
) -> frozenset[int]:
    """Return one-based IDs at or above the tissue threshold."""
    return frozenset(
        candidate_id
        for candidate_id, fraction in enumerate(fractions, start=1)
        if fraction >= minimum_fraction
    )
