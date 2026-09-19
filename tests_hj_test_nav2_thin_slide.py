"""Regression coverage for thin slides and sparse positive tissue in Nav2."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")

from PIL import Image

from vision_text_mas.geometry import Box, Magnification
from vision_text_mas.navigation_render import SlideLike
from vision_text_mas.onepass_navigation_nav2 import (
    Nav2Selection,
    _contact_sheet,
    _cross_scale_selection,
    _global_candidates,
)
from vision_text_mas.onepass_navigation_roots import prepare_root_grid
from vision_text_mas.wsi_search_tool import WSIObservation


class _SyntheticSlide:
    def __init__(self, dimensions: tuple[int, int], *, tissue_fraction: float) -> None:
        self._dimensions = dimensions
        self._tissue_fraction = tissue_fraction

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return (1.0,)

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        return Image.new("RGB", size, (200, 120, 160))

    def get_best_level_for_downsample(self, downsample: float) -> int:
        _ = downsample
        return 0

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        _ = (location, level)
        width, height = size
        image = Image.new("RGB", size, "white")
        stained_height = max(1, round(height * self._tissue_fraction))
        image.paste((200, 120, 160), (0, 0, width, stained_height))
        return image


def test_nav2_builds_nonempty_candidate_sheets_for_narrow_slides() -> None:
    slide: SlideLike = _SyntheticSlide((2_816, 27_648), tissue_fraction=0.40)
    grid = prepare_root_grid(slide, thumbnail=slide.get_thumbnail((1_024, 1_024)))

    x5_candidates, x20_candidates = _global_candidates(slide, root_grid=grid)

    assert x5_candidates
    assert x20_candidates
    assert all(candidate.scale == int(Magnification.X5) for candidate in x5_candidates)
    assert all(candidate.scale == int(Magnification.X20) for candidate in x20_candidates)
    assert _contact_sheet(x5_candidates).height > 0
    assert _contact_sheet(x20_candidates).height > 0


def test_nav2_keeps_best_positive_root_when_no_root_reaches_ten_percent() -> None:
    slide: SlideLike = _SyntheticSlide((16_384, 16_384), tissue_fraction=0.08)

    grid = prepare_root_grid(slide, thumbnail=slide.get_thumbnail((1_024, 1_024)))

    assert len(grid.eligible_ids) == 1
    chosen_id = next(iter(grid.eligible_ids))
    assert grid.fractions[chosen_id - 1] == max(grid.fractions)
    assert 0.0 < grid.fractions[chosen_id - 1] < 0.10


def _observation(crop_id: str, parent_crop_id: str | None, scale: int) -> WSIObservation:
    return WSIObservation(
        crop_id=crop_id,
        image=Image.new("RGB", (8, 8), "white"),
        scale=scale,
        box=Box(x=0, y=0, width=8, height=8),
        parent_crop_id=parent_crop_id,
        root_id=1,
        candidate_id=1,
        tissue_fraction=1.0,
        score=1.0,
    )


def test_cross_scale_selection_prioritizes_parents_supported_by_selected_details() -> None:
    """Given selected details, when routing, then their strongest parents lead x5."""
    x5 = tuple(_observation(f"x5-{index}", None, 5) for index in range(1, 6))
    x20 = (
        _observation("x20-1", "x5-2", 20),
        _observation("x20-2", "x5-4", 20),
        _observation("x20-3", "x5-2", 20),
        _observation("x20-4", "x5-5", 20),
    )
    selection = Nav2Selection(
        x5_ids=(1, 3, 4),
        x20_ids=(1, 2, 3),
        selection_reason="visible evidence",
    )

    routed = _cross_scale_selection(selection, x5, x20)

    assert routed.x5_ids == (2, 4, 1)


def test_cross_scale_selection_preserves_selected_detail_ids() -> None:
    """Given a routed bank, when selecting parents, then x20 evidence is unchanged."""
    x5 = tuple(_observation(f"x5-{index}", None, 5) for index in range(1, 4))
    x20 = (
        _observation("x20-1", "x5-2", 20),
        _observation("x20-2", "x5-3", 20),
    )
    selection = Nav2Selection(
        x5_ids=(1, 2),
        x20_ids=(2, 1),
        selection_reason="visible evidence",
    )

    routed = _cross_scale_selection(selection, x5, x20)

    assert routed.x20_ids == selection.x20_ids
