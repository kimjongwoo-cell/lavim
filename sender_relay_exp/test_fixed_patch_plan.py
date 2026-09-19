"""Regression tests for deterministic multi-scale OnePass allocations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_text_mas.latent_onepass import fixed_patch_plan

PatchBudget = Literal[8, 12, 20, 25]


@pytest.mark.parametrize(
    ("patch_budget", "expected"),
    [
        (8, (3, 5, 2)),
        (12, (4, 8, 4)),
        (20, (4, 16, 4)),
        (25, (5, 20, 5)),
    ],
)
def test_fixed_patch_plan_matches_its_requested_multiscale_budget(
    patch_budget: PatchBudget, expected: tuple[int, int, int]
) -> None:
    """Every supported budget yields its exact validated x5/x20 allocation."""
    plan = fixed_patch_plan("What is the diagnosis?", patch_budget=patch_budget)

    assert (
        plan.scale_plan.overview_patch_count,
        plan.scale_plan.detail_patch_count,
        plan.scale_plan.detail_anchor_count,
    ) == expected
    assert plan.scale_plan.overview_patch_count + plan.scale_plan.detail_patch_count == patch_budget
