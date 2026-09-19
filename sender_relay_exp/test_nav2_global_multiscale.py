"""Behavioral tests for Nav2 global multi-scale acquisition."""

from __future__ import annotations

from PIL import Image

from vision_text_mas.geometry import Box, Magnification
from vision_text_mas.latent_onepass import fixed_patch_plan
from vision_text_mas.onepass_navigation_nav2 import _contact_sheet, _fallback_ids, _prompt
from vision_text_mas.wsi_search_tool import WSIObservation


def _observation(*, crop_id: str, parent: str, score: float) -> WSIObservation:
    return WSIObservation(
        crop_id=crop_id,
        image=Image.new("RGB", (2, 2)),
        scale=int(Magnification.X20),
        box=Box(0, 0, 1024, 1024),
        parent_crop_id=parent,
        root_id=int(parent.removeprefix("p")),
        candidate_id=1,
        tissue_fraction=score,
        score=score,
    )


def test_contact_sheet_contains_every_actual_candidate() -> None:
    """Given global candidates, Navigator receives every crop in one image."""
    candidates = tuple(
        _observation(crop_id=f"c{index}", parent=f"p{index}", score=0.5)
        for index in range(1, 10)
    )

    sheet = _contact_sheet(candidates)

    assert sheet.size == (8 * 144, 2 * 144)


def test_contact_sheet_accepts_child_from_unselected_x5() -> None:
    """Given an x20-only parent, its actual pixels remain visible to Navigator."""
    candidate = _observation(crop_id="child", parent="p9", score=0.5)

    sheet = _contact_sheet((candidate,))

    assert sheet.getbbox() is not None


def test_nav2_prompt_preserves_latent_transport_marker() -> None:
    """Given Nav2, the latent client can replace its required prompt marker."""
    plan = fixed_patch_plan("What is shown?", patch_budget=12)

    prompt = _prompt(plan, x5_count=12, x20_count=24)

    assert prompt.startswith("Validated evidence plan: ")


def test_fallback_ids_stop_when_slide_has_fewer_regions_than_budget() -> None:
    """Given sparse tissue, fallback does not duplicate nonexistent evidence."""
    assert _fallback_ids(required=4, available=2) == (1, 2)
