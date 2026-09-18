"""Nav4: visual global x5/x20 candidate grids for one Navigator call."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import TypeAdapter

from vision_text_mas.contracts import PatchId, PatchMetadata, RoleCall
from vision_text_mas.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.navigation_render import SlideLike, save_image
from vision_text_mas.onepass_navigation_nav2 import (
    Nav2Selection,
    nav2_contact_sheet,
    nav2_cross_scale_selection,
    nav2_fallback_ids,
    nav2_global_candidates,
    nav2_patch,
)
from vision_text_mas.onepass_navigation_roots import RootGrid
from vision_text_mas.qwen_client import QwenJsonClient


def _prompt(
    plan: EvidencePlan,
    *,
    x5_count: int,
    x20_count: int,
) -> str:
    """Describe the two visual candidate grids supplied to Nav4."""
    scale = plan.scale_plan
    return (
        "Validated evidence plan: (carried in latent state)\n"
        f"Question focus: {plan.question_focus}\n"
        f"x5 overview target: {scale.overview_target}\n"
        f"x20 detail target: {scale.detail_target}\n"
        f"Image 1 is a numbered contact grid of {x5_count} actual x5 root "
        "candidates. Image 2 is a separate numbered contact grid of "
        f"{x20_count} actual x20 root candidates. Choose exactly "
        f"{min(scale.overview_patch_count, x5_count)} x5 IDs and "
        f"{min(scale.detail_patch_count, x20_count)} x20 IDs by inspecting "
        "the corresponding visual crops. The x20 grid is independent of the "
        "x5 selection; do not infer x20 IDs from x5 IDs. Prefer diagnostic "
        "tissue, spatial diversity, and complementary evidence. Never diagnose. "
        'Output only JSON: {"x5_ids": [1], "x20_ids": [2], '
        '"selection_reason": "visible evidence"}'
    )


def run_nav4_navigation(
    slide: SlideLike,
    *,
    client: QwenJsonClient,
    root_grid: RootGrid,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
    save_navigation_pngs: bool = True,
) -> tuple[tuple[PatchMetadata, ...], tuple[RoleCall, ...], SelectionTrace]:
    """Select x5 and independent x20 roots from two visual candidate grids."""
    x5_candidates, x20_candidates = nav2_global_candidates(slide, root_grid=root_grid)
    scale = evidence_plan.scale_plan
    x5_target = min(scale.overview_patch_count, len(x5_candidates))
    x20_target = min(scale.detail_patch_count, len(x20_candidates))

    def fallback(_outputs: tuple[str, ...], _error: str) -> Nav2Selection:
        return Nav2Selection(
            x5_ids=nav2_fallback_ids(
                required=x5_target,
                available=len(x5_candidates),
            ),
            x20_ids=nav2_fallback_ids(
                required=x20_target,
                available=len(x20_candidates),
            ),
            selection_reason="deterministic tissue fallback",
        )

    def validate(selection: Nav2Selection) -> str | None:
        valid = (
            len(selection.x5_ids) == x5_target
            and len(set(selection.x5_ids)) == len(selection.x5_ids)
            and all(1 <= index <= len(x5_candidates) for index in selection.x5_ids)
            and len(selection.x20_ids) == x20_target
            and len(set(selection.x20_ids)) == len(selection.x20_ids)
            and all(1 <= index <= len(x20_candidates) for index in selection.x20_ids)
        )
        return None if valid else "choose exact unique IDs visible in each grid"

    x5_grid = nav2_contact_sheet(x5_candidates)
    x20_grid = nav2_contact_sheet(x20_candidates)
    if save_navigation_pngs:
        _ = save_image(
            x5_grid,
            artifact_root / f"round_{round_number}_nav4_x5_root_grid.png",
        )
        _ = save_image(
            x20_grid,
            artifact_root / f"round_{round_number}_nav4_x20_root_grid.png",
        )

    call = client.generate_json(
        role="navigator",
        images=(x5_grid, x20_grid),
        image_labels=("x5-root-grid", "x20-root-grid"),
        system_prompt=(
            "You are Navigator. Select independent x5 and x20 pathology "
            "evidence from the two supplied visual candidate grids; never diagnose."
        ),
        user_prompt=_prompt(
            evidence_plan,
            x5_count=len(x5_candidates),
            x20_count=len(x20_candidates),
        ),
        output_adapter=TypeAdapter(Nav2Selection),
        final_tokens=512,
        semantic_validator=validate,
        fallback_factory=fallback,
        fallback_on_parse_failure=True,
    )
    selection = (
        nav2_cross_scale_selection(call.value, x5_candidates, x20_candidates)
        if os.environ.get("VLMAS_CROSS_SCALE_ROUTER", "0") == "1"
        else call.value
    )
    chosen = (
        *(x5_candidates[index - 1] for index in selection.x5_ids),
        *(x20_candidates[index - 1] for index in selection.x20_ids),
    )
    anchor_ids: dict[str, PatchId] = {}
    patches = tuple(
        nav2_patch(
            observation,
            rank=rank,
            round_number=round_number,
            anchor_ids=anchor_ids,
            artifact_root=artifact_root,
        )
        for rank, observation in enumerate(chosen, start=1)
    )
    trace = SelectionTrace(
        label=f"round-{round_number}-nav4-independent-root-grids",
        ranked_ids=selection.x5_ids,
        skipped_low_tissue_ids=(),
        selected_ids=selection.x5_ids,
    )
    return patches, (call.record,), trace


__all__ = ["run_nav4_navigation"]
