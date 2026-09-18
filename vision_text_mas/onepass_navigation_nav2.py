"""Nav2: one Navigator call jointly selects global x5 and x20 evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import Field, TypeAdapter

from vision_text_mas.contracts import (
    CandidateId,
    FrozenModel,
    NavigationAction,
    PatchId,
    PatchMetadata,
    RoleCall,
    RootCellId,
)
from vision_text_mas.geometry import Magnification
from vision_text_mas.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.navigation_render import SlideLike, save_image
from vision_text_mas.onepass_navigation_roots import RootGrid
from vision_text_mas.qwen_client import QwenJsonClient
from vision_text_mas.wsi_search_tool import TissueSearchTool, WSIObservation

MAX_X20_CANDIDATES = 72
CONTACT_COLUMNS = 8
CONTACT_TILE = 144


class Nav2Selection(FrozenModel):
    """Independent global selections for the two magnifications."""

    x5_ids: tuple[int, ...] = Field(min_length=1, max_length=8)
    x20_ids: tuple[int, ...] = Field(min_length=1, max_length=20)
    selection_reason: str = Field(min_length=1)


def _fallback_ids(*, required: int, available: int) -> tuple[int, ...]:
    """Return every available in-range ID up to the requested maximum."""
    if available < 1:
        raise RuntimeError("Nav2 requires at least one tissue candidate")
    return tuple(range(1, min(required, available) + 1))


def _cross_scale_selection(
    selection: Nav2Selection,
    x5_candidates: tuple[WSIObservation, ...],
    x20_candidates: tuple[WSIObservation, ...],
) -> Nav2Selection:
    """Align overview context with the Navigator's selected detail evidence."""
    x5_id_by_crop = {
        candidate.crop_id: candidate_id
        for candidate_id, candidate in enumerate(x5_candidates, start=1)
    }
    support_count: dict[int, int] = {}
    first_detail_rank: dict[int, int] = {}
    for detail_rank, detail_id in enumerate(selection.x20_ids):
        parent_crop_id = x20_candidates[detail_id - 1].parent_crop_id
        if parent_crop_id is None or parent_crop_id not in x5_id_by_crop:
            continue
        parent_id = x5_id_by_crop[parent_crop_id]
        support_count[parent_id] = support_count.get(parent_id, 0) + 1
        first_detail_rank.setdefault(parent_id, detail_rank)

    supported = sorted(
        support_count,
        key=lambda candidate_id: (
            -support_count[candidate_id],
            first_detail_rank[candidate_id],
            candidate_id,
        ),
    )
    routed = supported[: len(selection.x5_ids)]
    routed.extend(
        candidate_id
        for candidate_id in selection.x5_ids
        if candidate_id not in routed
    )
    return Nav2Selection(
        x5_ids=tuple(routed[: len(selection.x5_ids)]),
        x20_ids=selection.x20_ids,
        selection_reason=selection.selection_reason,
    )


def _contact_sheet(candidates: tuple[WSIObservation, ...]) -> Image.Image:
    """Render actual candidate crops with stable one-based IDs."""
    rows = (len(candidates) + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS
    sheet = Image.new(
        "RGB",
        (CONTACT_COLUMNS * CONTACT_TILE, rows * CONTACT_TILE),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for candidate_id, candidate in enumerate(candidates, start=1):
        col = (candidate_id - 1) % CONTACT_COLUMNS
        row = (candidate_id - 1) // CONTACT_COLUMNS
        crop = candidate.image.convert("RGB")
        crop.thumbnail((CONTACT_TILE, CONTACT_TILE))
        x = col * CONTACT_TILE
        y = row * CONTACT_TILE
        sheet.paste(crop, (x, y))
        draw.rectangle((x, y, x + 34, y + 20), fill="white")
        draw.text((x + 3, y + 2), str(candidate_id), fill="black")
    return sheet


def _global_candidates(
    slide: SlideLike,
    *,
    root_grid: RootGrid,
) -> tuple[tuple[WSIObservation, ...], tuple[WSIObservation, ...]]:
    """Build tissue-safe pools without conditioning x20 on selected x5."""
    tool = TissueSearchTool(root_grid=root_grid)
    x5 = tool.search_coarse(
        slide,
        query="",
        top_k=len(root_grid.eligible_ids),
    )
    x20 = tuple(
        child
        for parent in x5
        for child in tool.search_fine(slide, parent=parent, query="", top_k=2)
    )[:MAX_X20_CANDIDATES]
    return x5, x20


def _prompt(plan: EvidencePlan, *, x5_count: int, x20_count: int) -> str:
    """Build the established Nav2 selection prompt."""
    scale = plan.scale_plan
    template = json.dumps(
        {"x5_ids": [1], "x20_ids": [1], "selection_reason": "visible evidence"}
    )
    return (
        "Validated evidence plan: (carried in latent state)\n"
        f"Question focus: {plan.question_focus}\n"
        f"x5 evidence target: {scale.overview_target}\n"
        f"x20 evidence target: {scale.detail_target}\n"
        f"Image 1 contains {x5_count} global x5 candidates. Image 2 contains "
        f"{x20_count} global x20 candidates. IDs restart at 1 in each image. "
        f"Choose exactly {min(scale.overview_patch_count, x5_count)} x5 IDs and "
        f"{min(scale.detail_patch_count, x20_count)} x20 IDs by inspecting the "
        "actual crops. These are maxima: do not duplicate IDs when a sheet has "
        "fewer candidates. "
        "The x20 choices are independent and need not lie inside a chosen x5. "
        "Prefer diagnostic tissue, spatial diversity, and complementary evidence. "
        f"Do not diagnose. Output only JSON shaped as: {template}"
    )


def _patch(
    observation: WSIObservation,
    *,
    rank: int,
    round_number: int,
    anchor_ids: dict[str, PatchId],
    artifact_root: Path,
) -> PatchMetadata:
    patch_id = PatchId(f"R{round_number}-P{rank}")
    parent = observation.parent_crop_id
    anchor = patch_id if parent is None else anchor_ids.get(parent, patch_id)
    if parent is None:
        anchor_ids[observation.crop_id] = patch_id
    path = save_image(
        observation.image,
        artifact_root / f"round_{round_number}" / f"{patch_id}.png",
    )
    return PatchMetadata(
        patch_id=patch_id,
        round_number=round_number,
        rank=rank,
        action=NavigationAction.INITIAL,
        magnification=Magnification(observation.scale),
        box=observation.box,
        root_id=RootCellId(observation.root_id),
        source_x5_anchor=anchor,
        candidate_id=CandidateId(observation.candidate_id),
        tissue_fraction=observation.tissue_fraction,
        image_path=path,
    )


def run_nav2_navigation(
    slide: SlideLike,
    *,
    client: QwenJsonClient,
    root_grid: RootGrid,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
) -> tuple[tuple[PatchMetadata, ...], RoleCall, SelectionTrace]:
    """Select the final 4+8 bank in one visual Navigator call."""
    x5_candidates, x20_candidates = _global_candidates(slide, root_grid=root_grid)
    scale = evidence_plan.scale_plan
    x5_target = min(scale.overview_patch_count, len(x5_candidates))
    x20_target = min(scale.detail_patch_count, len(x20_candidates))

    def fallback(_outputs: tuple[str, ...], _error: str) -> Nav2Selection:
        return Nav2Selection(
            x5_ids=_fallback_ids(
                required=scale.overview_patch_count,
                available=len(x5_candidates),
            ),
            x20_ids=_fallback_ids(
                required=scale.detail_patch_count,
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
        return None if valid else "choose exact unique IDs visible in each sheet"

    call = client.generate_json(
        role="navigator",
        images=(_contact_sheet(x5_candidates), _contact_sheet(x20_candidates)),
        image_labels=("global-x5-candidates", "global-x20-candidates"),
        system_prompt=(
            "You are Navigator. Jointly select multi-scale pathology evidence "
            "from the supplied candidate sheets; never diagnose."
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
        _cross_scale_selection(call.value, x5_candidates, x20_candidates)
        if os.environ.get("VLMAS_CROSS_SCALE_ROUTER", "0") == "1"
        else call.value
    )
    chosen = (
        *(x5_candidates[index - 1] for index in selection.x5_ids),
        *(x20_candidates[index - 1] for index in selection.x20_ids),
    )
    anchor_ids: dict[str, PatchId] = {}
    patches = tuple(
        _patch(
            observation,
            rank=rank,
            round_number=round_number,
            anchor_ids=anchor_ids,
            artifact_root=artifact_root,
        )
        for rank, observation in enumerate(chosen, start=1)
    )
    trace = SelectionTrace(
        label=f"round-{round_number}-nav2-global-joint",
        ranked_ids=selection.x5_ids,
        skipped_low_tissue_ids=(),
        selected_ids=selection.x5_ids,
    )
    return patches, call.record, trace


# Public helpers shared by Nav4's independent two-grid selector.  Keeping the
# implementation in one place prevents the x5/x20 candidate and provenance
# rules from drifting between navigation variants.
nav2_contact_sheet = _contact_sheet
nav2_cross_scale_selection = _cross_scale_selection
nav2_fallback_ids = _fallback_ids
nav2_global_candidates = _global_candidates
nav2_patch = _patch
