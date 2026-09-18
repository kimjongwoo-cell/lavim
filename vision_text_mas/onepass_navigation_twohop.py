"""Opt-in second Navigator hop: per-anchor x20 selection from the real x5 crop.

Enabled with VLMAS_NAV_TWOHOP=1. Hop 1 stays the existing one-shot call (x5
roots chosen against scale_plan.overview_target). Hop 2 shows each selected x5
anchor crop with its numbered x20 grid and asks the model to pick detail cells
against scale_plan.detail_target, replacing the query-free round-robin
distribution. Tissue-snap (_choose_detail_id) remains the safety net, so an
ineligible model pick degrades to the nearest tissue-safe cell, never a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pydantic import Field, TypeAdapter, field_validator

from vision_text_mas.contracts import (
    CandidateId,
    FrozenModel,
    NavigationAction,
    PatchId,
    PatchMetadata,
    RoleCall,
)
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import Magnification, eligible_tissue_ids, zoom_grid
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.navigation_render import (
    SlideLike,
    numbered_grid,
    render_box,
    save_image,
    tissue_fractions,
)
from vision_text_mas.onepass_navigation_contracts import (
    DETAIL_FALLBACK_IDS,
    OneShotNavigation,
)
from vision_text_mas.onepass_navigation_details import (
    _choose_detail_id,
    _DetailGrid,
)
from vision_text_mas.onepass_navigation_roots import MINIMUM_TISSUE_FRACTION
from vision_text_mas.qwen_client import QwenJsonClient

TWO_HOP_NAVIGATOR_SYSTEM = (
    "You are Navigator. Select x20 detail cells inside one supplied x5 region "
    "against the plan's detail target; never answer or diagnose."
)


class TwoHopDetailSelection(FrozenModel):
    """Model-chosen x20 cells for a single x5 anchor."""

    detail_cell_ids: tuple[int, ...] = Field(min_length=1, max_length=20)
    selection_reason: str = Field(min_length=1)

    @field_validator("detail_cell_ids")
    @classmethod
    def require_x20_grid_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(candidate_id not in range(1, 17) for candidate_id in value):
            raise ValueError("x20 grid IDs must be between 1 and 16")
        return value


def two_hop_detail_prompt(
    *,
    evidence_plan: EvidencePlan,
    eligible_ids: frozenset[int],
    selection_count: int,
) -> str:
    scale = evidence_plan.scale_plan
    eligible_text = ", ".join(str(value) for value in sorted(eligible_ids))
    template = json.dumps(
        {"detail_cell_ids": [1], "selection_reason": "visible match to detail target"}
    )
    return (
        f"Detail target at x20: {scale.detail_target}\n"
        f"Question focus: {evidence_plan.question_focus}\n"
        "Image 1 is one selected x5 region. Image 2 is the same region with a "
        "4x4 numbered x20 grid (1 top-left, 16 bottom-right, row-major). "
        f"Tissue-eligible cell IDs: {eligible_text}.\n"
        f"Choose exactly {selection_count} unique cell IDs whose visible content "
        "best matches the detail target. Tissue amount alone is not relevance. "
        "Do not answer the question or diagnose. Output ONLY JSON with these "
        f"exact keys: {template}"
    )


def _anchor_allocation(
    anchors: tuple[PatchMetadata, ...],
    navigation: OneShotNavigation,
) -> dict[PatchId, list[int]]:
    """Mirror the deterministic round-robin budget split across anchors."""
    allocation: dict[PatchId, list[int]] = {a.patch_id: [] for a in anchors}
    for offset, requested_id in enumerate(navigation.detail_cell_ids):
        allocation[anchors[offset % len(anchors)].patch_id].append(requested_id)
    return allocation


def _query_anchor_cells(
    client: QwenJsonClient,
    *,
    anchor_crop: Image.Image,
    grid_image: Image.Image,
    evidence_plan: EvidencePlan,
    eligible_ids: frozenset[int],
    selection_count: int,
    hop1_ids: list[int],
) -> tuple[list[int], RoleCall]:
    def fallback(_outputs: tuple[str, ...], _last_error: str) -> TwoHopDetailSelection:
        """Keep the hop-1 allocation executable when hop-2 JSON fails."""
        ids = list(dict.fromkeys(hop1_ids))
        for candidate_id in DETAIL_FALLBACK_IDS:
            if len(ids) >= selection_count:
                break
            if candidate_id not in ids:
                ids.append(candidate_id)
        return TwoHopDetailSelection(
            detail_cell_ids=tuple(ids[:selection_count]),
            selection_reason="hop-1 allocation fallback",
        )

    def validate(output: TwoHopDetailSelection) -> str | None:
        if len(output.detail_cell_ids) != selection_count:
            return f"must choose exactly {selection_count} cell IDs"
        if len(set(output.detail_cell_ids)) != selection_count:
            return "cell IDs must be unique"
        return None

    call = client.generate_json(
        role="navigator_detail",
        images=(anchor_crop, grid_image),
        image_labels=("x5-anchor-crop", "x20-grid"),
        system_prompt=TWO_HOP_NAVIGATOR_SYSTEM,
        user_prompt=two_hop_detail_prompt(
            evidence_plan=evidence_plan,
            eligible_ids=eligible_ids,
            selection_count=selection_count,
        ),
        output_adapter=TypeAdapter(TwoHopDetailSelection),
        final_tokens=256,
        semantic_validator=validate,
        fallback_factory=fallback,
        fallback_on_parse_failure=True,
    )
    return list(call.value.detail_cell_ids), call.record


def materialize_x20_patches_twohop(
    slide: SlideLike,
    *,
    client: QwenJsonClient,
    x5_patches: tuple[PatchMetadata, ...],
    navigation: OneShotNavigation,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
) -> tuple[tuple[PatchMetadata, ...], tuple[RoleCall, ...]]:
    """Distribute x20 targets via one extra Navigator query per anchor."""
    anchor_count = evidence_plan.scale_plan.detail_anchor_count
    anchors = x5_patches[:anchor_count]
    allocation = _anchor_allocation(anchors, navigation)

    grids: dict[PatchId, _DetailGrid] = {}
    requests: list[tuple[PatchMetadata, int]] = []
    records: list[RoleCall] = []
    for anchor in anchors:
        boxes = zoom_grid(anchor.box, Magnification.X20)
        fractions = tissue_fractions(slide, boxes)
        eligible = eligible_tissue_ids(
            fractions,
            minimum_fraction=MINIMUM_TISSUE_FRACTION,
        )
        anchor_crop = render_box(slide, anchor.box, max_side=768)
        grid_image = numbered_grid(
            anchor_crop,
            parent=anchor.box,
            boxes=boxes,
            eligible_ids=eligible,
        )
        _ = save_image(
            grid_image,
            artifact_root
            / f"round_{round_number}_anchor_{anchor.patch_id}_x20_grid.png",
        )
        grids[anchor.patch_id] = _DetailGrid(
            boxes=boxes,
            fractions=fractions,
            eligible_ids=eligible,
        )
        hop1_ids = allocation[anchor.patch_id]
        if not hop1_ids:
            continue
        chosen, record = _query_anchor_cells(
            client,
            anchor_crop=anchor_crop,
            grid_image=grid_image,
            evidence_plan=evidence_plan,
            eligible_ids=eligible,
            selection_count=len(hop1_ids),
            hop1_ids=hop1_ids,
        )
        records.append(record)
        requests.extend((anchor, requested_id) for requested_id in chosen)

    used_ids: dict[PatchId, frozenset[int]] = {a.patch_id: frozenset() for a in anchors}
    details: list[PatchMetadata] = []
    for offset, (preferred, requested_id) in enumerate(requests):
        search = (preferred, *(anchor for anchor in anchors if anchor != preferred))
        selected_anchor: PatchMetadata | None = None
        selected_id: int | None = None
        for anchor in search:
            candidate_id = _choose_detail_id(
                grids[anchor.patch_id],
                requested_id=requested_id,
                used_ids=used_ids[anchor.patch_id],
            )
            if candidate_id is not None:
                selected_anchor, selected_id = anchor, candidate_id
                break
        if selected_anchor is None or selected_id is None:
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail="no tissue-safe x20 candidate remains after two-hop selection",
            )
        used_ids[selected_anchor.patch_id] = frozenset(
            (*used_ids[selected_anchor.patch_id], selected_id)
        )
        grid = grids[selected_anchor.patch_id]
        patch_id = PatchId(f"R{round_number}-P{len(x5_patches) + offset + 1}")
        box = grid.boxes[selected_id - 1]
        path = save_image(
            render_box(slide, box, max_side=512),
            artifact_root / f"round_{round_number}" / f"{patch_id}.png",
        )
        details.append(
            PatchMetadata(
                patch_id=patch_id,
                round_number=round_number,
                rank=len(x5_patches) + offset + 1,
                action=NavigationAction.INITIAL,
                magnification=Magnification.X20,
                box=box,
                root_id=selected_anchor.root_id,
                source_x5_anchor=selected_anchor.patch_id,
                candidate_id=CandidateId(selected_id),
                tissue_fraction=grid.fractions[selected_id - 1],
                image_path=path,
            )
        )
    return tuple(details), tuple(records)
