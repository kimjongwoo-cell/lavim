"""Independent root selection for x5 overview and x20 detail evidence."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Annotated, TypedDict

from PIL import Image
from pydantic import BeforeValidator, Field, TypeAdapter

from vision_text_mas.contracts import FrozenModel, PatchMetadata, RoleCall
from vision_text_mas.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.geometry import Magnification, eligible_tissue_ids, zoom_grid
from vision_text_mas.navigation_render import (
    SlideLike,
    numbered_grid,
    render_box,
    save_image,
    tissue_fractions,
)
from vision_text_mas.onepass_navigation_contracts import (
    JsonValue,
    OneShotNavigation,
)
from vision_text_mas.onepass_navigation_details import _DetailGrid
from vision_text_mas.onepass_navigation_details import materialize_x20_patches
from vision_text_mas.onepass_navigation_roots import (
    MINIMUM_TISSUE_FRACTION,
    RootGrid,
    materialize_x5_patches,
)
from vision_text_mas.qwen_client import QwenJsonClient
from vision_text_mas.onepass_navigation_scale_shift import shift_nav3_scales


class IndependentSelectionPayload(TypedDict):
    """JSON-compatible Navigator output after deterministic repair."""

    x5_root_ids: list[int]
    x20_root_ids: list[int]
    selection_reason: str


class IndependentScaleSelection(FrozenModel):
    """Two root choices that share a slide grid but not selected anchors."""

    x5_root_ids: tuple[int, ...] = Field(min_length=1, max_length=5)
    x20_root_ids: tuple[int, ...] = Field(min_length=1, max_length=5)
    selection_reason: str = Field(min_length=1)


class IndependentRootDetail(FrozenModel):
    """Two detail cells selected after viewing one independent x20 root."""

    root_id: int = Field(ge=1)
    detail_cell_ids: tuple[int, ...] = Field(min_length=2, max_length=2)


class IndependentDetailSelection(FrozenModel):
    """Per-root x20 choices from the second visual Navigator hop."""

    roots: tuple[IndependentRootDetail, ...] = Field(min_length=1, max_length=5)
    selection_reason: str = Field(min_length=1)


def interleave_detail_cells(
    selections: tuple[IndependentRootDetail, ...],
) -> tuple[int, ...]:
    """Preserve each root's choice through the round-robin materializer."""
    return tuple(
        selection.detail_cell_ids[pass_index]
        for pass_index in range(2)
        for selection in selections
    )


def _integer_ids(value: JsonValue | None, allowed: frozenset[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool) and item in allowed
    ]


def _fill_unique(values: list[int], fallback: tuple[int, ...], count: int) -> list[int]:
    result = list(dict.fromkeys(values))[:count]
    for candidate_id in fallback:
        if len(result) >= count:
            break
        if candidate_id not in result:
            result.append(candidate_id)
    return result


def repair_independent_selection(
    value: JsonValue,
    *,
    eligible_root_ids: frozenset[int],
    x5_count: int,
    x20_anchor_count: int,
) -> IndependentSelectionPayload:
    """Repair both root arrays separately, preserving independent choices."""
    raw = value if isinstance(value, dict) else {}
    root_fallback = tuple(sorted(eligible_root_ids))
    reason = raw.get("selection_reason")
    return IndependentSelectionPayload(
        x5_root_ids=_fill_unique(
            _integer_ids(raw.get("x5_root_ids"), eligible_root_ids),
            root_fallback,
            x5_count,
        ),
        x20_root_ids=_fill_unique(
            _integer_ids(raw.get("x20_root_ids"), eligible_root_ids),
            root_fallback,
            x20_anchor_count,
        ),
        selection_reason=(
            reason
            if isinstance(reason, str) and reason.strip()
            else "deterministic eligible-tissue fallback"
        ),
    )


def _prompt(plan: EvidencePlan, eligible_ids: frozenset[int]) -> str:
    scale = plan.scale_plan
    template = json.dumps(
        {
            "x5_root_ids": [1],
            "x20_root_ids": [2],
            "selection_reason": "visible evidence",
        }
    )
    return (
        "Validated evidence plan: (carried in latent state)\n"
        f"Question focus: {plan.question_focus}\n"
        f"x5 overview target: {scale.overview_target}\n"
        f"x20 detail target: {scale.detail_target}\n"
        f"Eligible root IDs: {sorted(eligible_ids)}. "
        f"Choose exactly {scale.overview_patch_count} roots for final x5 evidence "
        f"and independently choose exactly {scale.detail_anchor_count} roots in "
        "which x20 evidence should be sampled. The two root lists may overlap, "
        "but an x20 root must not be derived from or restricted to the x5 list. "
        "Do not choose x20 cells yet; each chosen x20 root will be inspected in "
        "a separate visual step. Prefer diagnostic tissue and spatial "
        "diversity. Never diagnose. Output only JSON shaped as: "
        f"{template}"
    )


def _detail_grids(
    slide: SlideLike,
    *,
    anchors: tuple[PatchMetadata, ...],
    artifact_root: Path,
    round_number: int,
    save_navigation_pngs: bool = True,
) -> tuple[_DetailGrid, ...]:
    grids: list[_DetailGrid] = []
    for anchor in anchors:
        boxes = zoom_grid(anchor.box, Magnification.X20)
        fractions = tissue_fractions(slide, boxes)
        eligible = eligible_tissue_ids(
            fractions,
            minimum_fraction=MINIMUM_TISSUE_FRACTION,
        )
        preview = render_box(slide, anchor.box, max_side=768)
        numbered_preview = numbered_grid(
            preview.copy(),
            parent=anchor.box,
            boxes=boxes,
            eligible_ids=eligible,
        )
        if save_navigation_pngs:
            _ = save_image(
                numbered_preview,
                artifact_root
                / f"round_{round_number}_nav3_root_{int(anchor.root_id)}_x20_grid.png",
            )
        grids.append(
            _DetailGrid(
                boxes=boxes,
                fractions=fractions,
                eligible_ids=eligible,
                preview=preview,
                numbered_preview=numbered_preview,
            )
        )
    return tuple(grids)


def _detail_prompt(
    plan: EvidencePlan,
    anchors: tuple[PatchMetadata, ...],
    grids: tuple[_DetailGrid, ...],
) -> str:
    root_lines = "; ".join(
        f"root {int(anchor.root_id)} eligible cells {sorted(grid.eligible_ids)}"
        for anchor, grid in zip(anchors, grids, strict=True)
    )
    template = json.dumps(
        {
            "roots": [{"root_id": 1, "detail_cell_ids": [3, 12]}],
            "selection_reason": "visible x20 evidence",
        }
    )
    return (
        f"Question focus: {plan.question_focus}\n"
        f"x20 detail target: {plan.scale_plan.detail_target}\n"
        "Images are supplied as an x5 anchor followed by its numbered 4x4 x20 "
        "grid, repeated once per independent root. Select exactly two distinct "
        "tissue-eligible cells separately for every root based on its own visible "
        "morphology. Do not reuse a global cell-number pattern across roots. "
        f"{root_lines}. Never diagnose. Output only JSON shaped as: {template}"
    )


def _select_details(
    client: QwenJsonClient,
    *,
    slide: SlideLike,
    anchors: tuple[PatchMetadata, ...],
    grids: tuple[_DetailGrid, ...],
    evidence_plan: EvidencePlan,
) -> tuple[IndependentDetailSelection, RoleCall]:
    root_ids = tuple(int(anchor.root_id) for anchor in anchors)

    def fallback(
        _outputs: tuple[str, ...], _error: str
    ) -> IndependentDetailSelection:
        roots = tuple(
            IndependentRootDetail(
                root_id=root_id,
                detail_cell_ids=(eligible[0], eligible[-1]),
            )
            for root_id, grid in zip(root_ids, grids, strict=True)
            if len(eligible := sorted(grid.eligible_ids)) >= 2
        )
        return IndependentDetailSelection(
            roots=roots,
            selection_reason="deterministic per-root tissue fallback",
        )

    def validate(output: IndependentDetailSelection) -> str | None:
        if tuple(item.root_id for item in output.roots) != root_ids:
            return "return every listed root exactly once and in the listed order"
        for item, grid in zip(output.roots, grids, strict=True):
            if len(set(item.detail_cell_ids)) != 2:
                return "choose two distinct cells for every root"
            if set(item.detail_cell_ids).difference(grid.eligible_ids):
                return "choose only tissue-eligible cells shown for each root"
        return None

    images = tuple(
        image
        for grid in grids
        for image in (grid.preview, grid.numbered_preview)
    )
    labels = tuple(
        label
        for root_id in root_ids
        for label in (f"root-{root_id}-x5", f"root-{root_id}-x20-grid")
    )
    call = client.generate_json(
        role="navigator_detail",
        images=images,
        image_labels=labels,
        system_prompt=(
            "You are Navigator. Inspect each independent pathology root and "
            "select its own x20 evidence cells; never diagnose."
        ),
        user_prompt=_detail_prompt(evidence_plan, anchors, grids),
        output_adapter=TypeAdapter(IndependentDetailSelection),
        final_tokens=384,
        semantic_validator=validate,
        fallback_factory=fallback,
        fallback_on_parse_failure=True,
    )
    return call.value, call.record


def _materialize_roots(
    slide: SlideLike,
    *,
    selection: IndependentScaleSelection,
    root_grid: RootGrid,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
    save_navigation_pngs: bool,
) -> tuple[tuple[PatchMetadata, ...], tuple[PatchMetadata, ...]]:
    x5_navigation = OneShotNavigation(
        root_ids=selection.x5_root_ids,
        detail_cell_ids=(1,),
        selection_reason=selection.selection_reason,
    )
    x5_patches = materialize_x5_patches(
        slide,
        root_grid=root_grid,
        navigation=x5_navigation,
        artifact_root=artifact_root,
        round_number=round_number,
        allow_partial=True,
        save_navigation_pngs=save_navigation_pngs,
    )
    x20_navigation = OneShotNavigation(
        root_ids=selection.x20_root_ids,
        detail_cell_ids=(1,),
        selection_reason=selection.selection_reason,
    )
    navigation_anchors = materialize_x5_patches(
        slide,
        root_grid=root_grid,
        navigation=x20_navigation,
        artifact_root=artifact_root / "nav3_x20_navigation_only",
        round_number=round_number,
        allow_partial=True,
        save_navigation_pngs=save_navigation_pngs,
    )
    return x5_patches, navigation_anchors


def _materialize_details(
    slide: SlideLike,
    *,
    x5_patches: tuple[PatchMetadata, ...],
    navigation_anchors: tuple[PatchMetadata, ...],
    detail_selection: IndependentDetailSelection,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
    prepared_grids: tuple[_DetailGrid, ...],
) -> tuple[PatchMetadata, ...]:
    navigation = OneShotNavigation(
        root_ids=tuple(int(anchor.root_id) for anchor in navigation_anchors),
        detail_cell_ids=interleave_detail_cells(detail_selection.roots),
        selection_reason=detail_selection.selection_reason,
    )
    raw_details = materialize_x20_patches(
        slide,
        x5_patches=navigation_anchors,
        navigation=navigation,
        evidence_plan=evidence_plan,
        artifact_root=artifact_root,
        round_number=round_number,
        prepared_grids=prepared_grids,
    )
    details = tuple(
        patch.model_copy(update={"source_x5_anchor": patch.patch_id})
        for patch in raw_details
    )
    return (*x5_patches, *details)


def run_nav3_navigation(
    slide: SlideLike,
    *,
    client: QwenJsonClient,
    thumbnail: Image.Image,
    root_grid: RootGrid,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
    save_navigation_pngs: bool = True,
) -> tuple[tuple[PatchMetadata, ...], tuple[RoleCall, ...], SelectionTrace]:
    """Select independent x5 and x20 roots in one visual model call."""
    scale = evidence_plan.scale_plan
    repair = partial(
        repair_independent_selection,
        eligible_root_ids=root_grid.eligible_ids,
        x5_count=scale.overview_patch_count,
        x20_anchor_count=scale.detail_anchor_count,
    )

    def fallback(outputs: tuple[str, ...], _error: str) -> IndependentScaleSelection:
        _ = outputs
        return IndependentScaleSelection.model_validate(repair({}))

    call = client.generate_json(
        role="navigator",
        images=(thumbnail, root_grid.image),
        image_labels=("whole-slide-thumbnail", "root-grid"),
        system_prompt=(
            "You are Navigator. Select independent x5 overview roots and x20 "
            "detail-search roots from one pathology slide; never diagnose."
        ),
        user_prompt=_prompt(evidence_plan, root_grid.eligible_ids),
        output_adapter=TypeAdapter(
            Annotated[IndependentScaleSelection, BeforeValidator(repair)]
        ),
        final_tokens=512,
        fallback_factory=fallback,
        fallback_on_parse_failure=True,
    )
    x5_patches, navigation_anchors = _materialize_roots(
        slide,
        selection=call.value,
        root_grid=root_grid,
        evidence_plan=evidence_plan,
        artifact_root=artifact_root,
        round_number=round_number,
        save_navigation_pngs=save_navigation_pngs,
    )
    grids = _detail_grids(
        slide,
        anchors=navigation_anchors,
        artifact_root=artifact_root,
        round_number=round_number,
        save_navigation_pngs=save_navigation_pngs,
    )
    detail_selection, detail_record = _select_details(
        client,
        slide=slide,
        anchors=navigation_anchors,
        grids=grids,
        evidence_plan=evidence_plan,
    )
    patches = _materialize_details(
        slide,
        x5_patches=x5_patches,
        navigation_anchors=navigation_anchors,
        detail_selection=detail_selection,
        evidence_plan=evidence_plan,
        artifact_root=artifact_root,
        round_number=round_number,
        prepared_grids=grids,
    )
    import os

    if os.environ.get("VLMAS_NAV3_X20_X40", "0") == "1":
        patches = shift_nav3_scales(
            slide,
            patches,
            artifact_root=artifact_root,
            round_number=round_number,
        )
    trace = SelectionTrace(
        label=f"round-{round_number}-nav3-independent-scales",
        ranked_ids=call.value.x5_root_ids,
        skipped_low_tissue_ids=(),
        selected_ids=call.value.x5_root_ids,
    )
    return patches, (call.record, detail_record), trace
