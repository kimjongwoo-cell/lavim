from pathlib import Path

from PIL import Image

import vision_text_mas.onepass_navigation_nav3 as nav3
import vision_text_mas.onepass_navigation_roots as roots
from vision_text_mas.contracts import PatchMetadata
from vision_text_mas.geometry import Box
from vision_text_mas.onepass_navigation_nav3 import (
    IndependentRootDetail,
    interleave_detail_cells,
    repair_independent_selection,
)
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.onepass_navigation_contracts import OneShotNavigation
from vision_text_mas.onepass_navigation_roots import materialize_x5_patches


def test_repairs_x5_and_x20_roots_independently() -> None:
    # Given: the model selects different valid roots for overview and detail.
    raw = {
        "x5_root_ids": [1, 2, 2],
        "x20_root_ids": [5, 6, 6],
        "selection_reason": "visible evidence",
    }

    # When: the selection is repaired to the requested branch budgets.
    repaired = repair_independent_selection(
        raw,
        eligible_root_ids=frozenset({1, 2, 3, 4, 5, 6}),
        x5_count=3,
        x20_anchor_count=3,
    )

    # Then: each root branch is deduplicated and filled without coupling them.
    assert repaired["x5_root_ids"] == [1, 2, 3]
    assert repaired["x20_root_ids"] == [5, 6, 1]


def test_filters_ineligible_roots_in_each_branch() -> None:
    # Given: both branches contain roots outside the tissue-eligible set.
    raw = {
        "x5_root_ids": [9, 4],
        "x20_root_ids": [8, 3],
        "selection_reason": "visible evidence",
    }

    # When: the model output crosses the navigation boundary.
    repaired = repair_independent_selection(
        raw,
        eligible_root_ids=frozenset({2, 3, 4}),
        x5_count=2,
        x20_anchor_count=2,
    )

    # Then: invalid roots are replaced independently with eligible roots.
    assert repaired["x5_root_ids"] == [4, 2]
    assert repaired["x20_root_ids"] == [3, 2]


def test_keeps_each_detail_choice_attached_to_its_independent_root() -> None:
    # Given: the second visual hop chose two cells separately for every root.
    selections = (
        IndependentRootDetail(root_id=8, detail_cell_ids=(2, 13)),
        IndependentRootDetail(root_id=11, detail_cell_ids=(4, 15)),
        IndependentRootDetail(root_id=22, detail_cell_ids=(6, 9)),
        IndependentRootDetail(root_id=29, detail_cell_ids=(7, 12)),
    )

    # When: selections are ordered for the existing round-robin materializer.
    detail_ids = interleave_detail_cells(selections)

    # Then: each pass across anchors retains the cell chosen for that anchor.
    assert detail_ids == (2, 4, 6, 7, 13, 15, 9, 12)


def test_nav3_keeps_available_x5_patches_when_candidates_run_out(
    monkeypatch,
) -> None:
    # Given: one tissue-safe patch can be materialized before candidates run out.
    class Patch:
        box = "available-box"

    calls = iter((Patch(), None))

    def materialize_once_then_fail(*args, **kwargs):
        patch = next(calls)
        if patch is None:
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail="no tissue-safe x5 candidate remains",
            )
        return patch

    monkeypatch.setattr(
        "vision_text_mas.onepass_navigation_roots._materialize_x5",
        materialize_once_then_fail,
    )

    # When: Nav3 explicitly permits a smaller evidence bundle.
    patches = materialize_x5_patches(
        None,
        root_grid=None,
        navigation=OneShotNavigation(
            root_ids=(1, 2),
            detail_cell_ids=(1,),
            selection_reason="test",
        ),
        artifact_root=None,
        round_number=1,
        allow_partial=True,
    )

    # Then: the available evidence is retained instead of failing the case.
    assert len(patches) == 1


def test_nav3_reuses_detail_preview_and_skips_temporary_grid_writes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Given: navigation artifacts are disabled and one independent root is prepared.
    preview = Image.new("RGB", (8, 8), (20, 30, 40))
    numbered = Image.new("RGB", (8, 8), (40, 30, 20))
    rendered: list[Box] = []
    saved: list[Path] = []
    anchor = PatchMetadata.model_construct(box=Box(x=0, y=0, width=2048, height=2048))
    monkeypatch.setattr(nav3, "zoom_grid", lambda *args, **kwargs: (Box(x=0, y=0, width=512, height=512),))
    monkeypatch.setattr(nav3, "tissue_fractions", lambda *args: (0.5,))

    def render_box(slide, box: Box, *, max_side: int) -> Image.Image:
        rendered.append(box)
        return preview.copy()

    monkeypatch.setattr(nav3, "render_box", render_box)
    monkeypatch.setattr(nav3, "numbered_grid", lambda *args, **kwargs: numbered.copy())
    monkeypatch.setattr(nav3, "save_image", lambda _image, path: saved.append(path) or path)

    # When: Nav3 prepares the independent detail grid for the model call.
    grids = nav3._detail_grids(
        None,
        anchors=(anchor,),
        artifact_root=tmp_path,
        round_number=1,
        save_navigation_pngs=False,
    )

    # Then: it renders the root once, keeps the exact model-view images in memory, and writes no temp PNG.
    assert len(rendered) == 1
    assert grids[0].preview.tobytes() == preview.tobytes()
    assert grids[0].numbered_preview.tobytes() == numbered.tobytes()
    assert saved == []


def test_x5_child_overlay_is_skipped_when_navigation_pngs_are_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Given: one eligible child patch and no persistent navigation artifacts.
    child = Box(x=1, y=1, width=8, height=8)
    saved: list[Path] = []
    monkeypatch.setattr(roots, "edge_anchored_grid", lambda *args, **kwargs: (child,))
    monkeypatch.setattr(roots, "tissue_fractions", lambda *args: (0.8,))
    monkeypatch.setattr(roots, "eligible_tissue_ids", lambda *args, **kwargs: frozenset({1}))
    monkeypatch.setattr(roots, "render_box", lambda *args, **kwargs: Image.new("RGB", (8, 8)))
    monkeypatch.setattr(roots, "numbered_grid", lambda *args, **kwargs: Image.new("RGB", (8, 8)))
    monkeypatch.setattr(roots, "save_image", lambda _image, path: saved.append(path) or path)

    # When: the same deterministic child selection runs with artifact saving disabled.
    selected = roots._best_x5_child(
        None,
        root=Box(x=0, y=0, width=4096, height=4096),
        root_id=1,
        used_boxes=frozenset(),
        artifact_root=tmp_path,
        round_number=1,
        save_navigation_pngs=False,
    )

    # Then: the selected child is identical while its unused overlay is not encoded to disk.
    assert selected == (child, 1, 0.8)
    assert saved == []
