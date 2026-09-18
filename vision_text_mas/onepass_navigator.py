"""One-call Navigator plus deterministic multi-scale materialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast, final

from PIL import Image

from vision_text_mas.contracts import EvidenceRound, NavigationAction, RoleCall
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.navigation_render import SlideLike, save_image
from vision_text_mas.onepass_navigation_materializer import (
    OneShotMaterialization,
    materialize_one_shot_bundle,
    prepare_root_grid,
)
from vision_text_mas.onepass_navigation_nav2 import run_nav2_navigation
from vision_text_mas.onepass_navigation_nav3 import run_nav3_navigation
from vision_text_mas.onepass_navigation_nav4 import run_nav4_navigation
from vision_text_mas.onepass_navigation_roots import RootGrid, materialize_x5_patches
from vision_text_mas.onepass_navigation_search import run_search_navigation
from vision_text_mas.onepass_navigation_twohop import materialize_x20_patches_twohop
from vision_text_mas.role_protocols import OneShotNavigationRole


@final
class OnePassNavigator:
    """Call Navigator once, then crop every planned patch deterministically."""

    def __init__(
        self,
        *,
        agent: OneShotNavigationRole,
        artifact_root: Path,
        save_navigation_pngs: bool = True,
    ) -> None:
        self._agent = agent
        self._artifact_root = artifact_root
        self._save_navigation_pngs = save_navigation_pngs
        self._records: tuple[RoleCall, ...] = ()
        self._traces: tuple[SelectionTrace, ...] = ()

    @property
    def records(self) -> tuple[RoleCall, ...]:
        return self._records

    @property
    def traces(self) -> tuple[SelectionTrace, ...]:
        return self._traces

    def initial(
        self,
        slide: SlideLike,
        *,
        thumbnail: Image.Image,
        evidence_plan: EvidencePlan,
        round_number: int,
    ) -> EvidenceRound:
        """Materialize the one planned eight-patch bundle without a retry loop."""
        cached_root_grid = getattr(slide, "cached_root_grid", None)
        root_grid: RootGrid = (
            cast(RootGrid, cached_root_grid(thumbnail))
            if callable(cached_root_grid)
            else prepare_root_grid(slide, thumbnail=thumbnail)
        )
        if self._save_navigation_pngs:
            _ = save_image(
                root_grid.image,
                self._artifact_root / f"round_{round_number}_roots.png",
            )
        if os.environ.get("VLMAS_NAV4", "0") == "1":
            nav4_client = getattr(self._agent, "_client", None)
            if nav4_client is None:
                raise PipelineFailure(
                    code=FailureCode.MODEL_EXECUTION,
                    stage="navigator",
                    detail="Nav4 requires a visual JSON client",
                )
            patches, records, trace = run_nav4_navigation(
                slide,
                client=nav4_client,
                root_grid=root_grid,
                evidence_plan=evidence_plan,
                artifact_root=self._artifact_root,
                round_number=round_number,
                save_navigation_pngs=self._save_navigation_pngs,
            )
            self._records = records
            self._traces = (trace,)
            return EvidenceRound(
                round_number=round_number,
                action=NavigationAction.INITIAL,
                patches=patches,
            )
        if os.environ.get("VLMAS_NAV3_INDEPENDENT", "0") == "1":
            nav3_client = getattr(self._agent, "_client", None)
            if nav3_client is None:
                raise PipelineFailure(
                    code=FailureCode.MODEL_EXECUTION,
                    stage="navigator",
                    detail="Nav3 requires a visual JSON client",
                )
            patches, records, trace = run_nav3_navigation(
                slide,
                client=nav3_client,
                thumbnail=thumbnail,
                root_grid=root_grid,
                evidence_plan=evidence_plan,
                artifact_root=self._artifact_root,
                round_number=round_number,
                save_navigation_pngs=self._save_navigation_pngs,
            )
            self._records = records
            self._traces = (trace,)
            return EvidenceRound(
                round_number=round_number,
                action=NavigationAction.INITIAL,
                patches=patches,
            )
        if os.environ.get("VLMAS_NAV2", "0") == "1":
            nav2_client = getattr(self._agent, "_client", None)
            if nav2_client is None:
                raise PipelineFailure(
                    code=FailureCode.MODEL_EXECUTION,
                    stage="navigator",
                    detail="Nav2 requires a visual JSON client",
                )
            patches, record, trace = run_nav2_navigation(
                slide,
                client=nav2_client,
                root_grid=root_grid,
                evidence_plan=evidence_plan,
                artifact_root=self._artifact_root,
                round_number=round_number,
            )
            self._records = (record,)
            self._traces = (trace,)
            return EvidenceRound(
                round_number=round_number,
                action=NavigationAction.INITIAL,
                patches=patches,
            )
        search_client = (
            getattr(self._agent, "_client", None)
            if os.environ.get("VLMAS_NAV_SEARCH", "0") == "1"
            else None
        )
        if search_client is not None:
            # VLMAS_NAV_SEARCH=1: the Navigator becomes a controller over a
            # frozen WsiSearchTool (query -> x5 -> observe -> query -> x20)
            # and the grid-pick select call below never happens.
            patches, records, trace = run_search_navigation(
                slide,
                client=search_client,
                root_grid=root_grid,
                evidence_plan=evidence_plan,
                artifact_root=self._artifact_root,
                round_number=round_number,
            )
            self._records = records
            self._traces = (trace,)
            return EvidenceRound(
                round_number=round_number,
                action=NavigationAction.INITIAL,
                patches=patches,
            )
        call = self._agent.select(
            thumbnail=thumbnail,
            root_grid=root_grid.image,
            evidence_plan=evidence_plan,
            root_candidate_ids=root_grid.eligible_ids,
        )
        self._records = (call.record,)
        two_hop_client = (
            getattr(self._agent, "_client", None)
            if os.environ.get("VLMAS_NAV_TWOHOP", "0") == "1"
            else None
        )
        if two_hop_client is not None:
            x5_patches = materialize_x5_patches(
                slide,
                root_grid=root_grid,
                navigation=call.value,
                artifact_root=self._artifact_root,
                round_number=round_number,
            )
            x20_patches, hop_records = materialize_x20_patches_twohop(
                slide,
                client=two_hop_client,
                x5_patches=x5_patches,
                navigation=call.value,
                evidence_plan=evidence_plan,
                artifact_root=self._artifact_root,
                round_number=round_number,
            )
            self._records = (call.record, *hop_records)
            materialized = OneShotMaterialization(
                patches=(*x5_patches, *x20_patches),
                root_ids=tuple(int(patch.root_id) for patch in x5_patches),
            )
        else:
            materialized = materialize_one_shot_bundle(
                slide,
                root_grid=root_grid,
                navigation=call.value,
                evidence_plan=evidence_plan,
                artifact_root=self._artifact_root,
                round_number=round_number,
            )
        actual_roots = materialized.root_ids
        self._traces = (
            SelectionTrace(
                label=f"round-{round_number}-one-shot-root-grid",
                ranked_ids=call.value.root_ids,
                skipped_low_tissue_ids=tuple(
                    root_id
                    for root_id in call.value.root_ids
                    if root_id not in actual_roots
                ),
                selected_ids=actual_roots,
            ),
        )
        return EvidenceRound(
            round_number=round_number,
            action=NavigationAction.INITIAL,
            patches=materialized.patches,
        )
