"""One-call Navigator plus deterministic multi-scale materialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import final

from PIL import Image

from vision_text_mas.contracts import EvidenceRound, NavigationAction, RoleCall
from vision_text_mas.navigation.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.navigation.navigation_render import SlideLike, save_image
from vision_text_mas.navigation.onepass_navigation_materializer import (
    OneShotMaterialization,
    materialize_one_shot_bundle,
    prepare_root_grid,
)
from vision_text_mas.navigation.onepass_navigation_roots import materialize_x5_patches
from vision_text_mas.navigation.onepass_navigation_twohop import (
    materialize_x20_patches_twohop,
)
from vision_text_mas.role_protocols import OneShotNavigationRole


@final
class OnePassNavigator:
    """Call Navigator once, then crop every planned patch deterministically."""

    def __init__(self, *, agent: OneShotNavigationRole, artifact_root: Path) -> None:
        self._agent = agent
        self._artifact_root = artifact_root
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
        root_grid = (
            cached_root_grid(thumbnail)
            if callable(cached_root_grid)
            else prepare_root_grid(slide, thumbnail=thumbnail)
        )
        _ = save_image(
            root_grid.image,
            self._artifact_root / f"round_{round_number}_roots.png",
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
