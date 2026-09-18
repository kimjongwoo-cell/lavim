"""Compose deterministic x5 and x20 rendering after one Navigator call."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_text_mas.contracts import PatchMetadata
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.navigation_render import SlideLike
from vision_text_mas.onepass_navigation_contracts import OneShotNavigation
from vision_text_mas.onepass_navigation_details import materialize_x20_patches
from vision_text_mas.onepass_navigation_roots import (
    RootGrid,
    materialize_x5_patches,
    prepare_root_grid,
)


@dataclass(frozen=True, slots=True)
class OneShotMaterialization:
    """Final evidence bundle plus the tissue-safe roots actually used."""

    patches: tuple[PatchMetadata, ...]
    root_ids: tuple[int, ...]


def materialize_one_shot_bundle(
    slide: SlideLike,
    *,
    root_grid: RootGrid,
    navigation: OneShotNavigation,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
) -> OneShotMaterialization:
    """Render all planned patches without another model invocation."""
    x5_patches = materialize_x5_patches(
        slide,
        root_grid=root_grid,
        navigation=navigation,
        artifact_root=artifact_root,
        round_number=round_number,
    )
    x20_patches = materialize_x20_patches(
        slide,
        x5_patches=x5_patches,
        navigation=navigation,
        evidence_plan=evidence_plan,
        artifact_root=artifact_root,
        round_number=round_number,
    )
    return OneShotMaterialization(
        patches=(*x5_patches, *x20_patches),
        root_ids=tuple(int(patch.root_id) for patch in x5_patches),
    )


__all__ = [
    "OneShotMaterialization",
    "RootGrid",
    "materialize_one_shot_bundle",
    "prepare_root_grid",
]
