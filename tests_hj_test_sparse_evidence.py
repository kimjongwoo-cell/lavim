"""Sparse slides may contribute fewer patches than the requested budget."""

from pathlib import Path

from vision_text_mas.contracts import (
    EvidenceRound,
    NavigationAction,
    PatchMetadata,
    PatchId,
)
from vision_text_mas.geometry import Box, Magnification


def test_evidence_round_accepts_actual_sparse_patch_count() -> None:
    patches = tuple(
        PatchMetadata(
            patch_id=PatchId(f"R1-P{rank}"),
            round_number=1,
            rank=rank,
            action=NavigationAction.INITIAL,
            magnification=Magnification.X5,
            box=Box(x=rank * 10, y=0, width=10, height=10),
            root_id=rank,
            source_x5_anchor=PatchId(f"R1-P{rank}"),
            candidate_id=rank,
            tissue_fraction=0.5,
            image_path=Path(f"patch-{rank}.png"),
        )
        for rank in range(1, 4)
    )

    result = EvidenceRound(
        round_number=1,
        action=NavigationAction.INITIAL,
        patches=patches,
    )

    assert len(result.patches) == 3
