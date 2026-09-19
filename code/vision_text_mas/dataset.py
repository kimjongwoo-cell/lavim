"""Validated WSI-VQA dataset and deterministic slide resolution boundaries."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from vision_text_mas.contracts import CaseInput, DatasetItem
from vision_text_mas.errors import FailureCode, PipelineFailure


def resolve_slide(slide_root: Path, slide_id: str) -> Path:
    """Resolve an explicit flat image or exactly one recursive TCGA DX1 slide."""
    if not slide_root.is_dir():
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="dataset",
            detail=f"slide root is not a directory: {slide_root}",
        )
    explicit_image = slide_root / slide_id
    if explicit_image.is_file() or explicit_image.is_dir():
        return explicit_image
    candidates = tuple(sorted(slide_root.rglob(f"{slide_id}*DX1*.svs")))
    if len(candidates) != 1:
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="dataset",
            detail=(
                f"expected exactly one DX1 slide for {slide_id}, "
                f"found {len(candidates)}"
            ),
        )
    return candidates[0]


def load_cases(
    dataset_path: Path,
    *,
    slide_root: Path,
    indices: tuple[int, ...],
) -> tuple[CaseInput, ...]:
    """Parse the complete dataset and resolve requested rows in caller order."""
    if len(set(indices)) != len(indices):
        raise PipelineFailure(
            code=FailureCode.SELECTION_CONTRACT,
            stage="dataset",
            detail="dataset indices must be unique",
        )
    try:
        raw = dataset_path.read_bytes()
    except OSError as error:
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="dataset",
            detail=f"failed to read {dataset_path}: {error}",
        ) from error
    try:
        items = TypeAdapter(list[DatasetItem]).validate_json(raw)
    except ValidationError as error:
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="dataset",
            detail=f"invalid dataset contract in {dataset_path}: {error}",
        ) from error
    cases: list[CaseInput] = []
    for index in indices:
        if index < 0 or index >= len(items):
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="dataset",
                detail=f"dataset index {index} is outside 0..{len(items) - 1}",
            )
        item = items[index]
        cases.append(
            CaseInput(
                dataset_index=index,
                item=item,
                slide_path=resolve_slide(slide_root, item.slide_id),
            )
        )
    return tuple(cases)
