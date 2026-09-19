"""Map the fixed 12-patch protocol onto real or shuffled tissue neighborhoods."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import torch

type PatchGroups = tuple[tuple[int, ...], ...]


class GroupMode(StrEnum):
    SPATIAL = "spatial"
    SHUFFLED = "shuffled"


@dataclass(frozen=True, slots=True)
class LayoutError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def make_groups(parents: tuple[int, ...], mode: GroupMode) -> PatchGroups:
    """Preserve patch coverage while optionally breaking every fine-parent link."""
    roots = tuple(i for i, parent in enumerate(parents) if parent < 0)
    children = [
        [i for i, parent in enumerate(parents) if parent == root] for root in roots
    ]
    if len(parents) != 12 or len(roots) != 4 or any(len(row) != 2 for row in children):
        raise LayoutError(
            f"Expected four parents with two children each; got {parents}"
        )
    match mode:
        case GroupMode.SPATIAL:
            assigned = children
        case GroupMode.SHUFFLED:
            shuffled = [child for row in children for child in row]
            rng = random.Random(42)
            for _ in range(1000):
                rng.shuffle(shuffled)
                if all(
                    parents[child] != roots[i // 2] for i, child in enumerate(shuffled)
                ):
                    break
            else:
                raise LayoutError("Could not form the seeded parent-deranged control")
            assigned = [shuffled[i : i + 2] for i in range(0, len(shuffled), 2)]
        case unreachable:
            assert_never(unreachable)
    return tuple((root, *row) for root, row in zip(roots, assigned, strict=True))


def patch_columns(
    columns: torch.Tensor, expected_patches: int
) -> tuple[torch.Tensor, ...]:
    """Split contiguous visual-token runs without losing carried-cache offsets."""
    boundaries = (columns[1:] != columns[:-1] + 1).nonzero().flatten() + 1
    parts = tuple(torch.tensor_split(columns, boundaries.cpu()))
    if len(parts) != expected_patches or any(part.numel() == 0 for part in parts):
        raise LayoutError(f"Expected {expected_patches} visual runs; got {len(parts)}")
    return parts
