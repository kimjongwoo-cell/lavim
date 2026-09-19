"""Training-free pathology-aware weights for retained visual KV tokens."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PathologyQuestionProfile:
    """Question-conditioned coarse/fine evidence routing coefficients."""

    enabled: bool
    coarse_boost: float
    fine_boost: float
    alpha_scale: float


def pathology_question_profile(prompt: str) -> PathologyQuestionProfile:
    """Route only morphology-grounded questions to the appropriate WSI scale."""
    question = prompt.splitlines()[0].lower().replace("_", " ")
    unsupported = (
        "estrogen receptor",
        "progesterone receptor",
        "her2",
        "receptor status",
        "survival",
        "vital status",
    )
    if any(term in question for term in unsupported):
        return PathologyQuestionProfile(False, 0.0, 0.0, 0.0)
    if "grade" in question:
        return PathologyQuestionProfile(True, 1.0, 0.6, 0.2)
    if "necrosis" in question and any(
        term in question for term in ("size", "extent", "percent", "%")
    ):
        return PathologyQuestionProfile(True, 1.0, 0.2, 0.2)
    architecture = (
        "size",
        "stage",
        "invasion",
        "margin",
        "architecture",
        "boundary",
    )
    if any(term in question for term in architecture):
        return PathologyQuestionProfile(True, 1.0, 0.2, 1.0)
    fine_morphology = ("diagnosis", "histologic", "subtype", "carcinoma", "tumor type")
    if any(term in question for term in fine_morphology):
        return PathologyQuestionProfile(True, 0.2, 1.0, 1.0)
    return PathologyQuestionProfile(True, 0.4, 0.6, 0.5)


def pathology_visual_patch_ids(
    columns: torch.Tensor,
    spans: object,
) -> torch.Tensor:
    """Map each retained visual KV column to its source WSI patch index.

    The Navigator materializes the patch hierarchy, and the cache span order is
    the authoritative patch order. This helper deliberately has no scoring:
    it only preserves the known coarse--fine relationship for attention
    reallocation at the next agent handoff.
    """
    if not isinstance(spans, list):
        return torch.empty(0, dtype=torch.long)
    retained_per_patch = [
        int(count)
        for token_type, count in spans
        if isinstance(token_type, str) and token_type == "V" and isinstance(count, int)
    ]
    if sum(retained_per_patch) != int(columns.numel()):
        return torch.empty(0, dtype=torch.long)
    pieces = [
        torch.full((count,), patch_index, dtype=torch.long)
        for patch_index, count in enumerate(retained_per_patch)
    ]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)


def build_pathology_reallocation_weights(
    columns: torch.Tensor,
    *,
    retained_per_patch: Sequence[int],
    original_per_patch: Sequence[int],
    magnifications: Sequence[int],
    parent_indices: Sequence[int],
    parent_anchor_strength: float,
    fine_evidence_boost: float = 0.0,
    coarse_evidence_boost: float = 0.0,
) -> torch.Tensor:
    """Weight retained evidence by compression and known coarse--fine ancestry.

    A patch compressed by a factor of ``r`` gets ``1 + log2(r)`` weight.  When
    a retained fine patch is strongly compressed, its known coarse parent gets
    a bounded anchor boost; unrelated patches are untouched.  The function
    never introduces columns, so it cannot undo the pruning budget.
    """
    if not 0.0 <= parent_anchor_strength <= 1.0:
        raise ValueError("parent anchor strength must be in [0, 1]")
    if fine_evidence_boost < 0.0:
        raise ValueError("fine evidence boost must be non-negative")
    if coarse_evidence_boost < 0.0:
        raise ValueError("coarse evidence boost must be non-negative")
    n_patches = len(retained_per_patch)
    if not (
        len(original_per_patch) == len(magnifications) == len(parent_indices) == n_patches
    ):
        raise ValueError("patch metadata must have the same length")
    if sum(int(count) for count in retained_per_patch) != int(columns.numel()):
        raise ValueError("retained patch counts must align with visual columns")

    patch_weights: list[float] = []
    for before, after in zip(original_per_patch, retained_per_patch, strict=True):
        if before < 1 or after < 1 or after > before:
            raise ValueError("patch token counts must satisfy 1 <= retained <= original")
        patch_weights.append(1.0 + math.log2(float(before) / float(after)))

    for patch, magnification in enumerate(magnifications):
        if magnification > 5:
            patch_weights[patch] += fine_evidence_boost
        else:
            patch_weights[patch] += coarse_evidence_boost

    for child, parent in enumerate(parent_indices):
        if magnifications[child] <= 5 or not 0 <= int(parent) < n_patches:
            continue
        child_weight = patch_weights[child]
        parent_index = int(parent)
        parent_weight = patch_weights[parent_index]
        patch_weights[parent_index] += parent_anchor_strength * max(
            0.0,
            child_weight - parent_weight,
        )

    pieces = [
        torch.full((int(count),), patch_weights[index], dtype=torch.float32)
        for index, count in enumerate(retained_per_patch)
    ]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)
