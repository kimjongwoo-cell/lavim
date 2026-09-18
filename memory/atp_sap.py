"""Training-free ATP/SAP-style visual-key selection for WSI cache boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ATPSAPSelection:
    """Inspectable adaptive threshold and retained-token counts."""

    redundant_threshold: float
    retained_by_redundancy: int
    retained_by_spatial_scaffold: int
    retained_total: int
    layer_thresholds: tuple[tuple[int, float], ...]
    retained_by_layer: tuple[tuple[int, int], ...]


def _normalize(values: torch.Tensor) -> torch.Tensor:
    """Map one non-empty score vector to [0, 1], keeping constant scores tied."""
    low = values.min()
    span = values.max() - low
    if float(span) <= torch.finfo(values.dtype).eps:
        return torch.ones_like(values)
    return (values - low) / span


def _cross_scores(
    attention_by_layer: dict[int, torch.Tensor],
    question_spans: list[tuple[int, int]] | None,
) -> dict[int, torch.Tensor]:
    """Return one question-to-visual attention score vector per decoder layer."""
    scores: dict[int, torch.Tensor] = {}
    for layer, attention in sorted(attention_by_layer.items()):
        if attention.ndim != 3 or attention.shape[-1] == 0:
            continue
        selected = attention.float()
        if question_spans:
            indices = [
                row
                for start, end in question_spans
                for row in range(max(0, start), min(end, selected.shape[1]))
            ]
            if indices:
                selected = selected[:, torch.tensor(indices, device=selected.device), :]
        scores[layer] = selected.mean(dim=(0, 1))
    if not scores:
        raise ValueError("ATP-SAP requires text-to-visual attention scores")
    return scores


def _spatial_scaffold(
    redundant_score: torch.Tensor,
    grid_shapes: tuple[tuple[int, int], ...],
    *,
    spatial_token_indices: tuple[torch.Tensor, ...] | None = None,
    cells_per_axis: int = 4,
) -> torch.Tensor:
    """Keep one high-scoring available token per occupied coarse spatial cell."""
    scaffold = torch.zeros_like(redundant_score, dtype=torch.bool)
    offset = 0
    if spatial_token_indices is not None and len(spatial_token_indices) != len(grid_shapes):
        raise ValueError("ATP-SAP spatial indices do not match image grids")
    for image_index, (height, width) in enumerate(grid_shapes):
        if height < 1 or width < 1:
            raise ValueError("ATP-SAP image grids do not match visual-token count")
        full_count = height * width
        indices = (
            spatial_token_indices[image_index].to(
                device=redundant_score.device, dtype=torch.long
            ).flatten()
            if spatial_token_indices is not None
            else torch.arange(full_count, device=redundant_score.device)
        )
        count = int(indices.numel())
        if bool((indices < 0).any()) or bool((indices >= full_count).any()):
            raise ValueError("ATP-SAP spatial token index is outside its image grid")
        if offset + count > redundant_score.numel():
            raise ValueError("ATP-SAP image grids do not match visual-token count")
        image_scores = redundant_score[offset:offset + count]
        rows = torch.div(indices, width, rounding_mode="floor")
        cols = indices.remainder(width)
        row_cells = min(cells_per_axis, height)
        col_cells = min(cells_per_axis, width)
        row_bins = torch.div(rows * row_cells, height, rounding_mode="floor")
        col_bins = torch.div(cols * col_cells, width, rounding_mode="floor")
        for row_index in range(row_cells):
            for col_index in range(col_cells):
                candidates = torch.nonzero(
                    (row_bins == row_index) & (col_bins == col_index),
                    as_tuple=True,
                )[0]
                if candidates.numel() == 0:
                    continue
                local = candidates[image_scores[candidates].argmax()]
                scaffold[offset + local] = True
        offset += count
    if offset != redundant_score.numel():
        raise ValueError("ATP-SAP image grids leave unmatched visual tokens")
    return scaffold


@torch.no_grad()
def atp_sap_keep_mask(
    visual_self_score: torch.Tensor | dict[int, torch.Tensor],
    text_to_visual_by_layer: dict[int, torch.Tensor],
    *,
    question_spans: list[tuple[int, int]] | None,
    grid_shapes: tuple[tuple[int, int], ...],
    spatial_token_indices: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ATPSAPSelection]:
    """Combine ATP-style redundancy scores with SAP coarse spatial coverage.

    The learned ATP threshold heads are unavailable for this frozen Qwen model.
    This training-free adaptation substitutes a per-instance mean-plus-half-std
    threshold; it does not claim to reproduce the learned ATP module.
    """
    cross_scores = _cross_scores(text_to_visual_by_layer, question_spans)
    if isinstance(visual_self_score, dict):
        self_scores = visual_self_score
        if not self_scores:
            raise ValueError("ATP-SAP requires visual self-attention scores")
        fallback_self_score = torch.stack(
            tuple(score.float().flatten() for score in self_scores.values())
        ).mean(dim=0)
    else:
        fallback_self_score = visual_self_score.float().flatten()
        self_scores = {}
    if fallback_self_score.numel() == 0:
        raise ValueError("ATP-SAP self score must not be empty")

    layer_scores: list[torch.Tensor] = []
    layer_masks: list[torch.Tensor] = []
    layer_thresholds: list[tuple[int, float]] = []
    layer_counts: list[tuple[int, int]] = []
    for layer, cross_score in cross_scores.items():
        self_score = self_scores.get(layer, fallback_self_score).float().flatten()
        if self_score.shape != cross_score.shape:
            raise ValueError("ATP-SAP self/cross scores must align on visual tokens")
        redundant_score = 0.5 * (_normalize(self_score) + _normalize(cross_score))
        threshold = redundant_score.mean() + 0.5 * redundant_score.std(unbiased=False)
        redundant_keep = redundant_score >= threshold
        layer_scores.append(redundant_score)
        layer_masks.append(redundant_keep)
        layer_thresholds.append((layer, float(threshold.item())))
        layer_counts.append((layer, int(redundant_keep.sum().item())))

    redundant_score = torch.stack(layer_scores).mean(dim=0)
    redundant_keep = torch.stack(layer_masks).any(dim=0)
    spatial_keep = _spatial_scaffold(
        redundant_score,
        grid_shapes,
        spatial_token_indices=spatial_token_indices,
    )
    keep = redundant_keep | spatial_keep
    if not bool(keep.any()):
        keep[redundant_score.argmax()] = True
    return keep, ATPSAPSelection(
        redundant_threshold=sum(value for _, value in layer_thresholds) / len(layer_thresholds),
        retained_by_redundancy=int(redundant_keep.sum().item()),
        retained_by_spatial_scaffold=int(spatial_keep.sum().item()),
        retained_total=int(keep.sum().item()),
        layer_thresholds=tuple(layer_thresholds),
        retained_by_layer=tuple(layer_counts),
    )
