"""Question-guided, rank-adaptive visual KV selection from SparseVLM."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SparseVLMSelection:
    """Observable decisions made by the SparseVLM rule."""

    rater_count: int
    rank: int
    drop_count: int
    visual_scores: torch.Tensor


@dataclass(frozen=True, slots=True)
class SparseVLMPathologyLayout:
    """Crop geometry used to preserve local and parent-scale context."""

    grid_shapes: tuple[tuple[int, int], ...]
    boxes: tuple[tuple[int, int, int, int] | None, ...]
    magnifications: tuple[int, ...]
    parent_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SparseVLMPathologySelection:
    """Final budget and pathology-context additions to a SparseVLM mask."""

    base_keep_count: int
    final_keep_count: int
    spatial_anchor_count: int
    neighbor_count: int
    parent_anchor_count: int
    context_added_count: int
    base_tokens_replaced_count: int


def _question_rows(
    attention: torch.Tensor, question_spans: list[tuple[int, int]] | None
) -> torch.Tensor:
    """Restrict ``[heads, text, vision]`` attention to question text rows."""
    if not question_spans:
        return attention
    indices = [
        index
        for start, end in question_spans
        for index in range(max(0, start), min(end, attention.shape[1]))
    ]
    if not indices:
        return attention
    return attention[:, torch.tensor(indices, device=attention.device), :]


@torch.no_grad()
def sparsevlm_keep_mask(
    query_to_vision: dict[int, torch.Tensor],
    *,
    question_spans: list[tuple[int, int]] | None = None,
    lambda_scale: float = 0.5,
) -> tuple[torch.Tensor, SparseVLMSelection]:
    """Apply SparseVLM's text-rater and rank-adaptive selection at one boundary.

    The original method progressively applies this rule at decoder layers. This
    function adapts one application to the Reasoner boundary, where the visual
    KV is physically compacted before latent reasoning begins.
    """
    if not query_to_vision:
        raise ValueError("SparseVLM requires question-to-vision attention")
    if not 0.0 <= lambda_scale <= 1.0:
        raise ValueError("SparseVLM lambda must lie in [0, 1]")

    matrices = [
        _question_rows(attention.float(), question_spans).mean(dim=0)
        for attention in query_to_vision.values()
        if attention.ndim == 3 and attention.shape[-1] > 0
    ]
    if not matrices:
        raise ValueError("SparseVLM attention must have shape [heads, text, vision]")
    priority = torch.cat(matrices, dim=0)
    visual_count = priority.shape[1]
    row_mass = priority.mean(dim=1)
    raters = row_mass >= row_mass.mean()
    if not bool(raters.any()):
        raters[row_mass.argmax()] = True
    rater_priority = priority[raters]
    rank = int(torch.linalg.matrix_rank(rater_priority).item())
    drop_count = min(
        visual_count - 1,
        max(0, int(lambda_scale * (visual_count - rank))),
    )
    keep_count = visual_count - drop_count
    score = rater_priority.mean(dim=0)
    order = torch.argsort(score, descending=True, stable=True)
    keep = torch.zeros(visual_count, dtype=torch.bool, device=priority.device)
    keep[order[:keep_count]] = True
    return keep, SparseVLMSelection(
        rater_count=int(raters.sum().item()),
        rank=rank,
        drop_count=drop_count,
        visual_scores=score,
    )


def _local_indices(
    row: int, col: int, height: int, width: int
) -> tuple[int, ...]:
    """Return in-bounds four-connected grid neighbors in stable order."""
    return tuple(
        next_row * width + next_col
        for next_row, next_col in (
            (row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)
        )
        if 0 <= next_row < height and 0 <= next_col < width
    )


@torch.no_grad()
def sparsevlm_pathology_keep_mask(
    base_keep: torch.Tensor,
    visual_scores: torch.Tensor,
    layout: SparseVLMPathologyLayout,
) -> tuple[torch.Tensor, SparseVLMPathologySelection]:
    """Add WSI local/scale anchors without replacing SparseVLM's adaptive budget.

    One strongest token per coarse tile and its strongest four-connected
    neighbor form a light spatial scaffold. A high-magnification crop also
    reserves the parent-scale token at its mapped center when crop geometry
    confirms the parent relation. These context anchors are mandatory; score
    ranking fills the original rank-adaptive budget, expanding it only when
    the scaffold itself is larger.
    """
    if base_keep.ndim != 1 or visual_scores.ndim != 1:
        raise ValueError("SparseVLM pathology masks and scores must be one-dimensional")
    if base_keep.numel() != visual_scores.numel():
        raise ValueError("SparseVLM pathology scores must match the visual mask")

    token_count = int(base_keep.numel())
    grid_shapes = layout.grid_shapes
    offsets: list[int] = []
    offset = 0
    for height, width in grid_shapes:
        if height < 1 or width < 1:
            raise ValueError("SparseVLM pathology grids must be positive")
        offsets.append(offset)
        offset += height * width
    if offset != token_count:
        raise ValueError("SparseVLM pathology grids must cover all visual tokens")
    image_count = len(grid_shapes)
    if not (
        len(layout.boxes) == image_count
        and len(layout.magnifications) == image_count
        and len(layout.parent_indices) == image_count
    ):
        raise ValueError("SparseVLM pathology crop metadata must match its grids")

    # Crop-level bookkeeping is Python-driven; move only the 1-D score vector
    # once so per-tile argmaxes do not synchronize the GPU hundreds of times.
    scores = visual_scores.detach().float().cpu()
    mandatory = torch.zeros(token_count, dtype=torch.bool, device=base_keep.device)
    spatial_anchors = 0
    neighbors = 0
    parent_anchors = 0

    for image_index, (height, width) in enumerate(grid_shapes):
        image_offset = offsets[image_index]
        row_bins = min(4, height)
        col_bins = min(4, width)
        for row_bin in range(row_bins):
            row_start = row_bin * height // row_bins
            row_end = (row_bin + 1) * height // row_bins
            for col_bin in range(col_bins):
                col_start = col_bin * width // col_bins
                col_end = (col_bin + 1) * width // col_bins
                tile = [
                    image_offset + row * width + col
                    for row in range(row_start, row_end)
                    for col in range(col_start, col_end)
                ]
                anchor = max(tile, key=lambda index: float(scores[index].item()))
                if not bool(mandatory[anchor]):
                    spatial_anchors += 1
                mandatory[anchor] = True
                row, col = divmod(anchor - image_offset, width)
                neighbor_candidates = _local_indices(row, col, height, width)
                if neighbor_candidates:
                    neighbor = max(
                        neighbor_candidates,
                        key=lambda index: float(scores[image_offset + index].item()),
                    )
                    absolute_neighbor = image_offset + neighbor
                    if not bool(mandatory[absolute_neighbor]):
                        neighbors += 1
                    mandatory[absolute_neighbor] = True

    for child_index, parent_index in enumerate(layout.parent_indices):
        if not (0 <= parent_index < image_count) or parent_index == child_index:
            continue
        child_mag = layout.magnifications[child_index]
        parent_mag = layout.magnifications[parent_index]
        child_box = layout.boxes[child_index]
        parent_box = layout.boxes[parent_index]
        if child_mag <= parent_mag or child_box is None or parent_box is None:
            continue
        child_x, child_y, child_width, child_height = child_box
        parent_x, parent_y, parent_width, parent_height = parent_box
        if min(child_width, child_height, parent_width, parent_height) <= 0:
            continue
        center_x = child_x + child_width / 2
        center_y = child_y + child_height / 2
        if not (
            parent_x <= center_x < parent_x + parent_width
            and parent_y <= center_y < parent_y + parent_height
        ):
            continue
        parent_height_tokens, parent_width_tokens = grid_shapes[parent_index]
        col = min(
            parent_width_tokens - 1,
            max(0, int((center_x - parent_x) / parent_width * parent_width_tokens)),
        )
        row = min(
            parent_height_tokens - 1,
            max(0, int((center_y - parent_y) / parent_height * parent_height_tokens)),
        )
        candidates = [
            offsets[parent_index] + neighbor_row * parent_width_tokens + neighbor_col
            for neighbor_row in range(max(0, row - 1), min(parent_height_tokens, row + 2))
            for neighbor_col in range(max(0, col - 1), min(parent_width_tokens, col + 2))
        ]
        if not candidates:
            continue
        selected_parent = max(candidates, key=lambda index: float(scores[index].item()))
        parent_anchors += 1
        mandatory[selected_parent] = True

    base_mask = base_keep.to(device=mandatory.device, dtype=torch.bool)
    context_added_count = int((mandatory & ~base_mask).sum().item())
    keep = mandatory.clone()
    target_count = max(int(base_keep.sum().item()), int(mandatory.sum().item()))
    if int(keep.sum().item()) < target_count:
        order = torch.argsort(scores, descending=True, stable=True).to(keep.device)
        available = order[~keep[order]]
        keep[available[: target_count - int(keep.sum().item())]] = True
    return keep, SparseVLMPathologySelection(
        base_keep_count=int(base_keep.sum().item()),
        final_keep_count=int(keep.sum().item()),
        spatial_anchor_count=spatial_anchors,
        neighbor_count=neighbors,
        parent_anchor_count=parent_anchors,
        context_added_count=context_added_count,
        base_tokens_replaced_count=int((base_mask & ~keep).sum().item()),
    )
