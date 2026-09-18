"""Training-free coordinate linkage between child and parent visual KV tokens."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def relation_spatial_context_keep(
    *,
    keep: np.ndarray,
    score: np.ndarray,
    roi_of_token: np.ndarray,
    parent_roi: np.ndarray,
    roi_grids: Sequence[tuple[int, int]],
    roi_boxes: Sequence[tuple[int, int, int, int] | None],
    radius: int,
    per_parent_cap: int,
) -> np.ndarray:
    """Add parent tokens spatially aligned with surviving child visual tokens."""
    expanded = np.asarray(keep, dtype=bool).copy()
    if radius < 0 or per_parent_cap < 1:
        return expanded
    for child_roi, parent in enumerate(parent_roi.tolist()):
        if parent < 0 or child_roi >= len(roi_grids) or parent >= len(roi_grids):
            continue
        if child_roi >= len(roi_boxes) or parent >= len(roi_boxes):
            continue
        child_box = roi_boxes[child_roi]
        parent_box = roi_boxes[parent]
        if child_box is None or parent_box is None:
            continue
        child_height, child_width = roi_grids[child_roi]
        parent_height, parent_width = roi_grids[parent]
        if min(child_height, child_width, parent_height, parent_width) < 1:
            continue
        parent_x, parent_y, parent_w, parent_h = parent_box
        child_x, child_y, child_w, child_h = child_box
        if parent_w < 1 or parent_h < 1 or child_w < 1 or child_h < 1:
            continue
        child_indices = np.flatnonzero(roi_of_token == child_roi)
        parent_indices = np.flatnonzero(roi_of_token == parent)
        if child_indices.size == 0 or parent_indices.size == 0:
            continue
        candidate_priority: dict[int, float] = {}
        for child_position in child_indices[expanded[child_indices]]:
            local_index = int(np.searchsorted(child_indices, child_position))
            child_row, child_col = divmod(
                local_index % (child_height * child_width), child_width
            )
            level0_x = child_x + (child_col + 0.5) * child_w / child_width
            level0_y = child_y + (child_row + 0.5) * child_h / child_height
            parent_col = min(
                parent_width - 1,
                max(0, int((level0_x - parent_x) * parent_width / parent_w)),
            )
            parent_row = min(
                parent_height - 1,
                max(0, int((level0_y - parent_y) * parent_height / parent_h)),
            )
            for row in range(
                max(0, parent_row - radius), min(parent_height, parent_row + radius + 1)
            ):
                for col in range(
                    max(0, parent_col - radius), min(parent_width, parent_col + radius + 1)
                ):
                    local_parent = row * parent_width + col
                    if local_parent >= parent_indices.size:
                        continue
                    parent_position = int(parent_indices[local_parent])
                    candidate_priority[parent_position] = max(
                        candidate_priority.get(parent_position, float("-inf")),
                        float(score[child_position]),
                    )
        ranked = sorted(
            candidate_priority,
            key=lambda position: (-candidate_priority[position], position),
        )
        expanded[ranked[:per_parent_cap]] = True
    return expanded
