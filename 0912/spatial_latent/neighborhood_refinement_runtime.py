"""Base-preserving reads of already-created regional latent K/V rows."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from spatial_latent.grouping import LayoutError
from spatial_latent.relation_attention import RelationStep, relation_step_sdpa
from spatial_latent.runtime import SpatialExperiment, StepTrace
from vision_text_mas import latent_hybrid_variants as hybrid


def refinement_definition(
    columns: torch.Tensor,
    *,
    step_index: int,
    gain: float,
) -> RelationStep:
    """Keep Base visual evidence while only even regional steps read prior latent."""
    return RelationStep(
        columns,
        include_previous=step_index % 2 == 1,
        log_gain=math.log(gain),
    )


class NeighborhoodRefinementExperiment(SpatialExperiment):
    """Read each regional latent once more with its unchanged visual neighborhood."""

    @contextmanager
    def latent_step(
        self,
        backbone: hybrid._VariantBackbone,
        vision_columns: torch.Tensor,
        visual_patch_ids: torch.Tensor | None = None,
        parent_patch_indices: tuple[int, ...] = (),
    ) -> Iterator[None]:
        """Preserve Base's four repeated groups and vary only latent-KV reads."""
        del visual_patch_ids, parent_patch_indices
        if not getattr(backbone, "_prune_morphology_enabled", False):
            yield None
            return
        step_index = len(self.steps)
        if step_index >= 10:
            raise LayoutError("Neighborhood refinement requires exactly 10 steps")
        if step_index == 0:
            self._prepare(backbone, vision_columns)
        if step_index < 8:
            group = self.groups[step_index // 2]
            columns = torch.cat([self.columns[index] for index in group])
            definition = refinement_definition(
                columns,
                step_index=step_index,
                gain=self.gain,
            )
            patch_indices = group
            visual_tokens = int(columns.numel())
        elif step_index == 8:
            definition = RelationStep(
                vision_columns[:0],
                False,
                math.log(self.gain),
                tail_offsets=(2, 4, 6, 8),
            )
            patch_indices = ()
            visual_tokens = 0
        else:
            definition = RelationStep(vision_columns[:0], True, math.log(self.gain))
            patch_indices = ()
            visual_tokens = 0
        with relation_step_sdpa(definition) as probe:
            yield None
        if probe.applied_calls != 13:
            raise LayoutError(
                f"Step {step_index}: neighborhood SDPA fired {probe.applied_calls}, "
                "expected 13"
            )
        self.steps.append(
            StepTrace(
                step=step_index + 1,
                patch_indices=patch_indices,
                visual_tokens=visual_tokens,
                biased_layer_calls=probe.applied_calls,
                cache_tokens=probe.key_count,
            )
        )
